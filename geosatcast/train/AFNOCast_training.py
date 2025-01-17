import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from geosatcast.models.autoencoder import Encoder, Decoder, VAE
from geosatcast.models.afnocast import AFNOCastLatent, AFNOCast
from torch.nn.parallel import DistributedDataParallel as DDP
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
from distribute_training import set_global_seed, setup_logger, get_dataloader, setup_distributed, load_checkpoint, save_model, load_vae, reduce_tensor 

def validate(model, val_loader, device, logger, writer, config, epoch):
    model.eval()
    total_loss = 0
    total_loss_per_ch = torch.zeros((11,))
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, loss_per_ch = compute_loss(afnocast, batch, device)
            total_loss += loss.item()
            total_loss_per_ch += loss_per_ch.cpu()
            num_batches += 1

    # Average the losses across all GPUs
    total_loss = reduce_tensor(torch.tensor(total_loss, device=device), device)
    total_loss_per_ch = reduce_tensor(total_loss_per_ch.to(device), device)

    avg_loss = total_loss / num_batches
    avg_loss_per_ch = total_loss_per_ch / num_batches
    
    if dist.get_rank() == 0:
        loss_str = ", ".join([f"{avg_loss_per_ch[i].item():.4f}" for i in range(11)])
        logger.info(f"Validation: Avg Loss {avg_loss:.4f}, Loss per ch {loss_str}")

        # Log validation losses to TensorBoard
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        for c in range(11):
            writer.add_scalar(f"Val/Loss_{c}", avg_loss_per_ch[c].item(), epoch)
    return avg_loss

def compute_loss(model, batch, device):
    in_steps = model.afnocast_latent.in_steps
    
    # open batch
    x, _, inv, sza = batch
    x = x.to(device, non_blocking=True)
    inv = inv.to(device, non_blocking=True)
    sza = sza.to(device, non_blocking=True)

    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
    # split input output sequences
    x, y = x[:,:,:in_steps], x[:,:,in_steps:]
    # encode y to compute loss in the latent space
    y = model.vae.encode(y)[0]
    y_pred = model.latent_forward(x, inv)
    res = (y - y_pred).abs()
    loss = res.mean()
    loss_per_ch = res.mean(dim=(0,2,3,4))
    return loss, loss_per_ch

def train(
    rank, 
    config, 
    afnocast,
    train_loader, 
    val_loader, 
    train_sampler, 
    optimizer, 
    scheduler, 
    logger,
    writer):

    device = f"cuda:{rank}"
    afnocast.to(device)
    afnocast = DDP(afnocast, device_ids=[rank])

    # Initialize GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Check if we should resume from a checkpoint
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(afnocast, optimizer, scheduler, checkpoint_path, logger, rank)

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")
    for epoch in range(start_epoch, config["Trainer"]["max_epochs"]):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        print(f"seed: {seed}")
        
        train_sampler.set_epoch(epoch)
        afnocast.train()
        
        total_loss = 0.0
        total_loss_per_ch = torch.zeros((11,))
        num_batches = 0
        
        in_steps = afnocast.afnocast_latent.in_steps
        for batch in train_loader:
            optimizer.zero_grad()
            # Mixed precision forward and loss computation
            with torch.amp.autocast('cuda'):
                loss, loss_per_ch = compute_loss(afnocast, batch, device)
            
            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss for averaging
            total_loss += loss.item()
            total_loss_per_ch += loss_per_ch.cpu()
            num_batches += 1

            if rank == 0:
                if num_batches % 50 == 0:
                    loss_str = ", ".join([f"{loss_per_ch[i].item():.4f}" for i in range(11)])
                    logger.info(f"Epoch {epoch}, Step {num_batches}: Loss {loss:.4f}, Loss Per CH {loss_str}")
                    writer.add_scalar("Train_minibatch/Loss", loss.item(), epoch * tot_num_batches + num_batches)
                    for c in range(11):
                        writer.add_scalar(f"Train_minibatch/Loss_{c}", loss_per_ch[c].item(), epoch * tot_num_batches + num_batches)

        # Average the losses across all GPUs
        total_loss = reduce_tensor(torch.tensor(total_loss, device=device), rank)
        total_loss_per_ch = reduce_tensor(total_loss_per_ch.to(device), rank)

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_loss_per_ch = total_loss_per_ch / num_batches
            # Log to TensorBoard
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            for c in range(11):
                writer.add_scalar(f"Train/Rec_Loss_{c}", avg_loss_per_ch[c], epoch)
            
            loss_str = ", ".join([f"{avg_loss_per_ch[i].item():.4f}" for i in range(11)])
            logger.info(f"Epoch {epoch}: Avg Loss {avg_loss:.4f}, Avg Loss Per CH {loss_str}")
            # Save the model at the end of each epoch
            save_model(afnocast, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch, config)

        val_loss = validate(afnocast, val_loader, device, logger, writer, config, epoch)
        scheduler.step(val_loss)
        
        # log the learning rate
        lr = scheduler.get_last_lr()[0] 
        logger.info(f"Epoch {epoch}: Learning Rate {lr:.4f}")
        if rank == 0:
            writer.add_scalar(f"Train/LR", lr, epoch)

def main():
    CONFIG_PATH = sys.argv[1]
    with open(CONFIG_PATH, "r") as f:
        config = load(f, Loader)

    local_rank = setup_distributed()
    
    log_dir = config["Trainer"]["log_dir"]
    logger = setup_logger(log_dir, local_rank, experiment_name=config["Experiment"])
    
    # TensorBoard writer (only for rank 0)
    writer = None
    if local_rank == 0:
        tensor_log_dir = os.path.join(log_dir, config["Experiment"])
        writer = SummaryWriter(log_dir=tensor_log_dir)
        logger.info(f"TensorBoard logs will be saved to {tensor_log_dir}")


    data_config = config['Dataset']
    train_years = data_config.pop("train_years")
    train_length = data_config.pop("train_length")
    val_years = data_config.pop("val_years")
    val_length = data_config.pop("val_length")
    
    train_dataloader, train_sampler = get_dataloader(
        **data_config, 
        years=train_years,
        length=train_length,
        validation=False)
    
    val_dataloader, _ = get_dataloader(
        **data_config, 
        years=val_years,
        length=val_length,
        validation=True)

    vae = load_vae(config["AFNOCast"].pop("VAE_ckpt_path"))
    afnocast_latent = AFNOCastLatent(**config["AFNOCast"])
    inv_encoder = Encoder(**config["Inv_Encoder"])
    afnocast = AFNOCast(
        afnocast_latent,
        vae,
        inv_encoder)

    optimizer = torch.optim.AdamW(list(afnocast.parameters()) + list(inv_encoder.parameters()), lr=config["Trainer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=config["Trainer"]["opt_patience"])

    train(
        local_rank, 
        config, 
        afnocast,
        train_dataloader, 
        val_dataloader, 
        train_sampler, 
        optimizer, 
        scheduler, 
        logger,
        writer)

if __name__ == "__main__":
    main()