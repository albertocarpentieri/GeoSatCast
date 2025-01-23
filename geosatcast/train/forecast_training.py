import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter

from geosatcast.models.autoencoder import Encoder, Decoder, AutoEncoder
from geosatcast.models.nowcast import NATCastLatent, AFNONATCastLatent, AFNOCastLatent, Nowcaster
from distribute_training import set_global_seed, setup_logger, get_dataloader, setup_distributed, load_checkpoint, save_model, load_vae, reduce_tensor, count_parameters


def validate(model, n_forecast_steps, val_loader, device, logger, writer, config, epoch):
    model.eval()
    total_loss = 0
    total_loss_per_ch = torch.zeros((11,))
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, loss_per_ch = compute_loss(model, batch, n_forecast_steps, device, per_ch=True)
            total_loss += loss.item()
            total_loss_per_ch += loss_per_ch.cpu()
            num_batches += 1

    # Average the losses across all GPUs
    total_loss = reduce_tensor(torch.tensor(total_loss, device=device), device)
    total_loss_per_ch = reduce_tensor(total_loss_per_ch.to(device), device)

    avg_loss = total_loss / num_batches
    avg_loss_per_ch = total_loss_per_ch / num_batches
    
    if dist.get_rank() == 0:
        loss_str = ", ".join([f"{avg_loss_per_ch[i].item():.6f}" for i in range(11)])
        logger.info(f"Validation: Avg Loss {avg_loss:.6f}, Loss per ch {loss_str}")

        # Log validation losses to TensorBoard
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        for c in range(11):
            writer.add_scalar(f"Val/Loss_{c}", avg_loss_per_ch[c].item(), epoch)
    return avg_loss

def compute_loss(model, batch, n_forecast_steps, device, per_ch=False):
    in_steps = model.module.latent_model.in_steps
    
    # open batch
    x, _, inv, sza = batch
    sza = sza[:, :, :in_steps+n_forecast_steps-1]
    x, y = x[:,:,:in_steps], x[:,:,in_steps:in_steps+n_forecast_steps]
    if per_ch:
        x = x.to(device, non_blocking=True).detach()
        inv = inv.to(device, non_blocking=True).detach()
        sza = sza.to(device, non_blocking=True).detach()
    else:
        x = x.to(device, non_blocking=True)
        inv = inv.to(device, non_blocking=True)
        sza = sza.to(device, non_blocking=True)

    y = y.to(device, non_blocking=True).detach()
    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
   
    yhat = model(x, inv, n_steps=n_forecast_steps)
    res = (y - yhat).abs()
    loss = res.mean()
    
    if per_ch:
        loss_per_ch = res.detach().mean(dim=(0,2,3,4))
        return loss, loss_per_ch
    else:
        return loss

def train(
    rank, 
    config, 
    model,
    train_loader, 
    val_loader, 
    train_sampler, 
    optimizer, 
    scheduler, 
    logger,
    writer):

    device = f"cuda:{rank}"
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Initialize GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Check if we should resume from a checkpoint
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger, rank)

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")
    
    n_forecast_steps = int(config["Trainer"].pop("n_steps"))
    
    for epoch in range(start_epoch, config["Trainer"]["max_epochs"]):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        print(f"seed: {seed}")
        
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_loss = 0.0
        total_loss_per_ch = torch.zeros((11,))
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            # Mixed precision forward and loss computation
            with torch.amp.autocast('cuda'):
                loss = compute_loss(model, batch, n_forecast_steps, device)
            
            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss for averaging
            total_loss += loss.item()
            num_batches += 1

            if rank == 0:
                if num_batches % 50 == 0:
                    logger.info(f"Epoch {epoch}, Step {num_batches}: Loss {loss:.6f}")
                    writer.add_scalar("Train_minibatch/Loss", loss.item(), epoch * tot_num_batches + num_batches)

        # Average the losses across all GPUs
        total_loss = reduce_tensor(torch.tensor(total_loss, device=device), rank)

        if rank == 0:
            avg_loss = total_loss / num_batches
            # Log to TensorBoard
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            logger.info(f"Epoch {epoch}: Avg Loss {avg_loss:.6f}")
            # Save the model at the end of each epoch
            save_model(model, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch, config)

        val_loss = validate(model, n_forecast_steps, val_loader, device, logger, writer, config, epoch)
        scheduler.step(val_loss)
        
        # log the learning rate
        lr = scheduler.get_last_lr()[0] 
        logger.info(f"Epoch {epoch}: Learning Rate {lr:.6f}")
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

    
    encoder = Encoder(**config["Encoder"])
    decoder = Decoder(**config["Decoder"])
    autoencoder = AutoEncoder(encoder, decoder)
    in_steps = config["Model"].pop("in_steps")
    model_type = config["Model_Type"]
    
    if model_type == "AFNO":
        latent_model = AFNOCastLatent(**config["Model"])

    elif model_type == "NAT":
        latent_model = NATCastLatent(**config["Model"])
    
    elif model_type == "AFNONAT":
        latent_model = AFNONATCastLatent(**config["Model"])
        print("mode:", latent_model.mode)

    model = Nowcaster(
        latent_model,
        autoencoder,
        in_steps=in_steps
    )
    if local_rank == 0:
        print(model)
        print(count_parameters(encoder))
        print(count_parameters(decoder))
        print(count_parameters(latent_model))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["Trainer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=config["Trainer"]["opt_patience"])

    train(
        local_rank, 
        config, 
        model,
        train_dataloader, 
        val_dataloader, 
        train_sampler, 
        optimizer, 
        scheduler, 
        logger,
        writer)

if __name__ == "__main__":
    main()