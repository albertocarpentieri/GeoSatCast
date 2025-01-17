import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from geosatcast.models.autoencoder import Encoder, Decoder, VAE
from geosatcast.utils import sample_from_standard_normal, kl_from_standard_normal, conv_nd
from torch.nn.parallel import DistributedDataParallel as DDP
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
from distribute_training import set_global_seed, setup_logger, get_dataloader, setup_distributed, load_checkpoint, save_model 

def validate(vae, val_loader, device, logger, writer, config, epoch):
    vae.eval()
    total_loss, rec_loss, kl_loss = 0, 0, 0
    rec_loss_per_ch = torch.zeros((11,))
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            x, *_ = batch
            x = x.to(device, non_blocking=True).detach()
            loss, r_loss, k_loss, r_loss_per_ch = compute_loss(vae, config["Loss"]["kl_weight"], x)
            total_loss += loss.item()
            rec_loss += r_loss.item()
            rec_loss_per_ch += r_loss_per_ch.cpu()
            kl_loss += k_loss.item()
            num_batches += 1

    # Average the losses across all GPUs
    total_loss = reduce_tensor(torch.tensor(total_loss).to(device), device)
    rec_loss = reduce_tensor(torch.tensor(rec_loss).to(device), device)
    kl_loss = reduce_tensor(torch.tensor(kl_loss).to(device), device)
    rec_loss_per_ch = reduce_tensor(rec_loss_per_ch.to(device), device)

    avg_loss = total_loss / num_batches
    avg_rec_loss = rec_loss / num_batches
    avg_kl_loss = kl_loss / num_batches
    avg_rec_per_ch_loss = rec_loss_per_ch / num_batches
    
    if dist.get_rank() == 0:
        rec_loss_str = ", ".join([f"{avg_rec_per_ch_loss[i].item():.4f}" for i in range(11)])
        logger.info(f"Validation: Avg Loss {avg_loss:.4f}, Avg Rec Loss {avg_rec_loss:.4f}, Avg KL Loss {avg_kl_loss:.4f}, Rec Loss per ch {rec_loss_str}")

        # Log validation losses to TensorBoard
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Rec_Loss", avg_rec_loss, epoch)
        writer.add_scalar("Val/KL_Loss", avg_kl_loss, epoch)
        for c in range(11):
            writer.add_scalar(f"Val/Rec_Loss_{c}", avg_rec_per_ch_loss[c].item(), epoch)
    return avg_rec_loss

def compute_loss(model, kl_weight, x):
    y_pred, mean, log_var = model(x)
    rec_res = (x - y_pred).abs()
    rec_loss = rec_res.mean()
    rec_loss_per_ch = rec_res.mean(dim=(0,2,3,4))
    kl_loss = kl_from_standard_normal(mean, log_var)
    return (1 - kl_weight) * rec_loss + kl_weight * kl_loss, rec_loss, kl_loss, rec_loss_per_ch

def train(
    rank, 
    config, 
    encoder, 
    decoder,
    vae, 
    train_loader, 
    val_loader, 
    train_sampler, 
    optimizer, 
    scheduler, 
    logger,
    writer):

    device = f"cuda:{rank}"
    vae.to(device)
    vae = DDP(vae, device_ids=[rank])

    # Initialize GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Check if we should resume from a checkpoint
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(vae, optimizer, scheduler, checkpoint_path, logger, rank)

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")
    for epoch in range(start_epoch, config["Trainer"]["max_epochs"]):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        print(f"seed: {seed}")
        
        train_sampler.set_epoch(epoch)
        vae.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl_loss = 0.0
        total_rec_loss_per_ch = torch.zeros((11,))
        num_batches = 0
        
        for batch in train_loader:
            x, *_ = batch
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Mixed precision forward and loss computation
            with torch.amp.autocast('cuda'):
                loss, rec_loss, kl_loss, rec_loss_per_ch = compute_loss(vae, config["Loss"]["kl_weight"], x)
            
            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss for averaging
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            total_rec_loss_per_ch += rec_loss_per_ch.cpu()
            num_batches += 1

            if rank == 0:
                if num_batches % 50 == 0:
                    rec_loss_str = ", ".join([f"{rec_loss_per_ch[i].item():.4f}" for i in range(11)])
                    logger.info(f"Epoch {epoch}, Step {num_batches}: Loss {loss:.4f}, Rec Loss {rec_loss:.4f}, KL Loss {kl_loss:.4f}, Rec Loss Per CH {rec_loss_str}")
                    writer.add_scalar("Train_minibatch/Loss", loss.item(), epoch * tot_num_batches + num_batches)
                    writer.add_scalar("Train_minibatch/Rec_Loss", rec_loss.item(), epoch * tot_num_batches + num_batches)
                    writer.add_scalar("Train_minibatch/KL_Loss", kl_loss.item(), epoch * tot_num_batches + num_batches)
                    for c in range(11):
                        writer.add_scalar(f"Train_minibatch/Rec_Loss_{c}", rec_loss_per_ch[c].item(), epoch * tot_num_batches + num_batches)

        # Average the losses across all GPUs
        total_loss = reduce_tensor(torch.tensor(total_loss).to(device), rank)
        total_rec_loss = reduce_tensor(torch.tensor(total_rec_loss).to(device), rank)
        total_kl_loss = reduce_tensor(torch.tensor(total_kl_loss).to(device), rank)
        total_rec_loss_per_ch = reduce_tensor(total_rec_loss_per_ch.to(device), rank)

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_rec_loss = total_rec_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches
            avg_rec_loss_per_ch = total_rec_loss_per_ch / num_batches
            # Log to TensorBoard
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Train/Rec_Loss", avg_rec_loss, epoch)
            writer.add_scalar("Train/KL_Loss", avg_kl_loss, epoch)
            for c in range(11):
                writer.add_scalar(f"Train/Rec_Loss_{c}", avg_rec_loss_per_ch[c], epoch)
            
            rec_loss_str = ", ".join([f"{avg_rec_loss_per_ch[i].item():.4f}" for i in range(11)])
            logger.info(f"Epoch {epoch}: Avg Loss {avg_loss:.4f}, Avg Rec Loss {avg_rec_loss:.4f}, Avg KL Loss {avg_kl_loss:.4f}, Avg Rec Loss Per CH {rec_loss_str}")
            # Save the model at the end of each epoch
            save_model(vae, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch, config)

        val_loss = validate(vae, val_loader, device, logger, writer, config, epoch)
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
    
    val_dataloader, val_sampler = get_dataloader(
        **data_config, 
        years=val_years,
        length=val_length,
        validation=True)

    encoder_config = config['Encoder']
    decoder_config = config['Decoder']
    vae_config = config['VAE']
    encoder = Encoder(**encoder_config)
    decoder = Decoder(
        in_dim=vae_config['hidden_width'],
        out_dim=encoder_config['in_dim'],
        **decoder_config)
    vae = VAE(encoder,
              decoder,
              **vae_config,
              encoded_channels=encoder_config['max_ch'],
        )
    
    if local_rank == 0:
        print('All models built')
        print(vae)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=config["Trainer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=config["Trainer"]["opt_patience"])

    train(
        local_rank, 
        config, 
        encoder, 
        decoder, 
        vae, 
        train_dataloader, 
        val_dataloader, 
        train_sampler, 
        optimizer, 
        scheduler, 
        logger,
        writer)

if __name__ == "__main__":
    main()
