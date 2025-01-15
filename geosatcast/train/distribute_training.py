import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from geosatcast.models.autoencoder import Encoder, Decoder, VAE
from geosatcast.utils import sample_from_standard_normal, kl_from_standard_normal, conv_nd
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from geosatcast.data.distributed_dataset import DistributedDataset, WorkerDistributedSampler
from yaml import load, Loader
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter
import logging

def setup_logger(log_dir, rank, experiment_name=None):
    """Set up a logger for distributed training."""
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.ERROR)  # Only rank 0 logs debug info

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    experiment_name = "training" if experiment_name is None else experiment_name
    # File handler (only for rank 0)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def get_dataloader(
    data_path,
    invariants_path,
    input_seq_len,           
    validation,
    years,
    length,
    field_size=128,
    num_workers=24,
    batch_size=8,
    ):

    # Get the current GPU and process information
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    dataset = DistributedDataset(
        data_path,
        invariants_path,
        "new_virtual",
        years,
        input_seq_len,
        None,
        np.arange(11),
        field_size,
        length,
        )

    sampler = WorkerDistributedSampler(
        dataset, 
        num_replicas=torch.distributed.get_world_size(),
        rank=local_rank,
        shuffle=not validation,
        seed=0)

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        sampler=sampler)
    
    return dataloader, sampler

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Validate local_rank
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(f"Invalid local_rank {local_rank}. Only {num_gpus} GPUs available.")
    
    device = f"cuda:{local_rank}"
    print(f"Setting device to {device}")
    torch.cuda.set_device(device)
    return local_rank

def load_checkpoint(vae, optimizer, scheduler, checkpoint_path, logger, rank):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    if rank == 0:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vae.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"]
    
    if rank == 0:
        logger.info(f"Resumed training from epoch {start_epoch}.")
    
    return start_epoch

def validate(vae, val_loader, device, logger, writer, config, epoch):
    vae.eval()
    total_loss, rec_loss, kl_loss = 0, 0, 0
    rec_loss_per_ch = torch.zeros((11,))
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            x, *_ = batch
            x = x.to(device, non_blocking=True).detach()
            loss, r_loss, k_loss, r_loss_per_ch = compute_loss(vae, config["Loss"]["kl_weigth"], x)
            total_loss += loss.item()
            rec_loss += r_loss.item()
            rec_loss_per_ch += r_loss_per_ch.cpu()
            kl_loss += k_loss.item()
            num_batches += 1

    # Average the losses across all GPUs
    total_loss = reduce_tensor(torch.tensor(total_loss).to(device), device)
    rec_loss = reduce_tensor(torch.tensor(rec_loss).to(device), device)
    kl_loss = reduce_tensor(torch.tensor(kl_loss).to(device), device)
    rec_loss_per_ch = reduce_tensor(torch.tensor(rec_loss_per_ch).to(device), device)

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
            writer.add_scalar(f"Val/Rec_Loss_{c}", rec_loss_per_ch[c].item(), epoch)
        
        
        writer.add_hparams(
            {
                f"{m}_{k}": config[m].get(k, 0)  # Use a default value if key is missing
                for m in ["Dataset", "Encoder", "Decoder", "VAE", "Loss", "Trainer"]
                for k in config[m]
            },
            {
                "Val/Rec_Loss": avg_rec_loss
            })
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
    for epoch in range(config["Trainer"]["max_epochs"]):
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
        total_rec_loss_per_ch = reduce_tensor(torch.tensor(total_rec_loss_per_ch).to(device), rank)

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
            save_model(vae, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch)

        # Validation and learning rate scheduling
        # if epoch % config["Trainer"]["val_interval"] == 0:
            # if rank == 0:
        val_loss = validate(vae, val_loader, device, logger, writer, config, epoch)
        scheduler.step(val_loss)

def save_model(vae, optimizer, scheduler, dirpath, model_id, epoch):
    model_path = os.path.join(dirpath, f"{model_id}_{epoch}.pt")
    if torch.distributed.get_rank() == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, model_path)
        print(f"Model saved at {model_path}")

def reduce_tensor(tensor, rank):
    """
    Reduce a tensor across all processes and return the sum.
    This function will be used to collect the total loss across all devices.
    """
    # Create a tensor for reduction
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # Average the loss across all devices
    tensor /= dist.get_world_size()
    return tensor

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
