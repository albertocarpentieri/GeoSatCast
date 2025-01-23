import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from geosatcast.data.distributed_dataset import DistributedDataset, WorkerDistributedSampler
from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from torch.utils.tensorboard import SummaryWriter
import logging
import random

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_global_seed(seed):
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch random
    torch.manual_seed(seed)
    # PyTorch on GPUs
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        validation,
        local_rank+1
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
        prefetch_factor=8,
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

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger, rank):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    if rank == 0:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"]
    
    if rank == 0:
        logger.info(f"Resumed training from epoch {start_epoch}.")
    
    return start_epoch

def save_model(model, optimizer, scheduler, dirpath, model_id, epoch, config):
    ckpt_folder = os.path.join(dirpath, f"{model_id}")
    os.makedirs(ckpt_folder, exist_ok=True)
    model_path = os.path.join(ckpt_folder, f"{model_id}_{epoch}.pt")
    if torch.distributed.get_rank() == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config
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

def load_vae(ckpt_path):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

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
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    vae.load_state_dict(state_dict)
    return vae

def load_nowcaster(ckpt_path, return_config=False):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    inv_encoder = Encoder(**config["Inv_Encoder"])
    vae = load_vae(config["VAE_ckpt_path"])

    model_type = config["Model_Type"]
    
    if model_type == "AFNO":
        latent_model = AFNOCastLatent(**config["Model"])

    elif model_type == "NAT":
        latent_model = NATCastLatent(**config["Model"])
    
    elif model_type == "AFNONAT":
        latent_model = AFNONATCastLatent(**config["Model"])

    model = Nowcaster(
        latent_model,
        vae,
        inv_encoder
    )
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    if return_config:
        return model, config
    return model