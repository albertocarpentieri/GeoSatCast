import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from geosatcast.data.distributed_dataset import DistributedDataset, WorkerDistributedSampler
from geosatcast.models.autoencoder import VAE, Encoder, Decoder, AutoEncoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.models.UNAT import UNAT
from geosatcast.models.predrnn import PredRNN, PredRNN_v2
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    LambdaLR
)
import logging
import random
from typing import Optional, Tuple, Any


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
    data_name="16b_virtual",
    field_size=128,
    num_workers=24,
    batch_size=8,
    prefetch_factor=8,
    seed=0,
    load_full=False,
    mask_sza=True,
    dtype=16,
    add_latlon=False,
    ):

    # Get the current GPU and process information
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    dataset = DistributedDataset(
        data_path,
        invariants_path,
        name=data_name,
        years=years,
        input_len=input_seq_len,
        output_len=None,
        channels=np.arange(11),
        field_size=field_size,
        length=length,
        validation=validation,
        load_full=load_full,
        rank=local_rank+1,
        mask_sza=mask_sza,
        dtype=dtype,
        add_latlon=add_latlon
        )
    
    if seed == "rank":
        seed = rank
    
    sampler = WorkerDistributedSampler(
        dataset, 
        num_samples=length,
        num_replicas=torch.distributed.get_world_size(),
        rank=rank,
        shuffle=not validation,
        seed=seed) # set to zero if you want all processes to go in the same time indices

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        # persistent_workers=True,
        prefetch_factor=prefetch_factor,
        sampler=sampler)
    
    return dataloader, sampler

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Validate rank
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(f"Invalid rank {rank}. Only {num_gpus} GPUs available.")
    
    device = f"cuda:{local_rank}"
    print(f"Setting device to {device}")
    torch.cuda.set_device(device)
    return rank, local_rank

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[LambdaLR],
    cosine_scheduler: Optional[CosineAnnealingLR],
    scheduler: Optional[ReduceLROnPlateau],
    scaler: Optional[torch.cuda.amp.GradScaler],
    checkpoint_path: str,
    logger: Any,
    rank: int
) -> int:
    """
    Loads model, optimizer, scaler, and multiple scheduler states from a checkpoint.
    Returns the epoch from which to resume (already 1-indexed in the checkpoint).
    """
    if rank == 0:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 1) Load model state
    try:
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()})
    except:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 2) Load optimizer state
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]  # This is epoch+1 from saving
    
    else:
        start_epoch = 0

    # 3) Load each scheduler state if present
    if "warmup_scheduler_state_dict" in checkpoint and warmup_scheduler is not None:
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])

    if "cosine_scheduler_state_dict" in checkpoint and cosine_scheduler is not None:
        cosine_scheduler.load_state_dict(checkpoint["cosine_scheduler_state_dict"])

    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 4) Load scaler state if present
    if "scaler_state_dict" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if rank == 0:
        logger.info(f"Resumed training from epoch {start_epoch}.")

    return start_epoch

def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[LambdaLR],
    cosine_scheduler: Optional[CosineAnnealingLR],
    scheduler: Optional[ReduceLROnPlateau],
    dirpath: str,
    model_id: str,
    epoch: int,
    config: dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    """
    Saves model, optimizer, multiple schedulers, scaler, and config to a checkpoint.
    """
    ckpt_folder = os.path.join(dirpath, f"{model_id}")
    os.makedirs(ckpt_folder, exist_ok=True)
    model_path = os.path.join(ckpt_folder, f"{model_id}_{epoch}.pt")

    ckpt = {
        "epoch": epoch + 1,  # so that when loaded, we resume at correct epoch
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config
    }

    # Save each scheduler's state if it exists
    if warmup_scheduler is not None:
        ckpt["warmup_scheduler_state_dict"] = warmup_scheduler.state_dict()

    if cosine_scheduler is not None:
        ckpt["cosine_scheduler_state_dict"] = cosine_scheduler.state_dict()

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    # Save the scaler state if present
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    torch.save(ckpt, model_path)
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

def load_latent_nowcaster(ckpt_path, return_config=False):
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

def load_nowcaster(ckpt_path, return_config=False, in_steps=2):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    encoder = Encoder(**config["Encoder"])
    decoder = Decoder(**config["Decoder"])
    model_type = config["Model_Type"]
    
    if model_type == "AFNO":
        latent_model = AFNOCastLatent(**config["Model"])

    elif model_type == "NAT":
        latent_model = NATCastLatent(**config["Model"])
    
    elif model_type == "AFNONAT":
        latent_model = AFNONATCastLatent(**config["Model"])

    model = Nowcaster(
        latent_model,
        encoder,
        decoder,
        in_steps=in_steps
    )
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    if return_config:
        return model, config
    return model

def load_predrnn(ckpt_path, return_config=False, in_steps=2):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    model = PredRNN_v2(
        **config["Model"],
        in_steps=in_steps
    )
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    if return_config:
        return model, config
    return model

def load_unatcast(ckpt_path, return_config=False, in_steps=2):
    """
    Loads model, optimizer, and scheduler states from a checkpoint.
    """
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    model = UNAT(
        **config["Model"],
        in_steps=in_steps
    )
    state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    if return_config:
        return model, config
    return model

class WarmupLambdaLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, lr_lambda):
        super().__init__(optimizer, lr_lambda)
        self.num_warmup_steps = num_warmup_steps

def Warmup_Scheduler(optimizer, num_warmup_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        return 1.0
    return WarmupLambdaLR(optimizer, num_warmup_steps, lr_lambda)