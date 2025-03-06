#!/usr/bin/env python
import os
import sys
import time
import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR
)
from yaml import load, Loader

from geosatcast.models.autoencoder import Encoder, Decoder, AutoEncoder
from geosatcast.models.nowcast import (
    NATCastLatent,
    AFNONATCastLatent,
    AFNOCastLatent,
    AdaFNOCastLatent,
    DummyLatent,
    Nowcaster,
)
from geosatcast.models.UNAT import UNAT
from geosatcast.models.predrnn import PredRNN_v2

from distribute_training import (
    set_global_seed,
    setup_logger,
    get_dataloader,
    setup_distributed,
    load_vae,
    reduce_tensor,
    count_parameters,
    Warmup_Scheduler,
    save_model,
    load_checkpoint
)

NUM_CHANNELS = 11

def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute the overall L2 norm of gradients of a model's parameters.
    Returns 0.0 if no gradients exist.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    total_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in parameters]), 2
    )
    return total_norm.item()

def log_gradients(model: nn.Module, writer: SummaryWriter, step: int) -> None:
    """
    Log gradients and weights histograms to TensorBoard.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad.detach().cpu(), step)
        writer.add_histogram(f'Weights/{name}', param.detach().cpu(), step)


def validate(
    model: nn.Module,
    n_forecast_steps: int,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Any,
    writer: SummaryWriter,
    config: dict,
    epoch: int
) -> torch.Tensor:
    """
    Run validation, compute average losses, and log per-channel metrics.
    """
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_mae_per_ch = torch.zeros((NUM_CHANNELS,))
    total_mse_per_ch = torch.zeros((NUM_CHANNELS,))
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            mae, mse, mae_per_ch, mse_per_ch = compute_loss(
                model, batch, n_forecast_steps, device, per_ch=True
            )
            total_mae += mae.item()
            total_mse += mse.item()
            total_mae_per_ch += mae_per_ch.cpu()
            total_mse_per_ch += mse_per_ch.cpu()
            num_batches += 1

    total_mae = reduce_tensor(torch.tensor(total_mae, device=device), device)
    total_mse = reduce_tensor(torch.tensor(total_mse, device=device), device)
    total_mae_per_ch = reduce_tensor(total_mae_per_ch.to(device), device)
    total_mse_per_ch = reduce_tensor(total_mse_per_ch.to(device), device)

    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    avg_mae_per_ch = total_mae_per_ch / num_batches
    avg_mse_per_ch = total_mse_per_ch / num_batches

    if dist.get_rank() == 0:
        ch_mae_str = ", ".join([f"{avg_mae_per_ch[i].item():.6f}" for i in range(NUM_CHANNELS)])
        ch_mse_str = ", ".join([f"{avg_mse_per_ch[i].item():.6f}" for i in range(NUM_CHANNELS)])
        logger.info(f"Validation: Avg MAE {avg_mae:.6f}, MAE per channel: {ch_mae_str}")
        logger.info(f"Validation: Avg MSE {avg_mse:.6f}, MSE per channel: {ch_mse_str}")
        writer.add_scalar("Val/MAE", avg_mae, epoch)
        writer.add_scalar("Val/MSE", avg_mse, epoch)
        for c in range(NUM_CHANNELS):
            writer.add_scalar(f"Val/MAE_{c}", avg_mae_per_ch[c].item(), epoch)
            writer.add_scalar(f"Val/MSE_{c}", avg_mse_per_ch[c].item(), epoch)

    monitor = config["Loss"].get("Monitor", "L1")
    if monitor == "L1":
        avg_loss = avg_mae
    elif monitor == "L2":
        avg_loss = avg_mse
    else:
        avg_loss = avg_mae  # fallback

    return avg_loss

def compute_loss(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    n_forecast_steps: int,
    device: str,
    per_ch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute loss given a batch.
    Splits the input into x and y based on in_steps and forecast steps.
    """
    in_steps = model.module.in_steps  # DDP wrapper, so access via model.module
    x, t, inv, sza = batch

    # Slice the data accordingly
    sza = sza[:, :, :in_steps + n_forecast_steps - 1]
    x, y = x[:, :, :in_steps], x[:, :, in_steps: in_steps + n_forecast_steps]

    if per_ch:
        x = x.to(device, non_blocking=True).float()
        inv = inv.to(device, non_blocking=True).float()
        sza = sza.to(device, non_blocking=True).float()
    else:
        x = x.to(device, non_blocking=True).float()
        inv = inv.to(device, non_blocking=True).float()
        sza = sza.to(device, non_blocking=True).float()

    y = y.to(device, non_blocking=True).detach().float()
    # Merge 'inv' and 'sza' along channel dimension
    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)

    # Forward
    yhat = model(x, inv, n_steps=n_forecast_steps)
    res = (y - yhat).abs()
    mae = res.mean()
    mse = (res ** 2).mean()

    if per_ch:
        mae_per_ch = res.detach().mean(dim=(0, 2, 3, 4))
        mse_per_ch = (res.detach() ** 2).mean(dim=(0, 2, 3, 4))
        return mae, mse, mae_per_ch, mse_per_ch

    return mae, mse, None, None

# ------------------------------------------------------------------------
# TRAIN FUNCTION
# ------------------------------------------------------------------------
def train(
    rank: int,
    local_rank: int,
    config: dict,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_sampler: Any,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[Warmup_Scheduler],
    cosine_scheduler: Optional[CosineAnnealingLR],
    scheduler: Optional[ReduceLROnPlateau],  # rename from 'reduce_on_plateau'
    start_cosine_epoch: int,
    T_max: int,
    logger: Any,
    writer: Optional[SummaryWriter] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    start_epoch: int = 0
) -> None:
    """
    Main training loop with optional warmup, optional cosine annealing,
    and optional reduce_on_plateau (called 'scheduler' here).
    """

    device = f"cuda:{local_rank}"
    
    # If no scaler provided, create one
    if scaler is None:
        scaler = torch.amp.GradScaler()

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")

    # Number of forecast steps
    n_forecast_steps = int(config["Trainer"].pop("n_steps"))

    max_epochs = config["Trainer"]["max_epochs"]
    global_step = start_epoch * len(train_loader)

    config_dtype = config["Trainer"].get("dtype", 32)
    if config_dtype == 32:
        dtype = torch.float32 
    elif config_dtype == 16:
        dtype = torch.bfloat16
    # else:
    #     dtype = torch.bfloat16

    # Start training
    for epoch in range(start_epoch, max_epochs):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)

        logger.info(f"Epoch {epoch} - Seed: {seed}")
        train_sampler.set_epoch(epoch)
        model.train()

        total_mae, total_mse = 0.0, 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                losses = compute_loss(model, batch, n_forecast_steps, device)

            if losses is None:
                continue

            mae, mse, _, _ = losses
            # Decide which loss to backprop
            loss_to_backprop = mae if config["Loss"]["Backprop"] == "L1" else mse
            scaler.scale(loss_to_backprop).backward()

            # Unscale and check for NaNs
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            if not math.isfinite(grad_norm):
                if rank == 0:
                    logger.info(
                        f"Batch {num_batches}: Non-finite grad norm ({grad_norm}). "
                        f"MAE: {mae.item()}, MSE: {mse.item()} => Skipping batch."
                    )
                scaler.update()
                optimizer.zero_grad()
                continue

            # if rank == 0:
            #     logger.info(f"Batch {num_batches}: Grad norm: {grad_norm:.4f}")

            # Gradient clipping if requested
            if config["Trainer"].get("gradient_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["Trainer"]["gradient_clip"],
                    error_if_nonfinite=False
                )
                # clipped_grad_norm = compute_grad_norm(model)
                # if rank == 0:
                #     logger.info(f"Batch {num_batches}: Clipped Grad norm: {clipped_grad_norm:.4f}")

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            total_mae += mae.item()
            total_mse += mse.item()
            num_batches += 1
            global_step += 1

            # Warmup scheduler step
            if warmup_scheduler and global_step <= warmup_scheduler.num_warmup_steps:
                warmup_scheduler.step()
            elif warmup_scheduler and global_step >= warmup_scheduler.num_warmup_steps:
                # Warmup is done
                warmup_scheduler = None

            # Logging
            if rank == 0:
                if num_batches % 50 == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {num_batches}: "
                        f"MAE {mae.item():.6f}, MSE {mse.item():.6f}"
                    )
                if writer is not None and num_batches % 10 == 0:
                    writer.add_scalar("Train_minibatch/MAE", mae.item(), global_step)
                    writer.add_scalar("Train_minibatch/MSE", mse.item(), global_step)
                    # log_gradients(model, writer, global_step)

        # End of epoch: compute average train metrics
        if num_batches > 0:
            total_mae_tensor = reduce_tensor(torch.tensor(total_mae, device=device), device)
            total_mse_tensor = reduce_tensor(torch.tensor(total_mse, device=device), device)
            avg_mae = total_mae_tensor / num_batches
            avg_mse = total_mse_tensor / num_batches
        else:
            avg_mae = torch.tensor(0.0, device=device)
            avg_mse = torch.tensor(0.0, device=device)

        if rank == 0:
            logger.info(f"Epoch {epoch}: Train Avg MAE {avg_mae:.6f}, Avg MSE {avg_mse:.6f}")
            if writer is not None:
                writer.add_scalar("Train/MAE", avg_mae, epoch)
                writer.add_scalar("Train/MSE", avg_mse, epoch)

        # Validation
        with torch.no_grad():
            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                val_loss = validate(
                    model,
                    n_forecast_steps,
                    val_loader,
                    device,
                    logger,
                    writer,
                    config,
                    epoch
                )

        # Saving checkpoint (only on rank 0)
        if rank == 0:
            # Additional logging
            writer.add_scalar("Train/MAE", avg_mae, epoch)
            writer.add_scalar("Train/MSE", avg_mse, epoch)
            logger.info(f"Epoch {epoch}: Avg MAE {avg_mae:.6f}, Avg MSE {avg_mse:.6f}")

            save_model(
                model=model,
                optimizer=optimizer,
                warmup_scheduler=warmup_scheduler,
                cosine_scheduler=cosine_scheduler,
                scheduler=scheduler,  # reduce-lr-on-plateau
                dirpath=config["Checkpoint"]["dirpath"],
                model_id=config["ID"],
                epoch=epoch,
                config=config,
                scaler=scaler
            )

        # Learning rate logging
        current_lr = optimizer.param_groups[0]["lr"]

        # 1) If warmup is done and we are at/after start_cosine_epoch, step CosineAnnealingLR
        if (not warmup_scheduler) and (epoch >= start_cosine_epoch) and cosine_scheduler and (epoch < start_cosine_epoch + T_max):
            # Typically CosineAnnealingLR is stepped once per epoch
            cosine_scheduler.step()

        # 2) Then step ReduceLROnPlateau with val_loss (if available)
        elif (not warmup_scheduler) and scheduler:
            scheduler.step(val_loss)

        # Log final LR after these operations
        final_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            logger.info(f"Epoch {epoch} done. LR was {current_lr:.6g}, now {final_lr:.6g}")
            if writer is not None:
                writer.add_scalar("Train/LR", final_lr, epoch)

        # Clear GPU cache
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main() -> None:
    set_global_seed(1996)
    CONFIG_PATH = sys.argv[1]
    with open(CONFIG_PATH, "r") as f:
        config = load(f, Loader)

    rank, local_rank = setup_distributed()
    log_dir = config["Trainer"]["log_dir"]
    logger = setup_logger(log_dir, rank, experiment_name=config["Experiment"])

    writer: Optional[SummaryWriter] = None
    if rank == 0:
        tensor_log_dir = os.path.join(log_dir, config["Experiment"])
        writer = SummaryWriter(log_dir=tensor_log_dir)
        logger.info(f"TensorBoard logs will be saved to {tensor_log_dir}")

    # ----------------------------------------------------------------
    # Build Dataloaders
    # ----------------------------------------------------------------
    data_config = config["Dataset"]
    train_years = data_config.pop("train_years")
    train_length = data_config.pop("train_length")
    val_years = data_config.pop("val_years")
    val_length = data_config.pop("val_length")

    train_dataloader, train_sampler = get_dataloader(
        **data_config,
        years=train_years,
        length=train_length,
        validation=False
    )
    val_dataloader, _ = get_dataloader(
        **data_config,
        years=val_years,
        length=val_length,
        validation=True
    )

    # ----------------------------------------------------------------
    # Build Model
    # ----------------------------------------------------------------
    model_type = config["Model_Type"]
    in_steps = config["Model"].pop("in_steps")

    if model_type == "AFNO":
        latent_model = AFNOCastLatent(**config["Model"])
    if model_type == "AdaFNO":
        latent_model = AdaFNOCastLatent(**config["Model"])
    elif model_type == "NAT":
        latent_model = NATCastLatent(**config["Model"])
    elif model_type == "AFNONAT":
        latent_model = AFNONATCastLatent(**config["Model"])
        logger.info(f"AFNONAT mode: {latent_model.mode}")
    elif model_type == "UNAT":
        model = UNAT(**config["Model"])
    elif model_type == "Dummy":
        latent_model = DummyLatent()
    elif model_type == "PredRNN_v2":
        model = PredRNN_v2(**config["Model"], in_steps=in_steps)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type in ["Dummy", "AFNO", "NAT", "AFNONAT", "AdaFNO"]:
        encoder = Encoder(**config["Encoder"])
        decoder = Decoder(**config["Decoder"])
        model = Nowcaster(latent_model, encoder, decoder, in_steps=in_steps)
        if rank == 0:
            logger.info(model)
            logger.info(f"Encoder parameters: {count_parameters(encoder)}")
            logger.info(f"Decoder parameters: {count_parameters(decoder)}")
            logger.info(f"Latent model parameters: {count_parameters(latent_model)}")
    else:
        if rank == 0:
            logger.info(model)
            logger.info(f"Model parameters: {count_parameters(model)}")

    device = f"cuda:{local_rank}"
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # ----------------------------------------------------------------
    # Optimizer & Optional Schedulers
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["Trainer"]["lr"])

    # 1) Warmup
    warmup_scheduler = None
    if config["Trainer"].get("warmup_steps", 0) > 0:
        warmup_scheduler = Warmup_Scheduler(optimizer, config["Trainer"]["warmup_steps"])

    # 2) Cosine
    T_max = config["Trainer"].get("T_max", 20)
    eta_min = config["Trainer"].get("eta_min", 1e-5)
    cosine_scheduler = None
    if config["Trainer"].get("use_cosine", True):
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    start_cosine_epoch = config["Trainer"].get("start_cosine_epoch", 5)

    # 3) Reduce On Plateau (named 'scheduler' here)
    scheduler = None
    if config["Trainer"].get("use_reduce_on_plateau", True):
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.25,
            patience=config["Trainer"]["opt_patience"]
        )

    # ----------------------------------------------------------------
    # Optionally load from checkpoint
    # ----------------------------------------------------------------
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    scaler = torch.amp.GradScaler()  # create a scaler for AMP
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            warmup_scheduler=warmup_scheduler,
            cosine_scheduler=cosine_scheduler,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=checkpoint_path,
            logger=logger,
            rank=rank
        )

    finetune_path = config["Checkpoint"].get("finetune_path", None)
    scaler = torch.amp.GradScaler()  # create a scaler for AMP
    if finetune_path and os.path.isfile(finetune_path):
        start_epoch = load_checkpoint(
            model=model,
            optimizer=None,
            warmup_scheduler=None,
            cosine_scheduler=None,
            scheduler=None,
            scaler=None,
            checkpoint_path=finetune_path,
            logger=logger,
            rank=rank
        )

    # ----------------------------------------------------------------
    # Start Training
    # ----------------------------------------------------------------
    train(
        rank=rank,
        local_rank=local_rank,
        config=config,
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        cosine_scheduler=cosine_scheduler,
        scheduler=scheduler,  # reduce-lr-on-plateau
        start_cosine_epoch=start_cosine_epoch,
        T_max=T_max,
        logger=logger,
        writer=writer,
        scaler=scaler,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    main()