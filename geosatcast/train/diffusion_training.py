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

from geosatcast.models.UNAT import UNAT, ViDiffUNAT, CondEncoder
from geosatcast.models.diffusion import VideoDiffusionModel

from distribute_training import (
    set_global_seed,
    setup_logger,
    get_dataloader,
    setup_distributed,
    reduce_tensor,
    count_parameters,
    Warmup_Scheduler,
    save_model,
    load_checkpoint,
    load_unatcast
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
    in_steps: int,
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
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            mse = compute_loss(
                model, in_steps, batch, n_forecast_steps, device, validation=True, batch_idx=num_batches
            )
            total_mse += mse.item()
            num_batches += 1

    total_mse = reduce_tensor(torch.tensor(total_mse, device=device), device)

    avg_mse = total_mse / num_batches

    if dist.get_rank() == 0:
        logger.info(f"Validation: Avg MSE {avg_mse:.6f}")
        writer.add_scalar("Val/MSE", avg_mse, epoch)
    return avg_mse

def compute_loss(
    model: nn.Module,
    in_steps: int,
    batch: Tuple[torch.Tensor, ...],
    n_forecast_steps: int,
    device: str,
    validation: bool = False,
    batch_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute loss given a batch.
    Splits the input into x and y based on in_steps and forecast steps.
    """
    x, _, inv, sza = batch

    # Slice the data accordingly
    sza = sza[:, :, :in_steps + n_forecast_steps - 1]
    x, y = x[:, :, :in_steps], x[:, :, in_steps: in_steps + n_forecast_steps]

    if validation:
        x = x.to(device, non_blocking=True).detach().type(torch.float32)
        inv = inv.to(device, non_blocking=True).detach().type(torch.float32)
        sza = sza.to(device, non_blocking=True).detach().type(torch.float32)
    else:
        x = x.to(device, non_blocking=True).type(torch.float32)
        inv = inv.to(device, non_blocking=True).type(torch.float32)
        sza = sza.to(device, non_blocking=True).type(torch.float32)

    y = y.to(device, non_blocking=True).detach().type(torch.float32)
    # Merge 'inv' and 'sza' along channel dimension
    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)

    # Forward
    if validation:
        g = torch.Generator()
        g.manual_seed(batch_idx)
        t = torch.randint(0, model.timesteps, x.shape[0], generator=g)
    else:
        t = torch.randint(0, model.timesteps, x.shape[0])
    mse = model.p_losses(y, x, inv, t)
    return mse

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
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

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
                loss_to_backprop = compute_loss(model, in_steps, batch, n_forecast_steps, device, batch_idx=num_batches)

            if loss_to_backprop is None:
                continue

            scaler.scale(loss_to_backprop).backward()

            # Unscale and check for NaNs
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            if not math.isfinite(grad_norm):
                if rank == 0:
                    logger.info(
                        f"Batch {num_batches}: Non-finite grad norm ({grad_norm}). "
                        f"MSE: {loss_to_backprop.item()} => Skipping batch."
                    )
                scaler.update()
                optimizer.zero_grad()
                continue


            # Gradient clipping if requested
            if config["Trainer"].get("gradient_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["Trainer"]["gradient_clip"],
                    error_if_nonfinite=False
                )


            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            total_mse += loss_to_backprop.item()
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
                        f"MSE {loss_to_backprop.item():.6f}"
                    )
                if writer is not None and num_batches % 10 == 0:
                    writer.add_scalar("Train_minibatch/MSE", loss_to_backprop.item(), global_step)
                    log_gradients(model, writer, global_step)

        # End of epoch: compute average train metrics
        if num_batches > 0:
            total_mse_tensor = reduce_tensor(torch.tensor(total_mse, device=device), device)
            avg_mse = total_mse_tensor / num_batches
        else:
            avg_mse = torch.tensor(0.0, device=device)

        if rank == 0:
            logger.info(f"Epoch {epoch}: Avg MSE {avg_mse:.6f}")
            if writer is not None:
                writer.add_scalar("Train/MSE", avg_mse, epoch)

        # Validation
        with torch.no_grad():
            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                val_loss = validate(
                    model,
                    in_steps,
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
            writer.add_scalar("Train/MSE", avg_mse, epoch)
            logger.info(f"Epoch {epoch}: Avg MSE {avg_mse:.6f}")

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

    unat = load_unatcast(config["Model"].pop("UNAT_path"))
    condencoder = CondEncoder(unat=unat, **config["CondEncoder"])
    denoiser = ViDiffUNAT(**config["Denoiser"])
    model = VideoDiffusionModel(denoiser,
        condencoder,
        **config["Model"])

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
        in_steps=in_steps,
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