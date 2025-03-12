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
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR
)
from torch.utils.tensorboard import SummaryWriter
from yaml import load, Loader

from geosatcast.models.UNAT import UNAT, ViDiffUNAT, CondEncoder
from geosatcast.models.diffusion import VideoDiffusionModel
from geosatcast.test.metrics import compute_crps
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

def compute_loss(
    model: nn.Module,
    in_steps: int,
    batch: Tuple[torch.Tensor, ...],
    n_forecast_steps: int,
    device: str,
    validation: bool = False,
    batch_idx: int = 0,
    rank: int = 0,
) -> torch.Tensor:
    """
    Compute MSE loss given a batch.
    Splits the input into x and y based on in_steps and forecast steps.
    """
    x, _, inv, sza = batch

    # Slice the data accordingly
    sza = sza[:, :, :in_steps + n_forecast_steps - 1]
    x_in = x[:, :, :in_steps]
    y = x[:, :, in_steps: in_steps + n_forecast_steps]

    # Move to device
    if validation:
        # Detach (optional, but can help ensure no grad is kept in validation)
        x_in = x_in.to(device, non_blocking=True).detach().float()
        inv = inv.to(device, non_blocking=True).detach().float()
        sza = sza.to(device, non_blocking=True).detach().float()
    else:
        x_in = x_in.to(device, non_blocking=True).float()
        inv = inv.to(device, non_blocking=True).float()
        sza = sza.to(device, non_blocking=True).float()

    y = y.to(device, non_blocking=True).detach().float()

    # Merge 'inv' and 'sza' along channel dimension
    # e.g. if inv is shape [B, C_inv, H, W], expand time dimension to match 'sza'
    # (Be sure this is correct for your actual data shapes.)
    inv_sza = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)

    # Sample random diffusion timesteps. For reproducibility in validation,
    # you can fix or seed them carefully. This can help debug NaNs.
    # Also ensure model.module.timesteps exists and is correct:
    timesteps = model.module.timesteps
    if validation:
        g = torch.Generator(device=device)
        g.manual_seed(batch_idx + rank)  # Some offset
        t = torch.randint(timesteps, (x_in.shape[0],), generator=g, device=device)
    else:
        t = torch.randint(timesteps, (x_in.shape[0],), device=device)

    # Forward diffusion loss
    mse_loss = model.module.p_losses(
        y, x_in, inv_sza, t, n_forecast_steps
    )

    return mse_loss

def validate(
    model: nn.Module,
    in_steps: int,
    n_forecast_steps: int,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Any,
    writer: SummaryWriter,
    config: dict,
    epoch: int,
    rank: int
) -> Optional[torch.Tensor]:
    """
    Run validation by computing the average MSE over the validation set,
    while avoiding mismatch across ranks. Summation is done across ranks.
    """
    model.eval()
    total_mse = 0.0
    total_batches = 0

    # Compute MSE over the validation set.
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                mse = compute_loss(
                    model, in_steps, batch, n_forecast_steps, device,
                    validation=True, batch_idx=batch_idx, rank=rank,
                )
        # Accumulate
        total_mse += mse.item()
        total_batches += 1

    # Now reduce across ranks: we want sum of MSE, sum of batches
    # so that we can compute an overall average
    tensor_out = torch.tensor([total_mse, total_batches], device=device, dtype=torch.float32)
    dist.all_reduce(tensor_out, op=dist.ReduceOp.SUM)

    global_mse_sum = tensor_out[0].item()
    global_batch_sum = tensor_out[1].item()

    # If no batches at all, skip
    if global_batch_sum == 0:
        # This can happen if the validation set is extremely small or
        # the distributed sampler is uneven. Just log a warning & return None.
        if rank == 0:
            logger.warning("No validation batches found across all ranks.")
        return None

    avg_mse = global_mse_sum / global_batch_sum

    if rank == 0:
        logger.info(f"Validation: Avg MSE {avg_mse:.6f}")
        writer.add_scalar("Val/MSE", avg_mse, epoch)

    return torch.tensor(avg_mse, device=device)

def train(
    rank: int,
    local_rank: int,
    config: dict,
    model: nn.Module,
    in_steps: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_sampler: Any,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: Optional[Any],
    cosine_scheduler: Optional[CosineAnnealingLR],
    scheduler: Optional[ReduceLROnPlateau],
    start_cosine_epoch: int,
    T_max: int,
    logger: Any,
    writer: Optional[SummaryWriter] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    start_epoch: int = 0
) -> None:
    """
    Main training loop with optional warmup, optional cosine annealing,
    and optional reduce_on_plateau (named 'scheduler').
    """

    device = f"cuda:{local_rank}"

    # If no scaler provided, create one
    if scaler is None:
        scaler = torch.amp.GradScaler()

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of train batches: {tot_num_batches}")

    # Number of forecast steps
    n_forecast_steps = int(config["Trainer"].pop("n_steps"))

    max_epochs = config["Trainer"]["max_epochs"]
    global_step = start_epoch * len(train_loader)

    config_dtype = config["Trainer"].get("dtype", 32)
    if config_dtype == 32:
        amp_dtype = torch.float32
    elif config_dtype == 16:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    for epoch in range(start_epoch, max_epochs):
        # Set seed for reproducibility
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        logger.info(f"Epoch {epoch} - Seed: {seed}")

        train_sampler.set_epoch(epoch)
        model.train()

        total_train_mse = 0.0
        num_batches = 0

        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = compute_loss(
                    model, in_steps, batch, n_forecast_steps, device,
                    validation=False, batch_idx=batch_idx
                )

            if loss is None or not torch.isfinite(loss):
                logger.warning(
                        f"Non-finite loss: {loss.item()} => Skipping batch."
                    )
                continue

            # Backprop in mixed precision
            scaler.scale(loss).backward()

            # Unscale so we can clip or check grad norm safely
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            if not math.isfinite(grad_norm):
                if rank == 0:
                    logger.warning(
                        f"Non-finite grad norm ({grad_norm}). "
                        f"Loss: {loss.item()} => Skipping batch."
                    )
                scaler.update()
                optimizer.zero_grad()
                continue

            # Optional gradient clip
            if config["Trainer"].get("gradient_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["Trainer"]["gradient_clip"],
                    error_if_nonfinite=False
                )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            total_train_mse += loss.item()
            num_batches += 1
            global_step += 1

            # Warmup scheduler step
            if warmup_scheduler and global_step <= warmup_scheduler.num_warmup_steps:
                warmup_scheduler.step()
            elif warmup_scheduler and global_step > warmup_scheduler.num_warmup_steps:
                # Warmup done
                warmup_scheduler = None

            # Logging
            if rank == 0:
                if batch_idx % 50 == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {batch_idx}: "
                        f"MSE {loss.item():.6f}, grad_norm={grad_norm:.3f}"
                    )
                if writer is not None and batch_idx % 50 == 0:
                    writer.add_scalar("Train_minibatch/MSE", loss.item(), global_step)
                    # log_gradients(model, writer, global_step)  # if you want frequent gradient histograms

        end_time = time.time()
        logger.info(f"Rank {rank} completed epoch {epoch} in {end_time - start_time:.2f}s")

        # Reduce train metrics across ranks
        if num_batches > 0:
            # For average MSE: sum it across ranks, sum the batch counts, then divide
            data_out = torch.tensor([total_train_mse, num_batches], device=device)
            dist.all_reduce(data_out, op=dist.ReduceOp.SUM)
            global_mse_sum = data_out[0].item()
            global_batch_count = data_out[1].item()
            avg_train_mse = global_mse_sum / max(global_batch_count, 1)
        else:
            # No batches for this rank - reduce anyway
            data_out = torch.tensor([0.0, 0.0], device=device)
            dist.all_reduce(data_out, op=dist.ReduceOp.SUM)
            global_mse_sum = data_out[0].item()
            global_batch_count = data_out[1].item()
            if global_batch_count == 0:
                avg_train_mse = float("nan")
            else:
                avg_train_mse = global_mse_sum / global_batch_count

        if rank == 0:
            logger.info(f"Epoch {epoch}: Avg Train MSE {avg_train_mse:.6f}")
            if writer is not None:
                writer.add_scalar("Train/MSE", avg_train_mse, epoch)

        # --------------------
        # Validation
        # --------------------
        with torch.no_grad():
            val_loss = validate(
                model,
                in_steps,
                n_forecast_steps,
                val_loader,
                device,
                logger,
                writer,
                config,
                epoch,
                rank
            )

        # Save checkpoint (rank 0 only)
        if rank == 0:
            # Log the average train MSE again, or any other metrics
            if writer is not None:
                writer.add_scalar("Train/MSE_epoch", avg_train_mse, epoch)
                        # Additional logging
            writer.add_scalar("Train/MSE", val_loss, epoch)
            logger.info(f"Epoch {epoch}: Avg MSE {val_loss:.6f}")

            # If you are returning None from validate() (e.g., no val data),
            # you might pass 0 or some default to the scheduler.
            val_for_scheduler = val_loss.item() if val_loss is not None else avg_train_mse

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

        # Update LR schedulers
        current_lr = optimizer.param_groups[0]["lr"]
        # 1) Cosine annealing
        if (not warmup_scheduler) and (epoch >= start_cosine_epoch) and cosine_scheduler and (epoch < start_cosine_epoch + T_max):
            cosine_scheduler.step()
        # 2) ReduceOnPlateau
        elif (not warmup_scheduler) and scheduler:
            val_for_scheduler = val_loss.item() if (val_loss is not None and torch.isfinite(val_loss)) else avg_train_mse
            scheduler.step(val_for_scheduler)

        final_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            logger.info(f"Epoch {epoch} done. LR from {current_lr:.6g} to {final_lr:.6g}")
            if writer is not None:
                writer.add_scalar("Train/LR", final_lr, epoch)

        # Clean up
        dist.barrier()
        torch.cuda.empty_cache()

def main():
    # ----------------------------------------------------------------
    # Setup & Configuration
    # ----------------------------------------------------------------
    set_global_seed(1996)
    CONFIG_PATH = sys.argv[1]
    with open(CONFIG_PATH, "r") as f:
        config = load(f, Loader)

    rank, local_rank = setup_distributed()  # your function
    log_dir = config["Trainer"]["log_dir"]
    logger = setup_logger(log_dir, rank, experiment_name=config["Experiment"])

    writer: Optional[SummaryWriter] = None
    if rank == 0:
        tensor_log_dir = os.path.join(log_dir, config["Experiment"])
        writer = SummaryWriter(log_dir=tensor_log_dir)
        logger.info(f"TensorBoard logs saved to {tensor_log_dir}")

    # ----------------------------------------------------------------
    # Build Dataloaders (ensure you use DistributedSampler for both!)
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
    val_dataloader, val_sampler = get_dataloader(
        **data_config,
        years=val_years,
        length=val_length,
        validation=True
    )
    # Make sure val_sampler is a distributed sampler or that you do single-rank validation

    if rank == 0:
        logger.info(f"Train DataLoader has {len(train_dataloader)} batches.")
        logger.info(f"Val DataLoader has {len(val_dataloader)} batches.")

    # ----------------------------------------------------------------
    # Build Model
    # ----------------------------------------------------------------
    in_steps = config["Model"].pop("in_steps")
    unat = load_unatcast(config["Model"].pop("UNAT_path"))
    condencoder = CondEncoder(unat=unat, **config["CondEncoder"])
    denoiser = ViDiffUNAT(**config["Denoiser"])
    model = VideoDiffusionModel(denoiser, condencoder, **config["Model"])
    model.to(f"cuda:{local_rank}")
    if rank == 0:
        logger.info(denoiser)
        logger.info(f"Denoiser parameters: {count_parameters(denoiser)}")
        logger.info(f"Model parameters: {count_parameters(model)}")
    
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

    # 3) Reduce On Plateau
    scheduler = None
    if config["Trainer"].get("use_reduce_on_plateau", True):
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.25,
            patience=config["Trainer"]["opt_patience"]
        )

    # ----------------------------------------------------------------
    # Optionally load checkpoint
    # ----------------------------------------------------------------
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    finetune_path = config["Checkpoint"].get("finetune_path", None)
    start_epoch = 0
    scaler = torch.amp.GradScaler('cuda')
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
    
    elif finetune_path and os.path.isfile(finetune_path):
        # Typically for fine-tuning you might skip loading the optimizer states
        load_checkpoint(
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
        scheduler=scheduler,
        start_cosine_epoch=start_cosine_epoch,
        T_max=T_max,
        logger=logger,
        writer=writer,
        scaler=scaler,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    main()
