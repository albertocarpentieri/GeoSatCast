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
from yaml import load, Loader

# Import your modules (adjust these imports as needed)
from geosatcast.models.predrnn import PredRNN

from distribute_training import (
    set_global_seed,
    setup_logger,
    get_dataloader,
    setup_distributed,
    load_checkpoint,
    save_model,
    load_vae,
    reduce_tensor,
    count_parameters,
    Warmup_Scheduler
)

# Constant for number of channels for per-channel losses.
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

    if config["Loss"]["Monitor"] == "L1":
        avg_loss = avg_mae
    elif config["Loss"]["Monitor"] == "L2":
        avg_loss = avg_mse
    else:
        avg_loss = avg_mae  # default fallback

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
    in_steps = model.module.in_steps  # DDP wrapper so access via module
    x, _, _, _ = batch
    x, y = x[:, :, :in_steps], x[:, :, in_steps: in_steps + n_forecast_steps]

    if per_ch:
        x = x.to(device, non_blocking=True).detach()
    else:
        x = x.to(device, non_blocking=True)

    y = y.to(device, non_blocking=True).detach()
    yhat = model(x, n_steps=n_forecast_steps)
    res = (y - yhat).abs()
    mae = res.mean()
    mse = (res ** 2).mean()

    if per_ch:
        mae_per_ch = res.detach().mean(dim=(0, 2, 3, 4))
        mse_per_ch = (res.detach() ** 2).mean(dim=(0, 2, 3, 4))
        return mae, mse, mae_per_ch, mse_per_ch

    return mae, mse, None, None

def train(
    rank: int,
    local_rank: int,
    config: dict,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_sampler: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_scheduler: Optional[Warmup_Scheduler],
    logger: Any,
    writer: SummaryWriter
) -> None:
    device = f"cuda:{local_rank}"
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    scaler = torch.amp.GradScaler()
    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")
    n_forecast_steps = int(config["Trainer"].pop("n_steps"))

    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, logger, rank)
        if warmup_scheduler and tot_num_batches * start_epoch < warmup_scheduler.num_warmup_steps:
            warmup_scheduler = Warmup_Scheduler(optimizer, config["Trainer"]["warmup_steps"] - tot_num_batches * start_epoch)
        
        
        # # 2. Manually override the learning rate after loading
        # lr = config["Trainer"]["lr"]
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr

    for epoch in range(start_epoch, config["Trainer"]["max_epochs"]):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        logger.info(f"Epoch {epoch} - Seed: {seed}")
        train_sampler.set_epoch(epoch)
        model.train()

        total_mae, total_mse = 0.0, 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                losses = compute_loss(model, batch, n_forecast_steps, device)

            if losses is None:
                continue

            mae, mse, _, _ = losses
            # Backward pass with scaling.
            if config["Loss"]["Backprop"] == "L1":
                scaler.scale(mae).backward()
            elif config["Loss"]["Backprop"] == "L2":
                scaler.scale(mse).backward()
            else:
                scaler.scale(mae).backward()

            # Always unscale to record inf checks.
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            if not math.isfinite(grad_norm):
                if rank == 0:
                    logger.info(f"Batch {num_batches}: Non-finite grad norm ({grad_norm}). Skipping batch.")
                scaler.update()  # now update finds the inf check recorded by unscale_()
                optimizer.zero_grad()
                continue

            # if rank == 0:
            #     logger.info(f"Batch {num_batches}: Grad norm: {grad_norm:.4f}")

            if config["Trainer"].get("gradient_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["Trainer"]["gradient_clip"], error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.update()

            total_mae += mae.item()
            total_mse += mse.item()
            num_batches += 1

            if warmup_scheduler and num_batches + tot_num_batches * epoch < warmup_scheduler.num_warmup_steps:
                warmup_scheduler.step()
            else:
                warmup_scheduler = None

            global_step = epoch * tot_num_batches + num_batches
            if rank == 0:
                if num_batches % 50 == 0:
                    logger.info(f"Epoch {epoch}, Step {num_batches}: MAE {mae.item():.6f}, MSE {mse.item():.6f}")
                if num_batches % 10 == 0:
                    writer.add_scalar("Train_minibatch/MAE", mae.item(), global_step)
                    writer.add_scalar("Train_minibatch/MSE", mse.item(), global_step)
                    log_gradients(model, writer, global_step)

        total_mae_tensor = reduce_tensor(torch.tensor(total_mae, device=device), device)
        total_mse_tensor = reduce_tensor(torch.tensor(total_mse, device=device), device)
        if num_batches > 0:
            avg_mae = total_mae_tensor / num_batches
            avg_mse = total_mse_tensor / num_batches
        else:
            avg_mae, avg_mse = torch.tensor(0.0), torch.tensor(0.0)

        if rank == 0:
            writer.add_scalar("Train/MAE", avg_mae, epoch)
            writer.add_scalar("Train/MSE", avg_mse, epoch)
            logger.info(f"Epoch {epoch}: Avg MAE {avg_mae:.6f}, Avg MSE {avg_mse:.6f}")
            save_model(model, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch, config, scaler)

        with torch.no_grad():
            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                val_loss = validate(model, n_forecast_steps, val_loader, device, logger, writer, config, epoch)

        if warmup_scheduler and num_batches + tot_num_batches * epoch < warmup_scheduler.num_warmup_steps:
            lr = warmup_scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}: Warm-up Learning Rate {lr:.6f}")
        else:
            scheduler.step(val_loss)
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}: Learning Rate {lr:.6f}")

        if rank == 0:
            writer.add_scalar("Train/LR", lr, epoch)

        torch.cuda.empty_cache()


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

    in_steps = config["Model"]["in_steps"]

    model = PredRNN(**config["Model"])
    
    if rank == 0:
        logger.info(model)
        logger.info(f"PredRNN model parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["Trainer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.25,
        patience=config["Trainer"]["opt_patience"]
    )
    warmup_scheduler = Warmup_Scheduler(optimizer, config["Trainer"]["warmup_steps"]) if config["Trainer"]["warmup_steps"] else None

    train(
        rank,
        local_rank,
        config,
        model,
        train_dataloader,
        val_dataloader,
        train_sampler,
        optimizer,
        scheduler,
        warmup_scheduler,
        logger,
        writer  # type: ignore
    )


if __name__ == "__main__":
    main()
