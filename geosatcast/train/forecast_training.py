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
from geosatcast.models.nowcast import NATCastLatent, AFNONATCastLatent, AFNOCastLatent, DummyLatent, Nowcaster
from distribute_training import set_global_seed, setup_logger, get_dataloader, setup_distributed, load_checkpoint, save_model, load_vae, reduce_tensor, count_parameters, Warmup_Scheduler

def validate(model, n_forecast_steps, val_loader, device, logger, writer, config, epoch):
    model.eval()
    
    total_mae = 0
    total_mae_per_ch = torch.zeros((11,))
    
    total_mse = 0
    total_mse_per_ch = torch.zeros((11,))

    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            mae, mse, mae_per_ch, mse_per_ch = compute_loss(model, batch, n_forecast_steps, device, per_ch=True)
            total_mae += mae.item()
            total_mae_per_ch += mae_per_ch.cpu()

            total_mse += mse.item()
            total_mse_per_ch += mse_per_ch.cpu()
            num_batches += 1

    # Average the losses across all GPUs
    total_mae = reduce_tensor(torch.tensor(total_mae, device=device), device)
    total_mae_per_ch = reduce_tensor(total_mae_per_ch.to(device), device)
    total_mse = reduce_tensor(torch.tensor(total_mse, device=device), device)
    total_mse_per_ch = reduce_tensor(total_mse_per_ch.to(device), device)

    avg_mae = total_mae / num_batches
    avg_mae_per_ch = total_mae_per_ch / num_batches

    avg_mse = total_mse / num_batches
    avg_mse_per_ch = total_mse_per_ch / num_batches
    
    if dist.get_rank() == 0:
        loss_str = ", ".join([f"{avg_mae_per_ch[i].item():.6f}" for i in range(11)])
        logger.info(f"Validation: Avg MAE {avg_mae:.6f}, MAE per ch {loss_str}")
        loss_str = ", ".join([f"{avg_mse_per_ch[i].item():.6f}" for i in range(11)])
        logger.info(f"Validation: Avg MSE {avg_mse:.6f}, MSE per ch {loss_str}")

        # Log validation losses to TensorBoard
        writer.add_scalar("Val/MAE", avg_mae, epoch)
        writer.add_scalar("Val/MSE", avg_mse, epoch)
        for c in range(11):
            writer.add_scalar(f"Val/MAE_{c}", avg_mae_per_ch[c].item(), epoch)
            writer.add_scalar(f"Val/MSE_{c}", avg_mse_per_ch[c].item(), epoch)
    
    if config["Loss"]["Monitor"] == "L1":
        avg_loss = avg_mae
    elif config["Loss"]["Monitor"] == "L2":
        avg_loss = avg_mse
    return avg_loss

def compute_loss(model, batch, n_forecast_steps, device, per_ch=False):
    in_steps = model.module.in_steps
    
    # open batch
    x, t, inv, sza = batch
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
    mae = res.mean()
    mse = (res**2).mean()
    
    # if mae >= 1:
    #     import datetime
    #     print("loss:", res.mean(dim=(1,2,3,4)))
    #     print("ground_truth:", y.mean(dim=(1,2,3,4)), y.std(dim=(1,2,3,4)))
    #     print(yhat.mean(dim=(1,2,3,4)), yhat.std(dim=(1,2,3,4)))
    #     with torch.no_grad():
    #         for i in range(n_forecast_steps):
    #             z = torch.cat((x, inv[:,:,:in_steps]), dim=1)
    #             print("concatenated:", z.mean(dim=(1,2,3,4)), z.std(dim=(1,2,3,4)))
    #             z = model.module.encoder(z)
    #             print("compressed:", z.mean(dim=(1,2,3,4)), z.std(dim=(1,2,3,4)))
    #             z = model.module.latent_model(z)
    #             print("latent_forecasted:", z.mean(dim=(1,2,3,4)), z.std(dim=(1,2,3,4)))
    #             z = model.module.decoder(z)
    #             print("decoded:", z.mean(dim=(1,2,3,4)), z.std(dim=(1,2,3,4)))
    #     print("time:", [t_b[0] for t_b in t])
    #     return None
    if per_ch:
        mae_per_ch = res.detach().mean(dim=(0,2,3,4))
        mse_per_ch = (res.detach()**2).mean(dim=(0,2,3,4))
        return mae, mse, mae_per_ch, mse_per_ch
    else:
        return mae, mse

def train(
    rank,
    local_rank, 
    config, 
    model,
    train_loader, 
    val_loader, 
    train_sampler, 
    optimizer, 
    scheduler, 
    warmup_scheduler,
    logger,
    writer):

    device = f"cuda:{local_rank}"
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Initialize GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    tot_num_batches = len(train_loader)
    logger.info(f"Total number of batches is: {tot_num_batches}")
    
    n_forecast_steps = int(config["Trainer"].pop("n_steps"))


    # Check if we should resume from a checkpoint
    checkpoint_path = config["Checkpoint"].get("resume_path", None)
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, logger, rank)
        if warmup_scheduler and tot_num_batches * start_epoch < warmup_scheduler.num_warmup_steps:
            warmup_scheduler = Warmup_Scheduler(optimizer, config["Trainer"]["warmup_steps"] -  tot_num_batches * start_epoch)

    
    
    for epoch in range(start_epoch, config["Trainer"]["max_epochs"]):
        seed = int((epoch + 1) ** 2) * (rank + 1)
        set_global_seed(seed)
        print(f"seed: {seed}")
        
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            # Mixed precision forward and loss computation
            with torch.amp.autocast('cuda', dtype=torch.float32):
                losses = compute_loss(model, batch, n_forecast_steps, device)
            
            if losses is not None:
                mae, mse = losses
                # Backpropagation with gradient scaling
                if config["Loss"]["Backprop"] == "L1":
                    scaler.scale(mae).backward()
                elif config["Loss"]["Backprop"] == "L2":
                    scaler.scale(mse).backward()
                        
                if config["Trainer"]["gradient_clip"] is not None:
                    scaler.unscale_(optimizer)  # Unscale the gradients before clipping
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["Trainer"]["gradient_clip"], error_if_nonfinite=False)
                
                if torch.isfinite(total_norm):
                    scaler.step(optimizer)
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"Gradient issue in {name}: contains NaN or Inf.")
                
                # optimizer.step()
                scaler.update()

                # Accumulate loss for averaging
                total_mae += mae.item()
                total_mse += mse.item()
                num_batches += 1
                
                if warmup_scheduler and num_batches + tot_num_batches * epoch < warmup_scheduler.num_warmup_steps:
                    warmup_scheduler.step()
                else:
                    warmup_scheduler = None
                    
                if rank == 0:
                    if num_batches % 50 == 0:
                        logger.info(f"Epoch {epoch}, Step {num_batches}: MAE {mae.item():.6f} MSE {mse.item():.6f}")
                    if num_batches % 10 == 0:
                        writer.add_scalar("Train_minibatch/MAE", mae.item(), epoch * tot_num_batches + num_batches)
                        writer.add_scalar("Train_minibatch/MSE", mse.item(), epoch * tot_num_batches + num_batches)
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f'Gradients/{name}', param.grad, epoch * tot_num_batches + num_batches)
                            if param.requires_grad:
                                writer.add_histogram(f'Weights/{name}', param, epoch * tot_num_batches + num_batches)
        
        # Average the losses across all GPUs
        total_mae = reduce_tensor(torch.tensor(total_mae, device=device), rank)
        total_mse = reduce_tensor(torch.tensor(total_mse, device=device), rank)

        if rank == 0:
            avg_mae = total_mae / num_batches
            avg_mse = total_mse / num_batches
            # Log to TensorBoard
            writer.add_scalar("Train/MAE", avg_mae, epoch)
            writer.add_scalar("Train/MSE", avg_mse, epoch)
            logger.info(f"Epoch {epoch}: Avg MAE {avg_mae:.6f}, Avg MSE {avg_mse:.6f}")
            # Save the model at the end of each epoch
            save_model(model, optimizer, scheduler, config["Checkpoint"]["dirpath"], config["ID"], epoch, config, scaler)
        
        with torch.no_grad():
            model.eval()
            val_loss = validate(model, n_forecast_steps, val_loader, device, logger, writer, config, epoch)
        
        
        if warmup_scheduler and num_batches + tot_num_batches * epoch < warmup_scheduler.num_warmup_steps:
            lr = warmup_scheduler.get_last_lr()[0] 
            logger.info(f"Epoch {epoch}: Warm-up Learning Rate {lr:.6f}")
        else:
            scheduler.step(val_loss)
            lr = scheduler.get_last_lr()[0] 
            logger.info(f"Epoch {epoch}: Learning Rate {lr:.6f}")
        
        if rank == 0:
            writer.add_scalar(f"Train/LR", lr, epoch)
        torch.cuda.empty_cache()

def main():
    set_global_seed(1996)
    CONFIG_PATH = sys.argv[1]
    with open(CONFIG_PATH, "r") as f:
        config = load(f, Loader)

    rank, local_rank = setup_distributed()
    
    log_dir = config["Trainer"]["log_dir"]
    logger = setup_logger(log_dir, rank, experiment_name=config["Experiment"])
    
    # TensorBoard writer (only for rank 0)
    writer = None
    if rank == 0:
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
    in_steps = config["Model"].pop("in_steps")
    model_type = config["Model_Type"]
    
    if model_type == "AFNO":
        latent_model = AFNOCastLatent(**config["Model"])

    elif model_type == "NAT":
        latent_model = NATCastLatent(**config["Model"])
    
    elif model_type == "AFNONAT":
        latent_model = AFNONATCastLatent(**config["Model"])
        print("mode:", latent_model.mode)
    
    elif model_type == "Dummy":
        latent_model = DummyLatent()

    model = Nowcaster(
        latent_model,
        encoder,
        decoder,
        in_steps=in_steps
    )
    if rank == 0:
        print(model)
        print(count_parameters(encoder))
        print(count_parameters(decoder))
        print(count_parameters(latent_model))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["Trainer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=config["Trainer"]["opt_patience"])
    if config["Trainer"]["warmup_steps"]:
        warmup_scheduler = Warmup_Scheduler(optimizer, config["Trainer"]["warmup_steps"])
    else:
        warmup_scheduler = None
    
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
        writer)

if __name__ == "__main__":
    main()