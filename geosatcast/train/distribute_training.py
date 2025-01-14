import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from geosatcast.blocks.ResBlock import ResBlock3D
from geosatcast.utils import sample_from_standard_normal, kl_from_standard_normal, conv_nd
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from geosatcast.data.dataset import WorkerDataset
from yaml import load, Loader
import pickle as pkl

class Encoder(nn.Module):
    def __init__(self, in_dim=1, levels=2, min_ch=64, max_ch=64, extra_resblock_levels=[], downsampling_mode='resblock', norm=None):
        super().__init__()
        self.max_ch = max_ch
        self.min_ch = min_ch
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[-1] = max_ch

        sequence = []
        kernel_size = (1, 3, 3)
        resample_factor = (1, 2, 2)
        res_block_fun = ResBlock3D

        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])

            if i in extra_resblock_levels:
                sequence.append(res_block_fun(in_channels, out_channels, resample=None, kernel_size=kernel_size, norm=norm))

            if downsampling_mode == 'resblock':
                sequence.append(res_block_fun(in_channels, out_channels, resample='down', kernel_size=kernel_size, resample_factor=resample_factor, norm=norm))
            elif downsampling_mode == 'stride':
                sequence.append(conv_nd(3, in_channels, out_channels, kernel_size=resample_factor, stride=resample_factor))

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, levels=2, min_ch=64, max_ch=64, extra_resblock_levels=[], upsampling_mode='stride', norm=None):
        super().__init__()
        self.max_ch = max_ch
        self.min_ch = min_ch
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[0] = out_dim
        channels[-1] = max_ch

        sequence = []
        kernel_size = (1, 3, 3)
        resample_factor = (1, 2, 2)
        stride_conv = nn.ConvTranspose3d
        res_block_fun = ResBlock3D

        for i in reversed(range(levels)):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])

            if upsampling_mode == 'stride':
                sequence.append(stride_conv(in_channels, in_channels, kernel_size=resample_factor, stride=resample_factor))
            elif upsampling_mode == 'resblock':
                sequence.append(res_block_fun(in_channels, in_channels, resample='up', kernel_size=kernel_size, resample_factor=resample_factor, norm=norm))

            if i in extra_resblock_levels:
                sequence.append(res_block_fun(in_channels, out_channels, resample=None, kernel_size=kernel_size, norm=norm))

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, hidden_width, kl_weight, encoded_channels):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.to_var = nn.Conv3d(encoded_channels, hidden_width, kernel_size=1)

    def encode(self, x):
        x = self.encoder(x)
        log_var = self.to_var(x)
        return x, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        z, log_var = self.encode(x)
        if sample_posterior:
            z = sample_from_standard_normal(z, log_var)
        return self.decode(z), z, log_var

    def compute_loss(self, x):
        y_pred, mean, log_var = self.forward(x)
        rec_loss = (x - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
        return (1 - self.kl_weight) * rec_loss + self.kl_weight * kl_loss, rec_loss, kl_loss

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    total_samples = len(dataset.global_indices)
    
    # Split the global indices among workers
    chunk_size = np.ceil(total_samples / num_workers)
    start_idx = int(worker_id * chunk_size)
    print(worker_id, start_idx)
    end_idx = int(min(start_idx + chunk_size, total_samples))
    worker_indices = dataset.global_indices[start_idx:end_idx]
    dataset.set_worker_indices(dataset.global_indices)

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
    

    dataset = WorkerDataset(
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

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
        shuffle=not validation)
    
    return dataloader

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def validate(vae, val_loader, device):
    vae.eval()
    total_loss, rec_loss, kl_loss = 0, 0, 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            x, *_ = batch
            x = x.to(device, non_blocking=True)
            loss, r_loss, k_loss = vae.module.compute_loss(x)
            total_loss += loss.item()
            rec_loss += r_loss.item()
            kl_loss += k_loss.item()
            num_batches += 1

    # Average the losses across all GPUs
    total_loss = reduce_tensor(torch.tensor(total_loss).to(device), device)
    rec_loss = reduce_tensor(torch.tensor(rec_loss).to(device), device)
    kl_loss = reduce_tensor(torch.tensor(kl_loss).to(device), device)

    if dist.get_rank() == 0:
        avg_loss = total_loss / num_batches
        avg_rec_loss = rec_loss / num_batches
        avg_kl_loss = kl_loss / num_batches
        print(f"Validation: Avg Loss {avg_loss:.4f}, Avg Rec Loss {avg_rec_loss:.4f}, Avg KL Loss {avg_kl_loss:.4f}")

def train(rank, config, encoder, decoder, vae, train_loader, val_loader, optimizer, scheduler):
    device = f"cuda:{rank}"
    vae.to(device)
    vae = DDP(vae, device_ids=[rank])

    for epoch in range(config["Trainer"]["max_epochs"]):
        vae.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            x, *_ = batch
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, rec_loss, kl_loss = vae.module.compute_loss(x)
            loss.backward()
            optimizer.step()

            # Accumulate loss for averaging
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        # Average the losses across all GPUs
        total_loss = reduce_tensor(torch.tensor(total_loss).to(device), rank)
        total_rec_loss = reduce_tensor(torch.tensor(total_rec_loss).to(device), rank)
        total_kl_loss = reduce_tensor(torch.tensor(total_kl_loss).to(device), rank)

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_rec_loss = total_rec_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches
            print(f"Epoch {epoch}: Avg Loss {avg_loss:.4f}, Avg Rec Loss {avg_rec_loss:.4f}, Avg KL Loss {avg_kl_loss:.4f}")

            # Save the model at the end of each epoch
            save_model(vae, config["Checkpoint"]["dirpath"], epoch)

        # Validation and learning rate scheduling
        if epoch % config["Trainer"]["val_interval"] == 0:
            if rank == 0:
                validate(vae, val_loader, device)
            scheduler.step()

def save_model(vae, dirpath, epoch):
    model_path = os.path.join(dirpath, f"vae_epoch_{epoch}.pt")
    # Save the model only from rank 0
    if torch.distributed.get_rank() == 0:
        torch.save(vae.state_dict(), model_path)
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

    train_years = data_config.pop("train_years")
    train_length = data_config.pop("train_length")
    val_years = data_config.pop("val_years")
    val_length = data_config.pop("val_length")
    
    train_dataloader = get_dataloader(
        **data_config, 
        years=train_years,
        length=train_length,
        validation=False)
    
    val_dataloader = get_dataloader(
        **data_config, 
        years=val_years,
        length=val_length,
        validation=True)

    encoder_config = config['Encoder']
    decoder_config = config['Decoder']
    vae_config = config['VAE']
    encoder = Encoder(**encoder_config)
    decoder = Decoder(
        in_dim=encoder_config['max_ch'],
        out_dim=encoder_config['in_dim'],
        **decoder_config)
    vae = VAE(encoder,
              decoder,
              **vae_config,
              encoded_channels=encoder_config['max_ch'],
        )
    data_config = config['Dataset']
    if local_rank == 0:
        print('All models built')
        print(vae)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=config["VAE"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=5)

    train(local_rank, config, encoder, decoder, vae, train_loader, val_loader, optimizer, scheduler)

if __name__ == "__main__":
    main()
