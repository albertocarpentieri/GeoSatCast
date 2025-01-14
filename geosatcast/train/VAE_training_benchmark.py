import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, get_worker_info
import torch
from yaml import load, Loader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.profilers import AdvancedProfiler
from geosatcast.models.autoencoder import Encoder, Decoder, VAE
from geosatcast.data.dataset import SimpleDataset
import pickle as pkl
import sys 
import numpy as np

CONFIG_PATH = sys.argv[1]

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
    

    dataset = SimpleDataset(
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
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        shuffle=not validation)
    
    return dataloader


def train(config):
    num_nodes = int(os.environ['SLURM_NNODES'])
    rank = int(os.environ['SLURM_NODEID'])
    print(rank, num_nodes)

    ID = config['ID']
    with open(config['Checkpoint']['dirpath'] + ID + '_config.pkl', 'wb') as o:
        pkl.dump(config, o)
    if rank == 0:
        print(config)

    encoder_config = config['Encoder']
    decoder_config = config['Decoder']
    vae_config = config['VAE']
    encoder = Encoder(**encoder_config)
    
    if rank == 0:
        print('Encoder built')

    decoder = Decoder(
        in_dim=encoder_config['max_ch'],
        out_dim=encoder_config['in_dim'],
        **decoder_config)
    
    if rank == 0:
        print('Decoder built')

    vae = VAE(encoder,
              decoder,
              **vae_config,
              encoded_channels=encoder_config['max_ch'],
        )
    data_config = config['Dataset']
    if rank == 0:
        print('All models built')
        print(vae)

    ckpt_config = config['Checkpoint']
    filename = ckpt_config.pop("filename")
    checkpoint_callback = ModelCheckpoint(
        **ckpt_config,
        filename=ID + '_' + filename,
    )

    early_stop_callback = EarlyStopping(monitor=ckpt_config['monitor'],
                                        patience=config['EarlyStopping']['patience'])

    tr_config = config['Trainer']
    logger = CSVLogger(ckpt_config['dirpath']+'logs', name=ID)
    resume_training = tr_config.pop('resume_training')
    profiler = AdvancedProfiler(filename="profiler_out.txt")
    trainer = Trainer(
        **tr_config,
        default_root_dir=ckpt_config['dirpath'],
        num_nodes=num_nodes,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        # deterministic=True,
        logger=logger,
        profiler=profiler
    )
    
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
    
    if rank == 0:
        print('Training started')

    torch.use_deterministic_algorithms(True, warn_only=True)
    
    if resume_training is None:
        trainer.fit(vae, train_dataloader, val_dataloader)
    else:
        trainer.fit(vae, train_dataloader, val_dataloader,
                    ckpt_path=resume_training)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    with open(CONFIG_PATH, 'r') as o:
        config = load(o, Loader)
    seed = config['seed']
    if seed is not None:
        seed_everything(int(seed), workers=True)
    train(config)