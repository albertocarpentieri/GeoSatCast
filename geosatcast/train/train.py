import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP

CONFIG_PATH = sys.argv[1]

def setup_distributed():
    """
    Initialize distributed training using SLURM environment variables.
    """
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """
    Destroy the process group.
    """
    dist.destroy_process_group()


def train_distributed(rank, world_size, model, train_dataset, val_dataset, criterion, optimizer, scheduler, num_epochs, batch_size):
    """
    Distributed training loop.
    """
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Use DistributedSampler for data loading
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()
            x, _, _, _ = batch
            x = x.to(device)

            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, _, _, _ = batch
                x = x.to(device)
                output = model(x)

        if scheduler:
            scheduler.step()

    cleanup_distributed()

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
    dataset.set_worker_indices(worker_indices)

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
        length
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