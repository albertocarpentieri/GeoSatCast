import xarray as xr
import h5py as h5
import numpy as np
import time
import os
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch
from geosatcast.data.utils import cos_zenith_angle_from_timestamp
import math

MEANS = [
    9.318155288696289,
    278.7270812988281,
    273.3865661621094,
    251.71925354003906,
    276.63323974609375,
    275.7444763183594,
    255.65524291992188,
    7.317643165588379,
    9.009994506835938,
    235.41212463378906,
    252.4060516357422
]

STDS = [
    14.593178749084473,
    18.606534957885742,
    18.29147720336914,
    14.870534896850586,
    20.109272003173828,
    20.402971267700195,
    13.62041187286377,
    10.912806510925293,
    13.091219902038574,
    7.859856605529785,
    11.093881607055664,
]

class DistributedDataset(Dataset):
    def __init__(
        self,
        data_path,
        invariants_path,
        name,
        years,
        input_len,
        output_len,
        channels=np.arange(11),
        field_size=256,
        length=10000,
        validation=False,
        rank=1
        ):
        
        self.data_path = data_path
        self.years = years
        self.output_len = output_len if output_len else 0
        self.input_len = input_len
        self.seq_len = self.input_len + self.output_len
        self.channels = channels
        self.field_size = field_size
        self.means = torch.from_numpy(np.array(MEANS)[None,:,None,None]).type(torch.float32)
        self.stds = torch.from_numpy(np.array(STDS)[None,:,None,None]).type(torch.float32)
        
        dem = xr.open_dataset(os.path.join(invariants_path, "dem.nc"))["DEM"].values
        dem[np.isnan(dem)] = -1
        dem = (dem - dem.mean()) / dem.std()

        lwmask = xr.open_dataset(os.path.join(invariants_path, "lwmask.nc"))["LWMASK"].values
        lwmask[lwmask==3] = 0
        self.inv = torch.from_numpy(np.stack((dem, lwmask)).astype(np.float32)).type(torch.float32)
        self.data_paths = {year: os.path.join(self.data_path, f"{year}_{name}.h5") for year in years}
        
        self.data_len = {}
        self.timestamps = {}
        for year in years: 
            with h5.File(self.data_paths[year]) as f:
                self.timestamps[year] = f["time"][:]
                self.data_len[year] = f["time"].shape[0]
        self.length = length
        self.get_latlon()
        self.generate_global_indices()
        self.validation = validation
        self.rank = rank

    def get_latlon(self):
        with h5.File(self.data_paths[self.years[0]], "r") as h:
            self.max_shape = h["fields"].shape[-2:]
            self.lon = h["longitude"][:]
            self.lat = h["latitude"][:]
        self.latlon_grid = np.stack(np.meshgrid(self.lon, self.lat))
    
    def generate_global_indices(self):
        # Precompute all valid (year, t_i, lat_i, lon_i) combinations
        indices = []
        for year in self.years:
            timestamps = self.timestamps[year]
            diff = timestamps[self.seq_len:] - timestamps[:-self.seq_len]
            idx = np.where(diff == 15*60*(self.seq_len))[0]
            for i in idx:
                indices.append((year, int(i)))
        self.indices = indices
        self.len_indices = len(indices)
        self.worker_indices = None
        self.len_worker_indices = None
        print("global indices computed")

    def set_worker_indices(self, indices):
        self.worker_indices = indices
        self.len_worker_indices = len(indices)
    
    def __len__(self):
        return self.length
    
    def get_data(self, year, t_i, lat_i, lon_i):
        # x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
        # with h5.File(self.data_paths[year], "r") as h:
        #     h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len,:,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
        
        x = np.empty((self.seq_len, 11, *self.max_shape), dtype=np.float32)
        with h5.File(self.data_paths[year], "r") as h:
            h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len,:], np.s_[:])
        x = x[:,:,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        t = self.timestamps[year][t_i:t_i+self.seq_len]

        inv = self.inv[:, None, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        sza = torch.from_numpy(np.stack([cos_zenith_angle_from_timestamp(
                t_, 
                self.latlon_grid[0, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size].flatten(),
                self.latlon_grid[1, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size].flatten())
                for t_ in t])).type(torch.float32)
        sza = sza.view(1,-1,self.field_size, self.field_size)
        t = torch.from_numpy(t).type(torch.float32)[None, ..., None, None]
        x = torch.from_numpy(x).type(torch.float32)
        x = (x - self.means) / self.stds
        x = x.permute(1, 0, 2, 3)
        return x, t, inv, sza
    
    def __getitem__(self, idx):
        if self.worker_indices is not None:
            i = int(idx % self.len_worker_indices)
            year, t_i = self.worker_indices[i]
        else:
            i = int(idx % self.len_indices)
            year, t_i = self.indices[i]
        
        if self.validation:
            sampler = np.random.default_rng(int(idx * self.rank)).integers
        else:
            sampler = np.random.randint
        lat_i = sampler(0, self.max_shape[-2] - self.field_size)
        lon_i = sampler(0, self.max_shape[-1] - self.field_size)        
        x, t, inv, sza = self.get_data(year, t_i, lat_i, lon_i)
        return x, t, inv, sza


class WorkerDistributedSampler(Sampler):
    """
    A distributed sampler that distributes dataset indices among both processes and workers.
    
    Arguments:
        dataset: The dataset to sample from.
        num_replicas: Total number of processes in the distributed setup.
        rank: Rank of the current process.
        shuffle: If True, shuffles the dataset indices before sampling.
        seed: Random seed for shuffling.
        drop_last: If True, drops the last incomplete batch if the dataset size
                   is not divisible by the number of replicas.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            num_replicas = torch.distributed.get_world_size()
        
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            rank = torch.distributed.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.indices = dataset.indices
        self.total_size = len(self.indices)
        
        # Compute the number of samples each process should handle
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        print(f"Defining sampler for replica {self.rank}")

    def __iter__(self):
        # Deterministic shuffling based on seed and epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            
        else:
            g = torch.Generator()
            g.manual_seed(0)

        iter_indices = torch.randperm(self.total_size, generator=g)[:self.num_samples].tolist()
        return iter(iter_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for the sampler. Useful for shuffling in distributed training.
        """
        self.seed = self.seed + epoch


if __name__ == "__main__":

    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data import get_worker_info
    from torch.utils.data.distributed import DistributedSampler
    from torch.profiler import profile, ProfilerActivity
    import numpy as np
    import time

    # Initialize the process group
    dist.init_process_group("nccl")

    # Get the current GPU and process information
    local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    num_workers = 16
    batch_size = 64
    pref_f = 8
    data_path = "/capstor/scratch/cscs/acarpent/SEVIRI/"
    invariants_path = "/capstor/scratch/cscs/acarpent/SEVIRI/invariants/"
    print("num workers:", num_workers)
    print("batch_size:", batch_size)
    print("prefetch factor:", pref_f)
    # Create the dataset
    dataset = DistributedDataset(
        data_path=data_path,
        invariants_path=invariants_path,
        name="new_virtual",
        years=[2017, 2018, 2019],
        input_len=1,
        output_len=None,
        field_size=256,
        length=batch_size * 100 * 4,
    )

    # Wrap the dataset with a DistributedSampler
    sampler = WorkerDistributedSampler(
        dataset, 
        num_replicas=torch.distributed.get_world_size(),
        rank=local_rank,
        shuffle=True,
        seed=0)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=pref_f,
        persistent_workers=True,
    )

    # Warm-up
    print("Starting warm-up...")
    warmup_start = time.time()
    b = 0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/capstor/scratch/cscs/acarpent/log')
    ) as prof:
        for batch in dataloader:
            sampler.set_epoch(0)
            x, t, inv, sza = batch
            x = x.to(device)
            t = t.to(device)
            inv = inv.to(device)
            sza = sza.to(device)
            b += 1
        print(f"Time to retrieve {b} batches: {time.time() - warmup_start}")
