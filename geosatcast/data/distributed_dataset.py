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
    252.4060516357422,
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
        load_full=False,
        rank=1,
        mask_sza=True,
        dtype=32,
        ):
        
        self.data_path = data_path
        self.years = years
        self.output_len = output_len if output_len else 0
        self.input_len = input_len
        self.seq_len = self.input_len + self.output_len
        self.channels = channels
        self.field_size = field_size
        self.mask_sza = mask_sza
        if dtype == 16:
            self.torch_dtype = torch.float16
            self.numpy_dtype = np.float16
        else:
            self.torch_dtype = torch.float32
            self.numpy_dtype = np.float32
        self.means = torch.from_numpy(np.array(MEANS)[:,None,None,None]).type(self.torch_dtype)
        self.stds = torch.from_numpy(np.array(STDS)[:,None,None,None]).type(self.torch_dtype)
        
        dem = xr.open_dataset(os.path.join(invariants_path, "dem.nc"))["DEM"].values
        dem[np.isnan(dem)] = -1
        dem = (dem - dem.mean()) / dem.std()

        lwmask = xr.open_dataset(os.path.join(invariants_path, "lwmask.nc"))["LWMASK"].values
        lwmask[lwmask==3] = 0
        self.inv = torch.from_numpy(np.stack((dem, lwmask)).astype(np.float32)).type(self.torch_dtype)
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
        self.load_full = load_full

    def get_latlon(self):
        with h5.File(self.data_paths[self.years[0]], "r") as h:
            self.max_shape = h["fields"].shape[1:3]
            self.lon = h["longitude"][:]
            self.lat = h["latitude"][:]
        # self.latlon_grid = np.stack(np.meshgrid(self.lon, self.lat))
    
    def generate_global_indices(self):
        # Precompute all valid (year, t_i, lat_i, lon_i) combinations
        indices = []
        for year in self.years:
            timestamps = self.timestamps[year]
            diff = timestamps[self.seq_len:] - timestamps[:-self.seq_len]
            idx = np.where(diff == 15*60*(self.seq_len))[0]
            print(f"Year {year}, indices {len(idx)}, timestamps {len(timestamps)}")
            for i in idx:
                indices.append((year, int(i)))
        self.indices = indices
        self.len_indices = len(indices)
        self.worker_indices = None
        self.len_worker_indices = None
        print(f"{self.len_indices} global indices computed")

    def set_worker_indices(self, indices):
        self.worker_indices = indices
        self.len_worker_indices = len(indices)
    
    def __len__(self):
        return self.len_indices
    
    def get_data(self, year, t_i, lat_i, lon_i):
        if not self.load_full:
            # x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
            x = np.empty((self.seq_len, self.field_size, self.field_size, 12), dtype=self.numpy_dtype)
            with h5.File(self.data_paths[year], "r") as h:
                h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size, :], np.s_[:])
        else:
            x = np.empty((self.seq_len, *self.max_shape, 12), dtype=self.numpy_dtype)
            with h5.File(self.data_paths[year], "r") as h:
                h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len,:], np.s_[:])
            x = x[:,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        t = self.timestamps[year][t_i:t_i+self.seq_len]

        inv = self.inv[:, None, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        t = torch.from_numpy(t).type(torch.float32)[None, ..., None, None]
        x = torch.from_numpy(x).type(self.torch_dtype)
        
        x = x.permute(3, 0, 1, 2).contiguous()
        x, sza = x[:-1], x[-1:]
        if self.mask_sza:
            x[[0,7,8]] = x[[0,7,8]] * (sza > - 0.07)
        x = (x - self.means) / self.stds
        return x, t, inv, sza
    
    def __getitem__(self, idx):
        year, t_i = self.indices[idx]
        
        if self.validation:
            sampler = np.random.default_rng(int(idx * self.rank)).integers
        else:
            sampler = np.random.randint
        lat_i = sampler(0, self.max_shape[-2] - self.field_size)
        lon_i = sampler(0, self.max_shape[-1] - self.field_size)        
        try:
            return self.get_data(year, t_i, lat_i, lon_i)
        except:
            raise Exception(f"{idx} relative to {t_i} of year {year} has problems")


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
    def __init__(
        self, 
        dataset, 
        num_replicas=None, 
        rank=None, 
        shuffle=True, 
        seed=0, 
        drop_last=False,
        num_samples=None):

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            num_replicas = torch.distributed.get_world_size()
        
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            rank = torch.distributed.get_rank()
        
        self.len_dataset = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.indices = dataset.indices
        self.total_size = len(self.indices)
        
        # Compute the number of samples each process should handle

        if num_samples is None:
            self.num_samples = math.floor(self.total_size / self.num_replicas)
        else:
            self.num_samples = num_samples 
        
        if self.num_samples > self.total_size // (self.num_replicas / 4):
            print(f"num_samples is bigger than total size / (num_replicas/4). num_samples: {self.num_samples}, total size: {self.total_size}, total_size / (num_replicas / 4): {self.total_size / (self.num_replicas / 4)}")
            self.num_samples = math.floor(self.total_size // (self.num_replicas / 4))
            

        self.q = self.rank // 4
        print(f"Defining sampler for replica {self.rank} of {self.num_replicas}")
        if self.rank == 0:
            print("Total Dataset Size:", self.total_size)
            print("Num samples:", self.num_samples)

    def __iter__(self):
        # Deterministic shuffling based on seed and epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
        else:
            g = torch.Generator()
            g.manual_seed(0)

        iter_indices = torch.randperm(self.total_size, generator=g)[self.q  * self.num_samples : (self.q + 1)  * self.num_samples].tolist() # all processes in one node get the same time indices
        # iter_indices = torch.randperm(self.total_size, generator=g)[self.rank * self.num_samples : (self.rank+1) * self.num_samples].tolist()
        # if self.rank % 4 == 0:
        print(self.rank, iter_indices[0], iter_indices[-1])
        return iter(iter_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for the sampler. Useful for shuffling in distributed training.
        """
        self.seed = self.seed + epoch


if __name__ == "__main__":
    pass
    