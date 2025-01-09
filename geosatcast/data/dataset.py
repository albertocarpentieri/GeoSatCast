import xarray as xr
import h5py as h5
import numpy as np
import time
import os
from torch.utils.data import Dataset
import torch
from geosatcast.data.utils import cos_zenith_angle_from_timestamp

class HRSDataset(Dataset):
    def __init__(
        self,
        data_path,
        name,
        invariants_path,
        years,
        input_len,
        output_len,
        channels=np.arange(11),
        field_size=256,
        sza=False,
        length=10000
        ):
        
        self.data_path = data_path
        self.invariants_path = invariants_path
        self.years = years
        self.output_len = output_len if output_len else 0
        self.input_len = input_len
        self.seq_len = self.input_len + self.output_len
        self.channels = channels
        self.field_size = field_size
        self.sza = sza
        self.length = length 
        
        self.data_paths = {year: os.path.join(self.data_path, f"{year}_{name}.h5") for year in years}
        self.data_len = {year: h5.File(self.data_paths[year])["time"].shape[0] for year in years}
        self.time_idx = {year: np.arange(self.data_len[year] - self.seq_len) for year in years}
        self.get_latlon()
        if invariants_path:
            dem = xr.open_dataset(os.path.join(invariants_path, "dem.nc"))["DEM"].values
            dem[np.isnan(dem)] = -1
            dem = (dem - dem.mean()) / dem.std()

            lwmask = xr.open_dataset(os.path.join(invariants_path, "lwmask.nc"))["LWMASK"].values
            lwmask[lwmask==3] = 0
            self.inv = torch.from_numpy(np.stack((dem, lwmask)).astype(np.float32))
        
    def get_latlon(self):
        with h5.File(self.data_paths[self.years[0]], "r") as h:
            self.max_shape = h["fields"].shape[-2:]
            self.lon = h["longitude"][:]
            self.lat = h["latitude"][:]
        self.latlon_grid = np.stack(np.meshgrid(self.lon, self.lat))
        
    def get_data(self, year, t_i, lat_i, lon_i):
        x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
        # x = np.empty((self.seq_len, 11, *self.max_shape), dtype=np.float32)
        t = np.empty((self.seq_len),dtype=np.int64)
        with h5.File(self.data_paths[year], "r", swmr=True) as h:
            h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
            # h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :], np.s_[:])
            h["time"].read_direct(t, np.s_[t_i:t_i+self.seq_len], np.s_[:])
        # x = x[:, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        return x, t
    
    def getitem(self, i):
        sample = True
        while sample:
            year = np.random.choice(self.years)
            t_i = np.random.randint(self.data_len[year] - self.seq_len)
            lat_i = np.random.randint(low=0, high=self.max_shape[-2]-self.field_size)
            lon_i = np.random.randint(low=0, high=self.max_shape[-1]-self.field_size)
            x, t =  self.get_data(year, t_i, lat_i, lon_i)
            if ~np.isnan(x).any() and t[-1] - t[0] == (self.seq_len-1)*15*60:
                sample = False
            else:
                # self.time_idx[year] = np.setdiff1d(self.time_idx[year], t_i)
                # print(year, t_i, "with nans, new time idx length:", len(self.time_idx[year]))
                print(year, t_i)
        x = torch.from_numpy(x)
        
        inv = self.inv[:, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        sza = torch.from_numpy(np.stack([cos_zenith_angle_from_timestamp(
                t_, 
                self.latlon_grid[0, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size].flatten(),
                self.latlon_grid[1, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size].flatten())
                for t_ in t]))
        
        sza = sza.view(-1,1,self.field_size, self.field_size)
        t = torch.from_numpy(t)
        return x, t[...,None,None,None], inv[None], sza
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        x, t, inv, sza = self.getitem(i)
        return x, t, inv, sza


class SimpleDataset(Dataset):
    def __init__(
        self,
        data_path,
        name,
        years,
        input_len,
        output_len,
        channels=np.arange(11),
        field_size=256,
        length=10000
        ):
        
        self.data_path = data_path
        self.years = years
        self.output_len = output_len if output_len else 0
        self.input_len = input_len
        self.seq_len = self.input_len + self.output_len
        self.channels = channels
        self.field_size = field_size
        
        
        self.data_paths = {year: os.path.join(self.data_path, f"{year}_{name}.h5") for year in years}
        self.data_len = {year: h5.File(self.data_paths[year])["time"].shape[0] for year in years}
        self.timestamps = {year: h5.File(self.data_paths[year])["time"][:] for year in years}
        
        if length is None:
            self.length = 0
            for year in years:
                self.length += self.data_len[year] - self.seq_len 
        else:
            self.length = length 
        self.time_idx = {year: np.arange(self.data_len[year] - self.seq_len) for year in years}
        self.get_latlon()
        
    def get_latlon(self):
        with h5.File(self.data_paths[self.years[0]], "r") as h:
            self.max_shape = h["fields"].shape[-2:]
            self.lon = h["longitude"][:]
            self.lat = h["latitude"][:]
        self.latlon_grid = np.stack(np.meshgrid(self.lon, self.lat))
        
    def get_data(self, year, t_i, lat_i, lon_i):
        x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
        # x = np.empty((self.seq_len, 11, *self.max_shape), dtype=np.float32)
        with h5.File(self.data_paths[year], "r") as h:
            h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
            # h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :], np.s_[:])
        t = self.timestamps[year][t_i:t_i+self.seq_len]
        # x = x[:, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        return x, t
    
    def getitem(self, i):
        sample = True
        while sample:
            year = np.random.choice(self.years)
            t_i = np.random.randint(self.data_len[year] - self.seq_len)
            lat_i = np.random.randint(low=0, high=self.max_shape[-2]-self.field_size)
            lon_i = np.random.randint(low=0, high=self.max_shape[-1]-self.field_size)
            x, t =  self.get_data(year, t_i, lat_i, lon_i)
            if ~np.isnan(x).any() and t[-1] - t[0] == (self.seq_len-1)*15*60:
                sample = False
            else:
                print(year, t_i)
        x = torch.from_numpy(x)
        
        t = torch.from_numpy(t)
        return x, t[...,None,None,None]
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        x, t = self.getitem(i)
        return x, t
    

class WorkerDataset(Dataset):
    def __init__(
        self,
        data_path,
        name,
        years,
        input_len,
        output_len,
        channels=np.arange(11),
        field_size=256,
        length=10000
        ):
        
        self.data_path = data_path
        self.years = years
        self.output_len = output_len if output_len else 0
        self.input_len = input_len
        self.seq_len = self.input_len + self.output_len
        self.channels = channels
        self.field_size = field_size
        
        
        self.data_paths = {year: os.path.join(self.data_path, f"{year}_{name}.h5") for year in years}
        self.data_len = {year: h5.File(self.data_paths[year])["time"].shape[0] for year in years}
        self.timestamps = {year: h5.File(self.data_paths[year])["time"][:] for year in years}
        
        if length is None:
            self.length = 0
            for year in years:
                self.length += self.data_len[year] - self.seq_len 
        else:
            self.length = length 
        self.time_idx = {year: np.arange(self.data_len[year] - self.seq_len) for year in years}
        self.get_latlon()
        self.global_indices = self.generate_global_indices()
        
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
            for t_i in range(self.data_len[year] - self.seq_len):
                # for lat_i in range(0, self.max_shape[-2] - self.field_size, 24):
                #     for lon_i in range(0, self.max_shape[-1] - self.field_size, 24):
                indices.append((year, t_i))
        return indices

    def set_worker_indices(self, indices):
        self.worker_indices = []
        for t in indices:
            for lat_i in range(0, self.max_shape[-2] - self.field_size, 24):
                for lon_i in range(0, self.max_shape[-1] - self.field_size, 24):
                    self.worker_indices.append((t[0], t[1], lat_i, lon_i))
    
    def __len__(self):
        return len(self.worker_indices)
    
    def get_data(self, year, t_i, lat_i, lon_i):
        x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
        with h5.File(self.data_paths[year], "r") as h:
            h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
        t = self.timestamps[year][t_i:t_i+self.seq_len]
        return x, t
    
    def __getitem__(self, idx):
        year, t_i, lat_i, lon_i = self.worker_indices[idx]
        x, t = self.get_data(year, t_i, lat_i, lon_i)
        return torch.from_numpy(x), torch.from_numpy(t)[..., None, None, None]
            
    def __len__(self):
        return self.length



if __name__ == "__main__":
    print("debugging")
    import os 
    from torch.utils.data import DataLoader
    from torch.utils.data import get_worker_info

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

    # Assign worker-specific indices to the dataset
    dataset = WorkerDataset(
        data_path="/capstor/scratch/cscs/acarpent/",
        name="virtual",
        years=[2021],
        input_len=8,
        output_len=None,
        length=None
        )
    
    
    print("dataset initialized")
    num_workers=16
    batch_size=32
    print(num_workers, batch_size)
    print(torch.cuda.is_available())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,  # Initialize workers with partitioned indices
        persistent_workers=True,
    )
    
    # Warm-up
    print("Starting warm-up...")
    warmup_start = time.time()
    for _ in dataloader:
        break
    print(f"Warm-up completed in {time.time() - warmup_start:.2f}s")

    for i in range(20):
        b = 0
        s = s_init = time.time()
        for batch in dataloader:
            b += 1
            st = time.time()
            print(st - s)
            s = st
            if b == 999:
                break
        print("##########################")
        print(i, time.time()-s_init)
        x, _ = batch 
        print(x.device, x.shape, b)
        print()

   