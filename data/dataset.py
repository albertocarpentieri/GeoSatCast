import xarray as xr
import h5py as h5
import numpy as np
import time
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from data.utils import cos_zenith_angle_from_timestamp


class Dataset(Dataset):
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
        max_shape=(),
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
        x = x[:, :, ]
        return x, t
    
    def getitem(self, i):
        year = np.random.choice(self.years)
        t_i = np.random.randint(low=0, high=self.data_len[year]-self.seq_len)
        lat_i = np.random.randint(low=0, high=self.max_shape[-2]-self.field_size)
        lon_i = np.random.randint(low=0, high=self.max_shape[-1]-self.field_size)

        s = time.time()
        x, t =  self.get_data(year, t_i, lat_i, lon_i)

        x = torch.from_numpy(x)
        
        inv = self.inv[:, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
        
        s = time.time()
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
        sample = True
        while sample:
            x, t, inv, sza = self.getitem(i)
            if ~torch.isnan(x).any() and t[-1] - t[0] == (self.seq_len-1)*15*60:
                sample = False
            else:
                print(i, "with nans")
                i += 1
        return x, t, inv, sza


if __name__ == "__main__":
    import os 
    from torch.utils.data import DataLoader
    np.random.seed(0)

    dataset = Dataset(data_path="/scratch/snx3000/acarpent/SEVIRI/",
                  name="virtual",
                  invariants_path="/scratch/snx3000/acarpent/SEVIRI_inv/",
                  years=[2017],
                  input_len=8,
                  output_len=None,
                  sza=True,
                  length=1024
                  )

    dataloader = DataLoader(
        dataset,
        batch_size=32, 
        shuffle=False, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=24, 
        pin_memory=True, 
        persistent_workers=True)
    
    s = time.time()
    for batch in dataloader:
        pass
    print(time.time()-s)