import xarray as xr
import h5py as h5
import numpy as np
import time
import os
from torch.utils.data import Dataset
import torch
from geosatcast.data.utils import cos_zenith_angle_from_timestamp

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

class WorkerDataset(Dataset):
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
        # self.files = {year: h5.File(self.data_paths[year], "r") for year in years}
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
            timestamps = self.timestamps[year]
            diff = timestamps[self.seq_len:] - timestamps[:-self.seq_len]
            idx = np.where(diff == 15*60*(self.seq_len))[0]
            for i in idx:
                indices.append((year, int(i)))
        self.len_worker_indices = len(indices)
        self.worker_indices = indices
        return indices

    def set_worker_indices(self, indices):
        self.worker_indices = indices
        self.len_worker_indices = len(indices)
    
    def __len__(self):
        return self.length
    
    def get_data(self, year, t_i, lat_i, lon_i):
        x = np.empty((self.seq_len, 11, self.field_size, self.field_size), dtype=np.float32)
        # self.files[year]["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len, :, lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
        # x = np.empty((self.seq_len, 11, *self.max_shape), dtype=np.float32)
        with h5.File(self.data_paths[year], "r") as h:
            h["fields"].read_direct(x, np.s_[t_i:t_i+self.seq_len,:,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size], np.s_[:])
        # x = x[:,:,lat_i:lat_i+self.field_size, lon_i:lon_i+self.field_size]
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
        # i = np.random.randint(0, self.len_worker_indices)
        i = int(idx % self.len_worker_indices)
        lat_i = np.random.randint(0, self.max_shape[-2] - self.field_size)
        lon_i = np.random.randint(0, self.max_shape[-1] - self.field_size)
        year, t_i = self.worker_indices[i]
        x, t, inv, sza = self.get_data(year, t_i, lat_i, lon_i)
        return x, t, inv, sza


class SimpleDataset(Dataset):
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
        # self.files = {year: h5.File(self.data_paths[year], "r") for year in years}
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
            timestamps = self.timestamps[year]
            diff = timestamps[self.seq_len:] - timestamps[:-self.seq_len]
            idx = np.where(diff == 15*60*(self.seq_len))[0]
            for i in idx:
                indices.append((year, int(i)))
        self.len_worker_indices = len(indices)
        return indices

    def set_worker_indices(self, indices):
        self.worker_indices = indices
        self.len_worker_indices = len(indices)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # i = np.random.randint(0, self.len_worker_indices)
        i = int(idx % self.len_worker_indices)
        lat_i = np.random.randint(0, self.max_shape[-2] - self.field_size)
        lon_i = np.random.randint(0, self.max_shape[-1] - self.field_size)
        year, t_i = self.worker_indices[i]
        x = torch.randn((11, self.seq_len, self.field_size, self.field_size), dtype=torch.float32)
        t = torch.randn((1, self.seq_len, 1, 1), dtype=torch.float32)
        inv = torch.randn((2, 1, self.field_size, self.field_size), dtype=torch.float32)
        sza = torch.randn((1, self.seq_len, self.field_size, self.field_size), dtype=torch.float32)
        return x, t, inv, sza


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
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    num_workers = 32
    batch_size = 64
    pref_f = 4
    data_path = "/capstor/scratch/cscs/acarpent/SEVIRI/"
    invariants_path = "/capstor/scratch/cscs/acarpent/SEVIRI/invariants/"

    # Create the dataset
    dataset = WorkerDataset(
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
    sampler = DistributedSampler(dataset)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=pref_f,
        # worker_init_fn=worker_init_fn,
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
            x, t, inv, sza = batch
            x = x.to(device)
            t = t.to(device)
            inv = inv.to(device)
            sza = sza.to(device)
            b += 1
        print(f"Time to retrieve {b} batches: {time.time() - warmup_start}")

        for batch in dataloader:
            x, t, inv, sza = batch
            x = x.to(device)
            t = t.to(device)
            inv = inv.to(device)
            sza = sza.to(device)
            b += 1
        print(f"Time to retrieve {b} batches: {time.time() - warmup_start}")

    print(prof.key_averages().table(sort_by="cpu_time_total"))

    # Cleanup
    dist.destroy_process_group()

# if __name__ == "__main__":
#     print("debugging")
#     import os 
#     from torch.utils.data import DataLoader
#     from torch.utils.data import get_worker_info
#     from torch.profiler import profile, record_function, ProfilerActivity


#     # h5.get_config().mpi = True


#     def worker_init_fn(worker_id):
#         worker_info = get_worker_info()
#         dataset = worker_info.dataset
#         num_workers = worker_info.num_workers
#         total_samples = len(dataset.global_indices)
        
#         # Split the global indices among workers
#         chunk_size = np.ceil(total_samples / num_workers)
#         start_idx = int(worker_id * chunk_size)
        
#         end_idx = int(min(start_idx + chunk_size, total_samples))
#         worker_indices = dataset.global_indices[start_idx:end_idx]
#         dataset.set_worker_indices(worker_indices)
#         print(worker_id, start_idx, len(dataset.worker_indices))

#     num_workers=16
#     batch_size=64
#     pref_f = 4

#     # Assign worker-specific indices to the dataset
#     dataset = WorkerDataset(
#         data_path="/capstor/scratch/cscs/acarpent/SEVIRI/",
#         invariants_path="/capstor/scratch/cscs/acarpent/SEVIRI/invariants/",
#         name="new_virtual",
#         years=[2017, 2018, 2019],
#         input_len=1,
#         output_len=None,
#         field_size=256,
#         length=batch_size*100
#         )
    
    
#     print("dataset initialized")
#     print(num_workers, batch_size, pref_f)
#     print(torch.cuda.is_available())

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         prefetch_factor=pref_f,
#         worker_init_fn=worker_init_fn,  # Initialize workers with partitioned indices
#         persistent_workers=True,
#     )
    
#     # Warm-up
#     print("Starting warm-up...")
#     warmup_start = time.time()
#     b = 0
#     with profile(
#         activities=[ProfilerActivity.CPU],
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('/capstor/scratch/cscs/acarpent/log')
#     ) as prof:
#         for batch in dataloader:
#             x, t, inv, sza = batch
#             x = x.to("cuda")
#             t = t.to("cuda")
#             inv = inv.to("cuda")
#             sza = sza.to("cuda")
#             b += 1
#         print(f"Time to retrieve {b} batches:", time.time() - warmup_start)
#         for batch in dataloader:
#             b += 1
#         print(f"Time to retrieve {b} batches:", time.time() - warmup_start)
#     print(prof.key_averages().table(sort_by="cpu_time_total"))

#     with profile(
#         activities=[ProfilerActivity.CPU],
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('/capstor/scratch/cscs/acarpent/log')
#     ) as prof:
#         for batch in dataloader:
#             b += 1
#         print(f"Time to retrieve {b} batches:", time.time() - warmup_start)
#         for batch in dataloader:
#             b += 1
#         print(f"Time to retrieve {b} batches:", time.time() - warmup_start)
#     print(prof.key_averages().table(sort_by="cpu_time_total"))


   