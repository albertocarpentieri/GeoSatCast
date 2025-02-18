import math
import xarray as xr
import h5py as h5
import numpy as np
import time
import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from collections import defaultdict

from geosatcast.data.utils import cos_zenith_angle_from_timestamp  # If needed

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
        """
        :param data_path: Directory containing multiple HDF5 files for each year
        :param invariants_path: Path to .nc files with DEM and LWMASK
        :param name: Suffix or pattern in the HDF5 filenames (e.g., '16b_datasets')
        :param years: List of years to include
        :param input_len: Number of timesteps to use as input
        :param output_len: Number of timesteps to predict (sequence length = input_len + output_len)
        :param channels: Channels to use from the 'fields' dataset
        :param field_size: Spatial size (height=width=field_size)
        :param length: Arbitrary maximum length
        :param validation: Whether this dataset is used for validation
        :param load_full: If True, read the entire 2D domain, then slice
        :param rank: For distributed sampling
        :param mask_sza: Whether to mask certain channels based on SZA
        :param dtype: Float precision (16 or 32)
        """
        super().__init__()
        
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
        
        # Store means and stds as torch tensors
        self.means = torch.from_numpy(np.array(MEANS)[:, None, None, None]).type(self.torch_dtype)
        self.stds = torch.from_numpy(np.array(STDS)[:, None, None, None]).type(self.torch_dtype)
        
        # Load invariants (DEM, LWMASK)
        dem = xr.open_dataset(os.path.join(invariants_path, "dem.nc"))["DEM"].values
        dem[np.isnan(dem)] = -1
        dem = (dem - dem.mean()) / dem.std()

        lwmask = xr.open_dataset(os.path.join(invariants_path, "lwmask.nc"))["LWMASK"].values
        lwmask[lwmask == 3] = 0
        
        self.inv = torch.from_numpy(np.stack((dem, lwmask)).astype(np.float32)).type(self.torch_dtype)
        
        # Prepare a dict of lists: each year -> list of h5 files
        self.data_paths = {}
        self.timestamps = {}  # year -> list of np arrays
        self.data_len = {}    # year -> list of lengths
        
        for year in years:
            # Example pattern: year_*.h5 or year_*_{name}.h5, adapt to your filenames
            pattern = os.path.join(self.data_path, f"{year}_{dtype}b_datasets/{year}_*.h5")
            file_list = sorted(glob.glob(pattern))
            self.data_paths[year] = file_list
            
            # read each file's timestamps
            file_timestamps = []
            file_lengths = []
            for fpath in file_list:
                with h5.File(fpath, "r") as f:
                    ts = f["time"][:]  # shape [T]
                    file_timestamps.append(ts)
                    file_lengths.append(len(ts))
            self.timestamps[year] = file_timestamps
            self.data_len[year] = file_lengths
        
        # We'll figure out the shape from the first file of the first year
        self.get_latlon()
        
        self.generate_global_indices()
        
        self.validation = validation
        self.rank = rank
        self.load_full = load_full
        self.length = length  # Not strictly required if we rely on self.len_indices
    
    def get_latlon(self):
        """
        Retrieves spatial dimensions and lat/lon from the first file
        (assuming all files share the same grid).
        """
        first_year = self.years[0]
        first_file = self.data_paths[first_year][0]
        with h5.File(first_file, "r") as h:
            # shape of 'fields' is [time, height, width, channels]
            # so shape[1:3] is the full domain size
            self.max_shape = h["fields"].shape[1:3]
            self.lon = h["longitude"][:]
            self.lat = h["latitude"][:]
    
    def generate_global_indices(self):
        """
        Build a list of (year, file_idx, t_i) that represent valid
        sequence start indices. We skip sequences that don't line up
        with the required 15-minute * seq_len gap.
        """
        indices = []
        
        # We'll store tuples: (year, file_idx, local_t_idx)
        for year in self.years:
            file_list = self.data_paths[year]
            time_arrays = self.timestamps[year]
            
            for file_idx, times in enumerate(time_arrays):
                if len(times) < self.seq_len:
                    continue
                # For example, you want consecutive times with a specific gap
                required_gap = 15 * 60 * self.seq_len  # 15 min * 60 sec * seq_len
                diff = times[self.seq_len:] - times[:-self.seq_len]
                valid_idx = np.where(diff == required_gap)[0]
                
                for i_local in valid_idx:
                    indices.append((year, file_idx, int(i_local)))
        
        self.indices = indices
        self.len_indices = len(indices)
        print(f"{self.len_indices} global indices computed.")
    
    def __len__(self):
        return self.len_indices
    
    def __getitem__(self, idx):
        # If you are using distributed sampling, you might fetch from self.worker_indices
        
        year, file_idx, t_i = self.indices[idx]
        
        if self.validation:
            sampler = np.random.default_rng(int(idx * self.rank)).integers
        else:
            sampler = np.random.randint
        
        lat_i = sampler(0, self.max_shape[0] - self.field_size)
        lon_i = sampler(0, self.max_shape[1] - self.field_size)
        
        fpath = self.data_paths[year][file_idx]
        
        try:
            return self.get_data(fpath, t_i, lat_i, lon_i)
        except Exception as e:
            raise Exception(f"{idx}: Error reading year={year}, file={fpath}, t_i={t_i}") from e
    
    def get_data(self, fpath, t_i, lat_i, lon_i):
        """
        Actually read the data from the given file fpath, at local index t_i.
        Slices out [seq_len, field_size, field_size, 12] from 'fields'.
        Then prepares Torch tensors and does standardization.
        """
        if not self.load_full:
            # Create the array to hold the read slice
            x = np.empty(
                (self.seq_len, self.field_size, self.field_size, 12),
                dtype=self.numpy_dtype
            )
            with h5.File(fpath, "r") as h:
                h["fields"].read_direct(
                    x,
                    np.s_[
                        t_i : t_i + self.seq_len,
                        lat_i : lat_i + self.field_size,
                        lon_i : lon_i + self.field_size,
                        :
                    ],
                    np.s_[:],
                )
                t_arr = h["time"][t_i : t_i + self.seq_len]
        else:
            # Read the entire domain, then slice
            with h5.File(fpath, "r") as h:
                x = np.empty(
                    (self.seq_len, *self.max_shape, 12),
                    dtype=self.numpy_dtype
                )
                h["fields"].read_direct(
                    x,
                    np.s_[t_i : t_i + self.seq_len, :],
                    np.s_[:],
                )
                t_arr = h["time"][t_i : t_i + self.seq_len]
            x = x[
                :,
                lat_i : lat_i + self.field_size,
                lon_i : lon_i + self.field_size,
                :
            ]
        
        # # Read the time array
        # with h5.File(fpath, "r") as h:
        #     t_arr = h["time"][t_i : t_i + self.seq_len]
        
        t = torch.from_numpy(t_arr).type(torch.float32)[None, ..., None, None]
        
        # Invariants for the subdomain
        inv = self.inv[:, None, lat_i : lat_i + self.field_size, lon_i : lon_i + self.field_size]
        
        # Convert to torch
        x = torch.from_numpy(x).type(self.torch_dtype)  # shape: [seq_len, H, W, 12]
        # Permute to [12, seq_len, H, W]
        x = x.permute(3, 0, 1, 2).contiguous()
        
        # Splitting out the last channel (sza) from the rest:
        x, sza = x[:-1], x[-1:]
        
        if self.mask_sza:
            # For example, mask channels 0,7,8 if sza <= -0.07
            x[[0, 7, 8]] = x[[0, 7, 8]] * (sza > -0.07)
        
        # Standardize
        x = (x - self.means) / self.stds
        
        # Return (input_data, times, invariants, sza)
        return x, t, inv, sza


class WorkerDistributedSampler(Sampler):
    """
    A sampler that:
      - Groups dataset indices by (year, file_idx).
      - Every epoch, randomly permutes *all* file keys.
      - Partitions the permuted file keys among ranks, so each rank exclusively
        processes a unique subset of files.
      - Within that subset, optionally shuffles sample indices themselves.
      - Ensures each rank yields exactly `num_samples` each epoch by:
         * Truncating if local_indices > num_samples
         * Cycling (replicating) if local_indices < num_samples

    Requirements:
      * dataset.indices must be a list of (year, file_idx, local_t_idx).
      * file_key = (year, file_idx).
      * `num_samples` must be provided (no default).
    
    By calling set_epoch(epoch) each epoch, the file-to-rank mapping changes
    (if shuffle=True), ensuring across epochs different ranks see different files.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        num_samples=None,  # Required: each rank must produce exactly this many samples
    ):
        super().__init__(dataset)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            num_replicas = torch.distributed.get_world_size()
        
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available.")
            rank = torch.distributed.get_rank()

        if num_samples is None:
            raise ValueError("`num_samples` must be specified so each rank produces exactly that many samples.")

        self.dataset = dataset
        self.indices = dataset.indices  # a list of (year, file_idx, local_t_idx)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = num_samples

        # 1. Group sample indices by file_key = (year, file_idx)
        file_to_indices = defaultdict(list)
        for global_idx, (year, fidx, local_idx) in enumerate(self.indices):
            file_key = (year, fidx)
            file_to_indices[file_key].append(global_idx)

        # Store a list of unique file_keys and the mapping from file_key -> global indices
        self.file_keys = sorted(file_to_indices.keys())
        self.file_to_indices = file_to_indices
        self.num_files = len(self.file_keys)

        self.epoch = 0  # track current epoch (for seeding)

        # Debug info at rank=0
        if self.rank == 0:
            print(f"[FileDistributedSampler] num_files: {self.num_files} | num_replicas: {self.num_replicas} | shuffle: {self.shuffle}")
            print(f"[FileDistributedSampler] Each rank must produce exactly num_samples={self.num_samples}")

    def __iter__(self):
        """
        Called by the DataLoader at the start of each epoch:
          1. Shuffle all file_keys if self.shuffle=True, else keep them in sorted order.
          2. Partition them among ranks.
          3. Gather the sample indices from these file_keys.
          4. Shuffle those local sample indices (if shuffle=True).
          5. Truncate or replicate to ensure we have exactly self.num_samples.
        """
        g = torch.Generator()
        # Combine seed + epoch so we get a new random state each epoch
        g.manual_seed(self.seed + self.epoch)  

        # (1) Shuffle file_keys if requested
        if self.shuffle:
            file_perm = torch.randperm(self.num_files, generator=g).tolist()
            shuffled_file_keys = [self.file_keys[i] for i in file_perm]
        else:
            # If not shuffling, we can keep them in sorted order
            shuffled_file_keys = self.file_keys
        
        # (2) Partition file_keys among ranks
        files_per_rank = math.ceil(self.num_files / self.num_replicas)
        start_file_idx = 0 #self.rank * files_per_rank
        end_file_idx = self.num_files #min(start_file_idx + files_per_rank, self.num_files)
        local_file_keys = shuffled_file_keys[start_file_idx:end_file_idx]
        print(local_file_keys)
        
        # (3) Gather all sample indices from those local_file_keys
        local_indices = []
        for fkey in local_file_keys:
            local_indices.extend(self.file_to_indices[fkey])
        local_size = len(local_indices)

        # (4) Shuffle local sample indices if needed
        if self.shuffle:
            idx_perm = torch.randperm(local_size, generator=g).tolist()
            local_indices = [local_indices[i] for i in idx_perm]

        # (5) Adjust to exactly self.num_samples
        #   - If local_size > num_samples, truncate
        #   - If local_size < num_samples, cycle/replicate
        if local_size >= self.num_samples:
            local_indices = local_indices[:self.num_samples]
        else:
            # Replicate from the front to reach num_samples
            if local_size == 0:
                # Edge case: if this rank has no data at all, we can't replicate
                # Could either raise error or produce an empty list
                raise ValueError(f"Rank {self.rank} has zero local samples, cannot replicate an empty list.")
            repeats_needed = self.num_samples - local_size
            # We'll cycle from the start
            # Another approach is to keep picking from local_indices at random
            for i in range(repeats_needed):
                local_indices.append(local_indices[i % local_size])

        return iter(local_indices)

    def __len__(self):
        """
        We always produce exactly `self.num_samples` local indices per epoch.
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Allows the user (typically the training loop) to set the epoch number.
        This modifies the seed used in __iter__, so each epoch can yield a new
        file partition + new local shuffle.
        """
        self.epoch = epoch


if __name__ == "__main__":
    pass
    