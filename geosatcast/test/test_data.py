#!/usr/bin/env python
import numpy as np
import torch
import pickle as pkl
import datetime
import argparse
import time

from geosatcast.data.distributed_dataset import DistributedDataset
from metrics import (
    compute_temporal_autocorrelation,
    compute_spatial_autocorrelation,
    compute_intra_channel_corr

)

##############################################
# Argument Parsing
##############################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0,
                        help="Which subset index to process?")
    parser.add_argument("--n_subsets", type=int, default=4,
                        help="Total number of subsets (processes).")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Which GPU to use (e.g. 0,1,2,3).")
    parser.add_argument("--in_steps", type=int, default=2,
                        help="Number of input timesteps.")
    parser.add_argument("--n_forecast_steps", type=int, default=8,
                        help="Number of forecast steps (appended to in_steps).")
    parser.add_argument("--field_size", type=int, default=256,
                        help="Spatial size of the subregion.")
    parser.add_argument("--lat_i", type=int, default=124,
                        help="Latitude index to extract.")
    parser.add_argument("--lon_i", type=int, default=124,
                        help="Longitude index to extract.")
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Number of samples to test.")
    args = parser.parse_args()
    return args

##############################################
# Main Analysis Routine
##############################################
def test():
    args = parse_args()

    subset_idx = args.subset
    total_subsets = args.n_subsets
    gpu_id = args.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    in_steps = args.in_steps
    n_forecast_steps = args.n_forecast_steps
    field_size = args.field_size
    lat_i = args.lat_i
    lon_i = args.lon_i
    n_samples = args.n_samples

    # Prepare the dataset (using the same parameters as your forecasting evaluation).
    dataset = DistributedDataset(
        data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
        invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
        name='32b_virtual',
        years=[2021],
        input_len=in_steps + n_forecast_steps,
        output_len=None,
        channels=np.arange(11),
        field_size=field_size,
        length=None,
        validation=True,
        rank=0
    )

    # Random sampling of dataset indices.
    np.random.seed(0)
    full_range = np.arange(len(dataset.indices))
    np.random.shuffle(full_range)
    full_range = full_range[:n_samples]

    # Partition indices into subsets.
    chunk_size = int(np.ceil(n_samples / total_subsets))
    start = subset_idx * chunk_size
    end = min(start + chunk_size, n_samples)
    indices = full_range[start:end]
    print(f"Subset {subset_idx}: Processing {len(indices)} samples out of {n_samples}")

    # Dictionaries/lists to accumulate metrics.
    results = {}
    temporal_vals = []
    spatial_vals_list = []  # will store dicts per sample
    power_vals_list = []    # list of tensors (for few examples)
    intra_vals = []

    # Loop over the chosen sample indices.
    for idx in indices:
        print(f"Processing dataset index: {idx}")
        year, t_i = dataset.indices[idx]

        # Retrieve data: x (shape: [C, T, H, W]), t, invariants, sza.
        x, t, inv, sza = dataset.get_data(
            year=year, t_i=t_i,
            lat_i=lat_i, lon_i=lon_i
        )
        t = t[0, :, 0, 0].numpy().astype(int)
        x = x.to(device)  # x is expected to have shape (C, T, H, W)
        sza = sza.to(device)
        sample = x if x.ndim == 4 else x[0]

        # --- SZA Filtering ---
        # In test_nowcast.py, channels 0, 7, and 8 are filtered on forecast steps (i.e. time indices from in_steps onward)
        if sza is not None:
            # Assume sza shape is (1, T, H, W)
            forecast_start = in_steps
            # Use the same sza mask as in your nowcast_test.py: filter where sza < 0.
            sza_mask = sza[:, forecast_start:, :, :]
            for c in [0, 7, 8]:
                # For forecast timesteps, set sample pixels to NaN if sza_mask < 0.
                sample[c, forecast_start:] = torch.where(sza_mask[0] < 0, torch.tensor(float('nan'), device=device), sample[c, forecast_start:])

        # --- Compute Metrics ---
        temp_acorr   = compute_temporal_autocorrelation(sample[:,forecast_start:], lags=[1,2,4,8])
        # Spatial autocorrelation for shifts 1, 2, and 3 (both horizontal and vertical)
        spat_acorr   = compute_spatial_autocorrelation(sample[:,forecast_start:], shifts=[1,2,4,8])
        # Compute power metric on this sample. (This can be expensive, so you might call it only for a few examples.)
        # power_metric = compute_power_metric(sample)  # returns a tensor of shape (C, T)
        intra_corr   = compute_intra_channel_corr(sample[:,forecast_start:])

        # Use the sample’s first timestamp as key (or use the index if conversion fails)
        try:
            init_time = datetime.datetime.utcfromtimestamp(t[in_steps])
        except Exception:
            init_time = f"index_{idx}"

        results[init_time] = {
            "time": t[in_steps:],
            "temporal_autocorrelation": temp_acorr,
            "spatial_autocorrelation": spat_acorr,  # dictionary of correlations
            "intra_channel_corr": intra_corr
        }
    

    # Save partial results to a pickle file.
    out_file = (
        f"/capstor/scratch/cscs/acarpent/test_results/"
        f"analysis_results_subset{subset_idx}_fs{field_size}_lat{lat_i}_lon{lon_i}.pkl"
    )
    with open(out_file, "wb") as o:
        pkl.dump(results, o)
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    test()