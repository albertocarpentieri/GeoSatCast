import os
import h5py
import numpy as np
import json
import pandas as pd
import re

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def check_h5_files(folder_path, folder_names, output_file="results.json"):
    missing_timestamps = []  # To track missing timestamps
    files_with_nans = []     # To track files containing NaNs in 'fields'
    files_with_negs = []
    channel_sums = None
    channel_squared_sums = None
    total_samples = 0
    
    for folder_name in folder_names:
        year = int(folder_name[:4])
        data_path = os.path.join(folder_path, folder_name)
        csv_df = pd.read_csv(os.path.join(folder_path, f"{year}.csv"), sep=";")
        print(csv_df)
        print(len(csv_df))

        # Get a sorted list of HDF5 files in the folder
        h5_files = sorted([f for f in os.listdir(data_path) if f.endswith('.h5')], key=natural_sort_key)
        print(len(h5_files))
        h5_files = [h5_files[i] for i in csv_df.index if csv_df.iloc[i, 2]>0]
        print(h5_files)
        for h5_file in h5_files:
            file_path = os.path.join(data_path, h5_file)
            print(f"Checking file: {file_path}")

            with h5py.File(file_path, 'r') as h5:
                # Check for missing timestamps
                if 'time' in h5:
                    timestamps = h5['time'][:]
                    diffs = np.diff(timestamps)
                    if not np.all(diffs == diffs[0]):
                        missing_timestamps.append(h5_file)

                # Check for NaNs in 'fields'
                if 'fields' in h5:
                    fields = h5['fields'][:]
                    time_values = h5['time'][:]
                    num_samples = fields.shape[0]  # Number of time samples
                    idx = np.arange(num_samples)
                    if np.isnan(fields).any():
                        nan_sum = np.isnan(fields).sum(axis=(1,2,3))
                        nan_idx = np.where(nan_sum)[0]
                        files_with_nans += [time_values[int(i)] for i in nan_idx]
                        num_samples -= len(nan_idx)
                        idx = [i for i in idx if i not in nan_idx]
                        print(h5_file, nan_idx)
                    
                    if (fields<0).any():
                        neg_sum = (fields < 0).sum(axis=(1,2,3))
                        neg_idx = np.where(neg_sum)[0]
                        files_with_negs += [time_values[int(i)] for i in neg_idx]
                        num_samples -= len(neg_idx)
                        idx = [i for i in idx if i not in neg_idx]
                        print(h5_file, neg_idx)
                    
                    print("The filtered data is ok:", ~np.isnan(fields[idx]).any(),  (fields[idx]>=0).all())
                    channel_min = np.nanmin(fields[idx], axis=(0, 2, 3))
                    channel_max = np.nanmax(fields[idx], axis=(0, 2, 3))
                    channel_sum = np.nansum(fields[idx], axis=(0, 2, 3))  # Sum over time, lat, lon
                    channel_squared_sum = np.nansum(fields[idx]**2, axis=(0, 2, 3))  # Sum of squares

                    if channel_sums is None:
                        channel_sums = channel_sum
                        channel_squared_sums = channel_squared_sum
                        channel_mins = channel_min
                        channel_maxs = channel_max    
                    else:
                        channel_mins = [float(min(channel_mins[i], channel_min[i])) for i in range(len(channel_min))]
                        channel_maxs = [float(max(channel_maxs[i], channel_max[i])) for i in range(len(channel_min))]
                        channel_sums += channel_sum
                        channel_squared_sums += channel_squared_sum
                    print(channel_mins, channel_maxs)

                    total_samples += num_samples * fields.shape[2] * fields.shape[3]  # Total spatial points
    # Calculate mean and std
    results = {}
    if channel_sums is not None:
        channel_means = channel_sums / total_samples
        channel_stds = np.sqrt((channel_squared_sums / total_samples) - channel_means**2)

        results["channel_stats"] = {
            f"channel_{i}": {"mean": float(mean), "std": float(std), "max": float(M), "min": float(m)}
            for i, (mean, std, M, m) in enumerate(zip(channel_means, channel_stds, channel_maxs, channel_mins))
        }

    # Save results
    results["missing_timestamps"] = missing_timestamps
    results["files_with_nans"] = files_with_nans
    results["files_with_negs"] = files_with_negs

    print(results)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Report results
    print("\nCheck complete! Results saved to", output_file)

    if missing_timestamps:
        print("\nFiles with missing timestamps:")
        for file in missing_timestamps:
            print(f"- {file}")
    else:
        print("\nNo missing timestamps found.")

    if files_with_nans:
        print("\nFiles with NaNs in 'fields':")
        for file in files_with_nans:
            print(f"- {file}")
    else:
        print("\nNo NaNs found in 'fields' dataset.")

    if "channel_stats" in results:
        print("\nMean and Standard Deviation per channel:")
        for channel, stats in results["channel_stats"].items():
            print(f"{channel}: Mean = {stats['mean']:.6f}, Std = {stats['std']:.6f}")

if __name__ == "__main__": 
    folder_path = "/capstor/scratch/cscs/acarpent/SEVIRI"
    folder_names = ["2017_weekly_datasets", "2018_weekly_datasets", "2019_weekly_datasets"]
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        output_file = "/capstor/scratch/cscs/acarpent/SEVIRI/SEVIRI_2017-2019.json"
        check_h5_files(folder_path, folder_names, output_file=output_file)
    else:
        print("Invalid folder path. Please try again.")
