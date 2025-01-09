import os
import h5py
import numpy as np
import json
import pandas as pd
import re

NON_HRV_BANDS = [
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
]

MIN_VALUES = [
    0,
    150,
    150,
    150,
    150,
    150,
    150,
    0,
    0,
    150,
    150
]

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def check_h5_files(folder_path, folder_names, output_file="results.json"):
    channel_sums = None
    channel_squared_sums = None
    total_samples = 0
    
    for folder_name in folder_names:
        year = int(folder_name[:4])
        data_path = os.path.join(folder_path, folder_name)


        # Get a sorted list of HDF5 files in the folder
        h5_files = sorted([f for f in os.listdir(data_path) if f.endswith('.h5') and "new" not in f], key=natural_sort_key)
        for h5_file in h5_files:
            file_path = os.path.join(data_path, h5_file)
            
            print(f"Checking file: {file_path}")
            if "new_"+h5_file in os.listdir(data_path):
                file_path = os.path.join(data_path, "new_"+h5_file)
                print("Switching to corrected data: ", file_path)
            
            # RETRIEVE DATA
            with h5py.File(file_path, 'r') as h5:
                data = h5["fields"][:]
                timestamps = h5["time"][:]
                lon = h5["longitude"][:]
                lat = h5["latitude"][:]
            print("data shape:", data.shape)
            # CORRECT DATA
            corrected = False
            for c in range(11):
                cond = (np.isnan(data[:,c])) | (data[:,c] < MIN_VALUES[c])
                nan_idx = np.where(cond)
                
                if len(nan_idx[0])>0:
                    time_nan_idx = nan_idx[0]
                    lat_nan_idx = nan_idx[1]
                    lon_nan_idx = nan_idx[2]

                    fill_lat_idx = np.array([lat_nan_idx + k for k in [-1, -1, -1, 0, 0, 0, 1, 1, 1]])
                    fill_lat_idx[fill_lat_idx<0] = 0
                    fill_lat_idx[fill_lat_idx>=data[:,c].shape[1]] = data[:,c].shape[1]-1
                    fill_lon_idx = np.array([lon_nan_idx + k for k in [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
                    fill_lon_idx[fill_lon_idx<0] = 0
                    fill_lon_idx[fill_lon_idx>=data[:,c].shape[2]] = data[:,c].shape[2]-1

                    stacked_x = np.nanmean(np.stack([data[:,c][time_nan_idx, fill_lat_i, fill_lon_i] for fill_lat_i, fill_lon_i in zip(fill_lat_idx, fill_lon_idx)], axis=0), axis=0)
                    data[:,c][nan_idx] = stacked_x
                    print(f"data has still nans in channel {c+1}:", np.isnan(data[:, c]).any())
                    
                    print("Now looking at minimum values")
                    
                    cond_sum = np.sum(data[:, c] < MIN_VALUES[c], axis=(1,2))
                    print(cond_sum)
                    replace_idx = (cond_sum < 0.05 * data.shape[-2]*data.shape[-1]) & (cond_sum > 0.)
                    print(replace_idx)
                    r = data[replace_idx, c]
                    replace_cond = r < MIN_VALUES[c]
                    print(replace_cond.sum(axis=(1,2)))
                    print(r[replace_cond])
                    r[replace_cond] = MIN_VALUES[c]
                    print(r[replace_cond])
                    data[replace_idx, c] = r
                    print(data[replace_idx, c][replace_cond])
                    # cond_sum = np.sum(data[:, c] < MIN_VALUES[c], axis=(1,2))
                    # print(cond_sum)
                    print(f"data has still negs in channel {c+1}:", (data[:, c] < MIN_VALUES[c]).any())

                    corrected = True
            
            # SAVE CORRECTED DATA
            if corrected:
                cond = (np.isnan(data)) | (data < np.array(MIN_VALUES)[None,:,None,None])
                cond_sum = np.sum(cond, axis=(1,2,3))
                idx = np.where(cond_sum==0)[0]
                
                
                data = data[idx]
                print("new corrected data shape:", data.shape)
                timestamps = timestamps[idx]
                print("Saving new_"+h5_file)
                with h5py.File(os.path.join(data_path, "new_"+h5_file), "w") as new_hf:
                    lat_dataset = new_hf.create_dataset(
                        "latitude",
                        data=lat.astype(np.float32),
                        )
                    lat_dataset.attrs["units"] = "degrees north"
                    
                    lon_dataset = new_hf.create_dataset(
                        "longitude",
                        data=lon.astype(np.float32),
                        )
                    lon_dataset.attrs["units"] = "degrees east"

                    time_dataset = new_hf.create_dataset(
                        "time",
                        data=timestamps,
                        )
                    time_dataset.attrs["units"] = "unix time [s]"
                    
                    channel_dataset = new_hf.create_dataset(
                        "channels",
                        data=NON_HRV_BANDS
                        )
                    
                    fields = new_hf.create_dataset(
                        "fields", 
                        data=data.astype(np.float32), 
                        )
                    fields.attrs["units"] = "Wm^-2"
                    fields.dims[0].attach_scale(time_dataset)
                    fields.dims[1].attach_scale(channel_dataset)
                    fields.dims[2].attach_scale(lat_dataset)
                    fields.dims[3].attach_scale(lon_dataset)

            # GET STATS
            channel_min = np.nanmin(data, axis=(0, 2, 3))
            channel_max = np.nanmax(data, axis=(0, 2, 3))
            channel_sum = np.nansum(data, axis=(0, 2, 3))
            channel_squared_sum = np.nansum(data**2, axis=(0, 2, 3))
            total_samples += data.shape[0] * data.shape[2] * data.shape[3]  # Total spatial points
            
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
            print("################################################################################")
    
    results = {}
    channel_means = channel_sums / total_samples
    channel_stds = np.sqrt((channel_squared_sums / total_samples) - channel_means**2)
    results["stats"] = {
        f"channel_{i}": {"mean": float(mean), "std": float(std), "max": float(M), "min": float(m)}
        for i, (mean, std, M, m) in enumerate(zip(channel_means, channel_stds, channel_maxs, channel_mins))
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    #     for h5_file in h5_files:
    #         file_path = os.path.join(data_path, h5_file)
    #         print(f"Checking file: {file_path}")

    #         with h5py.File(file_path, 'r') as h5:
    #             # Check for missing timestamps
    #             if 'time' in h5:
    #                 timestamps = h5['time'][:]
    #                 diffs = np.diff(timestamps)
    #                 if not np.all(diffs == diffs[0]):
    #                     missing_timestamps.append(h5_file)

    #             # Check for NaNs in 'fields'
    #             if 'fields' in h5:
    #                 fields = h5['fields'][:]
    #                 time_values = h5['time'][:]
    #                 num_samples = fields.shape[0]  # Number of time samples
    #                 idx = np.arange(num_samples)
    #                 if np.isnan(fields).any():
    #                     nan_sum = np.isnan(fields).sum(axis=(1,2,3))
    #                     nan_idx = np.where(nan_sum)[0]
    #                     files_with_nans += [time_values[int(i)] for i in nan_idx]
    #                     num_samples -= len(nan_idx)
    #                     idx = [i for i in idx if i not in nan_idx]
    #                     print(h5_file, nan_idx)
                    
    #                 if (fields<0).any():
    #                     neg_sum = (fields < 0).sum(axis=(1,2,3))
    #                     neg_idx = np.where(neg_sum)[0]
    #                     files_with_negs += [time_values[int(i)] for i in neg_idx]
    #                     num_samples -= len(neg_idx)
    #                     idx = [i for i in idx if i not in neg_idx]
    #                     print(h5_file, neg_idx)
                    
    #                 print("The filtered data is ok:", ~np.isnan(fields[idx]).any(),  (fields[idx]>=0).all())
    #                 channel_min = np.nanmin(fields[idx], axis=(0, 2, 3))
    #                 channel_max = np.nanmax(fields[idx], axis=(0, 2, 3))
    #                 channel_sum = np.nansum(fields[idx], axis=(0, 2, 3))  # Sum over time, lat, lon
    #                 channel_squared_sum = np.nansum(fields[idx]**2, axis=(0, 2, 3))  # Sum of squares

    #                 if channel_sums is None:
    #                     channel_sums = channel_sum
    #                     channel_squared_sums = channel_squared_sum
    #                     channel_mins = channel_min
    #                     channel_maxs = channel_max    
    #                 else:
    #                     channel_mins = [float(min(channel_mins[i], channel_min[i])) for i in range(len(channel_min))]
    #                     channel_maxs = [float(max(channel_maxs[i], channel_max[i])) for i in range(len(channel_min))]
    #                     channel_sums += channel_sum
    #                     channel_squared_sums += channel_squared_sum
    #                 print(channel_mins, channel_maxs)

    #                 total_samples += num_samples * fields.shape[2] * fields.shape[3]  # Total spatial points
    # # Calculate mean and std
    # results = {}
    # if channel_sums is not None:
    #     channel_means = channel_sums / total_samples
    #     channel_stds = np.sqrt((channel_squared_sums / total_samples) - channel_means**2)

    #     results["channel_stats"] = {
    #         f"channel_{i}": {"mean": float(mean), "std": float(std), "max": float(M), "min": float(m)}
    #         for i, (mean, std, M, m) in enumerate(zip(channel_means, channel_stds, channel_maxs, channel_mins))
    #     }

    # # Save results
    # results["missing_timestamps"] = missing_timestamps
    # results["files_with_nans"] = files_with_nans
    # results["files_with_negs"] = files_with_negs

    # print(results)

    # with open(output_file, "w") as f:
    #     json.dump(results, f, indent=4)

    # # Report results
    # print("\nCheck complete! Results saved to", output_file)

    # if missing_timestamps:
    #     print("\nFiles with missing timestamps:")
    #     for file in missing_timestamps:
    #         print(f"- {file}")
    # else:
    #     print("\nNo missing timestamps found.")

    # if files_with_nans:
    #     print("\nFiles with NaNs in 'fields':")
    #     for file in files_with_nans:
    #         print(f"- {file}")
    # else:
    #     print("\nNo NaNs found in 'fields' dataset.")

    # if "channel_stats" in results:
    #     print("\nMean and Standard Deviation per channel:")
    #     for channel, stats in results["channel_stats"].items():
    #         print(f"{channel}: Mean = {stats['mean']:.6f}, Std = {stats['std']:.6f}")

if __name__ == "__main__": 
    folder_path = "/capstor/scratch/cscs/acarpent/SEVIRI"
    folder_names = ["2020_weekly_datasets"]

    check_h5_files(folder_path, folder_names, output_file="/capstor/scratch/cscs/acarpent/SEVIRI_2020.json")
