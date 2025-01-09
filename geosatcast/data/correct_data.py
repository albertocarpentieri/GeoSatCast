import os
import h5py
import numpy as np
import json
import pandas as pd
import re

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def check_h5_files(folder_path, folder_names):
    for folder_name in folder_names:
        data_path = os.path.join(folder_path, folder_name)

        # Get a sorted list of HDF5 files in the folder
        h5_files = sorted([f for f in os.listdir(data_path) if f.endswith('.h5')], key=natural_sort_key)
        # print(len(h5_files))
        # h5_files = [h5_files[i] for i in csv_df.index if csv_df.iloc[i, 2]>0]
        # print(h5_files)
        for h5_file in h5_files:
            file_path = os.path.join(data_path, h5_file)
            print(f"Checking file: {file_path}")

            with h5py.File(file_path, 'r') as h5:
                data = h5["fields"][:]

            for c in range(11):
                cond = (np.isnan(data[:,c])) | (data[:,c] < 0)
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
            


if __name__ == "__main__": 
    folder_path = "/capstor/scratch/cscs/acarpent/SEVIRI"
    folder_names = ["2017_weekly_datasets", "2018_weekly_datasets", "2019_weekly_datasets"]

    check_h5_files(folder_path, folder_names)