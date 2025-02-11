from satpy import Scene, MultiScene
from satpy.multiscene import timeseries
import satpy
import pyresample as pr
import numpy as np
from pyproj import Proj
import time
import os
from datetime import datetime, timedelta, timezone
from geosatcast.data.utils import cos_zenith_angle_from_timestamp
import pandas as pd 
import h5py
import dask
import dask.array as da
# dask.config.set(num_workers=256)
# dask.config.set({"array.chunk-size": "64MiB"})
dask.config.set({"scheduler": "threads"})
NON_HRV_BANDS = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", 
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073"
]

YEAR = 2017
N_DAYS = 1
MIN_FILE_SIZE = 258 * 1024 * 1024 # 250 MB

# Function to extract time from filename
def get_time_from_filename(f):
    year, month, day, hour, minute = map(int, [
        f.split("-")[5][:4], f.split("-")[5][4:6], f.split("-")[5][6:8],
        f.split("-")[5][8:10], f.split("-")[5][10:12]
    ])
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

# # Function to read and process data
# @dask.delayed
# def read_data(filenames):
#     target_proj = Proj(proj='latlong', datum='WGS84')

#     # Define the area once
#     area_def = pr.create_area_def(
#         "custom_latlon", target_proj.srs, 
#         area_extent=[-16.125, 18.125, 36.125, 62.125],
#         resolution=(0.05, 0.05)
#     )

#     # Enable Dask
#     with satpy.config.set(
#         cache_dir='/workspace/cache', 
#         tmp_dir='/workspace/tmp',
#         data_dir='/workspace/data',
#         use_dask=True
#     ):
#         scn = MultiScene.from_files(filenames, reader="seviri_l1b_native")

#         # Load data in parallel
#         scn.load(NON_HRV_BANDS, use_dask=True)

#         # Faster resampling with bilinear interpolation
#         reprojected_scn = scn.resample(
#             area_def, resampler='bilinear', cache_dir='/workspace/proj_tmp/'
#         )

#         # Blend and convert to Xarray dataset
#         blended = reprojected_scn.blend(blend_function=timeseries)
#         ds = blended.to_xarray_dataset().drop_vars('crs').drop_attrs()

#         # Extract coordinates
#         lat, lon = ds["y"].values, ds["x"].values
#         time_vals = ds["time"].values

#         # Optimize memory
#         vars_data = [ds[var].values.astype(np.float16) for var in NON_HRV_BANDS]
#         combined_array = np.stack(vars_data, axis=-1)

#     return combined_array, time_vals, lat, lon

# @dask.delayed
def process_file(filenames):
    target_proj = Proj(proj='latlong', datum='WGS84')
    area_def = pr.create_area_def(
        "custom_latlon", target_proj.srs, 
        area_extent=[-16.125, 18.125, 36.125, 62.125],
        resolution=(0.05, 0.05)
    )
    with satpy.config.set(
        cache_dir='/capstor/scratch/cscs/acarpent/cache', 
        tmp_dir='/capstor/scratch/cscs/acarpent/tmp',
        data_dir='/capstor/scratch/cscs/acarpent/data',
        use_dask=True
    ):
        scn = MultiScene.from_files(filenames, reader="seviri_l1b_native")
        scn.load(NON_HRV_BANDS, use_dask=True)
        print("loaded")
        reprojected_scn = scn.resample(
            area_def, resampler='nearest', cache_dir='/capstor/scratch/cscs/acarpent/proj_tmp/'
        )
        print("resampled")
        # reprojected_scn.load(NON_HRV_BANDS, use_dask=True)
        reprojected_scn = reprojected_scn.blend(blend_function="composite")
        print("blended")
        ds = reprojected_scn.to_xarray_dataset().drop_vars('crs').drop_attrs()
        print("xarreyed")
        lat, lon = ds["y"].values, ds["x"].values
        time_vals = ds["time"].values
        vars_data = [ds[var].values.astype(np.float16) for var in NON_HRV_BANDS]
        combined_array = np.stack(vars_data, axis=-1)
    return combined_array, time_vals, lat, lon

# Function to read and process multiple files
def read_data(filenames):
    # # tasks = [process_file(fn) for fn in filenames]
    # results = np.concatenate(dask.compute(*tasks), axis=0)
    # print(results.shape)
    # # Combine results as needed
    results = process_file(filenames)
    print(results.shape)
    return results
# # Run in parallel
# filenames = [...]  # Your list of filenames
# data_array, timestamps, lat, lon = dask.compute(read_data(filenames))

if __name__ == "__main__":
    DATA_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI_DATA/HRSEVIRI{YEAR}/"
    DF_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI_16B/"
    SAVE_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI_16B/{YEAR}_weekly_datasets/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    start_date = datetime(YEAR, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(YEAR, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    
    filenames = [os.path.join(DATA_PATH, f) for f in sorted(os.listdir(DATA_PATH)) if ".nat" in f]
    time_array = [get_time_from_filename(f) for f in filenames]
    
    i = 0
    while start_date + timedelta(days=N_DAYS * i) < end_date:
        day_start = start_date + timedelta(days=N_DAYS * i)
        day_end = start_date + timedelta(days=N_DAYS * (i + 1))
        print(day_start, day_end)
    
        weekly_filenames = [filenames[j] for j, t in enumerate(time_array) 
                            if day_start <= t < day_end and os.path.getsize(filenames[j]) >= MIN_FILE_SIZE]
        
        if weekly_filenames:
            weekly_time_array = [get_time_from_filename(f) for f in weekly_filenames]
            data_array, timestamps, lat, lon = read_data(filenames)
            
            latlon_grid = np.stack(np.meshgrid(lon, lat, indexing='xy'))
            sza = np.stack([cos_zenith_angle_from_timestamp(
                    t_.timestamp(),
                    lon=latlon_grid[0].flatten(),
                    lat=latlon_grid[1].flatten())
                for t_ in weekly_time_array], axis=0).reshape(len(weekly_time_array), lat.shape[0], lat.shape[1], 1)
            data_array = np.concatenate((data_array, sza), axis=-1)
            
            corrected = False
            for c in range(11):
                c_data = data_array[:,:,:,c]
                cond = (np.isnan(c_data)) | (c_data < MIN_VALUES[c])
                nan_idx = np.where(cond)
                
                if len(nan_idx[0])>0:
                    time_nan_idx = nan_idx[0]
                    lat_nan_idx = nan_idx[1]
                    lon_nan_idx = nan_idx[2]

                    fill_lat_idx = np.array([lat_nan_idx + k for k in [-1, -1, -1, 0, 0, 0, 1, 1, 1]])
                    fill_lat_idx[fill_lat_idx<0] = 0
                    fill_lat_idx[fill_lat_idx>=c_data.shape[1]] = c_data.shape[1]-1
                    fill_lon_idx = np.array([lon_nan_idx + k for k in [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
                    fill_lon_idx[fill_lon_idx<0] = 0
                    fill_lon_idx[fill_lon_idx>=c_data.shape[2]] = c_data.shape[2]-1

                    stacked_x = np.nanmean(np.stack([c_data[time_nan_idx, fill_lat_i, fill_lon_i] for fill_lat_i, fill_lon_i in zip(fill_lat_idx, fill_lon_idx)], axis=0), axis=0)
                    
                    data_array[:,:,:,c][nan_idx] = stacked_x
                    print(f"data has still nans in channel {c+1}:", np.isnan(c_data).any())
                    print("Now looking at minimum values")
                    
                    cond_sum = np.sum(data_array[:,:,:,c] < MIN_VALUES[c], axis=(1,2))
                    
                    
                    replace_idx = (cond_sum < 0.05 * data_array.shape[-2]*data_array.shape[-1]) & (cond_sum > 0.)
                    r = data_array[replace_idx, :, :, c]
                    replace_cond = r < MIN_VALUES[c]
                    r[replace_cond] = MIN_VALUES[c]
                    data_array[replace_idx, :, :, c] = r
                    print(f"data has still negs in channel {c+1}:", (data_array[:,:,:,c] < MIN_VALUES[c]).any())

                    corrected = True
            
            if corrected:
                cond = (np.isnan(data_array)) | (data_array < np.array(MIN_VALUES)[None,:,None,None])
                cond_sum = np.sum(cond, axis=(1,2,3))
                idx = np.where(cond_sum==0)[0]
                
                
                data_array = data_array[idx]
                print("new corrected data shape:", data_array.shape)
                timestamps = timestamps[idx]
                print("Saving new_"+h5_file)


                save_path = os.path.join(SAVE_PATH, f"{YEAR}_{N_DAYS * i}-{N_DAYS * (i + 1)}.h5")
                os.system(f"lfs setstripe -c 4 -S 32M {save_path}")  # Apply Lustre striping
            
                with h5py.File(save_path, "w") as new_hf:
                    new_hf.create_dataset("latitude", data=lat.astype(np.float32))
                    new_hf.create_dataset("longitude", data=lon.astype(np.float32))
                    new_hf.create_dataset("time", data=[t_.timestamp() for t_ in weekly_time_array])
                    new_hf.create_dataset("channels", data=NON_HRV_BANDS + ["SZA"])
                    fields = new_hf.create_dataset("fields", data=data_array.astype(np.float16))
                    
                    fields.dims[0].attach_scale(time_dataset)
                    fields.dims[1].attach_scale(lat_dataset)
                    fields.dims[2].attach_scale(lon_dataset)
                    fields.dims[3].attach_scale(channel_dataset)

            avg = np.nanmean(data_array)
            nan = np.sum(np.isnan(data_array))
            shape = data_array.shape
            
            with open(os.path.join(DF_PATH, f"{YEAR}.csv"), "a") as f:
                f.write(f"{day_start}-{day_end};{avg};{nan};{shape}\n")

            print(f"Saved data chunk {i+1} containing data of shape {data_array.shape} in {time.time() - start_time} seconds") 
            print("###############################################################################################################################")
        i += 1
    