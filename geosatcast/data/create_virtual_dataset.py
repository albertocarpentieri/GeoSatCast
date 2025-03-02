import h5py
import os

def create_virtual_layout(file_paths, dataset_name, dtype, shape, axis=0):
    """Creates a virtual layout for a given dataset across multiple files."""
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    start = 0
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            dset = f[dataset_name]
            end = start + dset.shape[axis]
            layout[start:end, ...] = h5py.VirtualSource(file_path, dataset_name, shape=dset.shape)
            start = end

    return layout

def get_datasets_info(file_paths, dataset_name):
    """Get dataset info such as dtype and shape."""
    with h5py.File(file_paths[0], 'r') as f:
        dset = f[dataset_name]
        dtype = dset.dtype
        shape = list(dset.shape)
        shape[0] = sum([h5py.File(fp, 'r')[dataset_name].shape[0] for fp in file_paths])
    return dtype, tuple(shape)

def create_virtual_dataset(save_path, virtual_file_path):
    h5_files = [os.path.join(save_path, f) for f in sorted(os.listdir(save_path)) if f.endswith('.h5')]
    if len(h5_files) == 0:
        raise ValueError("No HDF5 files found to merge")

    with h5py.File(virtual_file_path, 'w') as f_virtual:
        dataset_names = ['latitude', 'longitude', 'time', 'fields']
        n_files = [1, 1, len(h5_files), len(h5_files)] # no need for multiple files to define lat/lon/channels 
        
        for i, dataset_name in enumerate(dataset_names):
            dtype, shape = get_datasets_info(h5_files[:n_files[i]], dataset_name)
            layout = create_virtual_layout(h5_files[:n_files[i]], dataset_name, dtype, shape, axis=0)
            f_virtual.create_virtual_dataset(dataset_name, layout)
        
        # Set units and attach scales for dimensions
        f_virtual['latitude'].attrs["units"] = "degrees north"
        f_virtual['longitude'].attrs["units"] = "degrees east"
        f_virtual['time'].attrs["units"] = "unix time [s]"

        f_virtual['fields'].dims[0].attach_scale(f_virtual['time'])
        f_virtual['fields'].dims[1].attach_scale(f_virtual['latitude'])
        f_virtual['fields'].dims[2].attach_scale(f_virtual['longitude'])
            # f_virtual['fields'].dims[3].attach_scale(f_virtual['channels'])

    print(f"Created virtual HDF5 file at {virtual_file_path}")

if __name__ == "__main__":
    for YEAR in [2021]:#[2017, 2018, 2019, 2020, 2021]:
        # SAVE_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI/{YEAR}_weekly_datasets/"
        # VIRTUAL_FILE_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI/{YEAR}_new_virtual.h5"
        SAVE_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI/{YEAR}_32b_datasets/" #f"/capstor/scratch/cscs/acarpent/SEVIRI/{YEAR}_16b_datasets/"
        VIRTUAL_FILE_PATH = f"/capstor/scratch/cscs/acarpent/SEVIRI/{YEAR}_32b_virtual.h5"
        create_virtual_dataset(SAVE_PATH, VIRTUAL_FILE_PATH)