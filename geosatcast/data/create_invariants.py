import xarray as xr
import numpy as np
from pyresample import geometry, kd_tree

ds = xr.open_dataset("/capstor/scratch/cscs/acarpent/2017_virtual.h5", engine="h5netcdf")

for filename, varname in zip(["NETCDF_LSASAF_USGS-IGBP_LWMASK_MSG-Disk_202001270900.nc", "NETCDF_LSASAF_USGS-IGBP_LANDCOV_MSG-Disk_202001270900.nc", "NETCDF_LSASAF_USGS_DEM_MSG-Disk_202001270900.nc"], 
                             ["LWMASK", "LANDCOV", "DEM"]):
    # Load the NetCDF file
    file_path = f'/capstor/scratch/cscs/acarpent/{filename}'
    dem_ds = xr.open_dataset(file_path)

    # Define the source grid
    source_lons, source_lats = np.meshgrid(dem_ds['lon'], dem_ds['lat'])
    source_def = geometry.SwathDefinition(lons=source_lons, lats=source_lats)

    # Define the target grid
    target_lons = ds['longitude'].values  # example target grid
    target_lats = ds['latitude'].values    # example target grid
    target_lons, target_lats = np.meshgrid(target_lons, target_lats)
    target_def = geometry.GridDefinition(lons=target_lons, lats=target_lats)

    # Perform the reprojection using nearest neighbor
    result = kd_tree.resample_nearest(source_def, dem_ds[varname].values, target_def, radius_of_influence=500000, fill_value=None)

    # Create a new xarray Dataset for the reprojected data
    reprojected_ds = xr.Dataset(
        {
            varname: (["lat", "lon"], result)
        },
        coords={
            "longitude": (["lon"], target_lons[0, :]),
            "latitude": (["lat"], target_lats[:, 0])
        }
    )

    reprojected_ds.to_netcdf(f"/capstor/scratch/cscs/acarpent/{varname.lower()}.nc")