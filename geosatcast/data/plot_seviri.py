#!/usr/bin/env python3
"""
Script to plot all channels of a SEVIRI native file using Satpy,
displaying each channel in a geostationary projection with its own colorbar
and associated metric (e.g. brightness temperature in Kelvin or reflectance).
 
Requirements:
    - Python 3.x
    - satpy
    - matplotlib
    - cartopy
    - numpy

Usage:
    python plot_all_seviri_channels_satpy.py
"""

import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from satpy import Scene

# --- User settings ---
NON_HRV_BANDS = [
    "VIS008", "WV_062", "IR_120", 
]
channel_names = [
    "0.8 µm", "6.2 µm", "12.0 µm", 
]
# Path to your native SEVIRI file.
native_file = '/capstor/scratch/cscs/acarpent/SEVIRI_DATA/HRSEVIRI2018/MSG2-SEVI-MSG15-0100-NA-20180507105742.713000000Z-NA.nat'  # Replace with your file path

# Specify the reader. For SEVIRI Level-1B, this is usually 'seviri_l1b'
reader = 'seviri_l1b_native'

# --- Load the scene with Satpy ---

# Create a Satpy Scene from the native file.
scn = Scene(filenames=[native_file], reader=reader)

# Determine all available channels/datasets.
# This returns a set of available dataset names.
# available_channels = sorted(scn.available_dataset_names())[1:]

# Load all channels.
scn.load(NON_HRV_BANDS)
print(scn[NON_HRV_BANDS[0]])
# --- Define the geostationary projection ---
# Typical geostationary settings: central_longitude=0 and satellite_height ~35786000 m.
geo_proj = ccrs.Geostationary(central_longitude=0, satellite_height=35786000)

# --- Determine subplot grid layout based on the number of channels ---
n_channels = len(NON_HRV_BANDS)
ncols = 3  # you can change this to suit your display preferences
nrows = math.ceil(n_channels / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 4),
                         subplot_kw={'projection': geo_proj})

# In case of a single row or column, ensure axes is a flat list.
if n_channels == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# --- Loop over each channel and plot ---
for idx, channel in enumerate(NON_HRV_BANDS):
    ax = axes[idx]

    # Get the data array for the channel.
    data_array = scn[channel].data

    # Try to get the extent from the dataset attributes (expected format: [min_x, max_x, min_y, max_y]).
    extent = scn[channel].attrs.get('extent', None)
    print(extent)
    if extent is None:
        # If extent is not provided, you may need to set it manually.
        # The following are placeholder values (in meters) and should be replaced by your file's extent.
        extent = [-5.566e+06, 5.566e+06, -5.566e+06, 5.566e+06]

    # Retrieve the physical unit or metric from the dataset attributes.
    # For example, brightness temperature might be in Kelvin.
    unit = scn[channel].attrs.get('units', 'unknown metric')

    # Plot the channel data.
    im = ax.imshow(data_array, origin='upper', extent=extent,
                   transform=geo_proj, cmap='viridis')

    # Add coastlines and borders for geographic context.
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Set subplot title with channel name and its unit.
    ax.set_title(f'{channel_names[idx]}', fontsize=12)

    # Add a colorbar for this subplot.
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=.5)
    cbar.set_label(f'{unit}')

# Turn off any unused subplots.
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('2018-05-07 10:57 UTC+0', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/capstor/scratch/cscs/acarpent/SEVIRI_plot.png", dpi=150)
plt.show()
