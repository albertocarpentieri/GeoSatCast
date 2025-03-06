import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import time

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster, load_predrnn
from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmse(x, y):
    return np.sqrt(np.nanmean((x - y) ** 2))

def ensure_dir(directory: str):
    """Creates the directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_models(paths, device="cuda"):
    """Load multiple models from a dictionary of name->path."""
    models = {}
    for name, path in paths.items():
        if "PREDRNN" in name.upper():
            models[name] = load_predrnn(path).to(device)
        else:
            models[name] = load_nowcaster(path).to(device)
        print(f"Loaded {name}, #Params: {count_parameters(models[name])}")
    return models

# --------------------------------------------------
# Forecasting
# --------------------------------------------------
def tvl1_forecast_with_stats(x_np, model_conf, n_forecast_steps, stds, means):
    """Compute TVL1 forecast, log time, stats, and return result with means/stds applied."""
    start_time = time.time()
    yhat_of = tvl1_forecast(x_np, model_conf, n_forecast_steps)
    print("TVL1 time:", time.time() - start_time,
          "TVL1 stats:", yhat_of.mean(), yhat_of.std())
    # scale results
    return yhat_of * stds + means


def process_forecast(x, inv, sza, in_steps, n_forecast_steps, models, stds, means, device="cuda"):
    """Compute forecasts for all loaded models."""
    # Move data to device
    x_t = x[None].to(device).detach()  # shape: (1, C, T, H, W)
    inv_t = inv[None].to(device).detach()
    sza_t = sza[None].to(device).detach()

    # Adjust sza / inv if needed
    sza_t = sza_t[:, :, :in_steps + n_forecast_steps - 1]
    inv_t = torch.cat((inv_t.expand(*inv_t.shape[:2], *sza_t.shape[2:]), sza_t), dim=1)

    forecasts = {}

    # Forecast from nowcaster models
    for name, model in models.items():
        start_time = time.time()
        with torch.no_grad():
            # if "PREDRNN" in name.upper():
                # predRNN signature: model(x, n_forecast_steps)
            if "AFNO" in name:
                yhat = model(x_t, inv_t, n_forecast_steps).detach().cpu().numpy()[0]
            else:
                yhat = model(x_t, inv_t, n_forecast_steps).detach().cpu().numpy()[0]
            # else:
                # AFNO, NAT, AFNONAT signature: model(x, inv, n_forecast_steps)
                # yhat = model(x_t, inv_t, n_forecast_steps).detach().cpu().numpy()[0]

        yhat = yhat * stds + means
        print(f"{name} time: {time.time() - start_time:.3f} s, "
              f"stats: mean={yhat.mean():.3f}, std={yhat.std():.3f}")
        forecasts[name] = yhat

    return forecasts

LAT = [18.15, 62.1]
LON = [-16.1, 36.1]
# --------------------------------------------------
# Plotting
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

# You can define a helper function to compute the lat/lon values based on the image size
def get_lat_lon_indices(lat_i, lon_i, field_size, data_shape):
    """
    Compute latitude and longitude for each pixel in the image.
    lat_i, lon_i are the starting indices, field_size is the size of the image in each direction,
    and data_shape is the shape of the data array (rows, cols).
    """
    lats = np.linspace(lat_i, lat_i + field_size, data_shape[0])
    lons = np.linspace(lon_i, lon_i + field_size, data_shape[1])
    return lats, lons

def plot_results_per_step(y, forecasts, t, channel_names, out_dir, stds, lats, lons):
    """One plot per time step, showing all channels in subplots with lat/lon labels."""
    model_names = list(forecasts.keys())
    n_models = len(model_names)
    n_channels = y.shape[0]
    n_steps = y.shape[1]

    # Get the latitude and longitude indices

    for j in range(n_steps):
        fig, ax = plt.subplots(
            2 + n_models, n_channels,
            figsize=(3 * n_channels, 3 * (2 + n_models)),
            sharex=True, sharey=True,
            constrained_layout=True
        )
        fig.suptitle(f"Forecast at step {j}, time={t[j]}")

        # Observed row
        for i in range(n_channels):
            img = ax[0, i].imshow(y[i, j], vmin=y[i].min(), vmax=y[i].max(), interpolation="none")
            ax[0, i].set_title(channel_names[i])
            ax[0, i].set_xlabel('Longitude')
            ax[0, i].set_ylabel('Latitude')
            ax[0, i].set_xticks(np.linspace(0, y.shape[2], 5))
            ax[0, i].set_xticklabels(np.round(np.linspace(LON[0], LON[-1], 5), 2))
            ax[0, i].set_yticks(np.linspace(0, y.shape[1], 5))
            ax[0, i].set_yticklabels(np.round(np.linspace(LAT[0], LAT[-1], 5), 2))
        ax[0, 0].set_ylabel("Observed")

        # Each model forecast row
        for m_idx, model_name in enumerate(model_names, start=1):
            forecast = forecasts[model_name]
            for i in range(n_channels):
                ax[m_idx, i].imshow(
                    forecast[i, j],
                    vmin=y[i].min(),
                    vmax=y[i].max(),
                    interpolation="none"
                )
                ax[m_idx, i].set_xlabel('Longitude')
                ax[m_idx, i].set_ylabel('Latitude')
                ax[m_idx, i].set_xticks(np.linspace(0, y.shape[2], 5))
                ax[m_idx, i].set_xticklabels(np.round(np.linspace(LON[0], LON[-1], 5), 2))
                ax[m_idx, i].set_yticks(np.linspace(0, y.shape[1], 5))
                ax[m_idx, i].set_yticklabels(np.round(np.linspace(LAT[0], LAT[-1], 5), 2))
            ax[m_idx, 0].set_ylabel(model_name)

        # Differences row (model - obs)
        for m_idx, model_name in enumerate(model_names, start=1):
            forecast = forecasts[model_name]
            for i in range(n_channels):
                img = ax[n_models + 1, i].imshow(
                    forecast[i, j] - y[i, j],
                    vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none"
                )
                if m_idx == 1:
                    ax[n_models + 1, 0].set_ylabel("Diff")

                # Put RMSE in the title
                local_rmse = rmse(forecast[i, j], y[i, j])
                ax[n_models + 1, i].set_title(f"RMSE: {local_rmse:.4f}")
        # Save figure
        fname = os.path.join(out_dir, f"forecast_{t[j].strftime('%Y%m%d_%H%M%S')}_step{j}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_results_per_channel(y, forecasts, t, channel_names, out_dir, lats, lons):
    """One plot per channel, showing multiple time steps and models with lat/lon labels."""
    model_names = list(forecasts.keys())
    n_models = len(model_names)
    n_channels = y.shape[0]
    n_steps = y.shape[1]

    # Define steps to plot
    steps = [0] + list(range(3, n_steps, 4))

    for ch_idx, channel in enumerate(channel_names):
        fig, axs = plt.subplots(
            len(steps), 1 + n_models,  # obs + models
            figsize=(4 * (n_models), 3 * len(steps)),
            sharex=True, sharey=True,
            constrained_layout=True
        )
        if n_steps == 1:
            axs = [axs]

        fig.suptitle(f"Channel: {channel}")

        for j, step in enumerate(steps):
            ax_obs = axs[j]

            # Observed image
            ax_obs[0].imshow(
                y[ch_idx, step], 
                interpolation="none",
                vmin=y[ch_idx].min(),
                vmax=y[ch_idx].max())
            ax_obs[0].set_title(f"Obs (t={t[step].strftime('%H:%M:%S')})")
            if step == steps[-1]:
                ax_obs[0].set_xlabel('Longitude')
            ax_obs[0].set_ylabel('Latitude')
            ax_obs[0].set_xticks(np.linspace(0, y[ch_idx].shape[2]-1, 5))
            ax_obs[0].set_xticklabels([str(round(lons[int(i)], 2)) + "˚" for i in np.linspace(0, y[ch_idx].shape[2]-1, 5)])
            ax_obs[0].set_yticks(np.linspace(0, y[ch_idx].shape[1]-1, 5))
            ax_obs[0].set_yticklabels([str(round(lats[int(i)], 2)) + "˚" for i in np.linspace(0, y[ch_idx].shape[1]-1, 5)])

            # Model forecasts
            for m_idx, model_name in enumerate(model_names, start=1):
                if ch_idx in [0, 7, 8]:
                    # Zero out pixels where observed image is 0.
                    forecasts[model_name][ch_idx, step][y[ch_idx, step] == 0] = 0
                
                ax_obs[m_idx].imshow(
                    forecasts[model_name][ch_idx, step],
                    interpolation="none",
                    vmin=y[ch_idx].min(),
                    vmax=y[ch_idx].max(),
                )
                ax_obs[m_idx].set_title(f"{model_name} (+ {15 * (step + 1)} min)")
                if step == steps[-1]:
                    ax_obs[m_idx].set_xlabel('Longitude')
                ax_obs[m_idx].set_xticks(np.linspace(0, y[ch_idx].shape[2]-1, 5))
                ax_obs[m_idx].set_xticklabels([str(round(lons[int(i)], 2)) + "˚" for i in np.linspace(0, y[ch_idx].shape[2]-1, 5)])
                ax_obs[m_idx].set_yticks(np.linspace(0, y[ch_idx].shape[1]-1, 5))
                ax_obs[m_idx].set_yticklabels([str(round(lats[int(i)], 2)) + "˚" for i in np.linspace(0, y[ch_idx].shape[1]-1, 5)])

        fname = os.path.join(out_dir, f"forecast_{channel}_{t[0].strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":
    device = "cuda"
    out_dir = "/capstor/scratch/cscs/acarpent/test_results/test_forecasts"
    ensure_dir(out_dir)

    # Setup model paths
    model_paths = {
        "AFNONATCast - small": "/capstor/scratch/cscs/acarpent/Checkpoints/AFNONATCast/AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2/AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13.pt",
        "AFNONATCast - large": "/capstor/scratch/cscs/acarpent/Checkpoints/AFNONATCast/AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2/AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14.pt",
        "PredRNN ++": "/capstor/scratch/cscs/acarpent/Checkpoints/predrnn/predrnn-inv-s2-fd_5-nh_64-v1-finetuned/predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45.pt"
    }

    # Load models
    models = load_models(model_paths, device=device)

    # Also define a TVL1 configuration
    tvl1_conf = {
        'tau': 0.15,
        'epsilon': 0.005,
        'gamma': 0,
        'warps': 10,
        'lambda': 0.05,
        'outer_iterations': 20,
        'inner_iterations': 20,
        'theta': 0.3,
        'nscales': 5,
        'median_filtering': 1,
        'scale_step': 0.5,
    }

    # We can optionally add TVL1 as a special method in the dictionary:
    # (We'll handle it outside or we can treat it as a separate forecast below)

    # Prepare dataset
    in_steps = 2
    n_forecast_steps = 16
    dataset = DistributedDataset(
        data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
        invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
        name='32b_virtual',
        years=[2021],
        input_len=in_steps + n_forecast_steps,
        output_len=None,
        channels=np.arange(11),
        field_size=768,
        length=None,
        validation=True,
        rank=0,
        mask_sza=True,
    )

    channel_names = [
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

    stds = dataset.stds.numpy().reshape(-1, 1, 1, 1)
    means = dataset.means.numpy().reshape(-1, 1, 1, 1)
    lats = dataset.lat 
    lons = dataset.lon
    # Indices to test
    test_indices = [4700, 5125, 5756]

    # Choose plotting mode: 'per_step' or 'per_channel'
    plot_mode = 'per_channel'  # or 'per_step'
    lat_i=56
    lon_i=138
    field_size=768

    lats = lats[lat_i:lat_i + field_size]
    lons = lons[lon_i:lon_i + field_size]
    for t_i in test_indices:
        # Retrieve data from dataset
        
        x, t, inv, sza = dataset.get_data(year=2021, t_i=t_i, lat_i=lat_i, lon_i=lon_i)
        # x shape: (C, T, H, W)
        # We'll split them: input is first 2, output is next 8
        # user had x[:,:in_steps], x[:, in_steps:in_steps+n_forecast_steps]
        # but let's rename them for clarity
        x_in = x[:, :in_steps]
        y_obs = x[:, in_steps : in_steps + n_forecast_steps]

        # Convert y_obs to np, scale up
        y_obs_np = y_obs.numpy() * stds + means

        # Times
        t_list = [datetime.datetime.utcfromtimestamp(t[0, in_steps + i, 0, 0].numpy().astype(int))
                  for i in range(n_forecast_steps)]

        # 1) TVL1 forecast
        yhat_tvl1 = tvl1_forecast_with_stats(
            x_in.numpy(), tvl1_conf, n_forecast_steps, stds, means
        )

        # 2) Other models
        forecasts = process_forecast(x_in, inv, sza, in_steps, n_forecast_steps, models, stds, means, device=device)
        # Insert TVL1 into the dictionary so we can plot it with the others
        forecasts["TVL1"] = yhat_tvl1

        # Now plot
        if plot_mode == 'per_step':
            plot_results_per_step(
                y_obs_np, forecasts, t_list, channel_names, out_dir, stds, lats, lons
            )
        elif plot_mode == 'per_channel':
            plot_results_per_channel(
                y_obs_np, forecasts, t_list, channel_names, out_dir, lats, lons
            )

    print("All plots done!")

