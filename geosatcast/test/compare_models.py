import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import List, Union
import re

# Original unit names and associated measures.
NON_HRV_BANDS = [
    "IR 1.6",
    "IR 3.9",
    "IR 8.7",
    "IR 9.7",
    "IR 10.8",
    "IR 12.0",
    "IR 13.4",
    "VIS 0.06",
    "VIS 0.08",
    "IR 6.2",
    "IR 7.3",
]

MEASURES = [
    "% Reflectivity",
    "K",
    "K",
    "K",
    "K",
    "K",
    "K",
    "% Reflectivity",
    "% Reflectivity",
    "K",
    "K",
]

# Helper function to extract the wavelength (or unit number) as a float.
def extract_wavelength(ch: str) -> float:
    """
    Convert a channel/unit string to a numeric value representing the wavelength.
    For IR channels like "IR_016", we assume the number represents 1.6 (i.e. divide by 10).
    For VIS channels like "VIS006", we assume the number represents 0.06 (i.e. divide by 100).
    For WV channels like "WV_062", we assume the number represents 6.2 (i.e. divide by 10).
    """
    if ch.startswith("IR"):
        # Remove "IR_" if present; e.g. "IR_016" -> "016"
        if "_" in ch:
            num_str = ch.split("_")[1]
        else:
            num_str = ch[2:]
        return float(num_str) / 10.0
    elif ch.startswith("VIS"):
        # Remove "VIS"; e.g. "VIS006" -> "006"
        num_str = ch[3:]
        return float(num_str) / 100.0
    elif ch.startswith("WV"):
        if "_" in ch:
            num_str = ch.split("_")[1]
        else:
            num_str = ch[2:]
        return float(num_str) / 10.0
    else:
        # Fallback: use first found number
        numbers = re.findall(r"\d+", ch)
        return float(numbers[0]) if numbers else 0.0

# Create a mapping for the original (unsorted) order.
channel_to_idx_unsorted = {name: i for i, name in enumerate(NON_HRV_BANDS)}

# Compute sorted channel order based on the extracted wavelength.
sorted_channels = sorted(NON_HRV_BANDS, key=lambda ch: extract_wavelength(ch))
# Reorder the MEASURES list accordingly.
sorted_measures = [MEASURES[channel_to_idx_unsorted[ch]] for ch in sorted_channels]

##############################################
# PARTIAL RESULTS AND AGGREGATION FUNCTIONS
##############################################
def load_partial_results(pkl_path: str):
    """
    Load a single partial .pkl file from nowcast_test.py, which contains:
      {
        "sample_metrics": {init_time: {...}, ...},
        "pixel_metrics": {
           "sum_diff": array,
           "sum_abs_diff": array,
           "sum_sq_diff": array,
           "count_valid": array
        }
      }
    """
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    return data

def merge_model_partials(partial_paths: List[str]):
    """
    Merges multiple partial .pkl files for a single model into a single dict:
      {
        "sample_metrics": merged_sample_dict,
        "pixel_metrics": {
           "mean_error_map": ...,
           "mae_map": ...,
           "rmse_map": ...
        }
      }
    """
    sum_diff_total = None
    sum_abs_diff_total = None
    sum_sq_diff_total = None
    count_valid_total = None

    merged_sample_metrics = {}

    for path in partial_paths:
        data = load_partial_results(path)

        # Merge sample_metrics
        for time_key, sample_dict in data["sample_metrics"].items():
            merged_sample_metrics[time_key] = sample_dict

        # Merge pixel partial sums
        pix = data["pixel_metrics"]
        if sum_diff_total is None:
            sum_diff_total = pix["sum_diff"].copy()
            sum_abs_diff_total = pix["sum_abs_diff"].copy()
            sum_sq_diff_total = pix["sum_sq_diff"].copy()
            count_valid_total = pix["count_valid"].copy()
        else:
            sum_diff_total += pix["sum_diff"]
            sum_abs_diff_total += pix["sum_abs_diff"]
            sum_sq_diff_total += pix["sum_sq_diff"]
            count_valid_total += pix["count_valid"]

    # Compute final mean_error, MAE, RMSE maps (shape: (channels, forecast_steps, H, W))
    mean_error_map = np.divide(
        sum_diff_total,
        count_valid_total,
        out=np.zeros_like(sum_diff_total, dtype=np.float32),
        where=(count_valid_total > 0)
    )
    mae_map = np.divide(
        sum_abs_diff_total,
        count_valid_total,
        out=np.zeros_like(sum_abs_diff_total, dtype=np.float32),
        where=(count_valid_total > 0)
    )
    rmse_map = np.divide(
        sum_sq_diff_total,
        count_valid_total,
        out=np.zeros_like(sum_sq_diff_total, dtype=np.float32),
        where=(count_valid_total > 0)
    )
    rmse_map = np.sqrt(rmse_map)

    result = {
        "sample_metrics": merged_sample_metrics,
        "pixel_metrics": {
            "mean_error_map": mean_error_map,
            "mae_map": mae_map,
            "rmse_map": rmse_map
        }
    }
    return result

def aggregate_sample_metrics(sample_metrics_dict):
    """
    Aggregates each metric across all init_times (samples).
    Returns a dict with arrays of shape (n_forecast_steps, n_channels) for each metric.
    """
    needed_keys = [
        "csi_above", "csi_below",
        "fss_above_s3", "fss_above_s7", "fss_above_s15",
        "fss_below_s3", "fss_below_s7", "fss_below_s15",
        "pearson_corr", "rmse", "mae", "mean_error"
    ]

    all_times = sorted(sample_metrics_dict.keys())
    if not all_times:
        raise ValueError("No sample metrics found to aggregate.")

    example_entry = sample_metrics_dict[all_times[0]]
    n_forecast_steps = len(example_entry["csi_above"])
    n_channels = len(example_entry["csi_above"][0])

    metric_arrays = {mk: [] for mk in needed_keys}

    for t_key in all_times:
        entry = sample_metrics_dict[t_key]
        for mk in needed_keys:
            arr = np.array(entry[mk])  # shape (n_forecast_steps, n_channels)
            metric_arrays[mk].append(arr)

    aggregated = {}
    for mk in needed_keys:
        stacked = np.array(metric_arrays[mk])  # shape (n_samples, n_forecast_steps, n_channels)
        mean_val = np.nanmean(stacked, axis=0)
        aggregated[mk] = mean_val

    aggregated["n_forecast_steps"] = n_forecast_steps
    aggregated["n_channels"] = n_channels
    return aggregated

##############################################
# MAIN COMPARISON FUNCTION
##############################################
def compare_models(
    model_pkl_paths: List[List[str]],
    model_names: List[str],
    epochs: List[Union[int, str]] = None,
    plot_save_dir="./plots",
    table_save_dir="./tables"
):
    """
    1) Merge partial files for each model.
    2) Aggregate sample_metrics and produce line plots using a custom layout.
       The channels (units) are sorted in increasing wavelength order.
       The layout is created automatically in a 4x3 grid (last slot reserved for legend).
    3) For pixel_metrics, produce one figure per metric and channel arranged as [forecast steps] x [models],
       and the channels are plotted in sorted order.
    """
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(table_save_dir, exist_ok=True)

    assert len(model_pkl_paths) == len(model_names), "Mismatch in model paths vs. names"
    if epochs is not None:
        assert len(epochs) == len(model_names), "Mismatch in epochs vs. names"

    # 1) Merge partial files per model.
    merged_results = []
    aggregated_samples = []
    for i, pkl_list in enumerate(model_pkl_paths):
        print(f"Merging partial files for model: {model_names[i]}")
        merged = merge_model_partials(pkl_list)
        merged_results.append(merged)
        agg_samples = aggregate_sample_metrics(merged["sample_metrics"])
        aggregated_samples.append(agg_samples)

    n_forecast_steps = aggregated_samples[0]["n_forecast_steps"]
    # Use the number of channels from the merged samples.
    n_channels = aggregated_samples[0]["n_channels"]
    lead_times = np.arange(n_forecast_steps)

    # Create the custom layout for sample metrics plots automatically.
    # We use the sorted channel order.
    n_total = len(sorted_channels)  # e.g. 11 channels
    nrows = 4
    ncols = 3
    # Fill row-by-row; last cell will be left for legend if channels < nrows*ncols.
    channel_groups = [ sorted_channels[i*ncols:(i+1)*ncols] for i in range(nrows) ]

    # Prepare a pivot table collection for CSV export.
    metrics_info = {
        "csi_above":    {"label": "CSI Above",     "suffix": "csi_above"},
        "csi_below":    {"label": "CSI Below",     "suffix": "csi_below"},
        "fss_above_s3": {"label": "FSS Above S=3", "suffix": "fss_above_s3"},
        "fss_above_s7": {"label": "FSS Above S=7", "suffix": "fss_above_s7"},
        "fss_above_s15":{"label": "FSS Above S=15","suffix": "fss_above_s15"},
        "fss_below_s3": {"label": "FSS Below S=3", "suffix": "fss_below_s3"},
        "fss_below_s7": {"label": "FSS Below S=7", "suffix": "fss_below_s7"},
        "fss_below_s15":{"label": "FSS Below S=15","suffix": "fss_below_s15"},
        "pearson_corr": {"label": "Pearson Corr",  "suffix": "corr"},
        "rmse":         {"label": "RMSE",          "suffix": "rmse"},
        "mae":          {"label": "MAE",           "suffix": "mae"},
        "mean_error":   {"label": "Mean Error",    "suffix": "mean_error"},
    }

    # ----------------------------------------------------------------
    # Plot each sample metric in the new layout.
    # ----------------------------------------------------------------
    for metric_key, info in metrics_info.items():
        metric_label = info["label"]
        file_suffix = info["suffix"]
        table_rows = []
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), sharex=True)
        axs = np.array(axs)
        if axs.ndim == 1:
            axs = axs[np.newaxis, :]

        # Determine y-axis limits for each row based on all models for channels in that row.
        row_ymins = []
        row_ymaxs = []
        for row_idx, ch_list in enumerate(channel_groups):
            row_vals = []
            for ch in ch_list:
                # Get the original index from the unsorted mapping.
                orig_idx = channel_to_idx_unsorted[ch]
                for m_i, agg_dict in enumerate(aggregated_samples):
                    metric_array = agg_dict[metric_key]  # shape (n_forecast_steps, n_channels)
                    channel_values = metric_array[:, orig_idx]
                    row_vals.append(channel_values)
            if len(row_vals) == 0:
                row_ymins.append(0)
                row_ymaxs.append(1)
            else:
                row_vals = np.concatenate(row_vals)
                row_min = np.nanmin(row_vals)
                row_max = np.nanmax(row_vals)
                margin = 0.05 * (row_max - row_min) if (row_max - row_min) > 0 else 1
                row_ymins.append(row_min - margin)
                row_ymaxs.append(row_max + margin)

        # Plot each row and each column.
        for row_idx, ch_list in enumerate(channel_groups):
            for col_idx in range(ncols):
                ax = axs[row_idx, col_idx]
                if col_idx >= len(ch_list):
                    ax.axis("off")
                    continue
                channel_name = ch_list[col_idx]
                orig_idx = channel_to_idx_unsorted[channel_name]
                for m_i, agg_dict in enumerate(aggregated_samples):
                    model_label = model_names[m_i]
                    epoch_label = str(epochs[m_i]) if epochs else None
                    line_label = f"{model_label} (ep={epoch_label})" if epoch_label else model_label
                    metric_array = agg_dict[metric_key]
                    channel_values = metric_array[:, orig_idx]
                    ax.plot(lead_times, channel_values,
                            marker="o",
                            label=line_label,
                            markerfacecolor='none')
                ax.set_title(channel_name)
                ax.set_ylim(row_ymins[row_idx], row_ymaxs[row_idx])
                if row_idx == (nrows - 1):
                    ax.set_xlabel("Lead Time")
                if col_idx == 0:
                    ax.set_ylabel(metric_label)
                # Save data for pivot table.
                for m_i, agg_dict in enumerate(aggregated_samples):
                    model_label = model_names[m_i]
                    epoch_label = str(epochs[m_i]) if epochs else "-"
                    metric_array = agg_dict[metric_key]
                    channel_values = metric_array[:, orig_idx]
                    for lt_i in range(n_forecast_steps):
                        table_rows.append({
                            "Metric": metric_label,
                            "Model": model_label,
                            "Epoch": epoch_label,
                            "Channel": channel_name,
                            "Lead_Time": lt_i,
                            f"{metric_label}": channel_values[lt_i]
                        })
        # Add legend in the last row, last column.
        legend_ax = axs[-1, -1]
        legend_ax.axis("off")
        handles, labels = [], []
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                if c_idx < len(channel_groups[r_idx]):
                    h, l = axs[r_idx, c_idx].get_legend_handles_labels()
                    if h and l:
                        handles = h
                        labels = l
                        break
            if handles:
                break
        legend_ax.legend(handles, labels, loc="center")

        fig.suptitle(f"{metric_label} vs Lead Time")
        fig.tight_layout()
        plot_path = os.path.join(plot_save_dir, f"compare_models_{file_suffix}_custom.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sample-metric plot for {metric_label}: {plot_path}")

        # Save pivot table CSV.
        df_metric = pd.DataFrame(table_rows)
        pivot_metric = df_metric.pivot_table(
            index="Lead_Time",
            columns=["Model", "Channel"],
            values=metric_label
        )
        csv_path = os.path.join(table_save_dir, f"compare_models_{file_suffix}_custom.csv")
        pivot_metric.to_csv(csv_path, float_format="%.4f")
        print(f"Saved table for {metric_label}: {csv_path}\n")

    # ----------------------------------------------------------------
    # 6) Produce pixel-wise maps as before, now iterating over sorted channels.
    # ----------------------------------------------------------------
    steps_to_plot = [0, 3, 7, 11, 15]  # pick whichever steps you want
    pixel_metric_keys = [
        ("rmse_map",        "RMSE"),
        ("mae_map",         "MAE"),
        ("mean_error_map",  "Mean Error")
    ]

    for (pix_key, pix_label) in pixel_metric_keys:
        # Loop over sorted channels instead of the original order.
        for sorted_idx, ch in enumerate(sorted_channels):
            orig_idx = channel_to_idx_unsorted[ch]
            fig, axs = plt.subplots(len(steps_to_plot), len(model_names),
                                    figsize=(4*len(model_names), 3.2*len(steps_to_plot)),
                                    sharex=True, sharey=True, constrained_layout=True)

            # Ensure axs is a 2D array.
            if len(steps_to_plot) == 1 and len(model_names) == 1:
                axs = np.array([[axs]])
            elif len(steps_to_plot) == 1:
                axs = axs[None, :]
            elif len(model_names) == 1:
                axs = axs[:, None]

            # Collect values to fix color scale across subplots.
            all_values = []
            for row_i, step in enumerate(steps_to_plot):
                for col_i, (model_label, merged) in enumerate(zip(model_names, merged_results)):
                    data_map = merged["pixel_metrics"][pix_key]  # shape (n_channels, n_forecast_steps, H, W)
                    if step < data_map.shape[1]:
                        arr = data_map[orig_idx, step]
                        valid_values = arr[~np.isnan(arr)]
                        all_values.append(valid_values)
            if len(all_values) == 0:
                vmin, vmax = 0, 1
            else:
                all_values_flat = np.concatenate(all_values)
                vmin = np.nanmin(all_values_flat)
                vmax = np.nanmax(all_values_flat)

            for row_i, step in enumerate(steps_to_plot):
                for col_i, (model_label, merged) in enumerate(zip(model_names, merged_results)):
                    ax = axs[row_i, col_i]
                    data_map = merged["pixel_metrics"][pix_key]
                    if step >= data_map.shape[1]:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(f"{model_label} - step={step} (No Data)")
                        continue
                    arr = data_map[orig_idx, step]
                    if "mean_error" in pix_key:
                        cmap = "bwr"
                        vmag = max(abs(vmin), abs(vmax))
                        vmin_ = -vmag
                        vmax_ = vmag
                    else:
                        cmap = "turbo"
                        vmin_ = vmin
                        vmax_ = vmax
                    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin_, vmax=vmax_)
                    if row_i == 0:
                        ax.set_title(f"{model_label}\nStep={step}")
                    else:
                        ax.set_title(f"Step={step}")

            cbar = fig.colorbar(im, ax=axs,
                                location="bottom",
                                orientation="horizontal",
                                fraction=0.02,
                                pad=0.02)
            cbar.set_label(f"{pix_label} {sorted_measures[sorted_idx]}")
            fig.suptitle(f"{ch}")
            out_map = os.path.join(plot_save_dir, f"compare_pixel_{pix_key}_{ch}.png")
            plt.savefig(out_map, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved pixel-wise {pix_label} maps: {out_map}")

    print("All comparisons complete.")


if __name__ == "__main__":
    """
    Example usage:

    python compare_models.py

    In a real workflow, you might parse arguments to handle multiple .pkl files dynamically.
    """
    # Paths to your .pkl metric files (one per model)
    # path = "/capstor/scratch/cscs/acarpent/test_results/"
    path = "/capstor/scratch/cscs/acarpent/validation_results/"


    # pkl_files = [
    #     [
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_3_val_fs768_lat56_lon138.pkl",
    #     ],
    #     [
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_3_val_fs768_lat56_lon138.pkl",
    #     ],
    #     [
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_3_val_fs768_lat56_lon138.pkl",
    #     ]
    # ]

    # pkl_files = [
    #     [
    #         path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1_99_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1_99_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1_99_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1_99_results_3_val_fs768_lat56_lon138.pkl",
    #     ],
    #     [
    #         path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1_95_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1_95_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1_95_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1_95_results_3_val_fs768_lat56_lon138.pkl",
    #     ],
    #     [
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1_99_results_0_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1_99_results_1_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1_99_results_2_val_fs768_lat56_lon138.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1_99_results_3_val_fs768_lat56_lon138.pkl",
    #     ]
    # ]

    pkl_files = [
        [
            path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1-finetuned_19_results_0_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1-finetuned_19_results_1_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1-finetuned_19_results_2_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-large-ks13-nb1-dd4_4_4-ud4_4-spherical_rope-192_768_768-L1-finetuned_19_results_3_val_fs768_lat56_lon138.pkl",
        ],
        [
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_1_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_2_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_3_val_fs768_lat56_lon138.pkl",
        ],
        [
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_35_results_0_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_35_results_1_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_35_results_2_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_35_results_3_val_fs768_lat56_lon138.pkl",
        ],
        [
            path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1-finetuned_18_results_0_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1-finetuned_18_results_1_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1-finetuned_18_results_2_val_fs768_lat56_lon138.pkl",
            path+"UNATCast-1024-s2-tss-dd048-ud40-ks5-skip-ls_0-L1-v1-finetuned_18_results_3_val_fs768_lat56_lon138.pkl",
        ],
    ]



    # Names of each model
    model_names = [
        # "UNATCast",
        # "PredRNN ++",
        # "AFNONATCast"
        "UNATCast - spherical",
        "PredRNN ++",
        "AFNONATCast",
        "UNATCast - RPB"
        
    ]

    compare_models(
        model_pkl_paths=pkl_files,
        model_names=model_names,
        epochs=None,
        plot_save_dir=os.path.join(path, "plots"),
        table_save_dir=os.path.join(path, "tables")
    )
