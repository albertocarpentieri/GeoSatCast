import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import List, Union

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

    Steps:
      - Combine 'sample_metrics' by simple dictionary update (keys are init_times).
      - Accumulate partial sums in 'pixel_metrics', then compute final maps.
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

    # Compute final mean_error, MAE, RMSE maps
    # shape: (channels, forecast_steps, H, W)
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
    Takes merged sample_metrics dict:
      {
         init_time1: { 'csi_above': [[...], ...], 'csi_below': [[...], ...], ...},
         init_time2: {...},
         ...
      }
    and aggregates each metric across all init_times (samples).

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

    # Inspect first sample to find shapes
    example_entry = sample_metrics_dict[all_times[0]]
    n_forecast_steps = len(example_entry["csi_above"])
    n_channels = len(example_entry["csi_above"][0])

    metric_arrays = {mk: [] for mk in needed_keys}

    for t_key in all_times:
        entry = sample_metrics_dict[t_key]
        for mk in needed_keys:
            arr = np.array(entry[mk])  # shape (n_forecast_steps, n_channels)
            metric_arrays[mk].append(arr)

    # Stack => shape (n_samples, n_forecast_steps, n_channels)
    # Then average => shape (n_forecast_steps, n_channels)
    aggregated = {}
    for mk in needed_keys:
        stacked = np.array(metric_arrays[mk])  # (n_samples, n_forecast_steps, n_channels)
        mean_val = np.nanmean(stacked, axis=0)
        aggregated[mk] = mean_val

    aggregated["n_forecast_steps"] = n_forecast_steps
    aggregated["n_channels"] = n_channels
    return aggregated


def compare_models(
    model_pkl_paths: List[List[str]],
    model_names: List[str],
    epochs: List[Union[int, str]] = None,
    plot_save_dir="./plots",
    table_save_dir="./tables"
):
    """
    1) Merge partial files for each model -> single result with sample_metrics + pixel_metrics.
    2) Aggregate sample_metrics -> produce line plots (one figure per metric) using a custom layout:
         - 2 rows (3+3) for the first 6 channels: IR_039, IR_087, IR_097, IR_108, IR_120, IR_134
         - 1 row for: IR_016, VIS006, VIS008
         - Last row plus legend for: WV_062, WV_073
       Each row has its own y-limits computed from the data in that row.
    3) For pixel_metrics (rmse_map, mae_map, mean_error_map), same as before: one figure per metric & channel,
       arranged in [selected forecast steps] x [models].
    """
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(table_save_dir, exist_ok=True)

    assert len(model_pkl_paths) == len(model_names), "Mismatch in model paths vs. names"
    if epochs is not None:
        assert len(epochs) == len(model_names), "Mismatch in epochs vs. names"

    # 1) Merge partial .pkl files per model
    merged_results = []
    aggregated_samples = []
    for i, pkl_list in enumerate(model_pkl_paths):
        print(f"Merging partial files for model: {model_names[i]}")
        merged = merge_model_partials(pkl_list)
        merged_results.append(merged)

        # 2) Aggregate sample-wise metrics
        agg_samples = aggregate_sample_metrics(merged["sample_metrics"])
        aggregated_samples.append(agg_samples)

    # We'll assume all have the same n_forecast_steps, n_channels
    n_forecast_steps = aggregated_samples[0]["n_forecast_steps"]
    n_channels = aggregated_samples[0]["n_channels"]
    lead_times = np.arange(n_forecast_steps)

    # Map from channel name -> index in aggregated arrays
    channel_to_idx = {name: i for i, name in enumerate(NON_HRV_BANDS)}

    # Custom layout for sample metrics line plots.
    #  - first 2 rows for these 6 channels:
    #       IR_039, IR_087, IR_097, IR_108, IR_120, IR_134
    #    grouped as: row0 -> [IR_039, IR_087, IR_097],
    #                row1 -> [IR_108, IR_120, IR_134]
    #  - 3rd row -> [IR_016, VIS006, VIS008]
    #  - 4th row -> [WV_062, WV_073] + 1 slot for legend
    channel_groups = [
        ["IR_039", "IR_087", "IR_097"],   # row0
        ["IR_108", "IR_120", "IR_134"],   # row1
        ["IR_016", "VIS006", "VIS008"],   # row2
        ["WV_062", "WV_073"]             # row3 (plus legend in col2)
    ]
    nrows = 4
    ncols = 3

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
    # Plot each metric in the new layout
    # ----------------------------------------------------------------
    for metric_key, info in metrics_info.items():
        metric_label = info["label"]
        file_suffix = info["suffix"]

        # Prepare for CSV pivot
        table_rows = []

        # Initialize figure
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), sharex=True)
        axs = np.array(axs)
        if axs.ndim == 1:
            axs = axs[np.newaxis, :]

        # ------------------------------------------------------------
        # 1) Determine row-wise min/max for auto y-lims
        # ------------------------------------------------------------
        row_ymins = []
        row_ymaxs = []
        for row_idx, ch_list in enumerate(channel_groups):
            # gather all values for these channels (across all models & leads)
            row_vals = []
            for ch in ch_list:
                c_idx = channel_to_idx[ch]
                for m_i, agg_dict in enumerate(aggregated_samples):
                    metric_array = agg_dict[metric_key]  # shape (n_forecast_steps, n_channels)
                    channel_values = metric_array[:, c_idx]  # shape (n_forecast_steps,)
                    row_vals.append(channel_values)
            if len(row_vals) == 0:
                # fallback
                row_ymins.append(0)
                row_ymaxs.append(1)
            else:
                row_vals = np.concatenate(row_vals)
                row_min = np.nanmin(row_vals)
                row_max = np.nanmax(row_vals)
                # add a little margin
                margin = 0.05*(row_max - row_min) if (row_max - row_min) > 0 else 1
                row_ymins.append(row_min - margin)
                row_ymaxs.append(row_max + margin)

        # ------------------------------------------------------------
        # 2) Plot each row
        # ------------------------------------------------------------
        for row_idx, ch_list in enumerate(channel_groups):
            for col_idx in range(ncols):
                ax = axs[row_idx, col_idx]

                # if no channel in this column for the row => turn off axis
                if col_idx >= len(ch_list):
                    # If this is the last row and the last col => legend
                    # row_idx=3 => WV_062, WV_073 => col=0,1 => col=2 => legend
                    if (row_idx == (nrows - 1)) and (col_idx == (ncols - 1)):
                        ax.axis("off")
                        continue
                    else:
                        ax.axis("off")
                        continue

                # We have a channel
                channel_name = ch_list[col_idx]
                c_idx = channel_to_idx[channel_name]

                # Plot each model's data for this channel
                for m_i, agg_dict in enumerate(aggregated_samples):
                    model_label = model_names[m_i]
                    epoch_label = str(epochs[m_i]) if epochs else None
                    line_label = f"{model_label} (ep={epoch_label})" if epoch_label else model_label

                    metric_array = agg_dict[metric_key]  # shape (n_forecast_steps, n_channels)
                    channel_values = metric_array[:, c_idx]
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

                # Collect data for CSV pivot
                for m_i, agg_dict in enumerate(aggregated_samples):
                    model_label = model_names[m_i]
                    epoch_label = str(epochs[m_i]) if epochs else "-"
                    metric_array = agg_dict[metric_key]
                    channel_values = metric_array[:, c_idx]
                    for lt_i in range(n_forecast_steps):
                        table_rows.append({
                            "Metric": metric_label,
                            "Model": model_label,
                            "Epoch": epoch_label,
                            "Channel": channel_name,
                            "Lead_Time": lt_i,
                            f"{metric_label}": channel_values[lt_i]
                        })

        # ------------------------------------------------------------
        # 3) Legend in the last row, last column
        # ------------------------------------------------------------
        legend_ax = axs[-1, -1]
        legend_ax.axis("off")  # blank
        # get handles/labels from the *first valid subplot*
        # you can also do it from e.g. axs[0,0] if you prefer
        handles, labels = [], []
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                if c_idx < len(channel_groups[r_idx]):  # a real channel
                    h, l = axs[r_idx, c_idx].get_legend_handles_labels()
                    if h and l:
                        handles = h
                        labels = l
                        break
            if handles:
                break
        legend_ax.legend(handles, labels, loc="center")

        # Final figure touches
        fig.suptitle(f"{metric_label} vs Lead Time (New Ordering)")
        fig.tight_layout()

        # 4) Save figure
        plot_path = os.path.join(plot_save_dir, f"compare_models_{file_suffix}_custom.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sample-metric plot for {metric_label}: {plot_path}")

        # 5) Save pivot table CSV
        df_metric = pd.DataFrame(table_rows)
        pivot_metric = df_metric.pivot_table(
            index="Lead_Time",
            columns=["Model", "Channel"],
            values=metric_label
        )
        csv_path = os.path.join(table_save_dir, f"compare_models_{file_suffix}_custom.csv")
        pivot_metric.to_csv(csv_path, float_format="%.4f")
        print(f"Saved table for {metric_label}: {csv_path}\n")

    # ---------------------------------------------------------------
    # 6) Produce pixel-wise maps as before
    # ---------------------------------------------------------------
    steps_to_plot = [0, 3, 7, 11, 15]  # pick whichever steps you want
    pixel_metric_keys = [
        ("rmse_map",        "RMSE"),
        ("mae_map",         "MAE"),
        ("mean_error_map",  "Mean Error")
    ]

    for (pix_key, pix_label) in pixel_metric_keys:
        for c in range(n_channels):
            # We'll build one figure for this metric & channel.
            # Rows = steps_to_plot, Cols = #models
            # Each cell => 2D imshow of the pixel map
            fig, axs = plt.subplots(len(steps_to_plot), len(model_names),
                                    figsize=(4*len(model_names), 3.2*len(steps_to_plot)),
                                    sharex=True, sharey=True, constrained_layout=True)

            # If we have only 1 row or 1 col, axs might not be a 2D array => unify
            if len(steps_to_plot) == 1 and len(model_names) == 1:
                axs = np.array([[axs]])
            elif len(steps_to_plot) == 1:
                axs = axs[None, :]  # shape (1, n_models)
            elif len(model_names) == 1:
                axs = axs[:, None]  # shape (n_steps, 1)

            # Collect all data to unify color scale across subplots
            all_values = []
            for row_i, step in enumerate(steps_to_plot):
                for col_i, (model_label, merged) in enumerate(zip(model_names, merged_results)):
                    data_map = merged["pixel_metrics"][pix_key]  # shape (n_channels, n_forecast_steps, H, W)
                    if step < data_map.shape[1]:
                        arr = data_map[c, step]  # shape (H, W)
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

                    arr = data_map[c, step]  # shape (H, W)
                    if "mean_error" in pix_key:
                        cmap = "bwr"
                        # Symmetric color scale for error
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

            # Single colorbar for the entire figure
            cbar = fig.colorbar(
                im, ax=axs,
                location="bottom",
                orientation="horizontal",
                fraction=0.02,
                pad=0.02
            )
            cbar.set_label(f"{pix_label} {MEASURES[c]}")

            fig.suptitle(f"{NON_HRV_BANDS[c]}")
            out_map = os.path.join(
                plot_save_dir,
                f"compare_pixel_{pix_key}_{NON_HRV_BANDS[c]}.png"
            )
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
    path = "/capstor/scratch/cscs/acarpent/test_results/"
    # pkl_files = [
    #     [
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned_61_results_0_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned_61_results_1_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned_61_results_2_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned_61_results_3_val_fs256_lat300_lon300.pkl",
    #     ],
    #     [
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_26_results_0_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_26_results_1_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_26_results_2_val_fs256_lat300_lon300.pkl",
    #         path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned_26_results_3_val_fs256_lat300_lon300.pkl",
    #     ],
    #     [
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs256_lat300_lon300.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_1_val_fs256_lat300_lon300.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_2_val_fs256_lat300_lon300.pkl",
    #         path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_3_val_fs256_lat300_lon300.pkl",
    #     ]
    # ]

    pkl_files = [
        [
            path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_0_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_1_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_2_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-512-s2-tss-ls_0-fd_2-ks_5-seq-L1-v1-finetuned-2_13_results_3_val_fs768_lat56_lon138.pkl",
        ],
        [
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_0_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_1_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_2_val_fs768_lat56_lon138.pkl",
            path+"AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1-finetuned-2_14_results_3_val_fs768_lat56_lon138.pkl",
        ],
        [
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
            path+"predrnn-inv-s2-fd_5-nh_64-v1-finetuned_45_results_0_val_fs768_lat56_lon138.pkl",
        ]
    ]


    # Names of each model
    model_names = [
        "AFNONATCast - small",
        "AFNONATCast - large",
        "PredRNN++",
    ]
    # (Optional) epochs for each model
    epochs = [61, 26, 45]

    compare_models(
        model_pkl_paths=pkl_files,
        model_names=model_names,
        epochs=None,
        plot_save_dir=os.path.join(path, "plots"),
        table_save_dir=os.path.join(path, "tables")
    )
