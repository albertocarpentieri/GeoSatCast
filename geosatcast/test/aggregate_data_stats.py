#!/usr/bin/env python
import os
import pickle as pkl
import numpy as np
import torch
import argparse
import datetime
import matplotlib.pyplot as plt

# Provided channel names.
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
# Mapping from channel index (as string) to channel name.
channel_names = {str(i): NON_HRV_BANDS[i] for i in range(len(NON_HRV_BANDS))}

##############################################
# Aggregation Functions
##############################################
def load_partial_results(partial_dir):
    """
    Load all partial result pickle files from the given directory.
    Each file is expected to contain a dictionary whose keys are sample IDs and whose values are dictionaries with keys:
      "temporal_autocorrelation", "spatial_autocorrelation", "intra_channel_corr", and "time".
    Returns a list of sample dictionaries.
    """
    partial_files = [os.path.join(partial_dir, f) for f in os.listdir(partial_dir) if f.endswith(".pkl")]
    all_samples = []
    for file in partial_files:
        with open(file, "rb") as f:
            partial = pkl.load(f)
        for sample_id, sample_data in partial.items():
            sample_data["_id"] = str(sample_id)
            all_samples.append(sample_data)
    return all_samples

def aggregate_results(samples):
    """
    Aggregate metrics over all samples while preserving channel information.
    
    Each sample is expected to have:
      - "temporal_autocorrelation": { channel: { lag: value } }
      - "spatial_autocorrelation": { channel: { shift_key: tensor (of shape (T,)) } }
      - "intra_channel_corr": tensor of shape (T, C, C) (to be averaged over time)
      - "time": an array of timestamps (we use the first element, converted via utcfromtimestamp)
    
    Additionally, for intra‑channel correlations we store every sample’s full (C, C) matrix.
    For every unique pair (i,j) with i<j, we also collect (timestamp, value) tuples.
    
    Returns a dictionary with keys:
      "temporal_autocorrelation": { channel: { lag: stats } }
      "spatial_autocorrelation": { channel: { shift: stats } }
      "intra_channel_corr": { channel: stats }   # per-channel scalar summary (average off-diagonals)
      "intra_channel_corr_matrix": { "average": ndarray, "q25": ndarray, "q75": ndarray }
      "intra_channel_pair": { "i_j": list of (timestamp, value) }
      "samples": list of dicts with "timestamp" and "intra_channel_corr" (vector)
    where stats is a dict with keys "average", "q25", and "q75".
    """
    temporal_by_channel = {}  # { channel: { lag: [values, ...] } }
    spatial_by_channel = {}   # { channel: { shift: [values, ...] } }
    intra_by_channel = {}     # { channel: [scalar, ...] }
    intra_pair_data = {}      # { "i_j": [(timestamp, value), ...] } for i<j
    intra_matrix_list = []    # list of (C, C) matrices (each sample's averaged matrix)
    samples_info = []         # List of dicts with "timestamp" and "intra_channel_corr" (vector)

    for sample in samples:
        ts = None
        if "time" in sample:
            try:
                # Assume sample["time"] is an array; use its first element.
                ts = datetime.datetime.utcfromtimestamp(float(sample["time"][0]))
            except Exception:
                ts = None
        if ts is None:
            try:
                ts = datetime.datetime.fromisoformat(str(sample.get("_id", "")))
            except Exception:
                ts = None
        
        # Temporal autocorrelation per channel.
        temp_acorr = sample.get("temporal_autocorrelation", {})
        for ch, ch_dict in temp_acorr.items():
            ch_key = str(ch)
            temporal_by_channel.setdefault(ch_key, {})
            for lag, val in ch_dict.items():
                lag_key = str(lag)
                temporal_by_channel[ch_key].setdefault(lag_key, []).append(val)
        
        # Spatial autocorrelation per channel.
        spat_acorr = sample.get("spatial_autocorrelation", {})
        for ch, ch_dict in spat_acorr.items():
            ch_key = str(ch)
            spatial_by_channel.setdefault(ch_key, {})
            for shift_key, tensor_val in ch_dict.items():
                if isinstance(tensor_val, torch.Tensor):
                    tensor_val = tensor_val.cpu().numpy()
                avg_val = np.nanmean(tensor_val)
                spatial_by_channel[ch_key].setdefault(shift_key, []).append(avg_val)
        
        # Intra-channel correlation.
        intra = sample.get("intra_channel_corr", None)
        if intra is not None:
            if isinstance(intra, torch.Tensor):
                intra = intra.cpu().numpy()
            # Average over time → (C, C)
            matrix_avg = np.nanmean(intra, axis=0)
            intra_matrix_list.append(matrix_avg)
            C = matrix_avg.shape[0]
            intra_vec = np.zeros(C)
            for i in range(C):
                row = matrix_avg[i, :]
                off_diag = row[np.arange(C) != i]
                intra_vec[i] = np.nanmean(off_diag)
            for i, val in enumerate(intra_vec):
                ch_key = str(i)
                intra_by_channel.setdefault(ch_key, []).append(val)
            # For each unique pair (i, j) with i<j, store (timestamp, value).
            for i in range(C):
                for j in range(i+1, C):
                    key = f"{i}_{j}"
                    intra_pair_data.setdefault(key, []).append((ts, matrix_avg[i, j]))
            samples_info.append({"timestamp": ts, "intra_channel_corr": intra_vec})
    
    def agg_stats(values):
        values = np.array(values, dtype=float)
        avg = np.nanmean(values)
        q25 = np.nanpercentile(values, 25)
        q75 = np.nanpercentile(values, 75)
        return {"average": avg, "q25": q25, "q75": q75}
    
    temporal_agg = {ch: {lag: agg_stats(vals) for lag, vals in lag_dict.items()}
                    for ch, lag_dict in temporal_by_channel.items()}
    spatial_agg = {ch: {shift: agg_stats(vals) for shift, vals in shift_dict.items()}
                   for ch, shift_dict in spatial_by_channel.items()}
    intra_agg = {ch: agg_stats(vals) for ch, vals in intra_by_channel.items()}
    
    # Aggregate the full intra-channel correlation matrices.
    if intra_matrix_list:
        mats = np.stack(intra_matrix_list, axis=0)  # shape: (n_samples, C, C)
        matrix_avg = np.nanmean(mats, axis=0)
        matrix_q25 = np.nanpercentile(mats, 25, axis=0)
        matrix_q75 = np.nanpercentile(mats, 75, axis=0)
        intra_matrix_agg = {"average": matrix_avg, "q25": matrix_q25, "q75": matrix_q75}
    else:
        intra_matrix_agg = None

    agg = {
        "temporal_autocorrelation": temporal_agg,
        "spatial_autocorrelation": spatial_agg,
        "intra_channel_corr": intra_agg,
        "intra_channel_corr_matrix": intra_matrix_agg,
        "intra_channel_pair": intra_pair_data,
        "samples": samples_info
    }
    return agg

##############################################
# Plotting Functions
##############################################
def plot_temporal_autocorrelation_all_channels(temporal_data, out_dir, channel_names):
    """
    Plot temporal autocorrelation curves (mean only) for all channels in one plot.
    """
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("tab20")
    for idx, ch in enumerate(sorted(temporal_data.keys(), key=lambda x: int(x))):
        ch_data = temporal_data[ch]
        lags = sorted(ch_data.keys(), key=lambda x: float(x))
        x = np.array([float(lag) for lag in lags])
        avg = np.array([ch_data[lag]["average"] for lag in lags])
        color = cmap(idx)
        plt.plot(x, avg, marker="o", color=color, label=channel_names.get(ch, f"Ch {ch}"))
    plt.xlabel("Lag")
    plt.ylabel("Temporal Autocorrelation")
    plt.title("Temporal Autocorrelation by Channel")
    plt.legend()
    out_path = os.path.join(out_dir, "temporal_autocorrelation_all_channels.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal autocorrelation plot to: {out_path}")

def plot_spatial_autocorrelation_both_directions(spatial_data, out_dir, channel_names):
    """
    Plot spatial autocorrelation curves (mean only, without error bounds) for both horizontal and vertical directions
    in a single figure with two subplots.
    """
    cmap = plt.get_cmap("tab20")
    fig, axs = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    directions = ["horizontal", "vertical"]
    for ax, direction in zip(axs, directions):
        for idx, ch in enumerate(sorted(spatial_data.keys(), key=lambda x: int(x))):
            ch_data = spatial_data[ch]
            filtered = {k: v for k, v in ch_data.items() if k.startswith(direction)}
            if not filtered:
                continue
            def shift_key_sort(k):
                try:
                    return float(k.split("_")[-1])
                except Exception:
                    return 0
            shifts = sorted(filtered.keys(), key=shift_key_sort)
            x = np.array([shift_key_sort(s) for s in shifts])
            avg = np.array([filtered[s]["average"] for s in shifts])
            color = cmap(idx)
            ax.plot(x, avg, marker="o", color=color, label=channel_names.get(ch, f"Ch {ch}"))
        ax.set_xlabel("Shift (pixels)")
        
        ax.set_title(f"{direction.capitalize()} Shifts")
        ax.legend()
    axs[0].set_ylabel("Spatial Autocorrelation")
    out_path = os.path.join(out_dir, "spatial_autocorrelation_both_directions.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial autocorrelation (both directions) plot to: {out_path}")

def plot_intra_corr_matrix_aggregated(intra_matrix_agg, out_dir, channel_names):
    """
    Plot the aggregated intra-channel correlation matrix (mean only) as a heatmap,
    with the numeric correlation values annotated in each cell.
    """
    if intra_matrix_agg is None:
        print("No aggregated intra-channel matrix available.")
        return
    # Use only the mean matrix.
    M = intra_matrix_agg["average"]
    N = M.shape[0]
    plt.figure(figsize=(8,6))
    im = plt.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Aggregated Intra-Channel Correlation Matrix (Mean)")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    # Set tick labels using channel names.
    ticks = np.arange(N)
    tick_labels = [channel_names.get(str(i), f"Ch {i}") for i in range(N)]
    plt.xticks(ticks, tick_labels, rotation=45, ha="right")
    plt.yticks(ticks, tick_labels)
    # Annotate each cell with the numeric value.
    for i in range(N):
        for j in range(N):
            text = f"{M[i, j]:.2f}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
    out_path = os.path.join(out_dir, "intra_channel_corr_matrix_mean.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated intra-channel correlation matrix (mean) plot to: {out_path}")

def plot_intra_corr_pairs_matrix_binned(intra_pair_data, bin_type, out_dir, channel_names):
    """
    Create an N x N grid of subplots for intra-channel pair correlations.
    For each pair (i, j) with i < j, bin the correlation values (using the timestamps from the tuple)
    by time-of-day ("hour") or by month ("day") and plot the binned average and quantiles.
    For i >= j, leave the subplot blank.
    """
    if bin_type == "hour":
        bins = np.linspace(0, 24, 25)
        xlabel = "Hour of Day"
    elif bin_type == "day":
        bins = np.linspace(0.5, 12.5, 13)
        xlabel = "Month"
    else:
        return

    N = len(channel_names)
    fig, axs = plt.subplots(N, N, figsize=(2.5*N, 2.5*N), squeeze=False)
    for i in range(N):
        for j in range(N):
            ax = axs[i, j]
            if i >= j:
                ax.axis("off")
                continue
            key = f"{i}_{j}"
            if key not in intra_pair_data:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue
            data = intra_pair_data[key]
            bin_vals = []
            vals = []
            for ts, val in data:
                if ts is None:
                    continue
                if bin_type == "hour":
                    b = ts.hour + ts.minute/60.0
                elif bin_type == "day":
                    b = ts.month
                bin_vals.append(b)
                vals.append(val)
            if len(bin_vals) == 0:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue
            bin_vals = np.array(bin_vals)
            vals = np.array(vals)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            stats = {}
            for k in range(len(bins)-1):
                mask = (bin_vals >= bins[k]) & (bin_vals < bins[k+1])
                if np.sum(mask) == 0:
                    continue
                subset = vals[mask]
                avg_val = np.nanmean(subset)
                q25_val = np.nanpercentile(subset, 25)
                q75_val = np.nanpercentile(subset, 75)
                stats[bin_centers[k]] = {"average": avg_val, "q25": q25_val, "q75": q75_val}
            if not stats:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue
            xs = np.array(sorted(stats.keys()))
            avgs = np.array([stats[x]["average"] for x in sorted(stats.keys())])
            q25s = np.array([stats[x]["q25"] for x in sorted(stats.keys())])
            q75s = np.array([stats[x]["q75"] for x in sorted(stats.keys())])
            ax.plot(xs, avgs, marker="o", markersize=3, linewidth=1)
            ax.fill_between(xs, q25s, q75s, alpha=0.3)
            ax.set_title(f"{channel_names.get(str(i), 'Ch '+str(i))} vs {channel_names.get(str(j), 'Ch '+str(j))}", fontsize=7)
            if i == N-1:
                ax.set_xlabel(xlabel, fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Corr", fontsize=6)
            else:
                ax.set_yticklabels([])
    fig.suptitle(f"Intra-Channel Pair Correlations vs. {xlabel}", fontsize=14)
    out_path = os.path.join(out_dir, f"intra_corr_pairs_matrix_binned_{bin_type}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved intra-channel pair {bin_type} matrix plot to: {out_path}")

##############################################
# Save Aggregated Results
##############################################
def save_aggregated_results(agg, out_file):
    with open(out_file, "wb") as f:
        pkl.dump(agg, f)
    print(f"Aggregated results saved to: {out_file}")

##############################################
# Main Routine
##############################################
def main():
    parser = argparse.ArgumentParser(description="Aggregate partial results and generate plots per channel and per channel pair.")
    parser.add_argument("--partial_dir", type=str, required=True,
                        help="Directory containing partial result .pkl files.")
    parser.add_argument("--out_dir", type=str, default="./aggregated_results",
                        help="Output directory for aggregated results and plots.")
    parser.add_argument("--agg_file", type=str, default="aggregated_results.pkl",
                        help="Filename for the aggregated results pickle file.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    samples = load_partial_results(args.partial_dir)
    print(f"Loaded {len(samples)} samples from partial results.")
    
    agg = aggregate_results(samples)
    
    agg_path = os.path.join(args.out_dir, args.agg_file)
    save_aggregated_results(agg, agg_path)
    
    # Plot temporal autocorrelation (mean only, no error bounds).
    plot_temporal_autocorrelation_all_channels(agg["temporal_autocorrelation"], args.out_dir, channel_names)
    # Plot spatial autocorrelation (both directions, in one figure, no error bounds).
    plot_spatial_autocorrelation_both_directions(agg["spatial_autocorrelation"], args.out_dir, channel_names)
    # Plot the aggregated intra-channel correlation matrix as 3 subplots.
    plot_intra_corr_matrix_aggregated(agg["intra_channel_corr_matrix"], args.out_dir, channel_names)
    # Plot intra-channel pair correlations binned by hour and by day.
    plot_intra_corr_pairs_matrix_binned(agg["intra_channel_pair"], "hour", args.out_dir, channel_names)
    plot_intra_corr_pairs_matrix_binned(agg["intra_channel_pair"], "day", args.out_dir, channel_names)

if __name__ == "__main__":
    main()
