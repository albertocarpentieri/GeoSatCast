# nowcast_test.py
import numpy as np
import torch
import pickle as pkl
import datetime
import argparse
import time
import os

from metrics import (
    critical_success_index_torch,
    fraction_skill_score_torch,
    pearson_correlation_torch,
    rmse_torch,
    mae_torch,
    mean_error_torch
)

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster, load_predrnn, load_unatcast
from geosatcast.data.distributed_dataset import DistributedDataset
from fvcore.nn import FlopCountAnalysis, flop_count_table

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0,
                        help="Which subset index to process?")
    parser.add_argument("--n_subsets", type=int, default=4,
                        help="How many total subsets?")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Which GPU to use (0,1,2,3)?")

    parser.add_argument("--model_names", type=str, default="AFNONATCast-1024-s2-tss-ls_0-fd_8-ks_5-seq-L1-v1",
                        help="Comma-separated list of model names.")
    parser.add_argument("--epochs", type=str, default="99",
                        help="Comma-separated list of epochs.")
    parser.add_argument("--ckpt_paths", type=str, default="/capstor/scratch/cscs/acarpent/Checkpoints",
                        help="Comma-separated list of ckpt paths.")

    parser.add_argument("--in_steps", type=int, default=2,
                        help="Number of input timesteps.")
    parser.add_argument("--n_forecast_steps", type=int, default=8,
                        help="Number of forecast steps to predict.")
    parser.add_argument("--field_size", type=int, default=256,
                        help="Spatial size of the subregion.")
    parser.add_argument("--lat_i", type=int, default=124,
                        help="Latitude index to extract.")
    parser.add_argument("--lon_i", type=int, default=124,
                        help="Longitude index to extract.")
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Number of samples to test.")
    parser.add_argument("--year", type=int, default=2020,
                        help="Data Year.")

    args = parser.parse_args()
    return args


def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    args = parse_args()

    subset_idx = args.subset
    total_subsets = args.n_subsets
    gpu_id = args.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Parse model names and epochs
    model_names = args.model_names.split(",")
    epochs = [int(e) for e in args.epochs.split(",")]
    ckpt_paths = args.ckpt_paths.split(",")

    in_steps = args.in_steps
    field_size = args.field_size
    n_forecast_steps = args.n_forecast_steps
    lat_i = args.lat_i
    lon_i = args.lon_i
    n_samples = args.n_samples
    year = args.year
    if year <= 2020:
        name = '16b_virtual'
        folder_name = "validation"
    else:
        name = '32b_virtual'
        folder_name = "test"

    add_latlon = False
    # Loop over multiple models/epochs
    for model_name, ckpt_path in zip(model_names, ckpt_paths):
        if "unat" in model_name.lower():
            add_latlon = True
        # Prepare dataset
        dataset = DistributedDataset(
            data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
            invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
            name=name,
            years=[year],
            input_len=in_steps + n_forecast_steps,
            output_len=None,
            channels=np.arange(11),
            field_size=field_size,
            length=None,
            validation=True,
            rank=0,
            add_latlon=add_latlon,
        )

        # Random sampling
        np.random.seed(0)
        full_range = np.arange(len(dataset.indices))
        np.random.shuffle(full_range)
        full_range = full_range[:n_samples]

        chunk_size = int(np.ceil(n_samples / total_subsets))
        start = subset_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        indices = full_range[start:end]
        
        for epoch in epochs:
            print(f"\n=== Testing Model: {model_name}, Epoch: {epoch} ===")

            # Load model
            if "predrnn" in model_name.lower():
                nowcaster = load_predrnn(
                    os.path.join(ckpt_path, f"{model_name}/{model_name}_{epoch}.pt"),
                    in_steps=in_steps
                ).to(device)
            elif "unat" in model_name.lower():
                nowcaster = load_unatcast(
                    os.path.join(ckpt_path, f"{model_name}/{model_name}_{epoch}.pt"), 
                    in_steps=in_steps,
                ).to(device)
            else:
                nowcaster = load_nowcaster(
                    os.path.join(ckpt_path, f"{model_name}/{model_name}_{epoch}.pt"),
                    in_steps=in_steps
                ).to(device)

            print(nowcaster)
            print("Number of parameters:", count_parameters(nowcaster))

            # --- PARTIAL ACCUMULATORS FOR PIXEL-WISE METRICS ---
            # shape: (channels=11, forecast_steps, field_size, field_size)
            sum_diff_map = torch.zeros((11, n_forecast_steps, field_size, field_size), device=device)
            sum_abs_diff_map = torch.zeros_like(sum_diff_map, device=device)
            sum_sq_diff_map = torch.zeros_like(sum_diff_map, device=device)
            count_valid_map = torch.zeros_like(sum_diff_map, device=device)  # to store counts

            # Dictionary for sample-wise metrics
            metric_dict = {}

            # Stats for denormalization
            stds = dataset.stds.reshape(-1, 1, 1, 1).to(device)
            means = dataset.means.reshape(-1, 1, 1, 1).to(device)
            # high_thresholds = means + stds / 2
            # low_thresholds = means - stds / 2

            # Evaluate each chosen sample
            flops = True
            for idx in indices:
                print(f"Processing dataset index: {idx}")
                year, t_i = dataset.indices[idx]

                batch = dataset.get_data(
                    year=year, t_i=t_i,
                    lat_i=lat_i, lon_i=lon_i
                )

                if len(batch) == 4:
                    x, t, inv, sza = batch
                    grid = None 
                elif len(batch) == 5:
                    x, t, inv, sza, grid = batch

                t = t[0, :, 0, 0].numpy().astype(int)
                x = x[None].to(device)   # [1, 11, time, H, W]
                inv = inv[None].to(device)
                sza = sza[None].to(device)
                if grid is not None:
                    grid = grid[None].to(device)

                x_in = x[:, :, :in_steps]
                y_true_ = x[:, :, in_steps:in_steps + n_forecast_steps]

                # Adjust invariants shape
                sza_inf = sza[:, :, :in_steps + n_forecast_steps - 1]
                sza_mask = sza[:, :, in_steps:]
                inv = torch.cat((inv.expand(*inv.shape[:2], *sza_inf.shape[2:]), sza_inf), dim=1)

                # Inference
                with torch.no_grad():
                    start_time = time.time()
                    if grid is not None:
                        y_pred_ = nowcaster(x_in, inv, grid, n_steps=n_forecast_steps)
                    else:
                        y_pred_ = nowcaster(x_in, inv, n_steps=n_forecast_steps)
                    y_pred_[0,0,sza_mask[0,0]<0] = torch.nan
                    y_pred_[0,7,sza_mask[0,0]<0] = torch.nan
                    y_pred_[0,8,sza_mask[0,0]<0] = torch.nan
                    print("Inference time (sec):", time.time() - start_time)

                    if flops:
                        if grid is not None:
                            flops = FlopCountAnalysis(nowcaster, (x_in, inv, grid, 1))
                        else:
                            flops = FlopCountAnalysis(nowcaster, (x_in, inv, 1))
                        print("FLOP Count Table:")
                        print(flop_count_table(flops))

                        # Compute total GFLOPs for the traced pass.
                        total_gflops = flops.total() / 1e9
                        print(f"Total GFLOPs (traced pass): {total_gflops:.2f} GFLOPs")
                        flops = False
                
                
                # Remove batch dim => [11, n_forecast_steps, H, W]
                y_true_ = y_true_[0] * stds + means
                y_pred_ = y_pred_[0] * stds + means

                # --- ACCUMULATE PARTIAL SUMS ---
                diffs = (y_pred_ - y_true_)                     # shape [11, n_forecast_steps, H, W]
                abs_diffs = torch.abs(diffs)
                sq_diffs = diffs ** 2

                # Valid = ~NaN in y_pred (or y_true, typically same shape)
                valid_mask = ~torch.isnan(y_pred_)
                # You might also exclude NaNs in y_true_ if that's relevant:
                valid_mask = valid_mask & ~torch.isnan(y_true_)

                # For valid pixels, accumulate sums
                # We do an in-place where valid_mask = True
                sum_diff_map += torch.where(valid_mask, diffs, torch.zeros_like(diffs))
                sum_abs_diff_map += torch.where(valid_mask, abs_diffs, torch.zeros_like(abs_diffs))
                sum_sq_diff_map += torch.where(valid_mask, sq_diffs, torch.zeros_like(sq_diffs))
                count_valid_map += valid_mask.type_as(sum_diff_map)

                # --- SAMPLE-WISE METRICS (spatially averaged) ---
                metric_dict_ = {
                    "time": t[in_steps:],
                    "loc": [lat_i, lon_i],
                    "csi_above": [],
                    "csi_below": [],
                    "fss_above_s3": [], "fss_below_s3": [],
                    "fss_above_s7": [], "fss_below_s7": [],
                    "fss_above_s15": [], "fss_below_s15": [],
                    "pearson_corr": [],
                    "rmse": [],
                    "mae": [],
                    "mean_error": []
                }

                for step in range(n_forecast_steps):
                    csi_above_step_s = []
                    csi_below_step_s = []
                    fss_above_s3_s = []
                    fss_below_s3_s = []
                    fss_above_s7_s = []
                    fss_below_s7_s = []
                    fss_above_s15_s = []
                    fss_below_s15_s = []
                    corr_step_s = []
                    rmse_step_s = []
                    mae_step_s = []
                    me_step_s = []

                    for c in range(11):
                        y_c_t = y_true_[c, step]   # [H, W]
                        yhat_c_t = y_pred_[c, step]
                        
                        high_thr = torch.quantile(y_c_t, .8)
                        low_thr = torch.quantile(y_c_t, .2)

                        csi_abv = critical_success_index_torch(y_c_t, yhat_c_t, high_thr, mode='above')
                        csi_blw = critical_success_index_torch(y_c_t, yhat_c_t, low_thr, mode='below')

                        fss_abv_s3 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=3, threshold=high_thr, mode='above')
                        fss_blw_s3 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=3, threshold=low_thr, mode='below')
                        fss_abv_s7 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=7, threshold=high_thr, mode='above')
                        fss_blw_s7 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=7, threshold=low_thr, mode='below')
                        fss_abv_s15 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=15, threshold=high_thr, mode='above')
                        fss_blw_s15 = fraction_skill_score_torch(y_c_t, yhat_c_t, scale=15, threshold=low_thr, mode='below')

                        corr_val = pearson_correlation_torch(y_c_t, yhat_c_t)
                        rmse_val = rmse_torch(y_c_t, yhat_c_t)
                        mae_val = mae_torch(y_c_t, yhat_c_t)
                        me_val = mean_error_torch(y_c_t, yhat_c_t)

                        csi_above_step_s.append(float(csi_abv.item()))
                        csi_below_step_s.append(float(csi_blw.item()))
                        fss_above_s3_s.append(float(fss_abv_s3.item()))
                        fss_below_s3_s.append(float(fss_blw_s3.item()))
                        fss_above_s7_s.append(float(fss_abv_s7.item()))
                        fss_below_s7_s.append(float(fss_blw_s7.item()))
                        fss_above_s15_s.append(float(fss_abv_s15.item()))
                        fss_below_s15_s.append(float(fss_blw_s15.item()))
                        corr_step_s.append(float(corr_val.item()))
                        rmse_step_s.append(float(rmse_val.item()))
                        mae_step_s.append(float(mae_val.item()))
                        me_step_s.append(float(me_val.item()))
    


                    metric_dict_["csi_above"].append(csi_above_step_s)
                    metric_dict_["csi_below"].append(csi_below_step_s)
                    metric_dict_["fss_above_s3"].append(fss_above_s3_s)
                    metric_dict_["fss_below_s3"].append(fss_below_s3_s)
                    metric_dict_["fss_above_s7"].append(fss_above_s7_s)
                    metric_dict_["fss_below_s7"].append(fss_below_s7_s)
                    metric_dict_["fss_above_s15"].append(fss_above_s15_s)
                    metric_dict_["fss_below_s15"].append(fss_below_s15_s)
                    metric_dict_["pearson_corr"].append(corr_step_s)
                    metric_dict_["rmse"].append(rmse_step_s)
                    metric_dict_["mae"].append(mae_step_s)
                    metric_dict_["mean_error"].append(me_step_s)

                init_time = datetime.datetime.utcfromtimestamp(t[in_steps])
                metric_dict[init_time] = metric_dict_

            # --- Final dictionary to save partial sums (NOT averaged yet) ---
            # We also store the sample-wise dictionary as usual
            results = {
                "sample_metrics": metric_dict,
                "pixel_metrics": {
                    # The partial sums and counts
                    "sum_diff": sum_diff_map.cpu().numpy(),
                    "sum_abs_diff": sum_abs_diff_map.cpu().numpy(),
                    "sum_sq_diff": sum_sq_diff_map.cpu().numpy(),
                    "count_valid": count_valid_map.cpu().numpy()
                }
            }

            # Save file (include field size, lat_i, lon_i in name)
            out_file = (
                # f"/capstor/scratch/cscs/acarpent/validation_results/"
                f"/capstor/scratch/cscs/acarpent/{folder_name}_results/"
                f"{model_name}_{epoch}_results_{subset_idx}"
                f"_val_fs{field_size}_lat{lat_i}_lon{lon_i}.pkl"
            )
            with open(out_file, "wb") as o:
                pkl.dump(results, o)
            print(f"Saved partial sums to: {out_file}")


if __name__ == "__main__":
    test()
