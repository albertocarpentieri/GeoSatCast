# tvl1_test.py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
# Import your updated metrics
from metrics import critical_success_index, fraction_skill_score, pearson_correlation

from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast
from scipy.stats import pearsonr

def test():
    in_steps = 2
    n_forecast_steps = 8

    dataset = DistributedDataset(
        data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
        invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
        name='new_virtual',
        years=[2020],
        input_len=in_steps + n_forecast_steps,
        output_len=None,
        channels=np.arange(11),
        field_size=512,
        length=None,
        validation=True,
        rank=0
    )

    np.random.seed(0)
    n_samples = 500
    indices = np.random.choice(np.arange(len(dataset.indices)), n_samples, replace=False)

    tvl1_metric_dict = {}
    
    stds = dataset.stds.numpy().reshape(-1,1,1,1)
    means = dataset.means.numpy().reshape(-1,1,1,1)
    
    thresholds = np.array([0.2, 0.3, 0.25, 0.4, 0.35, 0.3, 0.2, 0.15, 0.25, 0.5, 0.45])
    
    aggregated_rmse = np.zeros((11, n_forecast_steps, 512, 512))
    aggregated_mae = np.zeros((11, n_forecast_steps, 512, 512))
    aggregated_me = np.zeros((11, n_forecast_steps, 512, 512))
    
    for i in indices:
        year, t_i = dataset.indices[i]
        x, t, _, sza = dataset.get_data(year=year, t_i=t_i, lat_i=224, lon_i=224)
        t = t[0,:,0,0].numpy().astype(int)
        
        # x shape: [channels, total_steps, H, W]
        # We want the first in_steps as input, next n_forecast_steps as y
        x_in, y = x[:, :in_steps], x[:, in_steps:in_steps + n_forecast_steps]

        # TV-L1 config
        model_conf = {
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

        # Forecast
        yhat = tvl1_forecast(x_in.numpy(), model_conf, n_forecast_steps)
        
        # Convert back to original scale
        y    = y.numpy()    * stds + means
        yhat = yhat         * stds + means

        metric_dict_ = {
            "time": t[in_steps:],
            "loc": [224, 224],
            "rmse": [],
            "mae": [],
            "mean_error": [],
            "csi_above": [],
            "csi_below": [],
            "fss_above": [],
            "fss_below": [],
            "pearson_corr": []
        }

        for j in range(n_forecast_steps):
            rmse_map = np.sqrt((yhat[:, j] - y[:, j]) ** 2)
            mae_map = np.abs(yhat[:, j] - y[:, j])
            me_map = yhat[:, j] - y[:, j]
            
            aggregated_rmse[:, j] += rmse_map
            aggregated_mae[:, j] += mae_map
            aggregated_me[:, j]  += me_map
            
            # Example: channel-wise average metrics
            rmse = np.nanmean(rmse_map, axis=(1,2))
            mae  = np.nanmean(mae_map, axis=(1,2))
            mean_error = np.nanmean(me_map, axis=(1,2))

            # For each channel, compute "above" and "below" CSI/FSS
            csi_above = []
            csi_below = []
            fss_above = []
            fss_below = []
            corr_list = []

            for c in range(11):
                csi_above.append(critical_success_index(y[c, j], yhat[c, j], thresholds[c], mode='above'))
                csi_below.append(critical_success_index(y[c, j], yhat[c, j], thresholds[c], mode='below'))
                fss_above.append(fraction_skill_score(y[c, j], yhat[c, j], scale=3, 
                                                      threshold=thresholds[c], mode='above'))
                fss_below.append(fraction_skill_score(y[c, j], yhat[c, j], scale=3, 
                                                      threshold=thresholds[c], mode='below'))
                corr_list.append(pearson_correlation(y[c, j], yhat[c, j]))

            metric_dict_["rmse"].append(rmse)
            metric_dict_["mae"].append(mae)
            metric_dict_["mean_error"].append(mean_error)
            metric_dict_["csi_above"].append(csi_above)
            metric_dict_["csi_below"].append(csi_below)
            metric_dict_["fss_above"].append(fss_above)
            metric_dict_["fss_below"].append(fss_below)
            metric_dict_["pearson_corr"].append(corr_list)
        
        tvl1_metric_dict[datetime.datetime.utcfromtimestamp(t[in_steps])] = metric_dict_

    # Aggregated stats
    aggregated_rmse /= n_samples
    aggregated_mae  /= n_samples
    aggregated_me   /= n_samples
    
    # Save aggregated maps
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_rmse.npy", aggregated_rmse)
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_mae.npy", aggregated_mae)
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_me.npy", aggregated_me)
    
    # Save metrics dictionary
    with open("/capstor/scratch/cscs/acarpent/tvl1_results/results_validation.pkl", "wb") as o:
        pkl.dump(tvl1_metric_dict, o)
    print("Aggregated maps and metrics saved")

if __name__ == "__main__":
    test()
