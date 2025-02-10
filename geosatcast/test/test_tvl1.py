from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast
import numpy as np 
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter

np.random.seed(0)

def critical_success_index(y_true, y_pred, threshold):
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    hits = np.sum((y_pred >= threshold) & (y_true >= threshold))
    false_alarms = np.sum((y_pred >= threshold) & (y_true < threshold))
    misses = np.sum((y_pred < threshold) & (y_true >= threshold))

    return hits / (hits + false_alarms + misses + 1e-8)

def fraction_skill_score(y_true, y_pred, scale):
    def compute_fractions(field, scale):
        return uniform_filter(field, size=scale, mode='constant', cval=0)
    
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true * valid_mask
    y_pred = y_pred * valid_mask

    ft = compute_fractions(y_true, scale)
    fp = compute_fractions(y_pred, scale)

    numerator = np.sum((fp - ft) ** 2)
    denominator = np.sum(ft ** 2 + fp ** 2 + 1e-8)

    return 1 - (numerator / denominator)

def test():
    in_steps = 2
    n_forecast_steps = 8

    dataset = DistributedDataset(
        data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
        invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
        name='new_virtual',
        years=[2020],
        input_len=in_steps+n_forecast_steps,
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
        sza = sza.numpy()
        
        x, y = x[:,:in_steps], x[:,in_steps:in_steps+n_forecast_steps]

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
        
        yhat = tvl1_forecast(x.numpy(), model_conf, n_forecast_steps)
        
        y = y.numpy() * stds + means
        yhat = yhat * stds + means

        metric_dict_ = {
            "time": t[in_steps:],
            "loc": [224, 224],
            "rmse": [],
            "mae": [],
            "mean_error": [],
            "csi_low": [],
            "csi_high": [],
            "fss_low": [],
            "fss_high": [],
            "pearson_corr": []
        }
        
        for j in range(n_forecast_steps):
            rmse_map = np.sqrt((yhat[:, j] - y[:, j])**2)
            mae_map = np.abs(yhat[:, j] - y[:, j])
            me_map = yhat[:, j] - y[:, j]
            
            aggregated_rmse[:, j] += rmse_map
            aggregated_mae[:, j] += mae_map
            aggregated_me[:, j] += me_map
            
            rmse = np.nanmean(rmse_map, axis=(1, 2))
            mae = np.nanmean(mae_map, axis=(1, 2))
            mean_error = np.nanmean(me_map, axis=(1, 2))
            csi_low = [critical_success_index(y[c, j], yhat[c, j], thresholds[c]) for c in range(11)]
            csi_high = [critical_success_index(y[c, j], yhat[c, j], thresholds[c] * 1.5) for c in range(11)]
            fss_low = [fraction_skill_score(y[c, j], yhat[c, j], scale=3) for c in range(11)]
            fss_high = [fraction_skill_score(y[c, j], yhat[c, j], scale=10) for c in range(11)]
            pearson_corr = [pearsonr(y[c, j].flatten(), yhat[c, j].flatten())[0] if np.isfinite(y[c, j]).all() and np.isfinite(yhat[c, j]).all() else np.nan for c in range(11)]
            
            metric_dict_["rmse"].append(rmse)
            metric_dict_["mae"].append(mae)
            metric_dict_["mean_error"].append(mean_error)
            metric_dict_["csi_low"].append(csi_low)
            metric_dict_["csi_high"].append(csi_high)
            metric_dict_["fss_low"].append(fss_low)
            metric_dict_["fss_high"].append(fss_high)
            metric_dict_["pearson_corr"].append(pearson_corr)
        
        tvl1_metric_dict[datetime.datetime.utcfromtimestamp(t[in_steps])] = metric_dict_

    aggregated_rmse /= n_samples
    aggregated_mae /= n_samples
    aggregated_me /= n_samples
    
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_rmse.npy", aggregated_rmse)
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_mae.npy", aggregated_mae)
    np.save("/capstor/scratch/cscs/acarpent/tvl1_results/aggregated_me.npy", aggregated_me)
    
    with open("/capstor/scratch/cscs/acarpent/tvl1_results/results_validation.pkl", "wb") as o:
        pkl.dump(tvl1_metric_dict, o)
    print("Aggregated maps and metrics saved")

if __name__ == "__main__":
    test()