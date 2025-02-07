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
    
    rmse_map = np.zeros((11,n_forecast_steps,512,512))
    nan_map = np.zeros((11,n_forecast_steps,512,512))

    tvl1_metric_dict = {}
    
    stds = dataset.stds.numpy().reshape(-1,1,1,1)
    means = dataset.means.numpy().reshape(-1,1,1,1)
    
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

        # Mask channels 0, 7, 8 where sza > 90 degrees
        mask = sza > np.cos(np.deg2rad(90))
        y[[0, 7, 8], :, :, :][mask] = np.nan
        yhat[[0, 7, 8], :, :, :][mask] = np.nan

        square_res = (yhat-y)**2
        abs_res = np.abs(yhat-y)
        res = yhat - y
        
        nans = np.isnan(yhat)
        rmse_map += square_res
        nan_map += nans.astype(int)
        rmse = np.sqrt(np.nanmean(square_res, axis=(1,2,3)))
        mae = np.nanmean(abs_res, axis=(1,2,3))
        mean_error = np.nanmean(res, axis=(1,2,3))
        
        csi_low = critical_success_index(y, yhat, threshold=0.2)
        csi_high = critical_success_index(y, yhat, threshold=0.8)
        fss_low = fraction_skill_score(y, yhat, scale=3)
        fss_high = fraction_skill_score(y, yhat, scale=10)
        
        pearson_corr = []
        for c in range(11):
            if np.isnan(y[c]).all() or np.isnan(yhat[c]).all():
                pearson_corr.append(np.nan)
            else:
                pearson_corr.append(pearsonr(y[c].flatten(), yhat[c].flatten())[0])
        
        metric_dict_ = {
            "time": t[in_steps:],
            "loc": [224, 224],
            "rmse": rmse,
            "mae": mae,
            "mean_error": mean_error,
            "csi_low": csi_low,
            "csi_high": csi_high,
            "fss_low": fss_low,
            "fss_high": fss_high,
            "pearson_corr": pearson_corr
        }
        
        tvl1_metric_dict[datetime.datetime.utcfromtimestamp(t[in_steps])] = metric_dict_

    with open("/capstor/scratch/cscs/acarpent/tvl1_results/results_validation.pkl", "wb") as o:
        pkl.dump(tvl1_metric_dict, o)
    print("saved")

if __name__ == "__main__":
    test()
