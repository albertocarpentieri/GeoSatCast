import matplotlib.pyplot as plt 
import numpy as np
import torch 

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster
from geosatcast.data.distributed_dataset import DistributedDataset

import datetime
import pickle as pkl

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

np.random.seed(0)

def test():
    model_name = "AFNOCast-512-s2-tss-ls_0-fd_4-v1" #"AFNOCast-512-s2-tss-ls_0-fd_8-v1" #"NATCast-512-s2-tss-ls_0-ks_3-fd_4-v1"
    epoch = 32 #40 #35

    nowcaster = load_nowcaster(f"/capstor/scratch/cscs/acarpent/Checkpoints/{model_name.split('-')[0]}/{model_name}/{model_name}_{epoch}.pt").to("cuda")
    print(nowcaster)
    print(count_parameters(nowcaster))

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
    n_samples = 100
    indices = np.random.choice(np.arange(len(dataset.indices)), n_samples, replace=False) 

    rmse_map = np.zeros((11,n_forecast_steps,512,512))
    nan_map = np.zeros((11,n_forecast_steps,512,512))

    metric_dict = {}
    
    stds = dataset.stds.numpy().reshape(-1,1,1,1)
    means = dataset.means.numpy().reshape(-1,1,1,1)

    for i in indices:
        year, t_i = dataset.indices[i]
        x, t, inv, sza = dataset.get_data(year=year, t_i=t_i, lat_i=224, lon_i=224)
        x = x[None].to("cuda").detach()
        inv = inv[None].to("cuda").detach()
        sza = sza[None].to("cuda").detach()
        t = t[0,:,0,0].numpy().astype(int)

        sza = sza[:, :, :in_steps+n_forecast_steps-1]
        x, y = x[:,:,:in_steps], x[:,:,in_steps:in_steps+n_forecast_steps]

        inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
        with torch.no_grad():
            yhat = nowcaster(x, inv, n_forecast_steps)
        
        y = y[0].cpu().numpy() * stds + means
        yhat = yhat[0].cpu().numpy() * stds + means

        square_res = (yhat-y)**2
        abs_res = np.abs(yhat-y)
        res = yhat - y
        
        nans = np.isnan(yhat)

        rmse_map += square_res
        nan_map += nans.astype(int)
        rmse = np.sqrt(np.nanmean(square_res, axis=(1,2,3)))

        metric_dict_ = {"res":res, "nan":nans, "time":t[in_steps:], "loc":[224, 224]}
        metric_dict[datetime.datetime.utcfromtimestamp(t[in_steps])] = metric_dict_

    with open(f"/capstor/scratch/cscs/acarpent/nowcast_results/{model_name}_{epoch}_results_validation.pkl", "wb") as o:
        pkl.dump(metric_dict, o)
    print("saved")



    
    fig, ax = plt.subplots(n_forecast_steps, 11, figsize=(25, n_forecast_steps*2), sharex=True, sharey=True, constrained_layout=True)
    for j in range(n_forecast_steps):
        for i in range(11):
            rmse_map_ = np.sqrt(rmse_map[i,j] / (n_samples-nan_map[i,j]))
            rmse_map_[~np.isfinite(rmse_map_)] = np.nan
            img = ax[j,i].imshow(rmse_map_, interpolation="none", vmin=0, vmax=stds[i]/2)
            plt.colorbar(img, ax=ax[0,i], shrink=.4)
        
    fig.savefig(f"/capstor/scratch/cscs/acarpent/nowcast_results/{model_name}_{epoch}_{j}.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    test()