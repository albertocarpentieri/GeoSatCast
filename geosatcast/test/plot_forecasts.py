import matplotlib.pyplot as plt 
import numpy as np
import torch 

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster, load_predrnn
from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast
import datetime
import time

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
def rmse(x,y): return np.sqrt(np.nanmean((x-y)**2))

model_names = ["AFNO", "NAT", "AFNONAT", "PREDRNN", "TVL1"]
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
device = "cuda"
afnocast = load_nowcaster("/capstor/scratch/cscs/acarpent/Checkpoints/AFNOCast/AFNOCast-512-s2-tss-ls_0-fd_4-v1/AFNOCast-512-s2-tss-ls_0-fd_4-v1_30.pt").to(device)
print(count_parameters(afnocast))

natcast = load_nowcaster("/capstor/scratch/cscs/acarpent/Checkpoints/NATCast/NATCast-512-s2-tss-ls_0-ks_3-fd_4-v1/NATCast-512-s2-tss-ls_0-ks_3-fd_4-v1_35.pt").to(device)
print(count_parameters(natcast))

afnonatcast = load_nowcaster("/capstor/scratch/cscs/acarpent/Checkpoints/AFNONATCast/AFNONATCast-1024-s2-tss-ls_0-fd_12-ks_5-seq-v3/AFNONATCast-1024-s2-tss-ls_0-fd_12-ks_5-seq-v3_37.pt").to(device)
print(count_parameters(afnonatcast))

predrnn = load_predrnn("/capstor/scratch/cscs/acarpent/Checkpoints/PredRNN/predrnn-s2-fd_4-nh_64-v1/predrnn-s2-fd_4-nh_64-v1_99.pt").to(device)
print(count_parameters(predrnn))

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
    rank=0,
    mask_sza=True,
)
stds = dataset.stds.numpy().reshape(-1,1,1,1)
means = dataset.means.numpy().reshape(-1,1,1,1)
for t_i in [48, 124, 656, 845, 850, 12450, 24000]:
    x, t, inv, sza = dataset.get_data(year=2020, t_i=t_i, lat_i=224, lon_i=224)
    x, y = x[:,:in_steps], x[:,in_steps:in_steps+n_forecast_steps]

    t = [datetime.datetime.utcfromtimestamp(t[0,in_steps+i,0,0].numpy().astype(int)) for i in range(n_forecast_steps)]
    # tvl1 forecast
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
    start_time = time.time()
    yhat_of = tvl1_forecast(x.numpy(), model_conf, n_forecast_steps)
    print("OF time:", time.time() - start_time, "OF stats:", yhat_of.mean(), yhat_of.std())
    yhat_of = yhat_of * stds + means
    print(yhat_of.shape)

    # AFNNATCAST forecast
    x = x[None].to(device).detach()
    inv = inv[None].to(device).detach()
    sza = sza[None].to(device).detach()


    sza = sza[:, :, :in_steps+n_forecast_steps-1]
    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
    with torch.no_grad():
        start_time = time.time()
        yhat_afno = afnocast(x, inv, n_forecast_steps).detach().cpu().numpy()[0] * stds + means
        print("AFNO time:", time.time() - start_time, "AFNO stats:", yhat_afno.mean(), yhat_afno.std())
        
        start_time = time.time()
        yhat_nat = natcast(x, inv, n_forecast_steps).detach().cpu().numpy()[0] * stds + means
        print("NAT time:", time.time() - start_time, "NAT stats:", yhat_nat.mean(), yhat_nat.std())

        start_time = time.time()
        yhat_afnonat = afnonatcast(x, inv, n_forecast_steps).detach().cpu().numpy()[0] * stds + means
        print("AFNONAT time:", time.time() - start_time, "AFNONAT stats:", yhat_afnonat.mean(), yhat_afnonat.std())
        
        start_time = time.time()
        yhat_rnn = predrnn(x, n_forecast_steps).detach().cpu().numpy()[0] * stds + means
        print("RNN time:", time.time() - start_time, "RNN stats:", yhat_rnn.mean(), yhat_rnn.std())

    y = y.numpy() * stds + means
    print("y stats:", y.mean(), y.std())
    print(y.shape)
    for j in range(n_forecast_steps):
        fig, ax = plt.subplots(11, 11, figsize=(25, 25), sharex=True, sharey=True, constrained_layout=True)
        for c in range(1,6):
            ax[c, 0].set_ylabel(model_names[c-1])
        
        for c in range(6,11):
            ax[c, 0].set_ylabel(model_names[c-6]+" - obs")
        
        for i in range(11):
            ax[0, i].set_title(channel_names[i])
            
            ax[0, i].imshow(y[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[1, i].imshow(yhat_afno[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[2, i].imshow(yhat_nat[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[3, i].imshow(yhat_afnonat[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[4, i].imshow(yhat_rnn[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[5, i].imshow(yhat_of[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            
            
            ax[6, i].imshow(yhat_afno[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[6, i].set_title(f"RMSE: {rmse(yhat_afno[i,j], y[i,j]):.6f}")
            ax[7, i].imshow(yhat_nat[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[7, i].set_title(f"RMSE: {rmse(yhat_nat[i,j], y[i,j]):.6f}")
            ax[8, i].imshow(yhat_afnonat[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[8, i].set_title(f"RMSE: {rmse(yhat_afnonat[i,j], y[i,j]):.6f}")
            ax[9, i].imshow(yhat_rnn[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[9, i].set_title(f"RMSE: {rmse(yhat_rnn[i,j], y[i,j]):.6f}")
            ax[10, i].imshow(yhat_of[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[10, i].set_title(f"RMSE: {rmse(yhat_of[i,j], y[i,j]):.6f}")

        fig.savefig(f"/capstor/scratch/cscs/acarpent/images/forecast_{t[j]}_{j}.png", dpi=100, bbox_inches="tight")