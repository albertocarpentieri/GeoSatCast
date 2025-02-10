import matplotlib.pyplot as plt 
import numpy as np
import torch 

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster, load_predrnn
from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast
import datetime

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
afnonatcast = load_nowcaster("/capstor/scratch/cscs/acarpent/Checkpoints/AFNONATCast/AFNONATCast-512-s2-tss-ls_0-fd_8-ks_7-seq-v1/AFNONATCast-512-s2-tss-ls_0-fd_8-ks_7-seq-v1_59.pt").to("cuda")
print(count_parameters(afnonatcast))
predrnn = load_predrnn("/capstor/scratch/cscs/acarpent/Checkpoints/PredRNN/predrnn-v3/predrnn-v3_0.pt").to("cuda")


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
stds = dataset.stds.numpy().reshape(-1,1,1,1)
means = dataset.means.numpy().reshape(-1,1,1,1)
for t_i in [48, 12450, 850, 24000, 656]:
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
    yhat = tvl1_forecast(x.numpy(), model_conf, n_forecast_steps)
    yhat_of = yhat * stds + means
    print(yhat_of.shape)

    # AFNNATCAST forecast
    x = x[None].to("cuda").detach()
    inv = inv[None].to("cuda").detach()
    sza = sza[None].to("cuda").detach()


    sza = sza[:, :, :in_steps+n_forecast_steps-1]
    inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
    with torch.no_grad():
        yhat = afnonatcast(x, inv, n_forecast_steps)
        yhat_rnn = predrnn(x, n_forecast_steps)
    
    yhat = yhat.detach().cpu().numpy()[0] * stds + means
    yhat_rnn = yhat_rnn.detach().cpu().numpy()[0] * stds + means
    print(yhat.shape)

    y = y.numpy() * stds + means
    print(y.shape)
    for j in range(n_forecast_steps):
        fig, ax = plt.subplots(7, 11, figsize=(25, 16), sharex=True, sharey=True, constrained_layout=True)
        for i in range(11):
            ax[0, i].imshow(y[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[1, i].imshow(yhat[i,j], vmin=yhat[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[2, i].imshow(yhat_of[i,j], vmin=yhat[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            ax[3, i].imshow(yhat_rnn[i,j], vmin=yhat[i,j].min(), vmax=y[i,j].max(), interpolation="none")
            
            ax[4, i].imshow(yhat[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[5, i].imshow(yhat_of[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")
            ax[6, i].imshow(yhat_rnn[i,j] - y[i,j], vmin=-stds[i], vmax=stds[i], cmap="bwr", interpolation="none")

        fig.savefig(f"/capstor/scratch/cscs/acarpent/images/forecast_{t[j]}_{j}.png", dpi=100, bbox_inches="tight")