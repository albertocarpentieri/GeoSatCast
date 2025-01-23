import matplotlib.pyplot as plt 
import numpy as np
import torch 

from geosatcast.models.autoencoder import VAE, Encoder, Decoder
from geosatcast.models.nowcast import AFNOCastLatent, NATCastLatent, AFNONATCastLatent, Nowcaster
from geosatcast.train.distribute_training import load_nowcaster
from geosatcast.data.distributed_dataset import DistributedDataset

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
nowcaster = load_nowcaster("/capstor/scratch/cscs/acarpent/Checkpoints/AFNONATCast/AFNONATCast-2/AFNONATCast-2_37.pt").to("cuda")
print(nowcaster)
print(count_parameters(nowcaster))

in_steps = 2
n_forecast_steps = 8

dataset = DistributedDataset(
    data_path='/capstor/scratch/cscs/acarpent/SEVIRI',
    invariants_path='/capstor/scratch/cscs/acarpent/SEVIRI/invariants',
    name='new_virtual',
    years=[2020],
    input_len=in_steps+n_forecast_steps-1,
    output_len=None,
    channels=np.arange(11),
    field_size=512,
    length=None,
    validation=True,
    rank=0
)


x, t, inv, sza = dataset.get_data(year=2020, t_i=48, lat_i=224, lon_i=224)
print(x.shape, t.shape, inv.shape, sza.shape)
x = x[None].to("cuda").detach()
inv = inv[None].to("cuda").detach()
sza = sza[None].to("cuda").detach()


sza = sza[:, :, :in_steps+n_forecast_steps-1]
x, y = x[:,:,:in_steps], x[:,:,in_steps:in_steps+n_forecast_steps]

inv = torch.cat((inv.expand(*inv.shape[:2], *sza.shape[2:]), sza), dim=1)
with torch.no_grad():
    yhat = nowcaster(x, inv, n_forecast_steps)
    rec_y = nowcaster.vae(y, sample_posterior=False)[0]

print(yhat.shape)

for j in range(n_forecast_steps):
    fig, ax = plt.subplots(5, 11, figsize=(25, 12), sharex=True, sharey=True, constrained_layout=True)
    for i in range(11):
        ax[0, i].imshow(rec_y[0,i,j].cpu(), vmin=rec_y[0,i,j].min(), vmax=rec_y[0,i,j].max(), interpolation="none")
        ax[1, i].imshow(y[0,i,j].cpu(), vmin=y[0,i,j].min(), vmax=y[0,i,j].max(), interpolation="none")
        ax[2, i].imshow(yhat[0,i,j].cpu(), vmin=yhat[0,i,j].min(), vmax=y[0,i,j].max(), interpolation="none")
        ax[3, i].imshow(rec_y[0,i,j].cpu() - y[0,i,j].cpu(), vmin=-1, vmax=1, cmap="bwr", interpolation="none")
        ax[4, i].imshow(yhat[0,i,j].cpu() - y[0,i,j].cpu(), vmin=-1, vmax=1, cmap="bwr", interpolation="none")

    fig.savefig(f"/capstor/scratch/cscs/acarpent/AFNONATCast-2_37_forecast_example_step{j}.png", dpi=200, bbox_inches="tight")