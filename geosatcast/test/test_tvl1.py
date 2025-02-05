from geosatcast.data.distributed_dataset import DistributedDataset
from geosatcast.models.tvl1 import tvl1_forecast
import numpy as np 
import matplotlib.pyplot as plt

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

x, _, _, _ = dataset.get_data(year=2020, t_i=48, lat_i=224, lon_i=224)
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

# Random test data
yhat = tvl1_forecast(x.numpy(), model_conf, n_forecast_steps)
print(yhat.shape)
for j in range(n_forecast_steps):
    fig, ax = plt.subplots(2, 11, figsize=(25, 4), sharex=True, sharey=True, constrained_layout=True)
    for i in range(11):
        ax[0, i].imshow(y[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
        ax[1, i].imshow(yhat[i,j], vmin=y[i,j].min(), vmax=y[i,j].max(), interpolation="none")
        print(j, i, np.nanmean(np.abs(yhat[i,j]-y[i,j].numpy())))
    print()
    fig.savefig(f"/capstor/scratch/cscs/acarpent/tvl1_forecast_example_step{j}.png", dpi=200, bbox_inches="tight")