import time 
import h5py 
import numpy as np 

start_time = time.time()
with h5py.File("/capstor/scratch/cscs/acarpent/SEVIRI/2018_weekly_datasets/2018_0-3.h5", "r") as f:
    x = np.empty((20,11, 512, 512)).astype(np.float32)
    f["fields"].read_direct(x, np.s_[30:50,:,100:100+512,100:100+512], np.s_[:])
print(time.time()-start_time)
print(x[0,0])

start_time = time.time()
with h5py.File("/capstor/scratch/cscs/acarpent/SEVIRI_16B/2018_weekly_datasets/2018_0-1.h5", "r") as f:
    x = np.empty((20,512, 512,12)).astype(np.float16)
    f["fields"].read_direct(x, np.s_[30:50,100:100+512,100:100+512,:], np.s_[:])
print(time.time()-start_time)
print(x[0,:,:,0])

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, constrained_layout=True, figsize=(16,4))
for i in range(5):
    ax[0,i].imshow(x[i*2,:,:,7],vmin=x[:,:,:,7].min(), vmax=x[:,:,:,7].max())
    ax[1,i].imshow(x[i*2,:,:,-1],vmin=x[:,:,:,-1].min(),vmax=x[:,:,:,-1].max())
fig.savefig("/capstor/scratch/cscs/acarpent/new_dataset_try.png")
