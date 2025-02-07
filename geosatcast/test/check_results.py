import pickle as pkl 
import numpy as np 
import matplotlib.pyplot as plt

#TVL1
with open("/capstor/scratch/cscs/acarpent/tvl1_results/tvl1_results_validation.pkl", "rb") as o:
    tvl1 = pkl.load(o)

keys = list(tvl1.keys())


tvl1_rmse = np.sqrt(np.nanmean([tvl1[k]["res"]**2 for k in keys], axis=(0,3,4)))
del tvl1
print(tvl1_rmse.shape)

with open("/capstor/scratch/cscs/acarpent/nowcast_results/NATCast-512-s2-tss-ls_0-ks_3-fd_4-v1_35_results_validation.pkl", "rb") as o:
    natcast_4 = pkl.load(o)
nat4_rmse = np.sqrt(np.nanmean([natcast_4[k]["res"]**2 for k in keys], axis=(0,3,4)))
del natcast_4
print(nat4_rmse.shape)

with open("/capstor/scratch/cscs/acarpent/nowcast_results/AFNOCast-512-s2-tss-ls_0-fd_4-v1_32_results_validation.pkl", "rb") as o:
    afnocast_4 = pkl.load(o)
afno4_rmse = np.sqrt(np.nanmean([afnocast_4[k]["res"]**2 for k in keys], axis=(0,3,4)))
del afnocast_4
print(afno4_rmse.shape)

with open("/capstor/scratch/cscs/acarpent/nowcast_results/AFNOCast-512-s2-tss-ls_0-fd_8-v1_40_results_validation.pkl", "rb") as o:
    afnocast_8 = pkl.load(o)
afno8_rmse = np.sqrt(np.nanmean([afnocast_8[k]["res"]**2 for k in keys], axis=(0,3,4)))
del afnocast_8
print(afno8_rmse.shape)


fig, ax = plt.subplots(1,11,sharex=True,figsize=(32,4))
for i in range(11):
    ax[i].plot(nat4_rmse[i], label="NAT4")
    ax[i].plot(afno4_rmse[i], label="AFNO4")
    ax[i].plot(tvl1_rmse[i], label="TVL1")
    ax[i].plot(afno8_rmse[i], label="AFNO8")
plt.legend()
fig.savefig("/capstor/scratch/cscs/acarpent/nowcast_results/RMSE.png", dpi=200, bbox_inches="tight")
