from geosatcast.train.distribute_training import load_unatcast

model_name = "UNATCast-small-ks9-nb8-spherical_rope-192_1152-ud2_4-ud2-L1-v2"
folder = "UNATCast-small"
epoch = 0
for epoch in range(30):
    print(epoch)
    unatcast = load_unatcast("/capstor/scratch/cscs/acarpent/Checkpoints/"+folder+"/"+model_name+f"/{model_name}_{epoch}.pt")
    for name, param in unatcast.named_parameters():
        # if name == "down_blocks.0.nat_blocks.0.attn.rope.freq":
        #     print(name, param.detach().numpy().reshape(-1,2))
        if "gamma" in name:
            print(epoch, name, param.detach().square().mean().sqrt())
        if "alpha" in name:
            print(epoch, name, param.detach().square().mean().sqrt())
        if "skip" in name:
            print(epoch, name, param.detach().square().mean().sqrt())
        if "freq" in name:
            print(epoch, name, param.detach().square().mean().sqrt())