import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler,DDIMScheduler,PNDMScheduler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset
import h5py
from Dataset import DiffusionDataset
import random
import os
import sys

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

filepath='../Dataset/Test'
diffusiondataset = DiffusionDataset(filepath+"/Dataset_Test.h5")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise_scheduler=DDIMScheduler(beta_schedule='squaredcos_cap_v2')

total_steps=100
noise_scheduler.set_timesteps(total_steps)

LDPM = torch.load(r'LDPM.pt', weights_only=False)
LDPM=LDPM.to(device)
LDPM.eval()


TopoFormer=torch.load(r'TopoFormer.pt', weights_only=False)
TopoFormer=TopoFormer.to(device)
TopoFormer.eval()

RSV= torch.load(r'RSV.pt', weights_only=False)
RSV=RSV.to(device)
RSV.eval()

index=4012
x,y,z=diffusiondataset[index]

np.savetxt(f'./curve.txt',y)
np.savetxt(f'./scale.txt',z)

Y=y[1:,1].to(device).unsqueeze(0)/1000
Z=z.unsqueeze(0)
Z_expand = z.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(-1, -1, 32, 32).to(device)

diff_out=np.zeros((total_steps,3,32,32))
vae_out=np.zeros((total_steps,1,128,128))

while True:
    sample = torch.randn(1, 3, 32, 32).to(device)
    for i, t in enumerate(noise_scheduler.timesteps):
        t_=torch.Tensor([t]).to(device)
        model_input = noise_scheduler.scale_model_input(sample, t)
        with torch.no_grad():
            noise_pred = LDPM(sample, t_, Y.to(device),Z.to(device))
        scheduler_output = noise_scheduler.step(noise_pred, t, sample)
        sample = scheduler_output.prev_sample
        pred_x0 = scheduler_output.pred_original_sample
        pred_x0_eachstep= pred_x0.detach().cpu().numpy()
        diff_out[i,:,:,:]=pred_x0_eachstep
        pics_rec=TopoFormer.decoder(pred_x0)
        x1=pics_rec[0, :, :, :].detach().cpu().numpy().clip(0, 1)
        x1[x1<=0.5]=0
        x1[x1>0.5]=1
        vae_out[i,:,:,:]=x1

    inputs_discriminator = torch.cat((pred_x0, Z_expand), dim=1)
    loop = RSV(inputs_discriminator)
    prediction = loop[0].detach().cpu().numpy()[1:]
    observation = Y[0].cpu().numpy()
    error = np.mean(np.abs((prediction - observation) / observation)) * 100
    print(f'MAPE:{round(error,3)}%')
    if error < 10:
        np.savetxt(f'./imgs.txt', x1[0], fmt='%d')
        np.save(file=f"./diff_out.npy", arr=diff_out)
        np.save(file=f"./vae_out.npy", arr=vae_out)
        break