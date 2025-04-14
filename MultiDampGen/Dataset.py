import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionDataset(Dataset):
    def __init__(self,file):
        self.file_object=h5py.File(file,"r")
        self.feature=self.file_object.get('feature')
        self.label=self.file_object.get('label')
        self.size=self.file_object.get('size')
        pass
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, index):
        if (index >= len(self.feature)):
            raise IndexError()
        feature = np.array(self.feature[index],dtype=np.float32)
        label = np.array(self.label[index],dtype=np.float32)
        size=np.array(self.size[index],dtype=np.float32)
        feature=torch.Tensor(feature).type(torch.float32).unsqueeze(0)
        mask=torch.Tensor(label).type(torch.float32)
        scale=torch.Tensor(size).type(torch.float32).unsqueeze(0)
        return (feature,mask,scale)
    def plot_image(self,index):
        fig,(ax)=plt.subplots(nrows=1,ncols=1)
        ax.imshow(np.array(self.feature[index]),cmap='bone')
        plt.show()
        pass
    pass

if __name__=='__main__':
    diffusiondataset = DiffusionDataset("Dataset_Train.h5")
    index=1
    print(diffusiondataset[index][0].shape,diffusiondataset[index][1].shape,diffusiondataset[index][2].shape)
    print(diffusiondataset[index][2])
    print(diffusiondataset[index][1])
    diffusiondataset.plot_image(index)