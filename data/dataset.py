
import numpy as np
import pandas as pd
import torch

def GetDataset(name, base_path):

    data = np.loadtxt(base_path + "/UCI_Datasets/{}.txt".format(name))
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)


    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y
            
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]