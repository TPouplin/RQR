import numpy as np
import torch
import pandas as pd 


def GetDataset(name, base_path):
    if name == "cpu_act":
        data = pd.read_csv(base_path + "/cpu_act.arff", header=None)
        data.drop(columns=[21], inplace=True)
        data = np.array(data.values.tolist())
        
    elif name == "sulfur":
        data = pd.read_csv(base_path + "/sulfur.arff", header=None)
        data.drop(columns=[6], inplace=True)
        data = np.array(data.values.tolist())
            
    elif name == "miami":
        data = pd.read_csv(base_path + "/miami.arff", header=None)
        data.drop(columns=[2], inplace=True)
        data = data[[0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,3]]
        data = np.array(data.values.tolist())
        
    else:
        data = np.loadtxt(base_path + f"{name}.txt")
    
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