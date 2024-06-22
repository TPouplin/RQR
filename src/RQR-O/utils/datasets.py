"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
import sys

sys.path.insert(1, '..')


def GetDataset(name, base_path):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""
         
    UCI_datasets = ['kin8nm', 'naval', 'boston', 'energy', 'power', 'protein', 'wine', 'yacht', 'concrete' ]

    if name in UCI_datasets:
        data = np.loadtxt(base_path+name+'.txt')
        X = data[:, :-1]
        y = data[:, -1]
        
    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
    except Exception:
        raise Exception("invalid dataset")
    
    return X, y
    


def get_scaled_data(args):
    dataset_base_path = "./RQR/data/"
    dataset_name = args.data
    x,y = GetDataset(dataset_name, dataset_base_path) 
    data_size = len(x)  # data_size_per_dataset[dataset_name]
    idx = np.random.permutation(len(x))[:data_size]
    x = x[idx]
    y = y[idx]
    y = y.reshape(-1, 1)

    return scale_data_wrapper(x,y,args)


def scale_data_wrapper(x,y, args):
    test_ratio = args.test_ratio
    return scale_data(x, y, args.seed, test_ratio)


def scale_data(x,y, seed, test_size=0.1):
    x_train, x_te, y_train, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed)
    
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed)

    s_tr_x = StandardScaler().fit(x_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    mean_y_tr = np.mean(np.abs(y_tr))
    y_tr = torch.Tensor(y_tr/mean_y_tr)
    y_va = torch.Tensor(y_va/mean_y_tr)
    y_te = torch.Tensor(y_te/mean_y_tr)
    y_al = torch.Tensor(y/mean_y_tr)

    x_train = torch.cat([x_tr, x_va], dim=0)
    y_train = torch.cat([y_tr, y_va], dim=0)
    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al,
                              x_train=x_train, y_train=y_train)

    return out_namespace



