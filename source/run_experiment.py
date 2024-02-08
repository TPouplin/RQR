from source.parameters import best_parameter_dict
from data.dataset import GetDataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightning as L
from source.model import q_model
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime

default_parameters = {
    "dropout":0.1,
    "epochs":1000,
    "lr":0.0005,
    "penalty":0.,
    "scheduler1":0.999,
    "scheduler2":0.995,
    "batch_size":64
}

def run_experiment(config):
    
    best_parameters = best_parameter_dict[config["dataset_name"]][config["loss"]]
    
    for key in default_parameters:
        if key not in config:
            if key in best_parameters:
                config[key] = best_parameters[key]
            else:
                config[key] = default_parameters[key]
                
    
    X, y = GetDataset(config["dataset_name"], "data")
    
    L.pytorch.seed_everything(config["random_seed"], workers=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_ratio"], random_state=config["random_seed"])
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config["val_ratio"], random_state=config["random_seed"])
    
    
    # zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train)
    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)
    X_test = scalerX.transform(X_test)
    
    # scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train))
    y_train = y_train / mean_ytrain
    y_val = y_val / mean_ytrain
    y_test = y_test / mean_ytrain
    
    train_dataset = Dataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = Dataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = Dataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print("Data loaded")
    
    quantiles = [(1-config["coverage"])/2, 1-(1-config["coverage"])/2]
    
    
    
    model = q_model(quantiles=quantiles, in_shape=X_train.shape[1], hidden_size=64, dropout=config["dropout"], lr=config["lr"], loss = config["loss"])
    
    date = datetime.now().strftime("%Y%m%d%H%M%S")
        
    logger = TensorBoardLogger(save_dir="results/logs/", name=f'''{config["loss"]}_{config["dataset_name"]}_{config["random_seed"]}_{date}''')
    
    print("config : ", config)
    
    logger.log_hyperparams(config)
    
    trainer = L.Trainer(max_epochs=config["epochs"],accelerator="gpu", deterministic=True, logger=logger)
    trainer.fit(model, train_loader, val_loader, logger=logger)
    
    print("Model_trained")
    
    trainer.test(model, test_loader, logger=logger)
    
    