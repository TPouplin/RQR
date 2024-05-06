
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
from finetuning import fine_tuned_parameters, other_parameters
import os 
import pandas as pd 
from data.dataset import GetDataset


loss = ["RQR-W","SQR","SQRN","IR"]

test_name = "boston_0.9_RQR-W_0.1_0"

study = optuna.load_study(storage= "sqlite:///results/finetuning/recording.db", study_name =test_name)

print("ok")
        
