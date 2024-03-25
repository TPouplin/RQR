from source.run_experiment import run_experiment
from data.dataset import GetDataset
import numpy as np
from tqdm import tqdm

loss =  ["QR","WS","RQR-W","SQR","RQR-O","OQR","IR"]
dataset_names =  ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]

config = {}
config["verbose"] = True
config["finetuning"] = False


other_parameters = {
    "epochs": 800,
    "coverage": 0.9,
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "batch_size": 2000,    
}

    
for key in other_parameters:
    config[key] = other_parameters[key]
    

# "lr": [0.001,0.0001]
# "dropout": [0.0, 0.1, 0.2, 0.3]
# "penalty": [0.0001,0.001,0.01,0.1,1,10]  


config["dataset_name"] = "cpu_act"
config["loss"] = "RQR-W"

config["penalty"] = 1
config["lr"] =0.01
config["dropout"] =0.3


data = GetDataset(config["dataset_name"], "data")

coverage = []
width = []

for seed in tqdm(range(10), leave=False):
    config["random_seed"] = seed
    val_loss,results = run_experiment(config, data)
    print("seed : ", results)
    coverage.append(results["test_coverage"])
    width.append(results["test_width"])

print("coverage mean : ", np.mean(coverage))
print("coverage std : ", np.std(coverage)/np.sqrt(10))
print("width mean : ", np.mean(width))
print("width std : ", np.std(width))
