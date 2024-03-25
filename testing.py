
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
from finetuning import fine_tuned_parameters, other_parameters
import os 
import pandas as pd 
from data.dataset import GetDataset

loss = ["RQR-W"] #["QR","WS","RQR-W","SQR","RQR-O","OQR","IR"]


def testing():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')
    args = parser.parse_args()
    

    config = {}
    config["dataset_name"] = args.dataset_name
    config["verbose"] = True
    config["finetuning"] = False
    
    
    for key in other_parameters:
        config[key] = other_parameters[key]
    
    data = GetDataset(config["dataset_name"], "data")

    for l in loss:
        config["loss"] = l
        if l == "SQRN":
            l = "SQR"

        if not ( l in ['RQR-W', 'RQR-O', 'OQR', 'IR']):
            penalty = [0]
        else:
            penalty = fine_tuned_parameters["penalty"]
    
    
        for seed in range(7, args.n_seed):
            config["random_seed"] = seed

            best_length = 99999999
            coverage = 0
            tolerance = 2.5/100
            best_config  = {}
           
            for p in penalty:
                study = optuna.load_study(storage= "sqlite:///results/finetuning/recording_ultra_light_rebV2.db", study_name = args.dataset_name+ "_"+ l + "_" + str(p) + "_" + str(seed))
                best_params = study.best_params
                for key in best_params.keys():
                    config[key] =  best_params[key]
                config["penalty"] = p

                val_loss,results = run_experiment(config, data)
                if val_loss != study.best_value:
                    raise ValueError("The best value is not the same")
                if results["test_coverage"] < config["coverage"]*(1 + tolerance) and results["test_coverage"] > config["coverage"]*(1 - tolerance):
                    if coverage < config["coverage"]*(1 + tolerance) and coverage > config["coverage"]*(1 - tolerance): 
                        if best_length > results["test_width"]:
                            best_length = results["test_width"]
                            coverage = results["test_coverage"]
                            for key in best_params.keys():
                                best_config[key] =  best_params[key]
                    else:
                        best_length = results["test_width"]
                        coverage = results["test_coverage"]
                        for key in best_params.keys():
                            best_config[key] =  best_params[key]
                else:
                    if (results["test_coverage"] - config["coverage"])**2 < (coverage - config["coverage"])**2:
                        best_length = results["test_width"]
                        coverage = results["test_coverage"]
                        for key in best_params.keys():
                            best_config[key] =  best_params[key]
                torch.cuda.empty_cache()
                
            result_dict = {
                "dataset": args.dataset_name,
                "loss": config["loss"],
                "seed": seed,
                "coverage": coverage, 
                "length": best_length}
            for key in best_config.keys():
                result_dict[key] = best_config[key]

            if not os.path.exists("results/finetuning/test_results.csv"):
                pd.DataFrame([result_dict]).to_csv("results/finetuning/test_results.csv")
            else:
                
                df = pd.DataFrame([result_dict])
                old_df = pd.read_csv("results/finetuning/test_results.csv",index_col=0)
                pd.concat([old_df, df]).to_csv("results/finetuning/test_results.csv")
            
if __name__ == "__main__":
    testing()