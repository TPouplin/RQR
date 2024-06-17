
import optuna
import argparse
from src.run_experiment import run_experiment
import torch 
from finetuning import fine_tuned_parameters, other_parameters
import os 
import pandas as pd 
from data.dataset import GetDataset



def testing():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')
    parser.add_argument('--gpu', type=int, default=0, help='The id of the gpu to use')
    parser.add_argument('--coverage', type=float, default=0.9, help='Targeted coverage')
    parser.add_argument('--loss', type=str, default="QR", help='The name ofthe loss function')

    args = parser.parse_args()
    

    config = {}
    config["dataset_name"] = args.dataset_name
    config["verbose"] = True
    config["finetuning"] = False
    config["device"] = args.gpu
    
    for key in other_parameters:
        config[key] = other_parameters[key]
    
    config["coverage"] = args.coverage
    
    data = GetDataset(config["dataset_name"], "data")

    l = args.loss        
    output_path = f"results/finetuning/results_{args.dataset_name}_{l}.csv"

    config["loss"] = l
    if l == "SQRN":
        l = "SQRC"
        
    if not ( l in ['RQR-W', 'IR']):
        penalty = [0]
    else:
        penalty = fine_tuned_parameters["penalty"]    
    
    for seed in range(args.n_seed):

        
        config["random_seed"] = seed

        best_length = 99999999
        coverage = -10
        tolerance = 2.5/100
        best_config  = {}
        for p in penalty:
            study = optuna.load_study(storage= "sqlite:///results/finetuning/recording.db", study_name = args.dataset_name+ "_" +  str(args.coverage) + "_" + l + "_" + str(p) + "_" + str(seed))
            best_params = study.best_params
            for key in best_params.keys():
                config[key] =  best_params[key]
            config["penalty"] = p

            val_loss,results, results_val = run_experiment(config, data)
          
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
            "length": best_length,
            "target" : config["coverage"]}
        for key in best_config.keys():
            result_dict[key] = best_config[key]

        if not os.path.exists(output_path):
            pd.DataFrame([result_dict]).to_csv(output_path)
        else:
            old_df = pd.read_csv(output_path,index_col=0)
            df = pd.DataFrame([result_dict])
            pd.concat([old_df, df]).to_csv(output_path)
    
if __name__ == "__main__":
    testing()