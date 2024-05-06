
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
from finetuning import fine_tuned_parameters, other_parameters
import os 
import pandas as pd 
from data.dataset import GetDataset


loss = ["RQR-W","SQR","SQRN","IR","WS"]

def testing():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')
    parser.add_argument('--gpu', type=int, default=0, help='The id of the gpu to use')
    parser.add_argument('--coverage', type=float, default=0.9, help='Targeted coverage')

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


    

    for l in loss:
        
        output_path = f"results/finetuning/results_{args.dataset_name}_{l}.csv"

        config["loss"] = l
        if l == "SQRN":
            l = "SQR"
            
        if not ( l in ['RQR-W', 'RQR-O', 'OQR', 'IR']):
            penalty = [0]
        else:
            penalty = fine_tuned_parameters["penalty"]    
        
        for seed in range(args.n_seed):
        

            # if old_df[ (old_df["dataset"] == args.dataset_name) & (old_df["loss"] == l) & (old_df["seed"] == seed) & (old_df["target"] ==config["coverage"] )].shape[0] > 0:
            #     continue
            
            
            config["random_seed"] = seed

            best_length = 99999999
            coverage = 0
            tolerance = 2.5/100
            best_config  = {}
            for p in penalty:
                study = optuna.load_study(storage= "sqlite:///results/finetuning/recording.db", study_name = args.dataset_name+ "_" +  str(args.coverage) + "_" + l + "_" + str(p) + "_" + str(seed))
                best_params = study.best_params
                for key in best_params.keys():
                    config[key] =  best_params[key]
                config["penalty"] = p

                val_loss,results, results_val = run_experiment(config, data)
                if (results_val["test_coverage"] != study.best_trial.user_attrs["coverage"] or results_val["test_width"] != study.best_trial.user_attrs["width"]) and config["loss"] != "SQRN":
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