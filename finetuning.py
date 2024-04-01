
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
import numpy as np 
from data.dataset import GetDataset

fine_tuned_parameters = {
    "lr": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], #
    "dropout": [0,0.1, 0.2, 0.3,0.4],
    "penalty": [0.01,0.1,0.5,1,5, 10,15,20,30,40,50],   
}
    
other_parameters = {
    "epochs": 200,
    "coverage": 0.9,
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "batch_size": 10000,    
}


from source.model import objective_function


def objective(trial, dataset_name, loss, seed, p, data, device):
    config = {}
    config["dataset_name"] = dataset_name
    config["loss"] = loss
    config["random_seed"] = seed
    config["verbose"] = False
    config["finetuning"] = True
    config["device"] = device
   
    p = 1
    
    for key in fine_tuned_parameters:
        if (key == "penalty"):
            config[key] = p
        else:
            config[key] = trial.suggest_categorical(key, fine_tuned_parameters[key])

        
        
    for key in other_parameters:
        config[key] = other_parameters[key]
    
    
    if dataset_name == "nyc_taxi" or dataset_name == "medical_charges":
        config["batch_size"] = 20000
    
    
    
    try : 
        study = optuna.create_study(storage= "sqlite:///results/finetuning/recording_final.db", study_name = dataset_name+ "_"+ loss + "_" + str(p) + "_" + str(seed), direction='minimize', load_if_exists=True)
        trials = study.get_trials(optuna.trial.TrialState.COMPLETE)
        for old_trial in trials:
            if old_trial.params == trial.params:

                results = {"test_coverage": old_trial.user_attrs["coverage"], "test_width": old_trial.user_attrs["width"]}

                return objective_function(results["test_coverage"],results["test_width"])
        raise ValueError("The trial is not in the database")
        
    except:
        val_loss,results = run_experiment(config, data)
        
        trial.set_user_attr("coverage", results["test_coverage"])
        trial.set_user_attr("width", results["test_width"])
            
        return objective_function(results["test_coverage"],results["test_width"])



def fine_tuning():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--loss', type=str, default="QR", help='The name ofthe loss function')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')
    parser.add_argument('--gpu', type=int, default=0, help='The id of the gpu to use')

    args = parser.parse_args()

    n_trial = 35
    
    if not ( args.loss in ['RQR-W', 'RQR-O', 'OQR', 'IR']):
        penalty = [0]
    else:
        penalty = fine_tuned_parameters["penalty"]
    
    # penalty.reverse()
    
    data = GetDataset(args.dataset_name, "data")
    
    print("PARAMETER SET")
    
    seeds = np.arange(args.n_seed-1,-1,-1)
    # seeds = np.arange(0,10,1)
    
    print("SEEDS : ", seeds)
    for seed in seeds:
        for p in penalty:
            study = optuna.create_study(storage= "sqlite:///results/finetuning/recording_final_full.db", study_name = args.dataset_name+ "_"+ args.loss + "_" + str(p) + "_" + str(seed), direction='minimize', load_if_exists=True)
            currated_nb_trial = np.sum([ 1 if x.state != optuna.trial.TrialState.FAIL else 0 for x in study.trials])
            current_n_trial = int(max(0,n_trial - currated_nb_trial))
            print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss}, penalty {p} started. Remaining trials : {current_n_trial}/ {n_trial}")

            study.optimize(lambda trial: objective(trial, args.dataset_name, args.loss, seed, p, data, args.gpu), n_trials=current_n_trial, n_jobs=1)
            print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss} done")
            
            torch.cuda.empty_cache()

        
if __name__ == "__main__":
    fine_tuning()