
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
import numpy as np 

fine_tuned_parameters = {
    "lr": [0.01, 0.001], #
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "penalty": [0.1,1,10],   
}
    
other_parameters = {
    "epochs": 1000,
    "coverage": 0.9,
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "batch_size": 5000,    
}


def objective_function(coverage, width):
    if np.abs((coverage - 0.9)) > 0.9*2.5/100:
        obj = 100
    else:
       obj = width
    return obj


def objective(trial, dataset_name, loss, seed, p):
    config = {}
    config["dataset_name"] = dataset_name
    config["loss"] = loss
    config["random_seed"] = seed
    config["verbose"] = False
    config["finetuning"] = True
    
   
    
    for key in fine_tuned_parameters:
        if (key == "penalty"):
            config[key] = p
        else:
            config[key] = trial.suggest_categorical(key, fine_tuned_parameters[key])

        
        
    for key in other_parameters:
        config[key] = other_parameters[key]
        
    val_loss,results = run_experiment(config)
    

        
    return objective_function(results["test_coverage"],results["test_width"])



def fine_tuning():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--loss', type=str, default="QR", help='The name ofthe loss function')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')

    args = parser.parse_args()


    # n_trial = 140
    # for key in fine_tuned_parameters:
    #     n_trial *= len(fine_tuned_parameters[key])

    n_trial = 8
    
    if not ( args.loss in ['RQR-W', 'RQR-O', 'OQR', 'IR']):
        # n_trial /= len(fine_tuned_parameters["penalty"])
        penalty = [0]
    else:
        penalty = fine_tuned_parameters["penalty"]
        

    n_trial = int(n_trial)
    
    
    
    print("PARAMETER SET")
    for seed in range(args.n_seed):
        for p in penalty:
            study = optuna.create_study(storage= "sqlite:///results/finetuning/recording_ultra_light_reb_test.db", study_name = args.dataset_name+ "_"+ args.loss + "_" + str(p) + "_" + str(seed), direction='minimize', load_if_exists=True)
            currated_nb_trial = np.sum([ 1 if x.state != optuna.trial.TrialState.FAIL else 0 for x in study.trials])
            current_n_trial = max(0,n_trial - currated_nb_trial)
            print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss}, penalty {p} started. Remaining trials : {current_n_trial}/ {n_trial}")

            study.optimize(lambda trial: objective(trial, args.dataset_name, args.loss, seed, p), n_trials=current_n_trial, n_jobs=1)
            print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss} done")
            
            torch.cuda.empty_cache()

        
if __name__ == "__main__":
    fine_tuning()