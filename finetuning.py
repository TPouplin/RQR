
import optuna
import argparse
from source.run_experiment import run_experiment
import torch 
import numpy as np 
from data.dataset import GetDataset
from source.model import objective_function

fine_tuned_parameters = {
    "lr": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], #
    "dropout": [0.1, 0.2, 0.3,],
    "penalty" : [0.1,1,5,10,20,30,40,50]
}
    
other_parameters = {
    "epochs": 400,
    "test_ratio": 0.2,
    "val_ratio": 0.2,
    "batch_size": 10000,    
}




def objective(trial, dataset_name, loss, seed, p, coverage, data, device):
    config = {}
    config["dataset_name"] = dataset_name
    config["loss"] = loss
    config["random_seed"] = seed
    config["verbose"] = False
    config["finetuning"] = True
    config["device"] = device
       
    for key in fine_tuned_parameters:
        if (key == "penalty"):
            config[key] = p
        else:
            config[key] = trial.suggest_categorical(key, fine_tuned_parameters[key])

        
    for key in other_parameters:
        config[key] = other_parameters[key]
    
    config["coverage"] = coverage
    val_loss,results = run_experiment(config, data)
    
    trial.set_user_attr("coverage", results["test_coverage"])
    trial.set_user_attr("width", results["test_width"])
        
    return objective_function(results["test_coverage"],results["test_width"], config["coverage"])



def fine_tuning():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--dataset_name', type=str, default="boston", help='The name of the dataset')
    parser.add_argument('--loss', type=str, default="QR", help='The name ofthe loss function')
    parser.add_argument('--n_seed', type=int, default=10, help='Number of seed')
    parser.add_argument('--gpu', type=int, default=0, help='The id of the gpu to use')
    parser.add_argument('--coverage', type=float, default=0.9, help='Targeted coverage')
    args = parser.parse_args()

    n_trial = len(fine_tuned_parameters["lr"]) * len(fine_tuned_parameters["dropout"])
    
    if not ( args.loss in ['RQR-W', 'RQR-O', 'OQR', 'IR']):
        penalty = [0]
    else:
        penalty = fine_tuned_parameters["penalty"]
        
    data = GetDataset(args.dataset_name, "data")
    
    
    seeds = np.arange(0,args.n_seed,1)
    
    for seed in seeds:
        for p in penalty:
            currated_nb_trial = 0
            study = optuna.create_study(storage= "sqlite:///results/finetuning/recording.db", study_name = args.dataset_name+ "_" +  str(args.coverage) + "_" + args.loss + "_" + str(p) + "_" + str(seed), direction='minimize', load_if_exists=True)
            currated_nb_trial = int(np.sum([ 1 if x.state != optuna.trial.TrialState.FAIL else 0 for x in study.trials]))
            while currated_nb_trial < n_trial:
                print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss}, penalty {p} started. Remaining trials : {currated_nb_trial}/ {n_trial}")
                study.optimize(lambda trial: objective(trial, args.dataset_name, args.loss, seed, p, args.coverage, data, args.gpu), n_trials=1, n_jobs=1)
                currated_nb_trial = np.sum([ 1 if x.state != optuna.trial.TrialState.FAIL else 0 for x in study.trials])

                torch.cuda.empty_cache()

        print(f"Process seed {seed}, dataset {args.dataset_name}, loss {args.loss} done")
                

        
if __name__ == "__main__":
    fine_tuning()