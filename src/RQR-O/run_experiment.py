import subprocess
import os

def run_experiment(experiment_params):


    # & for windows or ; for mac
    if os.name == 'nt':
        separate = '&'
    else:
        separate = ';'

    command = f'cd .. {separate} python RQR/src/RQR-O/main.py --bs 1024 '

    command += ' --test_ratio 0.4 --nl 3 --dropout 0.1 '

    for param in ['hsic_mult', 'corr_mult', 'data', 'loss', 'method']:
        if param in experiment_params:
            command += f' --{param} {experiment_params[param]} '

    if 'seed' in experiment_params:
        if type(experiment_params["seed"]) == tuple:
            seed_begin, seed_end = experiment_params["seed"]
            seed_param = f' --seed_begin {seed_begin} --seed_end {seed_end} '
        else:
            seed = experiment_params["seed"]
            seed_param = f' --seed {seed} '
    else:
        seed_param = ''

    if 'lr' in experiment_params:
        command += f' --lr {experiment_params["lr"]} '
    
    if 'dropout' in experiment_params:
        command += f' --dropout {experiment_params["dropout"]} '
    
    if 'penalty' in experiment_params:
        command += f' --penalty {experiment_params["penalty"]} '
    
    if 'epochs' in experiment_params:
        command += f' --num_ep {experiment_params["epochs"]} '
    if "bs" in experiment_params:
        command += f' --bs {experiment_params["bs"]} '
    if "wd" in experiment_params:
        command += f' --wd {experiment_params["wd"]} '

    command += seed_param

    if 'save_training_results' in experiment_params and experiment_params['save_training_results']:
        command += ' --save_training_results 1'

    process = subprocess.Popen(command, shell=True)

    return process