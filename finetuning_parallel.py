import subprocess
from tqdm.contrib.concurrent import process_map 

    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

loss =   ["RQR-W","SQRC","IR","WS"]
dataset_names =  ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht", "miami","cpu_act", "sulfur"]
coverages = [0.9,0.7,0.5,0.3]

device = 0
n_seed = 10
n_job = 6

cmds = []
for d in dataset_names:
    for l in loss:
        for c in coverages:
            cmds += [['python', 'finetuning.py', "--dataset_name",   d, "--loss", l, "--coverage", str(c), "--n_seed", str(n_seed), "--gpu", str(device)]]

            
    
process_map(run_process, cmds, max_workers=n_job, chunksize=1, desc="Finetuning", leave=False)