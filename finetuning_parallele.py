import subprocess
from multiprocessing import Pool
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

loss =   ["RQR-W","SQR","IR","WS"] # ["QR","WS","RQR-W","SQR","RQR-O","OQR","IR"]
dataset_names =  ["miami","cpu_act","sulfur","boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
coverages = [0.9,0.7, 0.5, 0.3]

n_seed = 10
n_job = 3

processes = []
cmds = []
for d in dataset_names:
    for l in loss:
        for c in coverages:
            cmds += [['python', 'finetuning.py', "--dataset_name",   d, "--loss", l, "--coverage", str(c), "--n_seed", str(n_seed)]]


        
        
# pool = Pool(n_job)
 
# pool.map(run_process, cmds)
    
process_map(run_process, cmds, max_workers=n_job, chunksize=1, desc="Finetuning", leave=False)