import subprocess
from multiprocessing import Pool
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

loss =   ["RQR-W","SQR","IR","WS"] # ["QR","WS","RQR-W","SQR","RQR-O","OQR","IR"]
dataset_names =  ["sulfur","boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
dataset_names.reverse()

# dataset_names = ["protein"] #gpu 0
# dataset_names = ["naval"] # gpu 1
dataset_names = ["protein"]  # gpu 2
# dataset_names = ["protein"] # gpu 3
# dataset_names = ["miami","cpu_act"] # gpu 4


coverages = [0.9,0.7,0.5,0.3]

n_seed = 10
n_job = 2
device =3

processes = []
cmds = []
for d in dataset_names:
    for l in loss:
        for c in coverages:
            cmds += [['python', 'finetuning.py', "--dataset_name",   d, "--loss", l, "--coverage", str(c), "--n_seed", str(n_seed), "--gpu", str(device)]]


        
        
# pool = Pool(n_job)
 
# pool.map(run_process, cmds)
    
process_map(run_process, cmds, max_workers=n_job, chunksize=1, desc="Finetuning", leave=False)