import subprocess
from multiprocessing import Pool
import os


    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

dataset_names = ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht","sulfur","miami","cpu_act"]
loss = ["IR","WS","RQR-W","SQRC","SQRN"]

n_seed = 10
n_job = 6
device =2
coverage = 0.3
processes = []
cmds = []
for d in dataset_names:
    for l in loss:
        cmds += [['python', 'testing.py', "--dataset_name",   d, "--loss",l, "--n_seed", str(n_seed), "--gpu", str(device), "--coverage", str(coverage)]]
            

pool = Pool(n_job)
 
pool.map(run_process, cmds)
    
