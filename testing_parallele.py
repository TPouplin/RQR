import subprocess
from multiprocessing import Pool
import os


    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

dataset_names = ["power","yacht","protein"] # ["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
n_seed = 10
n_job = 4
device =0
coverage = 0.9
processes = []
cmds = []
for d in dataset_names:
    cmds += [['python', 'testing.py', "--dataset_name",   d, "--n_seed", str(n_seed), "--gpu", str(device), "--coverage", str(coverage)]]
            

pool = Pool(n_job)
 
pool.map(run_process, cmds)
    
