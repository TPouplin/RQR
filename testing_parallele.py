import subprocess
from multiprocessing import Pool
import os


    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

loss =  ["SQRN"] #["QR","WS","RQR-W","SQR","RQR-O","OQR","IR","SQRN"]
dataset_names =  ["boston"] #["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]

n_seed = 10
n_job = 30

processes = []
cmds = []
for d in dataset_names:
    cmds += [['python', 'testing.py', "--dataset_name",   d, "--n_seed", str(n_seed)]]
            

pool = Pool(n_job)
 
pool.map(run_process, cmds)
    
