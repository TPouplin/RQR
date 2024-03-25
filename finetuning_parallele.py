import subprocess
from multiprocessing import Pool
import os


    
def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()

loss =   ["RQR-W","SQR","QR"] #["SQR","WS","SWS"] # ["QR","WS","RQR-W","SQR","RQR-O","OQR","IR"]
dataset_names =  [ "cpu_act",]   #["boston", "concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht"]

n_seed = 10
n_job = 30

processes = []
cmds = []
for d in dataset_names:
    for l in loss:
            cmds += [['python', 'finetuning.py', "--dataset_name",   d, "--loss", l, "--n_seed", str(n_seed)]]

pool = Pool(n_job)
 
pool.map(run_process, cmds)
    
