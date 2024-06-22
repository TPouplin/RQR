## Relaxed Quantile Regression: Prediction Intervals for Asymmetric Noise

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/TPouplin/RQR/blob/main/LICENSE)


This repository contains the code associated with [our ICML 2024 paper](https://openreview.net/forum?id=L8nSGvoyvb) where we introduce a new loss to train Deep Learning models for interval prediction. The Relaxed Quantile Regression (RQR) loss directly learning of the intervals without the intermediary steps of estimating quantiles.

### Experiments
**Setup**

Clone this repository and navigate to the root folder.
```
git clone https://github.com/TPouplin/RQR.git
cd RQR
```
Using conda, create and activate a new environment. 
```
conda create -n <environment name> pip python
conda activate <environment name>
```
Then install the repository requirements.
```
pip install -r requirements.txt
```

**Data**

The datasets mentioned in the paper can be found in the data folder `src/data/`.
The datasets have to be unzipped e.g.
```
unzip data/datasets.zip -d data
```

**Quick Start**
To quickly get started with the RQR loss objective, check out [`quick_start.ipynb`](https://github.com/TPouplin/RQR/blob/main/quick_start.ipynb) which contains a simple implementation of the objective in Pytorch and an illustration on known noise distributions (i.e. reproducing Table 2 from the text).


**Running Width Experiments**

The hyperparameter fine-tuning can be run for all the datasets and losses proposed in the paper with the following command : 

```python finetuning_parallel.py```

Results and hyperparameters of these experiments are saved in an optuna database `results/finetuning/recording.db`.

The optimal hyperparameters found can be used to recreate the benchmark presented in the paper with the following command : 

```python testing_parallel.py```

**Running Orthogonal Experiments**

The orthogonal experiments implementation is a modified version of the official OQR repository : https://github.com/Shai128/oqr. 
It can be run for all the datasets and losses proposed in the paper with the following command : 

```python src/RQR-O/run_orthogonal_experiment.py```


### Citation
If you use this code, please cite the associated paper.
```
@inproceedings{
      pouplin2024relaxed,
      title={Relaxed Quantile Regression: Prediction Intervals for Asymmetric Noise},
      author={Thomas Pouplin and Alan Jeffares and Nabeel Seedat and Mihaela van der Schaar},
      booktitle={Forty-first International Conference on Machine Learning},
      year={2024},
      url={https://openreview.net/forum?id=L8nSGvoyvb}
}
```
