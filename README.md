## Relaxed Quantile Regression: Prediction Intervals for Asymmetric Noise

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD-blue.svg)](https://github.com/TPouplin/RQR/blob/main/LICENSE)


This repository contains the code associated with [our ICML 2024 paper](https://openreview.net/attachment?id=L8nSGvoyvb&name=pdf) where we introduce a new loss to train Deep Learning models for interval prediction. The Relaxed Quantile Regression (RQR) loss directly learning of the intervals without the intermediary steps of estimating quantiles.

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

The datasets mentionned in the paper can be found in the data folder `source/data/`.
The UCI datasets have to be unzip.

**Running**

The hyperparameter fine-tuning can be run for all the datasets and losses proposed in the paper with the following command : 

```python finetuning_parallele.py```

Results and hyperparameters of these experiments are saved in an optuna database `results/finetuning/recording.db`.

The optimal hyperparemeters found can be used to recreate the benchmark presented in the paper with the following command : 

```python testing_parallele.py```

### Citation
If you use this code, please cite the associated paper.
```
@misc{pouplin2024relaxed,
      title={Relaxed Quantile Regression: Prediction Intervals for Asymmetric Noise}, 
      author={Thomas Pouplin and Alan Jeffares and Nabeel Seedat and Mihaela van der Schaar},
      year={2024},
      eprint={2406.03258},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
