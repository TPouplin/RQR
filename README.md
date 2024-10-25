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

To quickly get started with the RQR loss objective, we provide the following resources.

A self-contained drop in loss function for Pytorch is provided in the following code block.

```python
class RQRW(torch.nn.Module):
    """
    RQR-W loss function for quantile regression.
    
    Args:
        alpha: float, default=0.9
            The desired coverage level in (0,1).
        lam: float, default=0.
            Width minimizing regularization parameter >= 0.
    
    See also:
        "Relaxed Quantile Regression: Prediction Intervals for Asymmetric Noise"
    """
    def __init__(self, alpha: float = 0.9, lam: float = 0.):
        super().__init__()
        self.alpha = alpha
        self.l = lam
        self.q1 = (1 - lam)/2
        self.q2 = 1 - (1 - lam)/2
 
    def forward(self, preds: torch.Tensor,  target: torch.Tensor) -> torch.Tensor:
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]

        diff_mu_1 = target - y_pred_q1 
        diff_mu_2 = target - y_pred_q2
        width = y_pred_q2 - y_pred_q1 
        RQR_loss = torch.maximum(diff_mu_1 * diff_mu_2 * (self.alpha + 2 * self.l),
                                 diff_mu_2 * diff_mu_1 * (self.alpha + 2 * self.l - 1))
        penalty = self.l * torch.square(width) * 0.5
        loss = RQR_loss + penalty

        return torch.mean(loss)
```

For an example usage in a self-contained notebook, check out  [`quick_start1.ipynb`](https://github.com/TPouplin/RQR/blob/main/quick_start1.ipynb).

To get started with this repositories implementation check out [`quick_start2.ipynb`](https://github.com/TPouplin/RQR/blob/main/quick_start2.ipynb) which provides an illustration on known noise distributions (i.e. reproducing Table 2 from the text).


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
