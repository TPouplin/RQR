import torch
import torch.nn as nn
import lightning as L
from source.loss import *
from source.metrics import *

class q_model(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 quantiles,
                 in_shape=1,
                 hidden_size=64,
                 dropout=0.1,
                 lr=0.0005,
                 loss="QR"):
 
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """
        super().__init__()
        self.num_quantiles = len(quantiles)
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.lr = lr
        self.loss = loss
        
        self.loss_fn = dict_loss[loss](quantiles)
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_quantiles),
        )

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if self.loss =="HQ" and m.bias.shape == torch.Size([2]):
                    m.bias = nn.Parameter(torch.Tensor([0, 1]))
                else:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return self.base_model(x)
    
    
    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        """ Training step
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        
        metrics = compute_metrics(y_hat, y)
        
        self.log("train_coverage", metrics["coverage"])
        self.log("train_width", metrics["interval_width"])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        
        metrics = compute_metrics(y_hat, y)
        
        self.log("val_coverage", metrics["coverage"])
        self.log("val_width", metrics["interval_width"])
        
        return loss
    
    def configure_optimizers(self):
        """ Configure optimizer
        """
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        """ Test step
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        
        metrics = compute_metrics(y_hat, y)
        
        self.log("test_coverage", metrics["coverage"])
        self.log("test_width", metrics["interval_width"])
            
        print( f"Metrics on test set  ! \n Coverage : {metrics['coverage']} \n Width : {metrics['interval_width']} \n Loss : {loss} \n")
    
        return loss
    
    