import torch
import torch.nn as nn
import lightning as L
from source.loss import *
from source.metrics import *

class Q_model(L.pytorch.LightningModule):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 coverage,
                 x_shape=1,
                 hidden_size=64,
                 dropout=0.1,
                 lr=0.0005,
                 loss="QR",
                 penalty=0,):
 
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """
        super().__init__()
        self.hidden_size = hidden_size
        
        
        if loss ==  "SQR":
            self.in_shape = x_shape+1
        else:
            self.in_shape = x_shape
        
        
        self.out_shape = 2
        self.dropout = dropout
        self.lr = lr
        self.loss = loss
        self.coverage = coverage
        
        self.loss_fn = dict_loss[loss](coverage, penalty=penalty)
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
            nn.Linear(self.hidden_size, self.out_shape),
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
        
        metrics = compute_metrics(y_hat, y)

        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """ Training step
        """
        loss, metrics = self.step(batch, batch_idx)
        
        self.log("train_loss", loss)
        self.log("train_coverage", metrics["coverage"])
        self.log("train_width", metrics["interval_width"])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step
        """

        loss, metrics = self.step(batch, batch_idx)
        
        self.log("val_loss", loss)
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
        loss, metrics = self.step(batch, batch_idx)
        
        self.log("test_loss", loss)        
        self.log("test_coverage", metrics["coverage"])
        self.log("test_width", metrics["interval_width"])
            
        # print( f"Metrics on test set  ! \n Coverage : {metrics['coverage']} \n Width : {metrics['interval_width']} \n Loss : {loss} \n")
    
        return loss
    

class SQ_model(Q_model):
    def step(self, batch, batch_idx, coverages):
        x, y = batch
        
        losses = []
        
        for c in coverages:
            x_c = torch.cat([x, c.repeat(x.shape[0], 1)], dim=1)
            y_hat = self(x_c)
            losses.append(self.loss_fn(y_hat, y, c))
            
        metrics = compute_metrics(y_hat, y)
        loss = torch.mean(torch.stack(losses))

        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """ Training step
        """
        
        coverages = torch.rand(30).to("cuda") #number of dcoverage sample, default value from the OQR implementation
        
        loss, metrics = self.step(batch, batch_idx, coverages)
        
        self.log("train_loss", loss)

        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step
        """
        
        coverages = torch.rand(30).to("cuda") #number of dcoverage sample, default value from the OQR implementation
        
        loss, metrics = self.step(batch, batch_idx, coverages)
        
        self.log("val_loss", loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """ Test step
        """
        
        coverages = torch.Tensor([self.coverage]).to("cuda")
        
        loss, metrics = self.step(batch, batch_idx, coverages)
        
        self.log("test_loss", loss)        
        self.log("test_coverage", metrics["coverage"])
        self.log("test_width", metrics["interval_width"])
            
        # print( f"Metrics on test set  ! \n Coverage : {metrics['coverage']} \n Width : {metrics['interval_width']} \n Loss : {loss} \n")
    
        return loss