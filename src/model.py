import torch
import torch.nn as nn
import lightning as L
from src.loss import *
from src.metrics import *
from tqdm import tqdm

def objective_function(coverage, width, target_coverage):
    if type(coverage) != torch.Tensor:
        coverage = torch.tensor(coverage)
    
    if torch.abs((coverage - target_coverage)) > target_coverage*2.5/100:
        obj= torch.abs((coverage - target_coverage))*100
    else:
       obj = width
    return obj

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
        
        
        if loss ==  "SQRC" or loss == "SQRN":
            self.in_shape = x_shape+1
            self.out_shape = 1
        else:
            self.in_shape = x_shape
            self.out_shape = 2
                
        self.dropout = dropout
        self.lr = lr
        self.loss = loss
        self.coverage = coverage
        self.quantiles = [(1-coverage)/2, 1-(1-coverage)/2]
        
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
        
        metrics = compute_metrics(y_hat, y[:,0])

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
        
        self.log("val_loss", loss,sync_dist=True)
        self.log("val_coverage", metrics["coverage"],sync_dist=True)
        self.log("val_width", metrics["interval_width"],sync_dist=True)
        self.log("val_objective", objective_function(metrics["coverage"], metrics["interval_width"], self.coverage),sync_dist=True)
        
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
        
        self.log("test_loss", loss,sync_dist=True)        
        self.log("test_coverage", metrics["coverage"],sync_dist=True)
        self.log("test_width", metrics["interval_width"],sync_dist=True)
                
        return loss
    
    def predict_step(self, batch, batch_idx):
        """ Prediction step
        """
        x, y = batch
        y_hat = self(x)
        return y_hat
    


class SQ_model(Q_model):
    def step(self, batch, batch_idx, quantiles, return_metrics=False, test=False):
        x, y = batch
        losses = []
        
        if self.narrowest and return_metrics and test:
            results = []
            
            bins = 10
            quantiles = torch.tensor(np.linspace(0, 1-self.coverage, bins)).to(self.device).float()
            result_all_quantiles = []
            for q in quantiles:
              
                
                x_l = torch.cat([x, q.repeat(x.shape[0], 1)], dim=1)
                x_u = torch.cat([x, (self.coverage+q).repeat(x.shape[0], 1)], dim=1)
                y_u = self(x_u)
                y_l = self(x_l)
                result_all_quantiles.append((y_l,y_u))
            
            y_hat = torch.tensor(np.zeros((x.shape[0],2))).to(x.device)
            
            for i in tqdm(range(x.shape[0])):
                min_width = float("inf") 
                best_q = None
                for q in range(len(quantiles)):
                    u =result_all_quantiles[q][1][i]
                    l = result_all_quantiles[q][0][i]
                    width =  u - l
                    if width < min_width:
                        min_width = width
                        best_q = (quantiles[q], self.coverage- quantiles[q])
                        y_hat[i] =  torch.cat([l,u], axis=0)
                    
                losses.append(self.loss_fn(y_hat, y, torch.tensor(best_q).to(batch[0].device)))
              
            results = [y_hat] 

            y_hat = torch.stack(results)            
        else:
            results = []
            for q in quantiles:
                x_q = torch.cat([x, q.repeat(x.shape[0], 1)], dim=1)
                y_hat = self(x_q)
                losses.append(self.loss_fn(y_hat, y, q))
                results.append(y_hat)
                
        loss = torch.mean(torch.stack(losses))

        if return_metrics:
        
            y_hat = torch.stack(results, dim=1).squeeze().to(self.device)
            metrics = compute_metrics(y_hat, y)
            return loss, metrics
        else:
            return loss
    
    def training_step(self, batch, batch_idx):
        """ Training step
        """
        
        quantiles = torch.rand(30).to(self.device) #number of coverage sample, default value from the OQR implementation
        
        loss = self.step(batch, batch_idx, quantiles)
        
        self.log("train_loss", loss,sync_dist=True)

        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step
        """
        
        quantiles = torch.rand(30).to(batch[0].device) #number of dcoverage sample, default value from the OQR implementation
        
        loss, metrics = self.step(batch, batch_idx, quantiles, return_metrics=True)
        
        self.log("val_loss", loss,sync_dist=True)
        self.log("val_coverage", metrics["coverage"],sync_dist=True)
        self.log("val_width", metrics["interval_width"],sync_dist=True)
        self.log("val_objective", objective_function(metrics["coverage"], metrics["interval_width"], self.coverage),sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        """ Test step
        """
        
        quantiles = torch.Tensor(self.quantiles).to(self.device)
        
        loss, metrics = self.step(batch, batch_idx, quantiles, return_metrics=True, test=True)
        
        self.log("test_loss", loss,sync_dist=True)      
        self.log("test_coverage", metrics["coverage"],sync_dist=True)
        self.log("test_width", metrics["interval_width"],sync_dist=True)
            
        # print( f"Metrics on test set  ! \n Coverage : {metrics['coverage']} \n Width : {metrics['interval_width']} \n Loss : {loss} \n")
    
        return loss
