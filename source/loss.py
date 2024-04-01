import torch
import torch.nn as nn
from source.OQR import *
import numpy as np

class loss(nn.Module):
    def __init__( self, coverage, penalty=0):
      
        super().__init__()
        self.c = coverage
        self.l = penalty
        self.quantiles = [(1-coverage)/2, 1-(1-coverage)/2]
        self.q1 = self.quantiles[0]
        self.q2 = self.quantiles[1]
        
    def forward(self, y_pred, target):
        pass
        

class RQRW_loss(loss):
 
    def forward(self, preds: torch.Tensor,  target: torch.Tensor) -> torch.Tensor:
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]

        errors1 =   target -y_pred_q1 
        errors2 =   target- y_pred_q2
        errors =y_pred_q2-y_pred_q1 
        loss1 = torch.maximum(errors1*errors2*(self.c+2*self.l), errors2*errors1*(self.c+2*self.l-1))
        loss2 = self.l*torch.square(errors)*0.5
        
        loss = loss1 + loss2

        return torch.mean(loss)
    
    

class RQRO_loss(loss):
 
    def forward(self, preds: torch.Tensor,  target: torch.Tensor) -> torch.Tensor:

        # print(preds.shape, target.shape)
        
        y_pred_q1= torch.min(preds, axis=1)[0].view(-1,1)
        y_pred_q2 = torch.max(preds, axis=1)[0].view(-1,1)
        
        
        errors1 =   target -y_pred_q1 
        errors2 =   target- y_pred_q2
        loss1 = torch.maximum(errors1*errors2*(self.c), errors2*errors1*(self.c+-1))
        
        
        loss2 = independence_penalty(target, y_pred_q1.view(1,-1), y_pred_q2.view(1,-1), pearsons_corr_multiplier=0, hsic_multiplier=self.l)
        
        loss = loss1 + loss2

        return torch.mean(loss)
    
    

    


class OQR_loss(loss):

    def forward(self, y_pred, target: torch.Tensor) -> torch.Tensor:
        
        y_pred_q1= torch.min(y_pred, axis=1)[0].view(-1,1)
        y_pred_q2 = torch.max(y_pred, axis=1)[0].view(-1,1)
        
        loss1 = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[::, i]
            loss1.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        loss1 = 2 * torch.cat(loss1, dim=2)
        
        loss2 = independence_penalty(target, y_pred_q1.view(1,-1), y_pred_q2.view(1,-1), pearsons_corr_multiplier=0, hsic_multiplier=self.l)
        
        loss = loss1 + loss2


        return torch.mean(loss)



class HQ_loss(loss):

    def forward(self, preds: torch.Tensor,  target: torch.Tensor) -> torch.Tensor:
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]
        
        n = y_pred_q1.shape[0]
        soften = torch.full((n,), 160).to(y_pred_q1.device)
        
        
        
        K_HU = torch.maximum(torch.Tensor([0.]).to(y_pred_q1.device), torch.sign(y_pred_q2-target))
        K_HL = torch.maximum(torch.Tensor([0.]).to(y_pred_q1.device), torch.sign(target-y_pred_q1))
        K_H = torch.mul(K_HU, K_HL)
        
        K_SU = torch.sigmoid(torch.mul(soften,y_pred_q2-target))
        K_SL = torch.sigmoid(torch.mul(soften,target-y_pred_q1))
        K_S = torch.mul(K_SU, K_SL)
        
        c = torch.sum(K_H)+ 0.001
        MPIW_capt = (1/c)*torch.sum(torch.mul(K_H,(y_pred_q2-y_pred_q1)))
        

        PICP_H = torch.mean(K_H)
        PICP_S = torch.mean(K_S)
        
        
        loss = MPIW_capt + self.l*n*(1/((1-self.c)*self.c))*torch.square(torch.maximum(torch.Tensor([0.]).to(y_pred_q1.device),self.c-PICP_S))

        
        return loss

class QR_loss(loss):

    def forward(self, y_pred, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[::, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return torch.mean(losses)


class Winkler_Loss(loss):
    def forward(self, preds, target):
     
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        c = self.q2-self.q1
        alpha = 1-c
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]
        
        below_1 = (y_pred_q1-target).gt(0)
        below_2 = (target-y_pred_q2).gt(0)
        
        # print(y_pred_q1.shape, y_pred_q2.shape, target.shape, below_1.shape, below_2.shape)
        loss = (y_pred_q2-y_pred_q1) + (2/alpha)*(y_pred_q1-target)*below_1  + (2/alpha)*(target-y_pred_q2)*below_2
        return loss.mean()
    

    
    
class S_Winkler_Loss(loss):
    def forward(self, preds, target, c=None):
        if c is None:
            c = self.coverage
 
        alpha = 1-c
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]
        
        below_1 = (y_pred_q1-target).gt(0)
        below_2 = (target-y_pred_q2).gt(0)
        
        # print(y_pred_q1.shape, y_pred_q2.shape, target.shape, below_1.shape, below_2.shape)
        loss = (y_pred_q2-y_pred_q1) + (2/alpha)*(y_pred_q1-target)*below_1  + (2/alpha)*(target-y_pred_q2)*below_2
        return loss.mean()
    
class SQR_loss(loss):
      def forward(self, y_pred, target: torch.Tensor, quantiles=None) -> torch.Tensor:
        losses = []
        if quantiles is None:
            quantiles = self.quantiles

        quantiles = quantiles.unsqueeze(-1)
        for i, q in enumerate(quantiles):
            errors = target - y_pred[::, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)

        return torch.mean(losses)
    
    
    
dict_loss = {"QR": QR_loss,
             "WS": Winkler_Loss,
             "RQR-W": RQRW_loss,
             "IR": HQ_loss,
             "SQR": SQR_loss,
             "RQR-O": RQRO_loss,
             "OQR" : OQR_loss,
             "SWS" : S_Winkler_Loss,
             "SQRN" : SQR_loss
}

