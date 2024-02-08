import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__( self, quantiles, penalty_w=0, penalty_o=0):
      
        super().__init__()
        self.c = 1-quantiles[0]*2
        self.lw = penalty_w
        self.lo = penalty_o
        self.quantiles = quantiles
        self.q1 = quantiles[0]
        self.q2 = quantiles[1]
        
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

class QR_loss(loss):

    def forward(self, y_pred, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[::, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return torch.mean(losses)



class HQ_loss(nn.Module):

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
        
        
        loss = MPIW_capt + self.penalty*n*(1/((1-self.c)*self.c))*torch.square(torch.maximum(torch.Tensor([0.]).to(y_pred_q1.device),self.c-PICP_S))

        
        return loss



class Winkler_Loss(nn.Module):
    def forward(self, preds, target):
     
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        c = self.q2-self.q1
        alpha = 1-c
        y_pred_q1 = preds[:, 0]
        y_pred_q2 = preds[:, 1]
        
        below_1 = (y_pred_q1-target).gt(-1)
        below_2 = (target-y_pred_q2).gt(-1)
        
        # print(y_pred_q1.shape, y_pred_q2.shape, target.shape, below_1.shape, below_2.shape)
        loss = (y_pred_q2-y_pred_q1) + (2/alpha)*(y_pred_q1-target)*below_1  + (2/alpha)*(target-y_pred_q2)*below_2
        return loss.mean()
    
    
dict_loss = {"QR": QR_loss,
             "RQR": RQRW_loss,
             "WS": Winkler_Loss,
             "RQR-W": RQRW_loss,
             "IR": HQ_loss
            #  "SQR":
            #  "RQR-O":
            #  "OQR" : 
}

