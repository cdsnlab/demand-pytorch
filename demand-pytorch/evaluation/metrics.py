import numpy as np
import torch.nn as nn 
import torch 

class MaskedMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.pow(preds - labels, 2)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedMAE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedRMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.pow(preds - labels, 2)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.sqrt(torch.mean(loss))

class MaskedMAPE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~(torch.abs(labels) < 1e-6)
        mask = mask.to(torch.float32).to(preds.device)
        labels = labels + (1-mask)
        loss = torch.abs(torch.divide(torch.subtract(preds, labels), labels))
        loss = loss * mask 
        return torch.mean(loss)

class RMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        loss = torch.pow(preds - labels, 2)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.sqrt(torch.mean(loss))

class MAPE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        loss = torch.abs(torch.divide(torch.subtract(preds, labels), labels))
        loss = torch.nan_to_num(loss)
        return torch.mean(loss)

class DMVSTNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mape = MaskedMAPE()
        self.rmse = RMSE()
    
    def forward(self, preds, labels):
        return self.rmse(preds, labels) + 10 * self.mape(preds, labels)