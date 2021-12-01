import torch 

def RMSELoss(yhat,y):
    return torch.sqrt(torch.nn.functional.mse_loss(yhat, y))