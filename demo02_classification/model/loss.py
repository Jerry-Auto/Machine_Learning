import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSE_loss(output, target):
    return nn.MSELoss(reduction='mean')(output, target)

def CrossEntropy_loss(output,target):
    return nn.CrossEntropyLoss() (output, target)

