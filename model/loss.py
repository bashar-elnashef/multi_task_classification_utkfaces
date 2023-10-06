import torch.nn as nn

class CustomMultiTaskLoss(nn.Module):
  def __init__(self, fns=None):
    super(CustomMultiTaskLoss, self).__init__()
    self.loss_fns = [getattr(nn, fn)() for fn in fns] 

  def forward(self, predictions, target):
      return [loss_fn(predictions[idx], target[idx]) for idx, loss_fn in enumerate(self.loss_fns)]