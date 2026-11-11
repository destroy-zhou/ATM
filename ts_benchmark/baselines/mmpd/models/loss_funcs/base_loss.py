import torch
import torch.nn as nn

class BaseLossFunc(nn.Module):
    def __init__(self):
        super(BaseLossFunc, self).__init__()

    def compute_loss(self, target_seq, dec_condition, *args, **kwargs):
        raise NotImplementedError("Each loss function must implement the compute loss method")

    @torch.no_grad()
    def predict(self, dec_condition, *args, **kwargs):
        raise NotImplementedError("Each loss function must implement the sample method")
