import torch
import torch.nn as nn
from einops import rearrange

class BackboneLossModel(nn.Module):
    def __init__(self, backbone, loss_func):
        super(BackboneLossModel, self).__init__()
        self.backbone = backbone
        self.loss_func = loss_func

    def forward(self, x_seq, y_seq, **kwargs):
        return self.compute_loss(x_seq, y_seq, **kwargs)

    def compute_loss(self, x_seq, y_seq, *args, **kwargs):
        """compute loss between bakcbone output and y_seq
        Args:
            x_seq scaled ts, shape: [batch_size, data_dim, in_len]
            y_seq: scaled target ts, shape: [batch_size, data_dim, out_len]

        Returns:
            loss between predicted y_seq and y_seq
        """

        dec_condition = self.backbone(x_seq, *args, **kwargs)
        loss = self.loss_func.compute_loss(y_seq, dec_condition, *args, **kwargs)

        return loss

    @torch.no_grad()
    def predict(self, x_seq, *args, **kwargs):
        """sample from the model
        Args:
            x_seq scaled ts, shape: [batch_size, data_dim, in_len]

        Returns:
            sampled y_seq
        """
        batch_size = x_seq.shape[0]
        dec_condition = self.backbone(x_seq, *args, **kwargs)

        sample_result = self.loss_func.predict(dec_condition, *args, **kwargs)
        deterministic_pred, multi_mode_pred, raw_samples = sample_result

        return deterministic_pred, multi_mode_pred, raw_samples
