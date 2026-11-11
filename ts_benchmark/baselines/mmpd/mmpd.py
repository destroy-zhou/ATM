"""TFB adapter for https://github.com/Thinklab-SJTU/MMPD.

Uses the official decoder backbone, weighted diffusion objective and point
prediction. Probabilistic GMM outputs are outside TFB's point forecast interface.
"""
import math

import torch
from torch import nn

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.mmpd.models.backbone_loss_model import BackboneLossModel
from ts_benchmark.baselines.mmpd.models.backbones.decoder_only_transformer import DecoderOnlyTransformer
from ts_benchmark.baselines.mmpd.models.loss_funcs.mmpd.mmpd_loss import MMPD_Loss
from ts_benchmark.baselines.mmpd.normalization import get_statistics, normalize, denormalize


MODEL_HYPER_PARAMS = {
    'seq_len': 96, 'horizon': 96, 'label_len': 48,
    'enc_in': 1, 'patch_size': 12,
    'd_model': 256, 'd_ff': 512, 'n_heads': 4, 'd_layers': 2,
    'dropout': 0.2, 'point_weight': 0.01, 'weighted': True,
    'd_diffusion': 256, 'diffusion_layers': 1, 'max_diffusion_steps': 1000,
    'beta_schedule': 'linear', 'radius': 3, 'num_sampling_steps': '20',
    'batch_size': 32, 'lr': 0.0001, 'lradj': 'cosine',
    'num_epochs': 20, 'patience': 5, 'num_workers': 1,
    'norm': True, 'loss': 'MMPD', 'use_amp': 0,
}


class _DiffusionOnlyCriterion(nn.Module):
    """The training loop receives the complete MMPD loss as additional_loss."""

    def forward(self, output, target):
        return output.new_zeros(())


class MMPD(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)
        if self.config.loss != 'MMPD':
            raise ValueError('MMPD requires loss="MMPD" to retain its diffusion objective')

    @property
    def model_name(self):
        return 'MMPD'

    def _init_model(self):
        config = self.config
        config.in_len = config.seq_len
        config.out_len = config.pred_len
        config.data_dim = config.enc_in
        config.num_sampling_steps = str(config.num_sampling_steps)
        if config.patch_size < 1 or config.radius < 1:
            raise ValueError('patch_size and radius must be positive')
        if config.d_model % config.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if config.d_diffusion < 2 or config.d_diffusion % 2:
            raise ValueError('d_diffusion must be a positive even integer')
        if not 0 <= config.point_weight <= 1:
            raise ValueError('point_weight must be between 0 and 1')
        return BackboneLossModel(DecoderOnlyTransformer(config), MMPD_Loss(config))

    def _init_criterion_and_optimizer(self):
        return _DiffusionOnlyCriterion(), torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _adjust_lr(self, optimizer, epoch, config):
        if config.lradj != 'cosine':
            return super()._adjust_lr(optimizer, epoch, config)
        # Official cosine schedule, with a 1% learning-rate floor.
        lr = config.lr * (0.99 * 0.5 * (1 + math.cos(math.pi * epoch / config.num_epochs)) + 0.01)
        for group in optimizer.param_groups:
            group['lr'] = lr

    def _compute_loss(self, input, target):
        x = input.transpose(1, 2)
        y = target[:, -self.config.pred_len:, :].transpose(1, 2)
        shift, scale = get_statistics(x)
        x, y = normalize(x, shift, scale), normalize(y, shift, scale)
        # Call forward through DataParallel when it is used by the base class.
        loss = self.model(x, y, point_weight=self.config.point_weight)
        if self.config.weighted:
            loss = scale.square() * loss
        return loss.mean()

    def _process(self, input, target, input_mark, target_mark):
        if self.model.training:
            loss = self._compute_loss(input, target)
            # Shape-only placeholder for the base loop's slicing. No point MSE
            # is added: the sole optimization objective is the official loss.
            output = input.new_zeros(input.shape[0], self.config.pred_len, input.shape[2])
            return {'output': output, 'additional_loss': loss}
        x = input.transpose(1, 2)
        shift, scale = get_statistics(x)
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        predicted, _, _ = module.predict(normalize(x, shift, scale), prob_pred=False)
        return {'output': denormalize(predicted, shift, scale).transpose(1, 2)}

    def validate(self, valid_data_loader, series_dim, criterion):
        # Official early stopping uses diffusion loss, not point prediction MSE.
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        losses = []
        try:
            with torch.no_grad():
                for input, target, _, _ in valid_data_loader:
                    losses.append(self._compute_loss(input.to(device), target.to(device)).item())
        finally:
            self.model.train(was_training)
        return sum(losses) / len(losses) if losses else float('inf')
