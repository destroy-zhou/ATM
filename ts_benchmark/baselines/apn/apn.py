"""Regular-series adapter for https://github.com/decisionintelligence/APN.

The regular-series variant retains globally learned per-channel boundaries and
time-aware patch aggregation, followed by a Flatten + Linear forecasting head.
This is an adaptation of APN, not the original irregular-series architecture.
"""
import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.apn.models.APN import RegularAPNModel


MODEL_HYPER_PARAMS = {
    'seq_len': 96, 'horizon': 96, 'enc_in': 1,
    'd_model': 32, 'dropout': 0.1,
    'apn_npatch': 48, 'apn_te_dim': 8, 'apn_tau_ratio': 0.25,
    # These two fields are required by the official constructor but unused
    # in its forward computation; they do not control depth or multi-head attention.
    'apn_nlayer': 1, 'apn_attn_heads': 1,
    'batch_size': 64, 'lr': 0.001, 'lradj': 'type3',
    'loss': 'MSE', 'num_epochs': 40, 'patience': 6,
    'num_workers': 0, 'norm': True, 'use_amp': 0,
    # History-window normalization, separate from the training-set StandardScaler.
    # No learnable affine transform: matches the validated normalization ablation.
    'revin': True,
}


class APN(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return 'APN'

    def _init_model(self):
        config = self.config
        if config.d_model < 2 or config.d_model % 2:
            raise ValueError('APN d_model must be a positive even integer')
        if config.apn_te_dim < 2 or config.apn_npatch < 1:
            raise ValueError('apn_te_dim must be >= 2 and apn_npatch must be positive')
        if config.seq_len < 1 or config.pred_len < 1:
            raise ValueError('seq_len and horizon must be positive')
        return RegularAPNModel(config)

    def _process(self, input, target, input_mark, target_mark):
        # Do not pass calendar feature zero as time: the official TAPA expects
        # a monotonically increasing scalar on the same scale as its boundaries.
        # A regular sequence is the fully observed special case of irregular data.
        batch, length, _ = input.shape
        observed = torch.isfinite(input)
        values = torch.where(observed, input, torch.zeros_like(input))
        if self.config.revin:
            # Statistics use only observed history, independently per sample/channel.
            # Stop gradients through statistics and accumulate half inputs in fp32.
            with torch.no_grad():
                stats = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
                count = observed.sum(dim=1, keepdim=True).clamp_min(1)
                mean = stats.sum(dim=1, keepdim=True) / count
                centered = torch.where(observed, stats-mean, torch.zeros_like(stats))
                scale = (centered.square().sum(dim=1, keepdim=True)/count + 1e-5).sqrt()
                mean, scale = mean.to(values.dtype), scale.to(values.dtype)
            values = torch.where(observed, (values-mean)/scale, torch.zeros_like(values))
        history_time = torch.arange(length, device=input.device, dtype=input.dtype) / length
        history_time = history_time.view(1, length, 1).expand(batch, -1, -1)
        output = self.model(values, history_time, observed.to(input.dtype))
        if self.config.revin:
            output = output*scale + mean
        return {'output': output}
