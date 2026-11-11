# -*- coding: utf-8 -*-
"""ATM adapter for the ts_benchmark forecasting interface."""

import torch
import torch.nn as nn

from ts_benchmark.baselines.atm.models.CompactATM import CompactATM
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)


MODEL_HYPER_PARAMS = {
    # DeepForecastingModelBase parameters.
    "use_amp": 0,
    "loss": "MAE",
    "batch_size": 32,
    "lradj": "type3",
    "lr": 0.00025,
    "num_epochs": 50,
    "num_workers": 1,
    "patience": 5,
    "norm": True,
    "grad_clip": 1.0,
    # Compatibility flags: only the current compact lightweight ATM is supported.
    "lightweight": True,
    "compact_patches": True,
    # Non-overlapping input units; 1 preserves point-wise tokenization.
    "small_patch_len": 4,
    # Share content embedding with the boundary MLP (otherwise separate Linear).
    "small_patch_share_embedding": False,
    "patch_mode": "adaptive_per_channel",
    "patch_len": 16,
    "patch_bins": 4,
    "patch_pooling": "boundary_query",
    # Optional independent K/V maps for boundary-query pooling.
    "patch_query_use_kv_projection": True,
    "patch_min_len": 1,
    "patch_max_len": 96,
    "boundary_threshold": 0.5,
    "boundary_temperature": 0.5,
    "boundary_ste_scale": 1,
    "use_ordinal_st": False,
    "boundary_network": "time_mlp",
    # Time-MLP readout: use h_t alone instead of [h_t, h_(t+1), difference].
    "boundary_mlp_point_head": False,
    # False: pooled patch -> one Linear -> broadcast, without scalar statistics.
    "patch_use_statistics": False,
    # Cross-variable mixing in the dense patch-broadcast prediction trunk.
    "broadcast_channel_mlp": True,
    # Initial residual weight for each variable's channel-MLP update.
    "broadcast_channel_scale_init": 1,
    # Learn per-variable residual weights; false keeps them fixed at the initial value.
    "broadcast_channel_scale_learnable": False,
    # 0 uses inter_dim; positive ratios use max(1, floor(seq_len * ratio)).
    "broadcast_time_hidden_ratio": 2.0,
    # Normalize the Time-MLP sublayer over its temporal axis instead of d_model.
    "broadcast_time_norm_t": False,
    # C -> C -> C instead of C -> 2C -> C in the broadcast channel MLP.
    "broadcast_channel_hidden_c": False,
    # 0 retains hidden_c behavior; positive ratios override it with floor(C * ratio).
    "broadcast_channel_hidden_ratio": 2.0,
    # Normalize the channel sublayer over C instead of d_model.
    "broadcast_channel_norm_c": False,
    # All sublayer residuals use the input of their own broadcast encoder block.
    "broadcast_residual_to_input": False,
    # Post-norm for boundary Time MLP and patch-broadcast Time/Channel MLP + FFN.
    "mlp_post_norm": False,
    "boundary_merge_weight": 0.01,
    # dense/sparse_peak: BCE; sparse_rank: ranking; waveform_lobe: half-wave BCE.
    "boundary_merge_mode": "waveform_lobe",
    # Fit-window half-width; waveform_lobe uses it as smoothing radius (2*w+1).
    "boundary_merge_window": 2,
    "boundary_merge_threshold": 0.15,
    "boundary_merge_temperature": 0.05,
    "boundary_semantic_weight": 0.0,
    "boundary_semantic_window": 8,
    "boundary_semantic_threshold": 0.25,
    "boundary_semantic_temperature": 0.03,
    "patch_readout_queries": 4,
    "patch_reconstruction_weight": 0.0,
    "patch_count_weight": 0.0,
    "patch_anchor_weight": 0.0,
    "patch_anchor_count": 128,
    # Total random non-self pair budget per batch, across all channels/buckets.
    # Zero retains the legacy N x patch_anchor_count path.
    "patch_anchor_pair_count": 1024,
    "patch_anchor_threshold": 0.95,
    "patch_anchor_variant": "channel",
    "encoder_layers": 1,
    "encoder_type": "forecast_token_attn",
    # Broadcast self.project patch content to points, followed by Time MLP + FFN.
    "prediction_backbone": "patch_broadcast_mlp",
    # 0 keeps all history points; otherwise adaptive-average-pool to this length
    # immediately before the patch_broadcast_mlp prediction head.
    "prediction_pool_len": 24,
    # 0: legacy D -> 1; (0,1): D -> max(1, floor(D*ratio)); 1: identity.
    "prediction_feature_ratio": 0.5,
    "forecast_patch_len": 24,
    "channel_attention": False,
    # ATM parameters.
    "task_name": "long_term_forecast",
    "label_len": 0,
    "d_model": 32,
    "inter_dim": 32,
    "n_heads": 2,
    # Accept aux_loss=0 in existing scripts; the legacy loss has been removed.
    "aux_loss": 0,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "activation": "gelu",
    "use_rope": False,
}


class ATM(DeepForecastingModelBase):
    """Adapt the repository's ATM model to ts_benchmark."""

    def __init__(self, **kwargs):
        config = dict(MODEL_HYPER_PARAMS)
        config.update(kwargs)
        if not config['lightweight'] or not config['compact_patches']:
            raise ValueError('Legacy ATM has been removed; lightweight and compact_patches must be true')
        if config['aux_loss'] != 0:
            raise ValueError('Legacy aux_loss has been removed; use the current patch/boundary loss weights')
        super().__init__(config)

    @property
    def model_name(self):
        return "ATM"

    @staticmethod
    def required_hyper_params() -> dict:
        # These are filled from rolling_forecast_config.json by the benchmark.
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        }

    def _init_model(self):
        return CompactATM(self.config)

    def _init_criterion_and_optimizer(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        output, auxiliary_loss = self.model(input)
        return {"output": output, "additional_loss": auxiliary_loss}
