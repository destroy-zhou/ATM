import torch
import torch.nn.functional as F

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.timemosaic.models.TimeMosaic import Model


MODEL_HYPER_PARAMS = {
    "task_name": "TimeMosaic",
    "is_training": 1,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "seq_len": 96,
    "label_len": 48,
    "horizon": 96,
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "factor": 3,
    "dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "activation": "gelu",
    "channel": "CI",
    "num_latent_token": 4,
    "patch_len_list": "[8,16,32]",
    "pre96": 32,
    "pre192": 64,
    "pre336": 168,
    "pre720": 240,
    "pre12": 6,
    "mask_ratio": 0.0,
    "mask_ratio_patch": 0.0,
    "patch_len": 16,
    "stride": 8,
    "route_balance_weight": 0.001,
    "mask_reconstruction_weight": 0.001,
    "batch_size": 32,
    "lr": 0.0001,
    "lradj": "type1",
    "loss": "MSE",
    "num_epochs": 10,
    "patience": 3,
    "num_workers": 1,
    "norm": True,
    "use_amp": 0,
}


class TimeMosaic(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeMosaic"

    def _init_model(self):
        return Model(self.config)

    def _model_module(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _build_mask(self, input):
        if self.config.mask_ratio > 0:
            return torch.rand_like(input) < self.config.mask_ratio
        if self.config.mask_ratio_patch > 0:
            module = self._model_module()
            batch, _, channels = input.shape
            mask = torch.rand(
                batch * channels,
                module.patch_num,
                device=input.device,
            ) < self.config.mask_ratio_patch
            return mask.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        return None

    def _process(self, input, target, input_mark, target_mark):
        dec_inp = torch.zeros_like(target[:, -self.config.pred_len :, :]).float()
        dec_inp = torch.cat(
            [target[:, : self.config.label_len, :], dec_inp], dim=1
        ).to(input.device)

        mask = self._build_mask(input) if self.model.training else None
        result = self.model(input, input_mark, dec_inp, target_mark, mask)

        if not self.model.training:
            output, _ = result
            return {"output": output}

        output, _, reconstructed = result
        module = self._model_module()
        assignment = module.patch_embedding.latest_cls_soft
        current_ratio = assignment.mean(dim=0)
        target_ratio = torch.full_like(current_ratio, 1.0 / current_ratio.numel())
        auxiliary_loss = self.config.route_balance_weight * F.mse_loss(
            current_ratio, target_ratio
        )

        if reconstructed is not None and mask is not None and mask.any():
            auxiliary_loss = auxiliary_loss + self.config.mask_reconstruction_weight * F.mse_loss(
                reconstructed[mask], input[mask]
            )

        return {"output": output, "additional_loss": auxiliary_loss}
