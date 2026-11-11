import pandas as pd
import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.emaformer.model import EMAformerModel


MODEL_HYPER_PARAMS = {
    "seq_len": 96,
    "horizon": 96,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "factor": 1,
    "dropout": 0.1,
    "activation": "gelu",
    "output_attention": False,
    "use_norm": True,
    "cycle": 24,
    "output_proj_dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "patience": 3,
    "lradj": "type1",
    "loss": "MSE",
    "num_workers": 1,
    "norm": True,
    "use_amp": 0,
}


class EMAformer(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "EMAformer"

    def _set_data_properties(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq is None:
            raise ValueError("Irregular time intervals")
        self.config.freq = freq
        channels = train_data.shape[1]
        self.config.enc_in = channels
        self.config.dec_in = channels
        self.config.c_out = channels
        self.config.label_len = self.config.seq_len // 2

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        self._set_data_properties(train_data)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        self._set_data_properties(train_data)

    def _init_model(self):
        return EMAformerModel(self.config)

    @staticmethod
    def _decode_mark(value, scale):
        return torch.round((value + 0.5) * scale).long()

    def _phase_index(self, input_mark):
        """Recover the first forecast step's daily/weekly phase from time features."""
        last = input_mark[:, -1, :]
        feature_count = last.shape[-1]

        if feature_count >= 5:  # minute or second frequency
            offset = 1 if feature_count == 6 else 0
            minute = self._decode_mark(last[:, offset], 59).clamp(0, 59)
            hour = self._decode_mark(last[:, offset + 1], 23).clamp(0, 23)
            steps_per_hour = max(self.config.cycle // 24, 1)
            minute_slot = torch.div(minute * steps_per_hour, 60, rounding_mode="floor")
            phase = hour * steps_per_hour + minute_slot + 1
        elif feature_count == 4:  # hourly frequency
            hour = self._decode_mark(last[:, 0], 23).clamp(0, 23)
            if self.config.cycle == 168:
                weekday = self._decode_mark(last[:, 1], 6).clamp(0, 6)
                phase = weekday * 24 + hour + 1
            else:
                phase = hour + 1
        elif feature_count >= 3:  # daily/business-day frequency
            day_of_year = self._decode_mark(last[:, -1], 365).clamp(0, 365)
            phase = day_of_year + 1
        else:
            phase = torch.zeros(last.shape[0], dtype=torch.long, device=last.device)

        return phase.remainder(self.config.cycle)

    def _process(self, input, target, input_mark, target_mark):
        del target, target_mark
        phase = self._phase_index(input_mark)
        output = self.model(input, phase)
        if isinstance(output, tuple):
            output = output[0]
        return {"output": output}
