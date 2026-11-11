"""Standard benchmark CLI, restricted to the last observed channel.

Accepts the same arguments as run_benchmark.py. No experiment manifests or saved
results are needed. Dataset lengths and split rules retain their original names.
"""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_last_channel(data, series_name):
    if data.shape[1] < 1:
        raise ValueError(f"No observed channels in {series_name}")
    selected = data.iloc[:, -1:].copy()
    print("UNIVARIATE_DATA " + json.dumps(dict(
        dataset=series_name, column=str(selected.columns[0]), length=len(selected),
        original_channels=data.shape[1], input_channels=1,
    )), flush=True)
    return selected


def main(arguments=None):
    from ts_benchmark.data.data_source import LocalForecastingDataSource
    from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

    arguments = list(sys.argv[1:] if arguments is None else arguments)
    original_loader = LocalForecastingDataSource._load_series
    original_tuner = DeepForecastingModelBase.single_forecasting_hyper_param_tune
    original_argv = sys.argv

    def load_last_channel(source, name):
        # Apply the ordinary SEER length/channel validation before selecting C=1.
        return select_last_channel(original_loader(source, name), name)

    try:
        LocalForecastingDataSource._load_series = load_last_channel
        # The default single-series tuner uses label_len=horizon, which produces
        # negative dataset slice starts for horizon>seq_len. Preserve shell-run
        # tuning instead: infer C=1, and use label_len=seq_len//2. Model-specific
        # overrides (e.g. EMAformer) remain untouched.
        DeepForecastingModelBase.single_forecasting_hyper_param_tune = (
            DeepForecastingModelBase.multi_forecasting_hyper_param_tune
        )
        sys.argv = [str(ROOT / "scripts/run_benchmark.py")] + arguments
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        LocalForecastingDataSource._load_series = original_loader
        DeepForecastingModelBase.single_forecasting_hyper_param_tune = original_tuner
        sys.argv = original_argv


if __name__ == "__main__":
    main()
