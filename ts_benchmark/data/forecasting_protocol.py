"""SEER (arXiv:2602.00589v1), Table 1, main multivariate benchmark.

Lengths count time points, not rows in a long-format CSV. Raw files are retained;
ETT uses the chronological prefix. Exchange is an explicit user-specified extension;
datasets not listed below are unchanged.
Ratios are implemented by rolling_forecast_config.json (tv=.8, train/tv below).
"""

SEER_DATASETS = {
    'ETTh1.csv': dict(length=14400, channels=7, freq='hourly', train_ratio_in_tv=.75),
    'ETTh2.csv': dict(length=14400, channels=7, freq='hourly', train_ratio_in_tv=.75),
    'ETTm1.csv': dict(length=57600, channels=7, freq='other', train_ratio_in_tv=.75),
    'ETTm2.csv': dict(length=57600, channels=7, freq='other', train_ratio_in_tv=.75),
    'Weather.csv': dict(length=52696, channels=21, freq='other', train_ratio_in_tv=.875),
    'Electricity.csv': dict(length=26304, channels=321, freq='hourly', train_ratio_in_tv=.875),
    'Solar.csv': dict(length=52560, channels=137, freq='other', train_ratio_in_tv=.75),
    'Traffic.csv': dict(length=17544, channels=862, freq='hourly', train_ratio_in_tv=.875),
    # User-specified extension, not a dataset in SEER Table 1.
    'Exchange.csv': dict(length=7588, channels=8, freq='daily', train_ratio_in_tv=.875),
}


def apply_seer_protocol(series_name, data):
    spec = SEER_DATASETS.get(series_name)
    if spec is None:
        return data
    if len(data) < spec['length']:
        raise ValueError(f"{series_name}: SEER requires {spec['length']} time points, got {len(data)}")
    if data.shape[1] != spec['channels']:
        raise ValueError(f"{series_name}: SEER requires {spec['channels']} channels, got {data.shape[1]}")
    return data.iloc[:spec['length']].copy()
