# ATM: Adaptive Tokenization with Lightweight Modeling for Time Series Forecasting

## Quickstart

### Installation

Create a Python environment and install the dependencies:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The requirements use PyTorch with CUDA 12.8. For CPU-only environments, install a CPU version of PyTorch instead.

### Train and evaluate model

Place the dataset CSV files under `./dataset/forecasting/` (e.g., `ETTh1.csv`, `Weather.csv`). Wide-format CSV files should contain a `date` column followed by the numerical variable columns.

Run the following commands with Bash. The scripts use GPU 0 by default; edit `--gpus` to select another GPU.

Multivariate forecasting (nine datasets, horizons 96, 192, 336, and 720):

```shell
bash ./scripts/ATM.sh
```

Univariate forecasting (ETTh2, ETTm2, Exchange, and Solar; the last channel only):

```shell
bash ./scripts/Univariate/ATM.sh
```

To test only one dataset or horizon, run the corresponding command from the script. Results are saved under `./result/` according to `--save-path`.

### Baselines

EMAformer, TimeMosaic, APN, MMPD, SRSNet, DUET, PatchMLP, HDMixer, and PatchTST are also included. Each model has a multivariate script under `scripts/` and a univariate script under `scripts/Univariate/`:

```shell
bash ./scripts/PatchTST.sh
bash ./scripts/Univariate/PatchTST.sh
```

Replace `PatchTST` with the desired model name. GPU IDs and model parameters follow each script; edit `--gpus` as needed.
