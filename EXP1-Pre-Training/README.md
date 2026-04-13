# NeuralCDE Multi-Lake Pre-Training

This project implements a reproducible NeuralCDE experiment stack for multi-lake pretraining on irregular lake ice observations. The prediction target is `total_ice_m`.

## Why NeuralCDE here

NeuralCDE is a natural fit for this task because the observations are irregularly sampled and each sample is a path through time rather than a fixed tabular row. The model consumes a continuous interpolation of the observed path, which keeps the ordering, spacing, and local dynamics of the observations.

For this dataset, forcing everything onto a uniform grid would blur short windows, distort long gaps, and inject extra interpolation assumptions before the model ever sees the data. Here we preserve the true observation times and let the controlled differential equation operate on the original irregular path.

The baseline intentionally does not use a lake embedding. For source pretraining we want a clean reference that relies on meteorology, seasonality, geography, and the irregular path itself rather than a strong identity feature that can reduce cross-lake generalization. A later fine-tuning stage can add lake-specific embeddings or adapters if that becomes useful.

## Default Features

The default input channels are:

1. `relative_time`
2. `doy_sin`
3. `doy_cos`
4. `latitude`
5. `longitude`
6. `Ten_Meter_Elevation_Wind_Speed_meterPerSecond`
7. `Air_Temperature_celsius`
8. `Relative_Humidity_percent`
9. `Shortwave_Radiation_Downwelling_wattPerMeterSquared`
10. `Longwave_Radiation_Downwelling_wattPerMeterSquared`
11. `Sea_Level_Barometric_Pressure_pascal`
12. `Surface_Level_Barometric_Pressure_pascal`
13. `Precipitation_millimeterPerDay`
14. `Snowfall_millimeterPerDay`

`total_ice_m` is never used as an input feature in the baseline.

## Experiment Layout

- `Exp-P0`: baseline NeuralCDE with Hermite interpolation
- `Exp-P1`: interpolation comparison (`hermite`, `linear`, `rectilinear`)
- `Exp-P2`: solver comparison (`dopri5`, `rk4`)
- `Exp-P3`: window length comparison (`7d`, `14d`, `30d`)
- `Exp-P4`: target transform comparison (`none`, `log1p`)
- `Exp-P5`: leave-one-lake-out cross-validation

## Project Structure

```text
EXP1-Pre-Training/
  configs/
  data/
  outputs/
  scripts/
  src/lakeice_ncde/
  tests/
```

## Environment

The expected Python environment is `SCI`.

```powershell
conda activate SCI
python -V
```

## Install

```powershell
conda activate SCI
cd S:\STU-Papers\My_Papers\Model-level-Paper-EXP2\EXP1-Pre-Training
python -m pip install -r requirements.txt
python -m pip install -e .
```

## End-To-End Commands

Validate the raw Excel file:

```powershell
python scripts/00_validate_data.py --config configs/experiments/exp_p0_baseline.yaml
```

Prepare the standardized dataframe:

```powershell
python scripts/01_prepare_dataframe.py --config configs/experiments/exp_p0_baseline.yaml
```

Create the default lake-aware split:

```powershell
python scripts/02_make_splits.py --config configs/experiments/exp_p0_baseline.yaml
```

By default, `split.seed: null` means each run creates a new random train/val/test split and saves it under a unique split directory. If you want the exact same split every time, set a fixed integer seed in [configs/base/split.yaml](S:/STU-Papers/My_Papers/Model-level-Paper-EXP2/EXP1-Pre-Training/configs/base/split.yaml).

If you create a split first and then want later steps to reuse that exact split, pass its directory name with `--split-name` to `03_build_windows.py`, `04_precompute_coeffs.py`, or `05_train.py`.

Build irregular lookback windows:

```powershell
python scripts/03_build_windows.py --config configs/experiments/exp_p0_baseline.yaml
```

Precompute interpolation coefficients:

```powershell
python scripts/04_precompute_coeffs.py --config configs/experiments/exp_p0_baseline.yaml
```

Run a debug training pass:

```powershell
python scripts/05_train.py --config configs/experiments/exp_p0_baseline.yaml --override configs/debug/debug_quick.yaml
```

Run the default training:

```powershell
python scripts/05_train.py --config configs/experiments/exp_p0_baseline.yaml
```

Run a lower-memory formal baseline with an explicit YAML config:

```powershell
python scripts/05_train.py --config configs/experiments/exp_p0_formal_stable.yaml
```

Evaluate a finished run:

```powershell
python scripts/06_evaluate.py --run-dir outputs/runs/exp_p0_baseline/<run_timestamp>
```

Regenerate all figures and the PDF report:

```powershell
python scripts/07_plot_results.py --run-dir outputs/runs/exp_p0_baseline/<run_timestamp>
```

Run Optuna tuning:

```powershell
python scripts/08_tune_optuna.py --config configs/tuning/optuna_baseline.yaml
```

Run leave-one-lake-out experiments:

```powershell
python scripts/09_run_lolo_cv.py --config configs/experiments/exp_p5_lolo.yaml
```

Run the smoke test:

```powershell
python scripts/99_smoke_test.py --config configs/experiments/exp_p0_baseline.yaml
```

## Outputs

Each run writes to its own directory under `outputs/runs/`. A successful run includes:

- `config_merged.yaml`
- `metrics.csv`
- `epoch_summary.csv`
- `best.ckpt`
- `latest.ckpt`
- `val_predictions.csv`
- `test_predictions.csv`
- `per_lake_metrics.csv`
- `run_summary.json`
- `figures/*.png`
- `figures/report.pdf`

The global experiment registry is saved to:

```text
outputs/runs/experiment_registry.csv
```

This file records experiment settings and outcomes so later tuning can build on earlier runs instead of starting from scratch.

## Fine-Tune Direction

This pretraining project is designed to be extended to target-lake fine-tuning later. The intended path is:

1. Keep the same preprocessing, windowing, and coefficient pipeline.
2. Load the pretrained checkpoint from this project.
3. Swap the split strategy to target-lake specific train/val/test.
4. Optionally add a lake embedding, lightweight adapter, or partial layer freezing.
5. Reuse the same evaluation and plotting stack to compare pretraining transfer against a from-scratch target-lake baseline.

## Method Sources

- NeuralCDE repository: [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE)
- torchcde repository: [patrick-kidger/torchcde](https://github.com/patrick-kidger/torchcde)
