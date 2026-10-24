# Regime-Gated Forecasting (RGF) for station-level PM2.5 prediction over India

Implementation code, quality-control rules, station list, fixed chronological splits and the
computed result tables for the paper

> Malge R, Swamy S, Sheshappa SN, Gupta S. *Regime-gated PM2.5 forecasting with regime-conditional
> conformal calibration for Indian monitoring stations.* Artificial Intelligence in Health
> (under review), 2026.

RGF forecasts hourly PM2.5 at 1, 6, 12 and 24 h ahead for every Central Pollution Control Board
(CPCB) station in the benchmark. The pollution regime at issue time conditions every stage:

1. **Fuzzy regime gating.** Three gradient-boosted experts (low / mid / high regime) are blended by
   piecewise-linear memberships with transition bands 50-70 and 220-280 ug/m3 around the Indian AQI
   breakpoints of 60 and 250 ug/m3 (`src/rgdf.py::memberships`).
2. **Severity-weighted expert risk.** Each expert is fit with sample weight
   `membership x min(1 + (y/250)^2, 20)` (`src/rgdf.py`).
3. **Regime-conditional conformal calibration.** Gated 5 % / 95 % quantile heads are calibrated on
   2024 per issue-time regime (Mondrian CQR), so 90 % coverage holds inside every regime
   (`src/conformal.py`).

All numbers in the paper come from this pipeline and the public CPCB archive; nothing is
hand-entered. The configuration reported as "RGF" was chosen on the 2024 calibration year only
(`src/evaluate.py::select_proposed`, `results/proposed.json`).

## Repository layout

```
src/        pipeline and analysis scripts
results/    computed metric tables, station list, QC statistics, training logs
data/       empty; raw and processed data are rebuilt locally (see data/README.md)
```

Not included (rebuilt by the pipeline): raw and QC'd panels (`data/`, about 1.6 GB), per-horizon
prediction parquets (`results/preds_h*.parquet`, about 0.2 GB) and model checkpoints.

## Data and quality control

Source: CPCB continuous monitoring data as republished by the
[india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi) archive (ODbL 1.0; data (c) CPCB).
`src/extract.py` reads the candidate-city rows of the yearly release parquets over HTTP with DuckDB.

| Rule | Definition |
|---|---|
| Q1 physical range | PM2.5 outside (0, 1500] ug/m3 set to missing |
| Q2 stuck sensor | runs of >= 8 identical consecutive non-zero 15-min values set to missing |
| Q3 hourly aggregation | mean of 15-min values; an hour is valid only if >= 2 of 4 quarter-hours survive |
| Q4 station inclusion | >= 60 % valid hours in 2019-2023, in 2024 and in 2025, each separately |

Result: 90 of 182 candidate stations in 15 cities, 4,707,587 valid station-hours
(`results/station_inclusion.csv`, `results/qc_stats.csv`).

| Split | Period | Use |
|---|---|---|
| train | 2019-01-08 to 2023-12-31 | fitting all models |
| calibration | 2024 | conformal calibration, early stopping, configuration selection |
| test | 2025 | single evaluation pass after all design decisions were frozen |

Issue times near split boundaries are trimmed by the horizon so no target crosses a split.

## Reproducing the results

Tested with Python 3.8 on Windows 11 (`requirements.txt`; LightGBM 4.6, PyTorch 1.9 + CUDA 11.1,
RTX 2060). Everything except the deep-learning baselines runs on the CPU.

```bash
pip install -r requirements.txt
python src/run_pipeline.py                 # main pipeline, stages 1-10 below
python src/run_pipeline.py --revision      # revision analyses, stages R1-R8 below
python src/run_pipeline.py --from rgdf.py  # resume from a stage
python src/run_pipeline.py --only evaluate.py
```

Main pipeline:

| # | Script | What it does | Approx. time |
|---|---|---|---|
| 1 | `extract.py` | download candidate-city rows for 2019-2025 into `data/raw/` | 15 min (network) |
| 2 | `qc_hourly.py` | QC rules Q1-Q4, hourly aggregation, station selection | 3 min |
| 3 | `train_eval.py` | features, statistical baselines, ridge, GBM univariate, unweighted fusion GBM | 50 min |
| 4 | `patch_seasonal_h6.py` | seasonal-naive column for h = 6 | seconds |
| 5 | `train_weighted.py` | severity-weighted single GBM (ablation "weighting only") | 30 min |
| 6 | `rgdf.py` | RGF experts, decomposition variant, gated quantile heads | 155 min |
| 7 | `dl_baselines.py` | LSTM and TCN baselines (GPU) | 15 min |
| 8 | `transition_analysis.py` | hard-gated variant and behaviour near regime transitions | 30 min |
| 9 | `evaluate.py` | configuration selection on 2024, all point / episode / warning / season / regime metrics | 1 min |
| 10 | `conformal.py` | raw, split-conformal, marginal CQR and regime-conditional CQR intervals | 1 min |

`qc_hourly_lowmem.py` is a drop-in replacement for stage 2 with identical outputs for machines with
less than about 16 GB of free RAM.

Revision analyses (added in response to review):

| # | Script | What it does | Paper |
|---|---|---|---|
| R1 | `cost_loss.py` | relative economic value of exceedance warnings vs cost-loss ratio | Table 7 |
| R2 | `sensitivity.py` | band half-widths x0.5 / x2, ramp exponent 1 / 3, cap 10 / none at 24 h | Table 11 |
| R3-R5 | `dl_baselines_v2.py --models lstm2 / tcn2 / transformer` | re-tuned LSTM and TCN (full passes, patience 5, LR schedule, deeper) and a patch-based Transformer encoder; checkpoints every epoch and resumes if interrupted | Section 3.12 |
| R6 | `dl_baselines_v2.py --merge` | append `lstm2`, `tcn2`, `transformer` to the prediction files | |
| R7 | `metrics_subset.py` | point, Severe-hour, escalation and warning metrics for the new columns | Tables 2, 5, 13 |
| R8 | `significance.py` | Diebold-Mariano tests (HAC, bandwidth h + 24) and stationary block-bootstrap 95 % intervals (mean block 168 h, 1,000 replicates) for MAE, Severe MAE and CSI | Table 12, Section 3.12 |

The reported Transformer was trained with `--amp` (mixed precision) for epochs 1-5, which include
its best checkpoint (epoch 3), and resumed in full precision for epochs 6-8 after CUDA
illegal-memory-access faults with PyTorch 1.9 on the RTX 2060. Full precision is the default.

Helpers: `revision_minimal.py` rebuilds only the prediction columns needed by R2-R8 (persistence,
unweighted fusion GBM and RGF at 6 and 24 h) when memory or time is limited; `repro_check.py`
compares rerun predictions with the stored result tables; `sanity_check.py` summarises a raw file.

Seeds are fixed (`random_state=42` for LightGBM, `torch.manual_seed(42)`). A full rerun from the
public archive reproduces the gradient-boosted results to within 0.01 ug/m3 in MAE. The LSTM and
TCN reproduce closely but not bit-for-bit, because GPU training is not deterministic.

### Model column names

Prediction columns use the working name `rgdf`. Mapping to the paper:

| Column | Name in the paper |
|---|---|
| `persistence`, `seasonal_naive`, `climatology`, `ridge` | statistical and linear baselines |
| `lgbm_uni` | GBM univariate |
| `stpf` | unweighted fusion GBM |
| `stpf_w` | weighting only (ablation) |
| `rgdf_nogate` | weighting + decomposition (ablation) |
| `rgdf_nodecomp` | **RGF, the proposed configuration** (weighting + fuzzy gating) |
| `rgdf` | weighting + gating + decomposition (ablation) |
| `rgdf_hard` | hard-gated variant (transition analysis) |
| `rgdf_q05`, `rgdf_q95` | gated quantile heads used by `conformal.py` |
| `lstm`, `tcn` | LSTM and TCN baselines |
| `lstm2`, `tcn2`, `transformer` | re-tuned LSTM, re-tuned TCN, patch Transformer |
| conformal `raw`, `scp`, `cqr`, `mond` | raw quantiles, split conformal, marginal CQR, regime-conditional CQR |

## Main results (2025 test year)

MAE in ug/m3; Severe = observed concentration above 250 ug/m3; CSI = critical success index of
24 h exceedance warnings at 250 ug/m3. Rows from `results/point_metrics.csv`,
`results/episode_metrics.csv`, `results/exceedance_metrics.csv` and, for the re-tuned models,
`results/metrics_subset.csv` / `results/exceedance_subset.csv`.

| Model | MAE 1 h | MAE 6 h | MAE 12 h | MAE 24 h | Severe MAE 6 h | Severe MAE 24 h | CSI 24 h |
|---|---|---|---|---|---|---|---|
| Persistence | 9.20 | 28.05 | 32.91 | 25.20 | 124.5 | 115.5 | 0.361 |
| Ridge | 8.25 | 23.13 | 24.39 | 23.69 | 105.9 | 113.6 | 0.361 |
| GBM univariate | 7.44 | 18.84 | 20.93 | 21.97 | 98.1 | 117.8 | 0.346 |
| Unweighted fusion GBM | 7.23 | 18.08 | 20.36 | 21.66 | 93.3 | 116.8 | 0.354 |
| LSTM | 7.64 | 18.56 | 20.46 | 21.74 | 105.9 | 120.3 | 0.343 |
| TCN | 7.60 | 17.91 | 20.14 | 21.84 | 92.3 | 107.1 | 0.391 |
| LSTM (re-tuned) | 7.64 | 18.46 | 20.91 | 22.75 | 89.9 | 106.2 | 0.389 |
| TCN (re-tuned) | 7.59 | 18.78 | 21.30 | 23.21 | 100.0 | 123.6 | 0.305 |
| Transformer (patch) | 7.87 | 17.99 | 20.43 | 22.03 | 93.0 | 116.6 | 0.338 |
| **RGF (proposed)** | 7.19 | 18.52 | 21.11 | 22.62 | 82.9 | 103.0 | 0.401 |

Significance (`results/significance.csv`): RGF's Severe-hour MAE is significantly lower than every
baseline at 6 h (p < 0.001). At 24 h it is significantly lower than all except the re-tuned LSTM
(difference -2.9 ug/m3, 95 % interval -9.1 to +2.6, p = 0.09), and its CSI is not significantly
different from the TCN or the re-tuned LSTM.

Coverage of nominal 90 % intervals at 24 h by issue-time regime (`results/conformal_summary.csv`;
coverage % / median width in ug/m3):

| Method | <=60 | 60-120 | 120-250 | >250 | Overall |
|---|---|---|---|---|---|
| Split conformal | 98.9 / 91 | 91.5 / 116 | 66.3 / 116 | 41.8 / 116 | 91.0 / 106 |
| Marginal CQR | 93.1 / 49 | 89.3 / 98 | 86.9 / 186 | 83.2 / 267 | 91.1 / 65 |
| **Regime-conditional CQR (proposed)** | 91.5 / 46 | 90.5 / 101 | 90.0 / 202 | 93.4 / 361 | 91.2 / 63 |

Other tables in `results/`: per-city and per-season MAE, issue-time regime and escalation / decay
subsets, transition-band analysis, feature importance, sensitivity, cost-loss value, training curves
and training / inference cost.

## Citation

See `CITATION.cff`. Please also credit the data source: CPCB via the india-cpcb-aqi archive (ODbL 1.0).

## License

Code: MIT License (`LICENSE`). The monitoring data remain subject to the CPCB terms and the ODbL 1.0
license of the india-cpcb-aqi archive.
