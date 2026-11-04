# Second test year: rolling-origin fold (train 2019-2022, calibrate 2023, test 2024)

Replication of the main pipeline with every split boundary moved back by one year, run to test
whether the findings depend on the 2025 test year (paper Section 3.14, Tables 15-17).

| Split | Period | Use |
|---|---|---|
| train | 2019-01-08 to 2022-12-31 | fitting all models |
| calibration | 2023 | conformal calibration, early stopping |
| test | 2024 | single evaluation pass; nothing tuned on 2024 |

Everything else is identical to the main pipeline: the same 90 stations
(`results/station_inclusion_fold1.csv`, the list selected under the main protocol), QC rules,
features, LightGBM hyper-parameters, severity weights, memberships, baselines, metrics and
significance protocol. The configuration reported as RGF (`rgdf_nodecomp`: severity weighting +
fuzzy gating, no decomposition) is fixed in advance rather than re-selected, so `proposed.json`
is not used. Not rebuilt for this fold: the decomposition variants, the hard-gated variant, the
re-tuned TCN and the Transformer. The LSTM and TCN use the configurations of
`src/dl_baselines.py` (main pipeline stage 7); the re-tuned LSTM (`lstm2`) was rebuilt with
`src/dl_baselines_v2.py` (main pipeline stages R3, R6 and R7) and is included in `significance.csv`.

## Scripts (`src/`)

| Script | Difference from the main pipeline |
|---|---|
| `run_fold2.py` | driver; `python fold2/src/run_fold2.py [--from <stage>]` |
| `extract.py` | years 2019-2024 |
| `extract_local.py`, `repair_raw.py` | extract from release files downloaded with `curl` when DuckDB's HTTP range reads stall, and re-download any truncated yearly file |
| `qc_hourly.py` | completeness spans 2019-2022 / 2023 / 2024; station list taken from the main fold |
| `qc_hourly_bycity.py` | drop-in for `qc_hourly.py` with the same rules, processing one city at a time (bounded memory and temporary disk) |
| `train_eval.py`, `train_weighted.py`, `dl_baselines.py` | split dates only |
| `rgdf_fold2.py` | RGF experts and gated q05 / q95 heads only (no decomposition variants) |
| `patch_seasonal_h6.py`, `rgdf.py`, `evaluate.py` | unchanged |
| `conformal.py`, `significance.py` | point column fixed to `rgdf_nodecomp` |
| `dl_baselines_v2.py --models lstm2`, `dl_baselines_v2.py --merge`, `metrics_subset.py` | split dates only; the re-tuned LSTM, its merge into the prediction files and its metrics (`metrics_subset.csv`, `exceedance_subset.csv`) |

Run with Python 3.14, pandas 3.0, LightGBM 4.7, PyTorch 2.13 (CUDA 12.6), DuckDB 2.0 on an
RTX 2070 Super; the only code changes needed for the newer libraries are the lower-case `"1h"`
frequency alias and an explicit float copy in the stuck-sensor check. Wall-clock time on a shared
6-core laptop: about 7 h (QC 3 min, fusion GBMs 83 min, weighting only 75 min, RGF 3 h,
LSTM + TCN 11 min, evaluation and tests 1 min); the re-tuned LSTM took a further 5.2 h on the
shared GPU (six epochs with early stopping, best epoch 1, as in the main fold).

## Result tables (`results/`)

Same files and column names as the main `results/` folder. `logs/` holds the stdout of each
stage. The 2024 test year holds 652,671 station-hours at 24 h, 26,587 of them Severe
(observed concentration above 250 ug/m3).

MAE in ug/m3; Severe = observed concentration above 250 ug/m3; CSI of 24 h exceedance warnings
at 250 ug/m3 (`point_metrics.csv`, `episode_metrics.csv`, `exceedance_metrics.csv`; for the re-tuned LSTM
`metrics_subset.csv`, `exceedance_subset.csv`):

| Model | MAE 1 h | MAE 6 h | MAE 12 h | MAE 24 h | Severe MAE 6 h | Severe MAE 24 h | POD 24 h | FAR 24 h | CSI 24 h |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | 9.85 | 29.44 | 34.44 | 27.31 | 128.0 | 129.6 | 0.531 | 0.469 | 0.361 |
| Ridge | 8.77 | 24.61 | 26.13 | 25.46 | 112.2 | 125.0 | 0.442 | 0.386 | 0.346 |
| GBM univariate | 8.12 | 20.57 | 22.88 | 24.04 | 110.2 | 134.1 | 0.380 | 0.346 | 0.316 |
| Unweighted fusion GBM | 7.81 | 19.60 | 22.17 | 23.71 | 105.2 | 133.7 | 0.372 | 0.317 | 0.317 |
| LSTM | 8.39 | 20.11 | 22.69 | 24.03 | 112.2 | 141.6 | 0.310 | 0.286 | 0.276 |
| TCN | 8.50 | 19.79 | 22.41 | 23.65 | 105.9 | 132.4 | 0.386 | 0.313 | 0.329 |
| LSTM (re-tuned) | 8.44 | 20.22 | 22.73 | 24.19 | 111.2 | 122.9 | 0.460 | 0.406 | 0.350 |
| Weighting only (ablation) | 7.87 | 20.01 | 22.88 | 24.36 | 94.0 | 115.4 | 0.538 | 0.437 | 0.379 |
| **RGF (proposed)** | 7.78 | 20.03 | 22.91 | 24.54 | 95.1 | 115.6 | 0.544 | 0.445 | 0.379 |

Significance at 24 h (`significance.csv`, RGF minus competitor, block-bootstrap 95 % intervals,
Diebold-Mariano p): Severe MAE -16.9 [-22.0, -10.7] (p < 0.001) vs TCN, -26.0 [-30.1, -20.6] vs
LSTM, -7.3 [-12.0, -3.0] (p < 0.001) vs the re-tuned LSTM, -18.1 [-23.1, -12.2] vs unweighted
fusion GBM; CSI +0.050 [+0.021, +0.080] vs TCN and +0.029 [+0.009, +0.052] vs the re-tuned LSTM;
overall MAE +0.90 [+0.24, +1.59] (p = 0.001) vs TCN and +0.36 [-0.05, +0.83] (p = 0.08) vs the
re-tuned LSTM. Gating vs weighting only: -0.09 [-0.15, -0.05] overall MAE at 1 h (p < 0.001),
+0.18 [+0.10, +0.26] at 24 h with no Severe-hour difference (+0.1 [-1.1, +1.3]).

Coverage of nominal 90 % intervals at 24 h by issue-time regime (`conformal_summary.csv`;
coverage % / median width in ug/m3):

| Method | <=60 | 60-120 | 120-250 | >250 | Overall |
|---|---|---|---|---|---|
| Split conformal | 98.7 / 93 | 91.3 / 120 | 67.8 / 120 | 39.9 / 120 | 90.5 / 110 |
| Marginal CQR | 92.4 / 51 | 88.5 / 99 | 86.3 / 179 | 81.0 / 283 | 90.2 / 68 |
| **Regime-conditional CQR (proposed)** | 89.5 / 46 | 91.2 / 108 | 91.0 / 205 | 86.3 / 326 | 90.0 / 67 |

Severe-regime coverage of the regime-conditional intervals at 1, 6, 12 and 24 h: 90.2, 88.3,
88.1 and 86.3 % (marginal CQR 85.1, 79.9, 80.3 and 81.0 %; split conformal 47.6, 44.3, 46.6 and
39.9 %).
