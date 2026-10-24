Data are not committed. Run `python src/extract.py` to download the candidate-city rows of the
india-cpcb-aqi yearly releases (2019-2025) into `data/raw/`, then `python src/qc_hourly.py` to build
`data/processed/hourly_pm25.parquet` (QC'd hourly panel of the retained stations).

Source: https://github.com/Vonter/india-cpcb-aqi (ODbL 1.0). Data (c) Central Pollution Control Board.
