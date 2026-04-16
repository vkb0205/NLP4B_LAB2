# Banking Intent - Quick Workflow

## 1) Prepare Data

```powershell
python scripts/preprocess_data.py --sample-size 4000 --output-dir sample_data
```

## 2) Run EDA

```powershell
python scripts/eda.py --data-dir sample_data --output-dir reports/eda --top-n 25
```

This generates:

- `reports/eda/report.md` with summary tables
- label distribution CSVs for train and test
- label share gap between train and test
- text length statistics for each split
- top word frequency tables for each split
