# Experiment Summary Script

This repo writes per-run outputs under `workspace/tools/experiments/experiment_*`.

## Generate summary + plots

1) (If needed) install plotting deps:

```bash
bash workspace/tools/setup_plot_env.sh
```

2) Generate summary CSV/JSON + plots:

```bash
python3 workspace/tools/summarize_experiments.py \
  --root workspace/tools/experiments \
  --out workspace/tools/experiments_summary \
  --per-experiment-plots
```

## Outputs

- `workspace/tools/experiments_summary/summary.csv`: one row per experiment, flattened metrics.
- `workspace/tools/experiments_summary/experiments.json`: full structured data.
- `workspace/tools/experiments_summary/REPORT.md`: quick table + embedded figures.
- `workspace/tools/experiments_summary/plots/`: cross-experiment plots.
- `workspace/tools/experiments_summary/plots/timeseries/`: per-experiment time series (optional).

