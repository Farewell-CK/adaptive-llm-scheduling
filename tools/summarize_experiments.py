#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ExperimentPaths:
    exp_dir: Path
    result_csv: Optional[Path]
    metrics_csv: Optional[Path]
    execution_log: Optional[Path]
    monitor_log: Optional[Path]


@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    qps_offered: Optional[float]
    minutes: Optional[int]
    timestamp: Optional[str]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "inf", "+inf", "-inf"}:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _mean(vals: List[float]) -> Optional[float]:
    return float(statistics.fmean(vals)) if vals else None


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return float(statistics.pstdev(vals))


def _parse_experiment_dir_name(name: str) -> ExperimentMeta:
    # Supports:
    # - experiment_qps1_0_min30_20251217_213059
    # - experiment_qps2.5_min40_...
    m = re.match(r"^experiment_qps(?P<qps>[0-9._]+)_min(?P<min>[0-9]+)_(?P<ts>[0-9_]+)$", name)
    if not m:
        return ExperimentMeta(name=name, qps_offered=None, minutes=None, timestamp=None)

    qps_raw = m.group("qps")
    qps = _safe_float(qps_raw.replace("_", "."))
    minutes = _safe_int(m.group("min"))
    ts = m.group("ts")
    return ExperimentMeta(name=name, qps_offered=qps, minutes=minutes, timestamp=ts)


def _find_experiment_paths(exp_dir: Path) -> ExperimentPaths:
    result_csv = next(iter(sorted(exp_dir.glob("result_*.csv"))), None)
    metrics_csv = next(iter(sorted(exp_dir.glob("metrics_*.csv"))), None)
    execution_log = exp_dir / "execution.log"
    monitor_log = exp_dir / "monitor.log"
    return ExperimentPaths(
        exp_dir=exp_dir,
        result_csv=result_csv if result_csv and result_csv.is_file() else None,
        metrics_csv=metrics_csv if metrics_csv and metrics_csv.is_file() else None,
        execution_log=execution_log if execution_log.is_file() else None,
        monitor_log=monitor_log if monitor_log.is_file() else None,
    )


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _summarize_results(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    statuses = [r.get("status", "").strip().lower() for r in rows]
    success_rows = [r for r in rows if r.get("status", "").strip().lower() == "success"]

    ttft_all = [_safe_float(r.get("ttft")) for r in rows]
    e2e_all = [_safe_float(r.get("e2e_latency")) for r in rows]
    ttft_success = [_safe_float(r.get("ttft")) for r in success_rows]
    e2e_success = [_safe_float(r.get("e2e_latency")) for r in success_rows]

    ttft_all_vals = sorted([x for x in ttft_all if x is not None])
    e2e_all_vals = sorted([x for x in e2e_all if x is not None])
    ttft_succ_vals = sorted([x for x in ttft_success if x is not None])
    e2e_succ_vals = sorted([x for x in e2e_success if x is not None])

    submit = [_safe_float(r.get("submit_time")) for r in rows]
    submit_vals = [x for x in submit if x is not None]

    end_times = []
    for r in rows:
        st = _safe_float(r.get("submit_time"))
        lat = _safe_float(r.get("e2e_latency"))
        if st is not None and lat is not None:
            end_times.append(st + lat)
    test_start = min(submit_vals) if submit_vals else None
    test_end = max(end_times) if end_times else None
    wall_s = (test_end - test_start) if (test_start is not None and test_end is not None and test_end >= test_start) else None

    prompt_tokens = [_safe_int(r.get("prompt_tokens")) for r in success_rows]
    completion_tokens = [_safe_int(r.get("completion_tokens")) for r in success_rows]
    prompt_total = sum(x for x in prompt_tokens if x is not None)
    completion_total = sum(x for x in completion_tokens if x is not None)
    tokens_total = prompt_total + completion_total

    success = len(success_rows)
    error = statuses.count("error")
    exception = statuses.count("exception")
    other = total - success - error - exception

    achieved_qps = (total / wall_s) if (wall_s and wall_s > 0) else None
    achieved_success_qps = (success / wall_s) if (wall_s and wall_s > 0) else None
    tokens_per_s = (tokens_total / wall_s) if (wall_s and wall_s > 0 and tokens_total > 0) else None

    return {
        "requests_total": total,
        "requests_success": success,
        "requests_error": error,
        "requests_exception": exception,
        "requests_other": other,
        "success_rate": (success / total) if total else None,
        "test_start_epoch": test_start,
        "test_end_epoch": test_end,
        "test_wall_seconds": wall_s,
        "achieved_qps_total": achieved_qps,
        "achieved_qps_success": achieved_success_qps,
        "prompt_tokens_total": prompt_total if prompt_total else None,
        "completion_tokens_total": completion_total if completion_total else None,
        "tokens_total": tokens_total if tokens_total else None,
        "tokens_per_second": tokens_per_s,
        "ttft_mean_all_s": _mean(ttft_all_vals),
        "ttft_p50_all_s": _percentile(ttft_all_vals, 50),
        "ttft_p90_all_s": _percentile(ttft_all_vals, 90),
        "ttft_p99_all_s": _percentile(ttft_all_vals, 99),
        "ttft_max_all_s": float(ttft_all_vals[-1]) if ttft_all_vals else None,
        "ttft_mean_success_s": _mean(ttft_succ_vals),
        "ttft_p50_success_s": _percentile(ttft_succ_vals, 50),
        "ttft_p90_success_s": _percentile(ttft_succ_vals, 90),
        "ttft_p99_success_s": _percentile(ttft_succ_vals, 99),
        "e2e_mean_all_s": _mean(e2e_all_vals),
        "e2e_p50_all_s": _percentile(e2e_all_vals, 50),
        "e2e_p90_all_s": _percentile(e2e_all_vals, 90),
        "e2e_p99_all_s": _percentile(e2e_all_vals, 99),
        "e2e_max_all_s": float(e2e_all_vals[-1]) if e2e_all_vals else None,
        "e2e_mean_success_s": _mean(e2e_succ_vals),
        "e2e_p50_success_s": _percentile(e2e_succ_vals, 50),
        "e2e_p90_success_s": _percentile(e2e_succ_vals, 90),
        "e2e_p99_success_s": _percentile(e2e_succ_vals, 99),
    }


def _summarize_metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def col_vals(name: str) -> List[float]:
        out = []
        for r in rows:
            v = _safe_float(r.get(name))
            if v is not None:
                out.append(v)
        return out

    elapsed = col_vals("elapsed_seconds")
    waiting = col_vals("vllm:num_requests_waiting")
    running = col_vals("vllm:num_requests_running")
    kv = col_vals("vllm:kv_cache_usage_perc")
    prompt_total = col_vals("vllm:prompt_tokens_total")
    gen_total = col_vals("vllm:generation_tokens_total")

    def rate(ts: List[float], ys: List[float]) -> List[float]:
        if len(ts) < 2 or len(ys) < 2:
            return []
        n = min(len(ts), len(ys))
        rates = []
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            dy = ys[i] - ys[i - 1]
            if dt > 0:
                rates.append(dy / dt)
        return rates

    prompt_tps = rate(elapsed, prompt_total)
    gen_tps = rate(elapsed, gen_total)
    total_tps = [a + b for a, b in zip(prompt_tps, gen_tps)] if prompt_tps and gen_tps else []

    def summarize_series(vals: List[float]) -> Dict[str, Any]:
        vals_sorted = sorted(vals)
        return {
            "mean": _mean(vals_sorted),
            "p50": _percentile(vals_sorted, 50),
            "p90": _percentile(vals_sorted, 90),
            "p99": _percentile(vals_sorted, 99),
            "max": float(vals_sorted[-1]) if vals_sorted else None,
        }

    out: Dict[str, Any] = {
        "metrics_samples": len(rows),
        "elapsed_max_s": max(elapsed) if elapsed else None,
        "waiting": summarize_series(waiting) if waiting else {},
        "running": summarize_series(running) if running else {},
        "kv_cache_usage_perc": summarize_series(kv) if kv else {},
        "prompt_tps": summarize_series(prompt_tps) if prompt_tps else {},
        "generation_tps": summarize_series(gen_tps) if gen_tps else {},
        "total_tps": summarize_series(total_tps) if total_tps else {},
    }
    return out


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _try_import_plotting():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        return False


def _plot_summary(summary_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def get_num(r: Dict[str, Any], k: str) -> Optional[float]:
        v = r.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def group_by_minutes() -> Dict[int, List[Dict[str, Any]]]:
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for r in summary_rows:
            mins = r.get("minutes")
            if mins is None:
                continue
            groups.setdefault(int(mins), []).append(r)
        for mins in groups:
            groups[mins].sort(key=lambda rr: (get_num(rr, "qps_offered") or 0))
        return groups

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_minutes()

    def lineplot(metric: str, title: str, ylabel: str, fname: str, series: List[Tuple[str, str]]):
        # series: [(label, metric_key)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for mins, rows in sorted(groups.items()):
            xs = [get_num(r, "qps_offered") for r in rows]
            for label, key in series:
                ys = [get_num(r, key) for r in rows]
                if all(y is None for y in ys):
                    continue
                ax.plot(xs, ys, marker="o", label=f"{mins}min {label}")
        ax.set_title(title)
        ax.set_xlabel("Offered QPS")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=160)
        plt.close(fig)

    lineplot(
        metric="ttft",
        title="TTFT vs Offered QPS (success only)",
        ylabel="Seconds",
        fname="ttft_vs_qps.png",
        series=[
            ("p50", "ttft_p50_success_s"),
            ("p90", "ttft_p90_success_s"),
            ("p99", "ttft_p99_success_s"),
        ],
    )

    lineplot(
        metric="e2e",
        title="E2E Latency vs Offered QPS (success only)",
        ylabel="Seconds",
        fname="e2e_vs_qps.png",
        series=[
            ("p50", "e2e_p50_success_s"),
            ("p90", "e2e_p90_success_s"),
            ("p99", "e2e_p99_success_s"),
        ],
    )

    lineplot(
        metric="success_rate",
        title="Success Rate vs Offered QPS",
        ylabel="Success Rate",
        fname="success_rate_vs_qps.png",
        series=[("", "success_rate")],
    )

    lineplot(
        metric="achieved_qps",
        title="Achieved QPS vs Offered QPS",
        ylabel="QPS",
        fname="achieved_qps_vs_qps.png",
        series=[
            ("total", "achieved_qps_total"),
            ("success", "achieved_qps_success"),
        ],
    )

    lineplot(
        metric="tokens_per_second",
        title="Throughput (Tokens/s) vs Offered QPS",
        ylabel="Tokens/s",
        fname="tokens_per_second_vs_qps.png",
        series=[("", "tokens_per_second")],
    )

    lineplot(
        metric="queue_waiting",
        title="vLLM Waiting Queue vs Offered QPS (from metrics)",
        ylabel="num_requests_waiting",
        fname="waiting_queue_vs_qps.png",
        series=[
            ("mean", "metrics.waiting.mean"),
            ("p90", "metrics.waiting.p90"),
            ("max", "metrics.waiting.max"),
        ],
    )

    lineplot(
        metric="kv_cache",
        title="KV Cache Usage vs Offered QPS (from metrics)",
        ylabel="kv_cache_usage_perc",
        fname="kv_cache_usage_vs_qps.png",
        series=[
            ("mean", "metrics.kv_cache_usage_perc.mean"),
            ("p90", "metrics.kv_cache_usage_perc.p90"),
            ("max", "metrics.kv_cache_usage_perc.max"),
        ],
    )


def _plot_per_experiment_timeseries(exp: Dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    exp_dir = Path(exp["exp_dir"])
    metrics_csv = exp.get("metrics_csv")
    if not metrics_csv:
        return

    rows = _read_csv_dicts(Path(metrics_csv))
    if not rows:
        return

    def get_series(key: str) -> List[float]:
        out = []
        for r in rows:
            v = _safe_float(r.get(key))
            out.append(v if v is not None else float("nan"))
        return out

    t = get_series("elapsed_seconds")
    waiting = get_series("vllm:num_requests_waiting")
    running = get_series("vllm:num_requests_running")
    kv = get_series("vllm:kv_cache_usage_perc")
    p_total = get_series("vllm:prompt_tokens_total")
    g_total = get_series("vllm:generation_tokens_total")

    def deriv(x: List[float], y: List[float]) -> List[float]:
        out = [float("nan")]
        for i in range(1, len(x)):
            dt = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            if dt and dt > 0 and not math.isnan(dy):
                out.append(dy / dt)
            else:
                out.append(float("nan"))
        return out

    p_tps = deriv(t, p_total)
    g_tps = deriv(t, g_total)
    total_tps = [a + b if (not math.isnan(a) and not math.isnan(b)) else float("nan") for a, b in zip(p_tps, g_tps)]

    ts_dir = out_dir / "plots" / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    qps = exp.get("qps_offered")
    mins = exp.get("minutes")
    title = f"{exp['name']} (qps={qps}, min={mins})"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, waiting, label="waiting")
    axes[0].plot(t, running, label="running")
    axes[0].set_ylabel("requests")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, kv, color="tab:purple", label="kv_cache_usage_perc")
    axes[1].set_ylabel("kv_cache")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, p_tps, label="prompt_tps")
    axes[2].plot(t, g_tps, label="generation_tps")
    axes[2].plot(t, total_tps, label="total_tps", linewidth=2)
    axes[2].set_ylabel("tokens/s")
    axes[2].set_xlabel("elapsed_seconds")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ts_dir / f"{exp['name']}.png", dpi=160)
    plt.close(fig)


def _write_report(summary_rows: List[Dict[str, Any]], out_dir: Path, generated_plots: bool, per_experiment_plots: bool) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (int,)):
            return str(x)
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return ""
            if abs(xf) >= 100:
                return f"{xf:.1f}"
            return f"{xf:.3f}"
        except Exception:
            return str(x)

    def get(r: Dict[str, Any], k: str) -> Any:
        return r.get(k)

    lines: List[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append(f"- Experiments: {len(summary_rows)}")
    lines.append(f"- CSV: `summary.csv`")
    lines.append(f"- JSON: `experiments.json`")
    if generated_plots:
        lines.append(f"- Plots: `plots/`")
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        for p in [
            "plots/ttft_vs_qps.png",
            "plots/e2e_vs_qps.png",
            "plots/success_rate_vs_qps.png",
            "plots/achieved_qps_vs_qps.png",
            "plots/tokens_per_second_vs_qps.png",
            "plots/waiting_queue_vs_qps.png",
            "plots/kv_cache_usage_vs_qps.png",
        ]:
            lines.append(f"![]({p})")
            lines.append("")
        if per_experiment_plots:
            lines.append("## Per-Experiment Time Series")
            lines.append("")
            lines.append("See `plots/timeseries/`.")
            lines.append("")
    else:
        lines.append("- Plots: skipped (matplotlib missing)")
        lines.append("")

    lines.append("## Key Table (success only)")
    lines.append("")
    cols = [
        ("minutes", "min"),
        ("qps_offered", "qps"),
        ("achieved_qps_success", "ach_qps"),
        ("success_rate", "succ_rate"),
        ("ttft_p50_success_s", "ttft_p50"),
        ("ttft_p90_success_s", "ttft_p90"),
        ("e2e_p50_success_s", "e2e_p50"),
        ("e2e_p90_success_s", "e2e_p90"),
        ("tokens_per_second", "tok/s"),
        ("metrics.waiting.mean", "wait_mean"),
        ("metrics.kv_cache_usage_perc.mean", "kv_mean"),
    ]
    lines.append("| " + " | ".join(c[1] for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in sorted(summary_rows, key=lambda rr: (rr.get("minutes") or 1e9, rr.get("qps_offered") or 1e9)):
        lines.append("| " + " | ".join(fmt(get(r, c[0])) for c in cols) + " |")
    lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize vLLM benchmark experiments and generate plots.")
    ap.add_argument("--root", type=str, default="workspace/tools/experiments", help="Root directory containing experiment_* folders")
    ap.add_argument("--out", type=str, default="workspace/tools/experiments_summary", help="Output directory for summary + plots")
    ap.add_argument("--per-experiment-plots", action="store_true", help="Generate per-experiment time series plots (more files)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support both:
    # - root/experiment_*
    # - root/<run_tag>/experiment_* (e.g., overnight runs grouped by folder)
    exp_dirs = sorted({p for p in root.rglob("experiment_*") if p.is_dir()})
    if not exp_dirs:
        print(f"No experiment_* directories found under: {root}")
        return 2

    experiments: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        meta = _parse_experiment_dir_name(exp_dir.name)
        paths = _find_experiment_paths(exp_dir)

        result_summary: Dict[str, Any] = {}
        if paths.result_csv:
            try:
                result_rows = _read_csv_dicts(paths.result_csv)
                result_summary = _summarize_results(result_rows)
            except Exception as e:
                result_summary = {"error": f"failed_to_parse_results: {e}"}

        metrics_summary: Dict[str, Any] = {}
        if paths.metrics_csv:
            try:
                metrics_rows = _read_csv_dicts(paths.metrics_csv)
                metrics_summary = _summarize_metrics(metrics_rows)
            except Exception as e:
                metrics_summary = {"error": f"failed_to_parse_metrics: {e}"}

        exp: Dict[str, Any] = {
            "name": meta.name,
            "qps_offered": meta.qps_offered,
            "minutes": meta.minutes,
            "timestamp": meta.timestamp,
            "exp_dir": str(paths.exp_dir),
            "result_csv": str(paths.result_csv) if paths.result_csv else None,
            "metrics_csv": str(paths.metrics_csv) if paths.metrics_csv else None,
            "execution_log": str(paths.execution_log) if paths.execution_log else None,
            "monitor_log": str(paths.monitor_log) if paths.monitor_log else None,
            **result_summary,
            "metrics": metrics_summary,
        }
        experiments.append(exp)

    # Write JSON (full)
    (out_dir / "experiments.json").write_text(json.dumps(experiments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Build flattened CSV
    flat_rows: List[Dict[str, Any]] = []
    for exp in experiments:
        base = {k: exp.get(k) for k in exp.keys() if k not in {"metrics"}}
        base.update(_flatten(exp.get("metrics") or {}, "metrics"))
        flat_rows.append(base)

    # Determine stable field order (key groups first)
    preferred = [
        "name",
        "qps_offered",
        "minutes",
        "timestamp",
        "requests_total",
        "requests_success",
        "success_rate",
        "test_wall_seconds",
        "achieved_qps_total",
        "achieved_qps_success",
        "tokens_total",
        "tokens_per_second",
        "ttft_p50_success_s",
        "ttft_p90_success_s",
        "ttft_p99_success_s",
        "e2e_p50_success_s",
        "e2e_p90_success_s",
        "e2e_p99_success_s",
        "metrics.waiting.mean",
        "metrics.waiting.p90",
        "metrics.waiting.max",
        "metrics.running.mean",
        "metrics.kv_cache_usage_perc.mean",
        "metrics.total_tps.mean",
    ]
    all_keys = sorted({k for r in flat_rows for k in r.keys()})
    fieldnames = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]

    # Sort rows for readability
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, str]:
        q = r.get("qps_offered")
        m = r.get("minutes")
        return (
            float(q) if q is not None else 1e9,
            float(m) if m is not None else 1e9,
            str(r.get("name") or ""),
        )

    flat_rows.sort(key=sort_key)

    _write_csv(out_dir / "summary.csv", flat_rows, fieldnames)

    # Optional plotting
    plotting_ok = _try_import_plotting()
    if not plotting_ok:
        msg = (
            "Plotting dependencies missing. Summary generated, but plots skipped.\n"
            "Install: python3 -m venv workspace/tools/.venv && "
            "workspace/tools/.venv/bin/pip install -U pip matplotlib\n"
            "Then rerun this script with that python.\n"
        )
        (out_dir / "PLOTS_SKIPPED.txt").write_text(msg, encoding="utf-8")
        print(msg.strip())
        _write_report(flat_rows, out_dir, generated_plots=False, per_experiment_plots=False)
        return 0

    _plot_summary(flat_rows, out_dir)
    if args.per_experiment_plots:
        for exp in experiments:
            _plot_per_experiment_timeseries(exp, out_dir)

    skipped_path = out_dir / "PLOTS_SKIPPED.txt"
    if skipped_path.exists():
        try:
            skipped_path.unlink()
        except Exception:
            pass

    _write_report(flat_rows, out_dir, generated_plots=True, per_experiment_plots=bool(args.per_experiment_plots))

    print(f"Summary written to: {out_dir / 'summary.csv'}")
    print(f"Plots written under: {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
