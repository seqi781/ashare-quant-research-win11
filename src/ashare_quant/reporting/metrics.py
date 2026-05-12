from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_curve(curve: pd.DataFrame) -> dict:
    r = curve["daily_ret"].fillna(0.0)
    eq = curve["equity"].ffill().fillna(1.0)
    ann_ret = float(eq.iloc[-1] ** (252 / max(len(eq), 1)) - 1)
    ann_vol = float(r.std(ddof=0) * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    win_rate = float((r > 0).mean())
    out = {
        "cumulative_return": float(eq.iloc[-1] - 1),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "days": int(len(curve)),
    }
    if "turnover" in curve.columns:
        out["avg_turnover"] = float(curve["turnover"].fillna(0.0).mean())
        out["annual_turnover"] = float(curve["turnover"].fillna(0.0).sum())
    if "benchmark_daily_ret" in curve.columns:
        br = curve["benchmark_daily_ret"].fillna(0.0)
        beq = curve.get("benchmark_equity", (1 + br).cumprod()).ffill().fillna(1.0)
        bench_ann = float(beq.iloc[-1] ** (252 / max(len(beq), 1)) - 1)
        bench_vol = float(br.std(ddof=0) * np.sqrt(252))
        diff = r - br
        tracking = float(diff.std(ddof=0) * np.sqrt(252))
        out["benchmark_cumulative_return"] = float(beq.iloc[-1] - 1)
        out["benchmark_ann_return"] = bench_ann
        out["benchmark_ann_vol"] = bench_vol
        out["benchmark_sharpe"] = float(bench_ann / bench_vol) if bench_vol > 1e-12 else 0.0
        out["excess_cumulative_return"] = float(eq.iloc[-1] / beq.iloc[-1] - 1) if float(beq.iloc[-1]) != 0.0 else 0.0
        out["alpha_ann"] = float(ann_ret - bench_ann)
        out["tracking_error"] = tracking
        out["information_ratio"] = float((ann_ret - bench_ann) / tracking) if tracking > 1e-12 else 0.0
    return out
