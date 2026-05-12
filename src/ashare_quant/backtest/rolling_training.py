from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ashare_quant.config.settings import SETTINGS
from ashare_quant.config.research_config import RollingBacktestCfg
from ashare_quant.reporting.metrics import evaluate_curve


ModelFactory = Callable[[], object]
SelectionFilter = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str = ""
    path: str = ""
    date_col: str = "trade_date"
    code_col: str = "ts_code"
    price_col: str = "close"
    daily_ret_col: str = "daily_ret"
    equity_col: str = "equity"


def add_forward_return_columns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    g = out.groupby("ts_code", group_keys=False)
    out["fwd_ret_1"] = g["close"].shift(-1) / out["close"] - 1
    for horizon in sorted(set(horizons)):
        out[f"fwd_ret_{horizon}"] = g["close"].shift(-horizon) / out["close"] - 1
    return out


def load_benchmark_series(
    raw: pd.DataFrame | None = None,
    spec: BenchmarkSpec | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    spec = spec or BenchmarkSpec()
    frame = pd.DataFrame()

    if spec.path:
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
    elif spec.name and raw is not None and spec.code_col in raw.columns:
        frame = raw[raw[spec.code_col].astype(str) == spec.name].copy()
    elif raw is not None:
        frame = raw.copy()
    else:
        return pd.DataFrame(columns=["trade_date", "benchmark_daily_ret", "benchmark_equity"])

    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "benchmark_daily_ret", "benchmark_equity"])

    if spec.date_col not in frame.columns:
        raise KeyError(f"benchmark date column missing: {spec.date_col}")

    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out[spec.date_col])
    if start_date:
        out = out[out["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["trade_date"] <= pd.to_datetime(end_date)]
    out = out.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    if spec.daily_ret_col in out.columns:
        out["benchmark_daily_ret"] = pd.to_numeric(out[spec.daily_ret_col], errors="coerce").fillna(0.0)
    elif spec.equity_col in out.columns:
        eq = pd.to_numeric(out[spec.equity_col], errors="coerce").ffill()
        out["benchmark_daily_ret"] = eq.pct_change().fillna(0.0)
    elif spec.price_col in out.columns:
        px = pd.to_numeric(out[spec.price_col], errors="coerce").ffill()
        out["benchmark_daily_ret"] = px.pct_change().fillna(0.0)
    else:
        raise KeyError("benchmark must contain one of daily_ret/equity/close style columns")

    out["benchmark_equity"] = (1 + out["benchmark_daily_ret"]).cumprod()
    return out[["trade_date", "benchmark_daily_ret", "benchmark_equity"]].reset_index(drop=True)


def _resolve_trade_dates(df: pd.DataFrame, start_date: str, end_date: str) -> list[pd.Timestamp]:
    dates = pd.to_datetime(df["trade_date"]).drop_duplicates().sort_values()
    dates = dates[(dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))]
    return dates.tolist()


def _select_weights(day: pd.DataFrame, score_col: str, top_n: int, weighting: str) -> pd.DataFrame:
    ranked = day.sort_values(score_col, ascending=False).head(top_n).copy()
    if ranked.empty:
        ranked["target_weight"] = []
        return ranked

    if weighting == "score":
        score = ranked[score_col].clip(lower=0.0)
        total = float(score.sum())
        if total <= 1e-12:
            ranked["target_weight"] = 1.0 / len(ranked)
        else:
            ranked["target_weight"] = score / total
    else:
        ranked["target_weight"] = 1.0 / len(ranked)
    return ranked


def _compute_turnover(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    union = set(prev_weights) | set(new_weights)
    return 0.5 * sum(abs(prev_weights.get(code, 0.0) - new_weights.get(code, 0.0)) for code in union)


def _compute_sell_weight(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    union = set(prev_weights) | set(new_weights)
    return sum(max(prev_weights.get(code, 0.0) - new_weights.get(code, 0.0), 0.0) for code in union)


def run_rolling_training_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    score_col: str,
    cfg: RollingBacktestCfg,
    model_factory: ModelFactory,
    benchmark: pd.DataFrame | None = None,
    selection_filter: SelectionFilter | None = None,
) -> dict[str, pd.DataFrame | dict]:
    if cfg.training_mode not in {"fixed", "expanding"}:
        raise ValueError(f"unsupported training_mode: {cfg.training_mode}")
    if cfg.weighting not in {"equal", "score"}:
        raise ValueError(f"unsupported weighting: {cfg.weighting}")

    data = add_forward_return_columns(df, [cfg.prediction_horizon_days])
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    trade_dates = _resolve_trade_dates(data, cfg.start_date, cfg.end_date)
    if not trade_dates:
        raise RuntimeError("no trade dates available in rolling backtest range")

    all_trade_dates = sorted(data["trade_date"].drop_duplicates().tolist())
    date_to_idx = {d: i for i, d in enumerate(all_trade_dates)}
    rebalance_dates = trade_dates[:: max(cfg.rebalance_frequency_days, 1)]

    curve_rows: list[dict] = []
    signal_rows: list[dict] = []
    action_rows: list[dict] = []
    node_rows: list[dict] = []

    equity = 1.0
    model = None
    last_retrain_trade_idx = -10**9
    weights: dict[str, float] = {}

    for node_idx, rebalance_date in enumerate(rebalance_dates):
        trade_idx = date_to_idx[rebalance_date]
        if cfg.training_mode == "fixed":
            train_start_idx = max(0, trade_idx - cfg.train_window_days)
        else:
            train_start_idx = 0

        train_dates = all_trade_dates[train_start_idx:trade_idx]
        if len(train_dates) < cfg.train_window_days:
            continue

        should_retrain = model is None or (trade_idx - last_retrain_trade_idx) >= max(cfg.retrain_frequency_days, 1)
        train_slice = data[data["trade_date"].isin(train_dates)].dropna(subset=feature_cols + [target_col]).copy()
        if len(train_slice) < cfg.min_train_rows:
            continue

        if should_retrain:
            model = model_factory()
            x_train = train_slice[feature_cols].astype(np.float32)
            y_train = train_slice[target_col].astype(np.float32)
            model.fit(x_train, y_train)
            last_retrain_trade_idx = trade_idx

        day = data[data["trade_date"] == rebalance_date].dropna(subset=feature_cols + ["close"]).copy()
        if selection_filter is not None:
            day = selection_filter(day)
        if day.empty or model is None:
            continue

        day[score_col] = model.predict(day[feature_cols].astype(np.float32))
        selected = _select_weights(day, score_col=score_col, top_n=cfg.top_n, weighting=cfg.weighting)
        new_weights = dict(zip(selected["ts_code"], selected["target_weight"]))
        turnover = _compute_turnover(weights, new_weights)
        sell_weight = _compute_sell_weight(weights, new_weights)

        prev_codes = set(weights)
        new_codes = set(new_weights)
        for _, row in selected.iterrows():
            action = "HOLD" if row["ts_code"] in prev_codes else "BUY"
            signal_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "ts_code": row["ts_code"],
                    "score": float(row[score_col]),
                    "target_weight": float(row["target_weight"]),
                    "signal": action,
                }
            )
            action_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "ts_code": row["ts_code"],
                    "action": action,
                    "prev_weight": float(weights.get(row["ts_code"], 0.0)),
                    "target_weight": float(row["target_weight"]),
                    "score": float(row[score_col]),
                }
            )
        for code in sorted(prev_codes - new_codes):
            action_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "ts_code": code,
                    "action": "SELL",
                    "prev_weight": float(weights.get(code, 0.0)),
                    "target_weight": 0.0,
                    "score": np.nan,
                }
            )

        next_rebalance_date = rebalance_dates[node_idx + 1] if (node_idx + 1) < len(rebalance_dates) else None
        node_dates = [d for d in trade_dates if d >= rebalance_date and (next_rebalance_date is None or d < next_rebalance_date)]

        for offset, trade_date in enumerate(node_dates):
            daily = data[data["trade_date"] == trade_date].set_index("ts_code")
            gross = 0.0
            for code, weight in new_weights.items():
                if code not in daily.index:
                    continue
                fr = daily.loc[code, "fwd_ret_1"]
                gross += float(weight) * (float(fr) if pd.notna(fr) else 0.0)

            daily_cost = 0.0
            if offset == 0:
                daily_cost = (
                    turnover * (SETTINGS.costs.commission_bps + SETTINGS.costs.slippage_bps) / 10000.0
                    + sell_weight * SETTINGS.costs.stamp_duty_sell_bps / 10000.0
                )
            net = gross - daily_cost
            equity *= 1 + net
            curve_rows.append(
                {
                    "trade_date": trade_date,
                    "daily_ret": net,
                    "equity": equity,
                    "turnover": daily_cost,
                    "n_holdings": len(new_weights),
                    "rebalance_flag": int(offset == 0),
                    "rebalance_date": rebalance_date,
                }
            )

        node_rows.append(
            {
                "rebalance_date": rebalance_date,
                "train_start_date": train_dates[0],
                "train_end_date": train_dates[-1],
                "train_days": len(train_dates),
                "train_rows": int(len(train_slice)),
                "retrained": int(should_retrain),
                "n_predictions": int(len(day)),
                "n_selected": int(len(selected)),
                "turnover": float(turnover),
            }
        )
        weights = new_weights

    curve = pd.DataFrame(curve_rows)
    if curve.empty:
        raise RuntimeError("rolling backtest produced empty curve")
    curve["equity"] = (1 + curve["daily_ret"]).cumprod()

    if benchmark is not None and not benchmark.empty:
        bench = benchmark.copy()
        bench["trade_date"] = pd.to_datetime(bench["trade_date"])
        curve = curve.merge(bench, on="trade_date", how="left")
        curve["benchmark_daily_ret"] = curve["benchmark_daily_ret"].fillna(0.0)
        curve["benchmark_equity"] = (1 + curve["benchmark_daily_ret"]).cumprod()
        curve["excess_daily_ret"] = curve["daily_ret"] - curve["benchmark_daily_ret"]
        curve["excess_equity"] = curve["equity"] / curve["benchmark_equity"].replace(0, np.nan)
    else:
        curve["excess_daily_ret"] = curve["daily_ret"]
        curve["excess_equity"] = curve["equity"]

    summary = {
        **evaluate_curve(curve),
        "config": asdict(cfg),
        "target_col": target_col,
        "score_col": score_col,
        "rebalance_nodes": int(curve["rebalance_flag"].sum()),
    }

    return {
        "curve": curve.sort_values("trade_date").reset_index(drop=True),
        "signals": pd.DataFrame(signal_rows).sort_values(["rebalance_date", "score"], ascending=[True, False]).reset_index(drop=True),
        "actions": pd.DataFrame(action_rows).sort_values(["rebalance_date", "ts_code"]).reset_index(drop=True),
        "nodes": pd.DataFrame(node_rows).sort_values("rebalance_date").reset_index(drop=True),
        "summary": summary,
    }
