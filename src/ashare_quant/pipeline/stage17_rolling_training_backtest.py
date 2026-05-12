from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ashare_quant.backtest.rolling_training import BenchmarkSpec, load_benchmark_series, run_rolling_training_backtest
from ashare_quant.config.research_config import load_config
from ashare_quant.factors.price_volume import add_basic_factors
from ashare_quant.pipeline.stage12_dual_signal_strategy import build_multifactor_frame, enrich_with_fundamentals
from ashare_quant.universe.filtering import apply_universe_filters


FEATURE_COLS = [
    "momentum_score",
    "quality_score",
    "liquidity_score",
    "value_score",
    "rule_score",
    "risk_score",
    "ret_20",
    "ret_60",
    "ret_120",
    "amount_20",
    "trend_gap_20_60",
    "trend_gap_60_120",
    "distance_ma20",
    "distance_ma60",
    "drawdown_20",
    "drawdown_60",
    "vol_20",
    "vol_60",
    "pe_ttm",
    "pb",
    "mv_total",
]


def _model_factory() -> object:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=160,
        max_depth=6,
        min_samples_leaf=80,
        l2_regularization=0.05,
        random_state=42,
    )


def _selection_filter(day: pd.DataFrame) -> pd.DataFrame:
    if "buyable_today" not in day.columns:
        return day
    return day[day["buyable_today"]].copy()


def run() -> None:
    cfg = load_config("configs/research.yaml")
    rolling = cfg.rolling_backtest
    path = Path(cfg.data.all_buyable_path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_parquet(path)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    benchmark = load_benchmark_series(
        raw=raw,
        spec=BenchmarkSpec(name=rolling.benchmark, path=rolling.benchmark_path),
        start_date=rolling.start_date,
        end_date=rolling.end_date,
    )

    raw = apply_universe_filters(raw)
    raw = add_basic_factors(raw)
    raw = enrich_with_fundamentals(raw)
    frame = build_multifactor_frame(raw)

    target_col = f"fwd_ret_{rolling.prediction_horizon_days}"
    if target_col not in frame.columns:
        frame = frame.sort_values(["ts_code", "trade_date"]).copy()
        frame[target_col] = frame.groupby("ts_code")["close"].shift(-rolling.prediction_horizon_days) / frame["close"] - 1

    frame = frame.dropna(subset=FEATURE_COLS + [target_col]).copy()
    result = run_rolling_training_backtest(
        df=frame,
        feature_cols=FEATURE_COLS,
        target_col=target_col,
        score_col="ml_score",
        cfg=rolling,
        model_factory=_model_factory,
        benchmark=benchmark,
        selection_filter=_selection_filter,
    )

    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)

    curve = result["curve"]
    signals = result["signals"]
    actions = result["actions"]
    nodes = result["nodes"]
    latest_date = signals["rebalance_date"].max() if not signals.empty else None
    latest_picks = signals[signals["rebalance_date"] == latest_date].copy() if latest_date is not None else pd.DataFrame()

    curve.to_parquet(out / "stage17_rolling_backtest_curve.parquet", index=False)
    signals.to_parquet(out / "stage17_rolling_signals.parquet", index=False)
    actions.to_parquet(out / "stage17_rolling_actions.parquet", index=False)
    nodes.to_parquet(out / "stage17_rolling_nodes.parquet", index=False)
    (out / "stage17_rolling_backtest_summary.json").write_text(
        json.dumps(
            {
                **result["summary"],
                "latest_rebalance_date": str(pd.to_datetime(latest_date).date()) if latest_date is not None else None,
                "latest_picks": latest_picks.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "trade_date": str(pd.to_datetime(latest_date).date()) if latest_date is not None else None,
                "summary": result["summary"],
                "n_latest_picks": int(len(latest_picks)),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    run()
