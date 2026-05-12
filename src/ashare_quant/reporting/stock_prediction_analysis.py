from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ashare_quant.config.research_config import load_config
from ashare_quant.factors.price_volume import add_basic_factors
from ashare_quant.pipeline.stage12_dual_signal_strategy import (
    TOP_N,
    apply_calibration,
    build_multifactor_frame,
    enrich_with_fundamentals,
    fit_predict_multifactor,
    load_calibration,
)
from ashare_quant.universe.filtering import apply_universe_filters


def _clean_symbol(symbol: str) -> str:
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if not digits:
        raise ValueError("symbol must contain digits")
    return digits.zfill(6)


def build_stock_prediction_frame(
    scored: pd.DataFrame,
    symbol: str,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    code = _clean_symbol(symbol)
    stock = scored[scored["ts_code"].astype(str).str.zfill(6) == code].copy()
    if stock.empty:
        raise RuntimeError(f"no predictions available for symbol {code}")

    stock = stock.sort_values("trade_date").reset_index(drop=True)
    stock["actual_ret_10"] = stock["fwd_ret_10"].astype(float)
    stock["predicted_ret_10"] = stock["expected_ret_10"].astype(float)
    stock["predicted_close_10"] = stock["close"].astype(float) * (1 + stock["predicted_ret_10"])
    stock["actual_close_10"] = stock["close"].astype(float) * (1 + stock["actual_ret_10"])
    stock["prediction_error"] = stock["predicted_ret_10"] - stock["actual_ret_10"]
    stock["abs_error"] = stock["prediction_error"].abs()
    stock["is_correct_direction"] = (stock["predicted_ret_10"] > 0) == (stock["actual_ret_10"] > 0)

    raw_signal = np.where(
        stock["buyable_today"] & (stock["stock_rank"] <= top_n) & (stock["predicted_ret_10"] >= buy_threshold),
        "BUY",
        np.where(
            (~stock["buyable_today"]) | (stock["predicted_ret_10"] <= sell_threshold),
            "SELL",
            "WATCH",
        ),
    )
    stock["raw_signal"] = raw_signal

    actions: list[str] = []
    in_position = False
    for signal in stock["raw_signal"].tolist():
        if signal == "BUY" and not in_position:
            actions.append("BUY")
            in_position = True
        elif signal == "SELL" and in_position:
            actions.append("SELL")
            in_position = False
        elif in_position:
            actions.append("HOLD")
        else:
            actions.append("WAIT")
    stock["action"] = actions
    stock["in_position"] = stock["action"].isin(["BUY", "HOLD"]).astype(int)
    stock["position_state"] = np.where(stock["in_position"] == 1, "IN", "OUT")

    stock["spot_ret_1"] = stock["close"].astype(float).pct_change().fillna(0.0)
    stock["strategy_ret_1"] = stock["spot_ret_1"] * stock["in_position"].shift(1).fillna(0).astype(float)
    stock["spot_equity"] = (1 + stock["spot_ret_1"]).cumprod()
    stock["strategy_equity"] = (1 + stock["strategy_ret_1"]).cumprod()

    trade_ids: list[int | None] = []
    trade_id = 0
    active_trade = 0
    for action, held in zip(stock["action"].tolist(), stock["in_position"].tolist()):
        if action == "BUY":
            trade_id += 1
            active_trade = trade_id
            trade_ids.append(active_trade)
        elif held == 1:
            trade_ids.append(active_trade if active_trade > 0 else None)
        else:
            trade_ids.append(None)
            if action == "SELL":
                active_trade = 0
    stock["trade_id"] = trade_ids

    stock["trend_delta_pred"] = stock["predicted_ret_10"].diff()
    stock["trend_delta_act"] = stock["actual_ret_10"].diff()
    stock["trend_match"] = np.sign(stock["trend_delta_pred"].fillna(0)) == np.sign(stock["trend_delta_act"].fillna(0))

    pred_sign = np.sign(stock["trend_delta_pred"])
    act_sign = np.sign(stock["trend_delta_act"])
    stock["pred_turning_point"] = (pred_sign.shift(1) * pred_sign < 0).fillna(False)
    stock["act_turning_point"] = (act_sign.shift(1) * act_sign < 0).fillna(False)
    return stock


def summarize_stock_prediction_frame(stock: pd.DataFrame, symbol: str) -> dict:
    matured = stock.dropna(subset=["predicted_ret_10", "actual_ret_10"]).copy()
    if matured.empty:
        raise RuntimeError("stock prediction frame has no matured rows")

    err = matured["prediction_error"].astype(float)
    pred = matured["predicted_ret_10"].astype(float)
    act = matured["actual_ret_10"].astype(float)

    trend_mask = matured["trend_delta_pred"].notna() & matured["trend_delta_act"].notna()
    trend_acc = float(matured.loc[trend_mask, "trend_match"].astype(float).mean()) if trend_mask.any() else None

    pred_tp = matured["pred_turning_point"].astype(bool)
    act_tp = matured["act_turning_point"].astype(bool)
    tp_hit = pred_tp & act_tp
    tp_precision = float(tp_hit.sum() / pred_tp.sum()) if int(pred_tp.sum()) > 0 else None
    tp_recall = float(tp_hit.sum() / act_tp.sum()) if int(act_tp.sum()) > 0 else None

    buy_rows = matured[matured["action"] == "BUY"].copy()
    sell_rows = matured[matured["action"] == "SELL"].copy()
    held_rows = matured[matured["in_position"] == 1].copy()

    latest = stock.iloc[-1]
    return {
        "symbol": _clean_symbol(symbol),
        "start_date": str(pd.to_datetime(stock["trade_date"].min()).date()),
        "end_date": str(pd.to_datetime(stock["trade_date"].max()).date()),
        "rows": int(len(stock)),
        "matured_rows": int(len(matured)),
        "latest_trade_date": str(pd.to_datetime(latest["trade_date"]).date()),
        "latest_close": float(latest["close"]),
        "latest_predicted_ret_10": float(latest["predicted_ret_10"]),
        "latest_actual_ret_10": float(latest["actual_ret_10"]) if pd.notna(latest["actual_ret_10"]) else None,
        "latest_stock_rank": int(latest["stock_rank"]),
        "latest_action": str(latest["action"]),
        "latest_raw_signal": str(latest["raw_signal"]),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "directional_accuracy": float(matured["is_correct_direction"].astype(float).mean()),
        "corr": float(np.corrcoef(pred, act)[0, 1]) if len(matured) > 1 else None,
        "trend_direction_accuracy": trend_acc,
        "turning_point_precision": tp_precision,
        "turning_point_recall": tp_recall,
        "buy_signal_count": int(len(buy_rows)),
        "sell_signal_count": int(len(sell_rows)),
        "profitable_buy_signal_rate": float((buy_rows["actual_ret_10"] > 0).mean()) if not buy_rows.empty else None,
        "avg_buy_signal_actual_ret_10": float(buy_rows["actual_ret_10"].mean()) if not buy_rows.empty else None,
        "avg_sell_signal_actual_ret_10": float(sell_rows["actual_ret_10"].mean()) if not sell_rows.empty else None,
        "strategy_cumulative_return": float(stock["strategy_equity"].iloc[-1] - 1),
        "spot_cumulative_return": float(stock["spot_equity"].iloc[-1] - 1),
        "max_rank": int(stock["stock_rank"].max()),
        "min_rank": int(stock["stock_rank"].min()),
        "avg_rank": float(stock["stock_rank"].mean()),
        "holding_day_ratio": float(held_rows.shape[0] / matured.shape[0]) if not matured.empty else None,
    }


def render_stock_prediction_chart(stock: pd.DataFrame, symbol: str, out_path: Path) -> Path:
    code = _clean_symbol(symbol)
    x = stock.copy()
    x["trade_date"] = pd.to_datetime(x["trade_date"])

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Close Price With Actions",
            "Predicted vs Actual 10-Day Return",
            "Prediction Error",
            "Signal Position State",
            "Single-Stock Strategy Equity vs Spot Equity / Rank",
        ),
        specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(x=x["trade_date"], y=x["close"], mode="lines", name="close", line=dict(color="#0f766e", width=2)),
        row=1,
        col=1,
    )

    for action, color, symbol_name in [("BUY", "#16a34a", "triangle-up"), ("SELL", "#dc2626", "triangle-down"), ("HOLD", "#2563eb", "circle")]:
        subset = x[x["action"] == action]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["trade_date"],
                y=subset["close"],
                mode="markers",
                name=action,
                marker=dict(color=color, symbol=symbol_name, size=10),
                text=[f"{action} {code}" for _ in range(len(subset))],
            ),
            row=1,
            col=1,
        )

    buy_sell = x[x["action"].isin(["BUY", "SELL"])].copy()
    if not buy_sell.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_sell["trade_date"],
                y=buy_sell["close"],
                mode="text",
                name="trade_labels",
                text=buy_sell["action"],
                textposition="top center",
                textfont=dict(size=10, color="#111827"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=x["trade_date"], y=x["predicted_ret_10"], mode="lines", name="pred_ret_10", line=dict(color="#7c3aed", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x["trade_date"], y=x["actual_ret_10"], mode="lines", name="actual_ret_10", line=dict(color="#ea580c", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x["trade_date"], y=x["prediction_error"], name="prediction_error", marker_color="#64748b"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x["trade_date"],
            y=x["in_position"],
            mode="lines",
            line=dict(color="#0ea5e9", width=2, shape="hv"),
            fill="tozeroy",
            name="in_position",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x["trade_date"],
            y=x["strategy_equity"],
            mode="lines",
            name="strategy_equity",
            line=dict(color="#16a34a", width=2.5),
        ),
        row=5,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x["trade_date"],
            y=x["spot_equity"],
            mode="lines",
            name="spot_equity",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ),
        row=5,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x["trade_date"],
            y=x["stock_rank"],
            mode="lines",
            name="stock_rank",
            line=dict(color="#1d4ed8", width=1.5, dash="dash"),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Single Stock Prediction Analysis: {code}",
        template="plotly_white",
        height=1450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_yaxes(title_text="Return", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="Error", tickformat=".1%", row=3, col=1)
    fig.update_yaxes(title_text="Position", tickvals=[0, 1], row=4, col=1)
    fig.update_yaxes(title_text="Equity", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rank", autorange="reversed", row=5, col=1, secondary_y=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def run_single_stock_prediction_analysis(
    symbol: str,
    out_dir: str | Path = "reports",
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    top_n: int = TOP_N,
) -> dict:
    cfg = load_config("configs/research.yaml")
    data_path = Path(cfg.data.all_buyable_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    raw = pd.read_parquet(data_path)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["ts_code"] = raw["ts_code"].astype(str).str.zfill(6)
    raw = apply_universe_filters(raw)
    raw = add_basic_factors(raw)
    raw = enrich_with_fundamentals(raw)

    feature_frame = build_multifactor_frame(raw)
    scored, model_meta = fit_predict_multifactor(feature_frame)
    calibration = load_calibration(Path(out_dir))
    calibration["model_name"] = str(model_meta.get("model", "multifactor_top10"))
    scored = apply_calibration(scored, calibration)

    stock = build_stock_prediction_frame(
        scored,
        symbol=symbol,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        top_n=top_n,
    )
    summary = summarize_stock_prediction_frame(stock, symbol)

    code = _clean_symbol(symbol)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts_path = out_root / f"stage12_stock_{code}_prediction_timeseries.parquet"
    summary_path = out_root / f"stage12_stock_{code}_prediction_summary.json"
    chart_path = out_root / f"stage12_stock_{code}_prediction_chart.html"

    stock.to_parquet(ts_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    render_stock_prediction_chart(stock, code, chart_path)

    return {
        "summary": summary,
        "timeseries_path": str(ts_path),
        "summary_path": str(summary_path),
        "chart_path": str(chart_path),
        "recent_actions": stock[["trade_date", "close", "predicted_ret_10", "actual_ret_10", "stock_rank", "raw_signal", "action", "in_position", "strategy_equity"]]
        .tail(10)
        .copy(),
    }
