from __future__ import annotations

import pandas as pd

from ashare_quant.config.research_config import load_config
from ashare_quant.reporting.user_decision_service import (
    build_recommendation_table,
    build_single_stock_review,
    simulate_from_date,
)


def _sample_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
    rows = []
    closes_a = [10.0, 10.5, 11.4, 11.7, 11.6]
    closes_b = [8.0, 7.8, 7.6, 7.5, 7.4]
    for idx, trade_date in enumerate(dates):
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "000001",
                "name": "Alpha",
                "open": closes_a[idx] - 0.1,
                "high": closes_a[idx] + 0.3,
                "low": closes_a[idx] - 0.2,
                "close": closes_a[idx],
                "amount": 1e9,
                "buyable_today": True,
                "stock_rank": 1,
                "rule_score": 0.76,
                "risk_score": 0.24,
                "final_score": 0.32,
                "ml_score": 0.05,
                "expected_ret_10": 0.12,
                "fwd_ret_10": 0.10,
                "direction": "预计上涨",
                "predicted_close_10": closes_a[idx] * 1.12,
                "actual_close_10": closes_a[idx] * 1.10,
                "prediction_error": 0.02,
            }
        )
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "000002",
                "name": "Beta",
                "open": closes_b[idx] + 0.05,
                "high": closes_b[idx] + 0.12,
                "low": closes_b[idx] - 0.15,
                "close": closes_b[idx],
                "amount": 1e9,
                "buyable_today": idx < 2,
                "stock_rank": 2,
                "rule_score": 0.51,
                "risk_score": 0.48,
                "final_score": 0.14,
                "ml_score": -0.01,
                "expected_ret_10": -0.03,
                "fwd_ret_10": -0.05,
                "direction": "预计下跌",
                "predicted_close_10": closes_b[idx] * 0.97,
                "actual_close_10": closes_b[idx] * 0.95,
                "prediction_error": 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_build_recommendation_table_returns_t_plus_1_plan():
    cfg = load_config("configs/research.yaml")
    frame = _sample_frame()

    result = build_recommendation_table(frame, selected_date="2025-01-02", cfg=cfg, initial_capital=500000.0)

    assert result["effective_trade_date"] == "2025-01-02"
    assert result["buy_date"] == "2025-01-03"
    assert result["rows"][0]["ts_code"] == "000001"
    assert result["rows"][0]["suggested_shares"] > 0


def test_simulate_from_date_generates_completed_trade():
    cfg = load_config("configs/research.yaml")
    frame = _sample_frame()

    result = simulate_from_date(frame, start_date="2025-01-02", cfg=cfg, initial_capital=300000.0)

    assert not result["curve"].empty
    assert result["summary"]["trade_count"] >= 1
    assert "pnl" in result["trades"].columns


def test_build_single_stock_review_resolves_prediction_and_exit():
    cfg = load_config("configs/research.yaml")
    frame = _sample_frame()

    review = build_single_stock_review(frame, symbol="000001", selected_date="2025-01-02", cfg=cfg, initial_capital=200000.0)

    assert review["symbol"] == "000001"
    assert review["buy_date"] == "2025-01-03"
    assert review["shares"] > 0
    assert review["exit_reason"] in {"止盈卖出", "止损卖出", "到期卖出"}

