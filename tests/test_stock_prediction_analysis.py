import pandas as pd

from ashare_quant.reporting.stock_prediction_analysis import (
    build_stock_prediction_frame,
    summarize_stock_prediction_frame,
)


def _sample_scored():
    dates = pd.bdate_range("2025-01-01", periods=6)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["600397"] * 6,
            "close": [10, 10.2, 10.1, 10.5, 10.4, 10.7],
            "expected_ret_10": [0.04, 0.03, -0.03, 0.05, 0.01, -0.04],
            "fwd_ret_10": [0.05, 0.02, -0.01, 0.03, -0.02, -0.05],
            "stock_rank": [5, 7, 20, 4, 12, 15],
            "buyable_today": [True, True, True, True, True, False],
            "rule_score": [0.7] * 6,
            "risk_score": [0.3] * 6,
        }
    )


def test_build_stock_prediction_frame_actions():
    stock = build_stock_prediction_frame(_sample_scored(), "600397", buy_threshold=0.02, sell_threshold=-0.02, top_n=10)
    assert stock["action"].tolist()[:4] == ["BUY", "HOLD", "SELL", "BUY"]
    assert "prediction_error" in stock.columns
    assert "strategy_equity" in stock.columns
    assert "in_position" in stock.columns


def test_summarize_stock_prediction_frame():
    stock = build_stock_prediction_frame(_sample_scored(), "600397", buy_threshold=0.02, sell_threshold=-0.02, top_n=10)
    summary = summarize_stock_prediction_frame(stock, "600397")
    assert summary["symbol"] == "600397"
    assert summary["buy_signal_count"] >= 1
    assert summary["matured_rows"] == len(stock)
    assert "strategy_cumulative_return" in summary
