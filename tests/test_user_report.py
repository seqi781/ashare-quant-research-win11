from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_quant.reporting import user_report


def test_render_html_report_writes_embedded_images(tmp_path, monkeypatch):
    payload = {
        "simulation": {
            "summary": {
                "selected_start_date": "2025-01-02",
                "effective_start_trade_date": "2025-01-02",
                "latest_trade_date": "2025-01-08",
                "total_pnl": 12800.0,
                "return_rate": 0.0128,
                "max_drawdown": -0.031,
                "trade_count": 3,
                "win_rate": 2 / 3,
                "ending_asset": 1012800.0,
            },
            "curve": pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
                    "total_asset": [1000000.0, 1008000.0, 1012800.0],
                    "drawdown": [0.0, -0.005, 0.0],
                }
            ),
            "trades": pd.DataFrame(
                [
                    {
                        "ts_code": "000001",
                        "buy_date": "2025-01-03",
                        "sell_date": "2025-01-06",
                        "exit_reason": "止盈卖出",
                        "pnl": 8200.0,
                        "realized_return": 0.081,
                    }
                ]
            ),
        },
        "recommendations": {
            "summary": {
                "candidate_count": 1,
                "capital_usage_rate": 0.1,
                "explanation": "测试说明",
            },
            "rows": [
                {
                    "ts_code": "000001",
                    "direction": "预计上涨",
                    "buy_date": "2025-01-03",
                    "suggested_shares": 1000,
                    "suggested_amount": 100000.0,
                    "expected_profit": 12000.0,
                    "max_possible_loss": 7000.0,
                    "predicted_ret_10": 0.12,
                    "actual_ret_10": 0.10,
                    "conviction_badge": "高优先级",
                }
            ]
        },
        "stock_review": {
            "symbol": "000001",
            "selected_date": "2025-01-02",
            "effective_trade_date": "2025-01-02",
            "prediction_error": 0.02,
            "series": pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
                    "close": [10.0, 10.6, 11.1],
                    "expected_ret_10": [0.12, 0.10, 0.08],
                    "fwd_ret_10": [0.10, 0.09, 0.07],
                }
            ),
        },
        "education": {
            "rule_book": [{"title": "T+1", "detail": "测试规则"}],
            "steps": [{"title": "先选日期", "detail": "测试步骤"}],
            "risk_focus": ["测试风险"],
        },
    }

    monkeypatch.setattr(user_report, "build_dashboard_payload", lambda **kwargs: payload)
    monkeypatch.setattr(user_report, "export_dashboard_payload", lambda payload, out_path: Path(out_path))

    out = user_report.render_html_report(
        selected_date="2025-01-02",
        stock_symbol="000001",
        stock_date="2025-01-02",
        report_dir=tmp_path,
    )

    html = out.read_text(encoding="utf-8")
    demo = out.with_name("user_friendly_quant_demo.html").read_text(encoding="utf-8")
    assert out.exists()
    assert "data:image/png;base64" in html
    assert "A股网页量化交易系统报告" in html
    assert "A股网页量化交易系统演示页" in demo


def test_render_html_report_handles_unmatured_actual_returns(tmp_path, monkeypatch):
    payload = {
        "simulation": {
            "summary": {
                "selected_start_date": "2026-03-10",
                "effective_start_trade_date": "2026-03-10",
                "latest_trade_date": "2026-03-18",
                "total_pnl": 1000.0,
                "return_rate": 0.001,
                "max_drawdown": -0.01,
                "trade_count": 1,
                "win_rate": 1.0,
                "ending_asset": 1001000.0,
            },
            "curve": pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                    "total_asset": [1000000.0, 1001000.0],
                    "drawdown": [0.0, 0.0],
                }
            ),
            "trades": pd.DataFrame(
                [
                    {
                        "ts_code": "600397",
                        "buy_date": "2026-03-11",
                        "sell_date": "2026-03-12",
                        "exit_reason": "到期卖出",
                        "pnl": 1000.0,
                        "realized_return": 0.01,
                    }
                ]
            ),
        },
        "recommendations": {
            "summary": {
                "candidate_count": 1,
                "capital_usage_rate": 0.2,
                "explanation": "测试未成熟收益场景",
            },
            "rows": [
                {
                    "ts_code": "600397",
                    "direction": "预计上涨",
                    "buy_date": "2026-03-11",
                    "suggested_shares": 1000,
                    "suggested_amount": 100000.0,
                    "expected_profit": 2000.0,
                    "max_possible_loss": 7000.0,
                    "predicted_ret_10": 0.02,
                    "actual_ret_10": None,
                    "conviction_badge": "观察级",
                }
            ],
        },
        "stock_review": {
            "symbol": "600397",
            "selected_date": "2026-03-10",
            "effective_trade_date": "2026-03-10",
            "prediction_error": None,
            "series": pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2026-03-10", "2026-03-11"]),
                    "close": [10.0, 10.1],
                    "expected_ret_10": [0.02, 0.01],
                    "fwd_ret_10": [None, None],
                }
            ),
        },
        "education": {
            "rule_book": [{"title": "T+1", "detail": "测试规则"}],
            "steps": [{"title": "先选日期", "detail": "测试步骤"}],
            "risk_focus": ["测试风险"],
        },
    }

    monkeypatch.setattr(user_report, "build_dashboard_payload", lambda **kwargs: payload)
    monkeypatch.setattr(user_report, "export_dashboard_payload", lambda payload, out_path: Path(out_path))

    out = user_report.render_html_report(
        selected_date="2026-03-10",
        stock_symbol="600397",
        stock_date="2026-03-10",
        report_dir=tmp_path,
    )

    assert out.exists()
