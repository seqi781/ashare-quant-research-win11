import pandas as pd

from ashare_quant.backtest.rolling_training import BenchmarkSpec, load_benchmark_series, run_rolling_training_backtest
from ashare_quant.config.research_config import RollingBacktestCfg


class IdentityModel:
    def fit(self, x, y):
        self.fitted_rows = len(x)
        return self

    def predict(self, x):
        return x["feature"].to_numpy()


def _factory():
    return IdentityModel()


def _sample_frame():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for di, d in enumerate(dates):
        for code, bonus in [("AAA", 0.3), ("BBB", 0.2), ("CCC", 0.1)]:
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "close": 10 + di + bonus,
                    "feature": di + bonus,
                    "target": di / 100 + bonus / 10,
                }
            )
    return pd.DataFrame(rows)


def test_rolling_backtest_uses_only_past_data():
    df = _sample_frame()
    cfg = RollingBacktestCfg(
        start_date="2024-01-01",
        end_date="2024-01-31",
        train_window_days=3,
        rebalance_frequency_days=2,
        prediction_horizon_days=1,
        retrain_frequency_days=2,
        training_mode="fixed",
        top_n=2,
        weighting="equal",
        benchmark="",
        benchmark_path="",
        min_train_rows=3,
    )

    res = run_rolling_training_backtest(
        df=df,
        feature_cols=["feature"],
        target_col="target",
        score_col="score",
        cfg=cfg,
        model_factory=_factory,
    )

    nodes = res["nodes"]
    assert not nodes.empty
    assert (pd.to_datetime(nodes["train_end_date"]) < pd.to_datetime(nodes["rebalance_date"])).all()
    assert (nodes["train_days"] == 3).all()


def test_benchmark_series_and_summary_metrics():
    df = _sample_frame()
    cfg = RollingBacktestCfg(
        start_date="2024-01-01",
        end_date="2024-01-31",
        train_window_days=3,
        rebalance_frequency_days=2,
        prediction_horizon_days=1,
        retrain_frequency_days=2,
        training_mode="expanding",
        top_n=1,
        weighting="score",
        benchmark="",
        benchmark_path="",
        min_train_rows=3,
    )
    bench_raw = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-01", periods=10),
            "close": [100, 101, 100, 102, 103, 104, 103, 105, 106, 107],
        }
    )
    benchmark = load_benchmark_series(spec=BenchmarkSpec(), raw=bench_raw, start_date="2024-01-01", end_date="2024-01-31")

    res = run_rolling_training_backtest(
        df=df,
        feature_cols=["feature"],
        target_col="target",
        score_col="score",
        cfg=cfg,
        model_factory=_factory,
        benchmark=benchmark,
    )

    summary = res["summary"]
    curve = res["curve"]
    assert "benchmark_ann_return" in summary
    assert "information_ratio" in summary
    assert "benchmark_daily_ret" in curve.columns
