from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class DataCfg:
    all_buyable_path: str
    start_date: str
    end_date: str

@dataclass(frozen=True)
class BacktestCfg:
    top_n: int
    hold_days: int
    commission_bps: float
    slippage_bps: float
    stamp_duty_sell_bps: float

@dataclass(frozen=True)
class RiskCfg:
    stop_loss: float
    take_profit_partial: float
    max_positions: int


@dataclass(frozen=True)
class RollingBacktestCfg:
    start_date: str
    end_date: str
    train_window_days: int
    rebalance_frequency_days: int
    prediction_horizon_days: int
    retrain_frequency_days: int
    training_mode: str
    top_n: int
    weighting: str
    benchmark: str
    benchmark_path: str
    min_train_rows: int = 2000

@dataclass(frozen=True)
class ResearchCfg:
    data: DataCfg
    backtest: BacktestCfg
    risk: RiskCfg
    rolling_backtest: RollingBacktestCfg


def load_config(path: str = 'configs/research.yaml') -> ResearchCfg:
    p = Path(path)
    d = yaml.safe_load(p.read_text(encoding='utf-8'))
    rolling = d.get('rolling_backtest', {})
    rolling_defaults = {
        'start_date': d['data']['start_date'],
        'end_date': d['data']['end_date'],
        'train_window_days': 252,
        'rebalance_frequency_days': d['backtest']['hold_days'],
        'prediction_horizon_days': d['backtest']['hold_days'],
        'retrain_frequency_days': d['backtest']['hold_days'],
        'training_mode': 'fixed',
        'top_n': d['backtest']['top_n'],
        'weighting': 'equal',
        'benchmark': '000300.SH',
        'benchmark_path': '',
        'min_train_rows': 2000,
    }
    rolling_defaults.update(rolling)
    return ResearchCfg(
        data=DataCfg(**d['data']),
        backtest=BacktestCfg(**d['backtest']),
        risk=RiskCfg(**d['risk']),
        rolling_backtest=RollingBacktestCfg(**rolling_defaults),
    )
