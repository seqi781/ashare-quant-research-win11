from ashare_quant.config.research_config import load_config

def test_load_config():
    c = load_config('configs/research.yaml')
    assert c.backtest.top_n > 0
    assert c.data.all_buyable_path.endswith('.parquet')
    assert c.rolling_backtest.train_window_days > 0
    assert c.rolling_backtest.training_mode in {'fixed', 'expanding'}
