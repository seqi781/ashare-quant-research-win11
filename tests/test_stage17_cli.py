import json
from pathlib import Path

import pandas as pd

from ashare_quant.reporting.stage17_cli import _load_json, _load_parquet


def test_stage17_cli_loaders(tmp_path: Path):
    summary_path = tmp_path / "stage17_rolling_backtest_summary.json"
    nodes_path = tmp_path / "stage17_rolling_nodes.parquet"

    summary_path.write_text(json.dumps({"status": "ok", "latest_picks": []}), encoding="utf-8")
    pd.DataFrame([{"rebalance_date": "2026-03-03", "train_rows": 1000}]).to_parquet(nodes_path, index=False)

    summary = _load_json(summary_path)
    nodes = _load_parquet(nodes_path)

    assert summary["status"] == "ok"
    assert int(nodes.iloc[0]["train_rows"]) == 1000
