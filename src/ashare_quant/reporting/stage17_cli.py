from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_REPORT_DIR = Path("reports")


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _print_summary(summary: dict) -> None:
    cfg = summary.get("config", {})
    print("Stage17 Rolling Backtest")
    print(f"Latest rebalance: {summary.get('latest_rebalance_date', '-')}")
    print(f"Range: {cfg.get('start_date', '-')} -> {cfg.get('end_date', '-')}")
    print(
        "Mode: "
        f"{cfg.get('training_mode', '-')} | train_window={cfg.get('train_window_days', '-')} | "
        f"rebalance={cfg.get('rebalance_frequency_days', '-')} | horizon={cfg.get('prediction_horizon_days', '-')}"
    )
    print(
        "Perf: "
        f"cum={_fmt_pct(summary.get('cumulative_return'))}, "
        f"ann={_fmt_pct(summary.get('ann_return'))}, "
        f"mdd={_fmt_pct(summary.get('max_drawdown'))}, "
        f"sharpe={_fmt_num(summary.get('sharpe'))}, "
        f"win={_fmt_pct(summary.get('win_rate'))}"
    )
    print(
        "Flow: "
        f"days={summary.get('days', '-')}, "
        f"nodes={summary.get('rebalance_nodes', '-')}, "
        f"avg_turnover={_fmt_pct(summary.get('avg_turnover'))}, "
        f"annual_turnover={_fmt_num(summary.get('annual_turnover'))}"
    )
    if "benchmark_ann_return" in summary:
        print(
            "Benchmark: "
            f"ann={_fmt_pct(summary.get('benchmark_ann_return'))}, "
            f"cum={_fmt_pct(summary.get('benchmark_cumulative_return'))}, "
            f"alpha={_fmt_pct(summary.get('alpha_ann'))}, "
            f"ir={_fmt_num(summary.get('information_ratio'))}"
        )


def _print_latest_picks(summary: dict, limit: int) -> None:
    picks = summary.get("latest_picks", [])[:limit]
    print("")
    print(f"Latest Picks Top {len(picks)}")
    if not picks:
        print("(empty)")
        return
    frame = pd.DataFrame(picks)
    keep = [c for c in ["ts_code", "score", "target_weight", "signal"] if c in frame.columns]
    if "target_weight" in frame.columns:
        frame["target_weight"] = frame["target_weight"].map(_fmt_pct)
    if "score" in frame.columns:
        frame["score"] = frame["score"].map(lambda v: _fmt_num(v, 6))
    print(frame[keep].to_string(index=False))


def _print_recent_nodes(nodes: pd.DataFrame, limit: int) -> None:
    print("")
    print(f"Recent Nodes {min(limit, len(nodes))}")
    if nodes.empty:
        print("(empty)")
        return
    show = nodes.tail(limit).copy()
    for col in ["rebalance_date", "train_start_date", "train_end_date"]:
        if col in show.columns:
            show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d")
    if "turnover" in show.columns:
        show["turnover"] = show["turnover"].map(_fmt_num)
    keep = [c for c in ["rebalance_date", "train_start_date", "train_end_date", "train_rows", "retrained", "n_selected", "turnover"] if c in show.columns]
    print(show[keep].to_string(index=False))


def _print_recent_actions(actions: pd.DataFrame, limit: int) -> None:
    print("")
    print(f"Recent Actions {min(limit, len(actions))}")
    if actions.empty:
        print("(empty)")
        return
    show = actions.tail(limit).copy()
    if "rebalance_date" in show.columns:
        show["rebalance_date"] = pd.to_datetime(show["rebalance_date"]).dt.strftime("%Y-%m-%d")
    for col in ["prev_weight", "target_weight"]:
        if col in show.columns:
            show[col] = show[col].map(_fmt_pct)
    if "score" in show.columns:
        show["score"] = show["score"].map(lambda v: _fmt_num(v, 6))
    keep = [c for c in ["rebalance_date", "action", "ts_code", "prev_weight", "target_weight", "score"] if c in show.columns]
    print(show[keep].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View stage17 rolling backtest results from the command line.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory containing stage17 report files.")
    parser.add_argument("--top", type=int, default=10, help="How many latest picks to show.")
    parser.add_argument("--nodes", type=int, default=5, help="How many recent rebalance nodes to show.")
    parser.add_argument("--actions", type=int, default=10, help="How many recent actions to show.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_dir = Path(args.report_dir)

    summary = _load_json(report_dir / "stage17_rolling_backtest_summary.json")
    nodes = _load_parquet(report_dir / "stage17_rolling_nodes.parquet")
    actions = _load_parquet(report_dir / "stage17_rolling_actions.parquet")

    _print_summary(summary)
    _print_latest_picks(summary, max(args.top, 0))
    _print_recent_nodes(nodes, max(args.nodes, 0))
    _print_recent_actions(actions, max(args.actions, 0))


if __name__ == "__main__":
    main()
