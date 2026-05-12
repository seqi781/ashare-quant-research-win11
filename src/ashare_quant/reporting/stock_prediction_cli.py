from __future__ import annotations

import argparse

import pandas as pd

from ashare_quant.reporting.stock_prediction_analysis import run_single_stock_prediction_analysis


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze single-stock prediction quality and trading signals.")
    parser.add_argument("--symbol", required=True, help="A-share code, for example 600397 or 688343.")
    parser.add_argument("--report-dir", default="reports", help="Output directory for generated artifacts.")
    parser.add_argument("--buy-threshold", type=float, default=0.02, help="BUY when expected 10-day return is above this threshold.")
    parser.add_argument("--sell-threshold", type=float, default=-0.02, help="SELL when expected 10-day return is below this threshold.")
    parser.add_argument("--top-n", type=int, default=10, help="Only treat ranks within top N as BUY candidates.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_single_stock_prediction_analysis(
        symbol=args.symbol,
        out_dir=args.report_dir,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        top_n=args.top_n,
    )
    summary = result["summary"]
    recent = result["recent_actions"].copy()
    recent["trade_date"] = pd.to_datetime(recent["trade_date"]).dt.strftime("%Y-%m-%d")
    recent["predicted_ret_10"] = recent["predicted_ret_10"].map(_fmt_pct)
    recent["actual_ret_10"] = recent["actual_ret_10"].map(_fmt_pct)

    print(f"Single Stock Prediction Analysis: {summary['symbol']}")
    print(f"Range: {summary['start_date']} -> {summary['end_date']}")
    print(
        "Latest: "
        f"date={summary['latest_trade_date']}, "
        f"close={_fmt_num(summary['latest_close'], 3)}, "
        f"rank={summary['latest_stock_rank']}, "
        f"raw_signal={summary['latest_raw_signal']}, "
        f"action={summary['latest_action']}, "
        f"pred_ret_10={_fmt_pct(summary['latest_predicted_ret_10'])}"
    )
    print(
        "Prediction: "
        f"mae={_fmt_pct(summary['mae'])}, "
        f"rmse={_fmt_pct(summary['rmse'])}, "
        f"bias={_fmt_pct(summary['bias'])}, "
        f"dir_acc={_fmt_pct(summary['directional_accuracy'])}, "
        f"corr={_fmt_num(summary['corr'])}"
    )
    print(
        "Trend: "
        f"trend_acc={_fmt_pct(summary['trend_direction_accuracy'])}, "
        f"turn_precision={_fmt_pct(summary['turning_point_precision'])}, "
        f"turn_recall={_fmt_pct(summary['turning_point_recall'])}"
    )
    print(
        "Signals: "
        f"buy_count={summary['buy_signal_count']}, "
        f"sell_count={summary['sell_signal_count']}, "
        f"profitable_buy_rate={_fmt_pct(summary['profitable_buy_signal_rate'])}, "
        f"avg_buy_actual_ret_10={_fmt_pct(summary['avg_buy_signal_actual_ret_10'])}, "
        f"avg_sell_actual_ret_10={_fmt_pct(summary['avg_sell_signal_actual_ret_10'])}"
    )
    print(
        "Trading: "
        f"strategy_cum={_fmt_pct(summary['strategy_cumulative_return'])}, "
        f"spot_cum={_fmt_pct(summary['spot_cumulative_return'])}, "
        f"hold_ratio={_fmt_pct(summary['holding_day_ratio'])}, "
        f"rank_range={summary['min_rank']}->{summary['max_rank']}, "
        f"avg_rank={_fmt_num(summary['avg_rank'], 1)}"
    )
    print("")
    print("Recent Actions")
    print(recent.to_string(index=False))
    print("")
    print(f"Summary JSON: {result['summary_path']}")
    print(f"Timeseries:   {result['timeseries_path']}")
    print(f"Chart HTML:   {result['chart_path']}")


if __name__ == "__main__":
    main()
