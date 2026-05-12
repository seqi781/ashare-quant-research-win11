from __future__ import annotations

import argparse

from ashare_quant.reporting.user_report import render_html_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a user-friendly A-share quant HTML report.")
    parser.add_argument("--start-date", required=True, help="Simulation start date, for example 2025-01-02.")
    parser.add_argument("--stock", required=True, help="Stock code for single-stock review, for example 600397.")
    parser.add_argument("--stock-date", required=True, help="Review date for the selected stock.")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Initial capital for simulation.")
    parser.add_argument("--report-dir", default="reports", help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = render_html_report(
        selected_date=args.start_date,
        stock_symbol=args.stock,
        stock_date=args.stock_date,
        initial_capital=args.initial_capital,
        report_dir=args.report_dir,
    )
    print(path)
    print(path.with_name("user_friendly_quant_demo.html"))


if __name__ == "__main__":
    main()
