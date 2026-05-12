#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_DATE="${1:-2025-01-02}"
STOCK_CODE="${2:-600397}"
STOCK_DATE="${3:-2025-01-02}"
INITIAL_CAPITAL="${4:-1000000}"

exec uv run python -m ashare_quant.reporting.user_report_cli \
  --start-date "$START_DATE" \
  --stock "$STOCK_CODE" \
  --stock-date "$STOCK_DATE" \
  --initial-capital "$INITIAL_CAPITAL" \
  --report-dir reports
