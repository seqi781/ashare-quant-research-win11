from __future__ import annotations

from pathlib import Path
import json
import time

import akshare as ak
import pandas as pd


CHECK_EVERY = 50
SLEEP_SEC = 0.03


def to_sina_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "data" / "stage4_all_buyable"
    data_path = base / "market_daily_all_buyable_20210101_20260314.parquet"
    sym_path = base / "buyable_symbols.csv"
    progress_path = base / "ak_incremental_progress.json"
    meta_path = base / "refresh_meta.json"
    err_path = base / "refresh_errors.log"

    if not data_path.exists() or not sym_path.exists():
        raise FileNotFoundError("missing parquet dataset or buyable_symbols.csv")

    old = pd.read_parquet(data_path)
    old["trade_date"] = pd.to_datetime(old["trade_date"])
    last_date = old["trade_date"].max().normalize()
    end_date = pd.Timestamp.today().normalize()

    start_ts = time.perf_counter()

    if last_date >= end_date:
        elapsed = time.perf_counter() - start_ts
        meta = {
            "mode": "akshare_incremental",
            "status": "up_to_date",
            "last_date_before": str(last_date.date()),
            "date_max_after": str(last_date.date()),
            "elapsed_seconds": round(elapsed, 3),
            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False))
        return

    sym = pd.read_csv(sym_path)
    if "ts_code" in sym.columns:
        symbols = sym["ts_code"].astype(str).str.zfill(6).drop_duplicates().tolist()
    else:
        symbols = sym["symbol"].astype(str).str.zfill(6).drop_duplicates().tolist()

    appended = []
    errs: list[str] = []
    target_start = last_date + pd.Timedelta(days=1)

    for idx, code in enumerate(symbols, start=1):
        try:
            df = ak.stock_zh_a_daily(symbol=to_sina_symbol(code), adjust="qfq")
            if df is None or df.empty:
                continue
            x = df.copy()
            x["trade_date"] = pd.to_datetime(x["date"])
            x = x[(x["trade_date"] >= target_start) & (x["trade_date"] <= end_date)].copy()
            if x.empty:
                continue
            out = pd.DataFrame(
                {
                    "trade_date": x["trade_date"],
                    "open": x["open"].astype(float),
                    "high": x["high"].astype(float),
                    "low": x["low"].astype(float),
                    "close": x["close"].astype(float),
                    "volume": x["volume"].astype(float),
                    "amount": x["amount"].astype(float),
                }
            )
            out["ts_code"] = code
            out["is_st"] = False
            out["is_suspended"] = False
            out["up_limit"] = out["close"] * 1.1
            out["down_limit"] = out["close"] * 0.9
            out["data_vendor"] = "sina_daily"
            appended.append(out)
        except Exception as exc:
            errs.append(f"{code}: {type(exc).__name__}: {exc}")

        if idx % CHECK_EVERY == 0 or idx == len(symbols):
            elapsed = time.perf_counter() - start_ts
            progress = {
                "mode": "akshare_incremental",
                "progress_symbols": idx,
                "total_symbols": len(symbols),
                "new_rows_raw": int(sum(len(x) for x in appended)),
                "errors": len(errs),
                "elapsed_seconds": round(elapsed, 2),
                "target_start": str(target_start.date()),
                "target_end": str(end_date.date()),
            }
            progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(progress, ensure_ascii=False), flush=True)
        time.sleep(SLEEP_SEC)

    if appended:
        new = pd.concat(appended, ignore_index=True)
        full = pd.concat([old, new], ignore_index=True)
        full = full.sort_values(["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)
        full["trade_date"] = pd.to_datetime(full["trade_date"]).dt.strftime("%Y-%m-%d")
        full.to_parquet(data_path, index=False)
    else:
        full = old.copy()
        full["trade_date"] = pd.to_datetime(full["trade_date"]).dt.strftime("%Y-%m-%d")

    elapsed = time.perf_counter() - start_ts
    meta = {
        "mode": "akshare_incremental",
        "status": "completed",
        "last_date_before": str(last_date.date()),
        "date_min_after": str(full["trade_date"].min()),
        "date_max_after": str(full["trade_date"].max()),
        "rows_after": int(len(full)),
        "symbols_after": int(full["ts_code"].astype(str).nunique()),
        "new_rows_raw": int(sum(len(x) for x in appended)),
        "fetch_errors": len(errs),
        "elapsed_seconds": round(elapsed, 3),
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if errs:
        err_path.write_text("\n".join(errs[-3000:]), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
