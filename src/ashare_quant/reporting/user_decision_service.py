from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from ashare_quant.config.research_config import ResearchCfg, load_config
from ashare_quant.factors.price_volume import add_basic_factors
from ashare_quant.pipeline.stage12_dual_signal_strategy import (
    apply_calibration,
    build_multifactor_frame,
    enrich_with_fundamentals,
    fit_predict_multifactor,
    load_calibration,
)
from ashare_quant.universe.filtering import apply_universe_filters


DEFAULT_CACHE_NAME = "stage18_user_decision_predictions.parquet"
DEFAULT_REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class PendingOrder:
    symbol: str
    planned_date: pd.Timestamp
    signal_date: pd.Timestamp
    budget: float
    predicted_ret_10: float
    rule_score: float
    risk_score: float
    rank: int


@dataclass(frozen=True)
class Position:
    symbol: str
    buy_signal_date: pd.Timestamp
    buy_date: pd.Timestamp
    buy_price: float
    shares: int
    cost_basis: float
    predicted_ret_10: float
    stop_loss_price: float
    take_profit_price: float
    max_hold_date: pd.Timestamp


def _normalize_symbol(symbol: str) -> str:
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if not digits:
        raise ValueError("symbol must contain digits")
    return digits.zfill(6)


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _resolve_trade_date(dates: list[pd.Timestamp], selected_date: str | pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    target = pd.Timestamp(selected_date).normalize()
    eligible = [d for d in dates if d <= target]
    if eligible:
        resolved = eligible[-1]
        return resolved, resolved != target
    return dates[0], dates[0] != target


def _next_trade_date(dates: list[pd.Timestamp], current_date: pd.Timestamp) -> pd.Timestamp | None:
    for trade_date in dates:
        if trade_date > current_date:
            return trade_date
    return None


def _calc_drawdown(series: pd.Series) -> pd.Series:
    equity = series.astype(float)
    return equity / equity.cummax() - 1.0


def _risk_badge(score: float) -> str:
    if score <= 0.25:
        return "低风险"
    if score <= 0.5:
        return "中等风险"
    return "偏高风险"


def _conviction_badge(rank: int, predicted_ret_10: float) -> str:
    if rank <= 3 and predicted_ret_10 >= 0.08:
        return "高优先级"
    if rank <= 8 and predicted_ret_10 >= 0.03:
        return "中优先级"
    return "观察级"


def _build_reason_text(row: pd.Series) -> str:
    rank = int(row["stock_rank"])
    predicted_ret = float(row["expected_ret_10"])
    risk_score = float(row["risk_score"])
    return (
        f"排名第{rank}，预测10日收益 {_fmt_pct(predicted_ret)}，"
        f"规则分 {float(row['rule_score']):.2f}，风险分 {risk_score:.2f}。"
    )


def _build_recommendation_summary(rows: list[dict], initial_capital: float) -> dict:
    total_amount = float(sum(row["suggested_amount"] for row in rows))
    total_expected_profit = float(sum(row["expected_profit"] for row in rows))
    total_max_loss = float(sum(row["max_possible_loss"] for row in rows))
    count = len(rows)
    return {
        "candidate_count": count,
        "capital_in_use": total_amount,
        "capital_usage_rate": total_amount / initial_capital if initial_capital > 0 else 0.0,
        "expected_profit_total": total_expected_profit,
        "expected_return_on_used_capital": total_expected_profit / total_amount if total_amount > 0 else 0.0,
        "max_loss_total": total_max_loss,
        "headline": (
            f"当天共有 {count} 只推荐股。"
            if count
            else "当天没有满足规则的买入候选。"
        ),
        "explanation": (
            f"若等权执行，预计使用资金 {total_amount:,.0f}，"
            f"对应预计收益 {total_expected_profit:,.0f}，"
            f"按止损规则估算的最大风险 {total_max_loss:,.0f}。"
            if count
            else "系统未找到同时满足可买、排名靠前、预测收益为正的股票。"
        ),
    }


def _build_simulation_story(summary: dict) -> str:
    pnl = float(summary["total_pnl"])
    return_rate = float(summary["return_rate"])
    drawdown = float(summary["max_drawdown"])
    trades = int(summary["trade_count"])
    win_rate = float(summary["win_rate"])
    direction = "赚到钱" if pnl >= 0 else "出现亏损"
    return (
        f"从 {summary['effective_start_trade_date']} 开始按系统规则执行，"
        f"到 {summary['latest_trade_date']} 为止总体{direction}，"
        f"累计收益 {_fmt_pct(return_rate)}，最大回撤 {_fmt_pct(drawdown)}，"
        f"共完成 {trades} 笔交易，胜率 {_fmt_pct(win_rate)}。"
    )


def _build_stock_review_story(review: dict) -> str:
    actual_ret = review["actual_ret_10"]
    actual_text = _fmt_pct(actual_ret) if actual_ret is not None else "尚未走完观察期"
    pnl = float(review["pnl"])
    return (
        f"{review['symbol']} 在 {review['effective_trade_date']} 的判断为“{review['predicted_direction']}”，"
        f"系统预计 10 日收益 {_fmt_pct(review['predicted_ret_10'])}，"
        f"真实 10 日收益 {actual_text}。"
        f"若按模拟规则在 {review['buy_date']} 买入并在 {review['exit_date']} {review['exit_reason']}，"
        f"本次盈亏 {pnl:,.0f}，收益率 {_fmt_pct(review['return_rate'])}。"
    )


def _build_rule_book(cfg: ResearchCfg) -> list[dict]:
    return [
        {
            "title": "T+1 执行",
            "detail": "信号在当天产生，下一交易日开盘价尝试买入，避免把当日收盘后的信息假装提前知道。",
        },
        {
            "title": "止盈止损",
            "detail": f"达到 {_fmt_pct(cfg.risk.take_profit_partial)} 止盈，或回撤到 {abs(cfg.risk.stop_loss) * 100:.2f}% 止损就卖出。",
        },
        {
            "title": "持仓上限",
            "detail": f"最多同时持有 {cfg.risk.max_positions} 只股票，避免资金过度集中。",
        },
        {
            "title": "持有期限",
            "detail": f"单只股票最长持有 {cfg.backtest.hold_days} 个交易日，信号转弱也会提前卖出。",
        },
    ]


def _build_user_steps() -> list[dict]:
    return [
        {"title": "先选日期", "detail": "系统会定位到这一天或最近一个交易日。"},
        {"title": "看当天建议", "detail": "重点看买什么、建议买多少、预计能赚多少、最多可能亏多少。"},
        {"title": "看历史模拟", "detail": "验证如果当时真的照做，到今天总收益、回撤、胜率会怎样。"},
        {"title": "做单股复盘", "detail": "输入一只股票和某一天，检查系统当时的预测是否靠谱。"},
    ]


def build_prediction_cache(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    config_path: str = "configs/research.yaml",
) -> pd.DataFrame:
    cfg = load_config(config_path)
    data_path = Path(cfg.data.all_buyable_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    raw = pd.read_parquet(data_path)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["ts_code"] = raw["ts_code"].astype(str).str.zfill(6)

    raw = apply_universe_filters(raw)
    raw = add_basic_factors(raw)
    raw = enrich_with_fundamentals(raw)

    feature_frame = build_multifactor_frame(raw)
    scored, model_meta = fit_predict_multifactor(feature_frame)
    calibration = load_calibration(Path(report_dir))
    calibration["model_name"] = str(model_meta.get("model", "multifactor_top10"))
    scored = apply_calibration(scored, calibration)

    keep_cols = [
        "trade_date",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "buyable_today",
        "stock_rank",
        "rule_score",
        "risk_score",
        "final_score",
        "ml_score",
        "expected_ret_10",
        "fwd_ret_10",
    ]
    keep_cols = [col for col in keep_cols if col in scored.columns]
    out = scored[keep_cols].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    if "name" not in out.columns:
        out["name"] = out["ts_code"]
    out["direction"] = out["expected_ret_10"].apply(lambda value: "预计上涨" if float(value) >= 0 else "预计下跌")
    out["predicted_close_10"] = out["close"] * (1 + out["expected_ret_10"])
    out["actual_close_10"] = out["close"] * (1 + out["fwd_ret_10"])
    out["prediction_error"] = out["expected_ret_10"] - out["fwd_ret_10"]
    out.attrs["model_name"] = calibration["model_name"]
    return out.sort_values(["trade_date", "stock_rank", "ts_code"]).reset_index(drop=True)


def load_or_build_prediction_cache(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    config_path: str = "configs/research.yaml",
    cache_name: str = DEFAULT_CACHE_NAME,
) -> pd.DataFrame:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    cache_path = report_path / cache_name
    cfg = load_config(config_path)
    data_path = Path(cfg.data.all_buyable_path)

    if cache_path.exists() and data_path.exists() and cache_path.stat().st_mtime >= data_path.stat().st_mtime:
        frame = pd.read_parquet(cache_path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame.sort_values(["trade_date", "stock_rank", "ts_code"]).reset_index(drop=True)

    frame = build_prediction_cache(report_dir=report_path, config_path=config_path)
    frame.to_parquet(cache_path, index=False)
    return frame


def build_recommendation_table(
    frame: pd.DataFrame,
    selected_date: str | pd.Timestamp,
    cfg: ResearchCfg,
    initial_capital: float,
) -> dict:
    trade_dates = sorted(pd.to_datetime(frame["trade_date"]).drop_duplicates().tolist())
    if not trade_dates:
        raise RuntimeError("prediction cache is empty")

    effective_date, used_nearest = _resolve_trade_date(trade_dates, selected_date)
    buy_date = _next_trade_date(trade_dates, effective_date)
    if buy_date is None:
        raise RuntimeError("selected date is already the latest trade date and has no T+1 buy date")

    day = frame[frame["trade_date"] == effective_date].copy()
    picks = day[(day["buyable_today"]) & (day["expected_ret_10"] > 0)].sort_values(
        ["stock_rank", "expected_ret_10"],
        ascending=[True, False],
    ).head(cfg.risk.max_positions)

    max_positions = max(cfg.risk.max_positions, 1)
    per_position_budget = initial_capital / max_positions
    buy_rows: list[dict] = []
    buy_day = frame[frame["trade_date"] == buy_date].set_index("ts_code")

    for _, row in picks.iterrows():
        code = str(row["ts_code"]).zfill(6)
        if code not in buy_day.index:
            continue
        next_open = float(buy_day.loc[code, "open"])
        shares = int(per_position_budget // next_open // 100 * 100)
        amount = shares * next_open
        predicted_ret_10 = float(row["expected_ret_10"])
        risk_score = float(row["risk_score"])
        rank = int(row["stock_rank"])
        buy_rows.append(
            {
                "trade_date": effective_date.strftime("%Y-%m-%d"),
                "buy_date": buy_date.strftime("%Y-%m-%d"),
                "ts_code": code,
                "name": row.get("name", code),
                "direction": row["direction"],
                "predicted_ret_10": predicted_ret_10,
                "actual_ret_10": float(row["fwd_ret_10"]) if pd.notna(row["fwd_ret_10"]) else None,
                "current_close": float(row["close"]),
                "planned_buy_price": next_open,
                "planned_sell_rule": f"止盈 {_fmt_pct(cfg.risk.take_profit_partial)} / 止损 {abs(cfg.risk.stop_loss) * 100:.2f}% / 最长持有 {cfg.backtest.hold_days} 天",
                "suggested_shares": shares,
                "suggested_amount": amount,
                "score": float(row["final_score"]),
                "rank": rank,
                "risk_score": risk_score,
                "rule_score": float(row["rule_score"]),
                "risk_badge": _risk_badge(risk_score),
                "conviction_badge": _conviction_badge(rank, predicted_ret_10),
                "expected_profit": amount * predicted_ret_10,
                "max_possible_loss": amount * abs(float(cfg.risk.stop_loss)),
                "predicted_close_10": float(row["predicted_close_10"]),
                "actual_close_10": float(row["actual_close_10"]) if pd.notna(row["actual_close_10"]) else None,
                "reason": _build_reason_text(row),
            }
        )

    summary = _build_recommendation_summary(buy_rows, initial_capital=initial_capital)
    return {
        "selected_date": str(pd.Timestamp(selected_date).date()),
        "effective_trade_date": effective_date.strftime("%Y-%m-%d"),
        "used_nearest_trade_date": used_nearest,
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "rows": buy_rows,
        "summary": summary,
    }


def simulate_from_date(
    frame: pd.DataFrame,
    start_date: str | pd.Timestamp,
    cfg: ResearchCfg,
    initial_capital: float = 1_000_000.0,
) -> dict:
    trade_dates = sorted(pd.to_datetime(frame["trade_date"]).drop_duplicates().tolist())
    if not trade_dates:
        raise RuntimeError("prediction cache is empty")

    start_trade_date, used_nearest = _resolve_trade_date(trade_dates, start_date)
    day_map = {day: df.copy() for day, df in frame.groupby("trade_date")}

    cash = float(initial_capital)
    positions: dict[str, Position] = {}
    pending_orders: list[PendingOrder] = []
    trade_records: list[dict] = []
    curve_rows: list[dict] = []

    commission_rate = cfg.backtest.commission_bps / 10000.0
    slippage_rate = cfg.backtest.slippage_bps / 10000.0
    sell_tax_rate = cfg.backtest.stamp_duty_sell_bps / 10000.0

    for trade_date in trade_dates:
        if trade_date < start_trade_date:
            continue
        day = day_map[trade_date].copy().sort_values("stock_rank")
        day_index = day.set_index("ts_code")

        still_pending: list[PendingOrder] = []
        for order in pending_orders:
            if order.planned_date != trade_date:
                still_pending.append(order)
                continue
            if order.symbol not in day_index.index or order.symbol in positions:
                continue
            open_price = float(day_index.loc[order.symbol, "open"])
            gross_price = open_price * (1 + commission_rate + slippage_rate)
            shares = int(order.budget // gross_price // 100 * 100)
            if shares <= 0:
                continue
            turnover = shares * open_price
            fee = turnover * commission_rate
            slippage = turnover * slippage_rate
            total_cost = turnover + fee + slippage
            if total_cost > cash:
                continue
            cash -= total_cost
            max_hold_date = trade_dates[min(trade_dates.index(trade_date) + cfg.backtest.hold_days - 1, len(trade_dates) - 1)]
            positions[order.symbol] = Position(
                symbol=order.symbol,
                buy_signal_date=order.signal_date,
                buy_date=trade_date,
                buy_price=open_price,
                shares=shares,
                cost_basis=total_cost,
                predicted_ret_10=order.predicted_ret_10,
                stop_loss_price=open_price * (1 + cfg.risk.stop_loss),
                take_profit_price=open_price * (1 + cfg.risk.take_profit_partial),
                max_hold_date=max_hold_date,
            )
        pending_orders = still_pending

        for symbol, position in list(positions.items()):
            if symbol not in day_index.index:
                continue
            row = day_index.loc[symbol]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if trade_date <= position.buy_date:
                continue

            exit_reason = None
            exit_price = None
            if low <= position.stop_loss_price:
                exit_reason = "止损卖出"
                exit_price = position.stop_loss_price
            elif high >= position.take_profit_price:
                exit_reason = "止盈卖出"
                exit_price = position.take_profit_price
            elif trade_date >= position.max_hold_date:
                exit_reason = "到期卖出"
                exit_price = close
            elif bool(row["stock_rank"] > cfg.backtest.top_n or not row["buyable_today"]):
                exit_reason = "信号转弱卖出"
                exit_price = close

            if exit_reason is None:
                continue

            turnover = position.shares * float(exit_price)
            fee = turnover * commission_rate
            slippage = turnover * slippage_rate
            tax = turnover * sell_tax_rate
            net_cash = turnover - fee - slippage - tax
            cash += net_cash
            pnl = net_cash - position.cost_basis
            trade_records.append(
                {
                    "ts_code": symbol,
                    "buy_signal_date": position.buy_signal_date.strftime("%Y-%m-%d"),
                    "buy_date": position.buy_date.strftime("%Y-%m-%d"),
                    "sell_date": trade_date.strftime("%Y-%m-%d"),
                    "buy_price": position.buy_price,
                    "sell_price": float(exit_price),
                    "shares": position.shares,
                    "predicted_ret_10": position.predicted_ret_10,
                    "realized_return": net_cash / position.cost_basis - 1,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "holding_days": int((trade_date - position.buy_date).days),
                }
            )
            del positions[symbol]

        open_slots = max(cfg.risk.max_positions - len(positions) - len(pending_orders), 0)
        if open_slots > 0:
            candidates = day[day["buyable_today"]].sort_values(["stock_rank", "expected_ret_10"], ascending=[True, False])
            next_date = _next_trade_date(trade_dates, trade_date)
            if next_date is not None:
                budget_per_order = min(cash / max(open_slots, 1), initial_capital / max(cfg.risk.max_positions, 1))
                for _, row in candidates.iterrows():
                    code = str(row["ts_code"]).zfill(6)
                    if code in positions or any(order.symbol == code for order in pending_orders):
                        continue
                    if int(row["stock_rank"]) > cfg.backtest.top_n or float(row["expected_ret_10"]) <= 0:
                        continue
                    pending_orders.append(
                        PendingOrder(
                            symbol=code,
                            planned_date=next_date,
                            signal_date=trade_date,
                            budget=budget_per_order,
                            predicted_ret_10=float(row["expected_ret_10"]),
                            rule_score=float(row["rule_score"]),
                            risk_score=float(row["risk_score"]),
                            rank=int(row["stock_rank"]),
                        )
                    )
                    open_slots -= 1
                    if open_slots <= 0:
                        break

        market_value = 0.0
        for symbol, position in positions.items():
            if symbol in day_index.index:
                market_value += position.shares * float(day_index.loc[symbol, "close"])
        total_asset = cash + market_value
        curve_rows.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "market_value": market_value,
                "total_asset": total_asset,
                "n_positions": len(positions),
                "n_pending": len(pending_orders),
            }
        )

    curve = pd.DataFrame(curve_rows)
    curve["daily_return"] = curve["total_asset"].pct_change().fillna(0.0)
    curve["cum_return"] = curve["total_asset"] / initial_capital - 1.0
    curve["drawdown"] = _calc_drawdown(curve["total_asset"])

    trades = pd.DataFrame(trade_records)
    latest_asset = float(curve["total_asset"].iloc[-1]) if not curve.empty else initial_capital
    summary = {
        "selected_start_date": str(pd.Timestamp(start_date).date()),
        "effective_start_trade_date": start_trade_date.strftime("%Y-%m-%d"),
        "used_nearest_trade_date": used_nearest,
        "latest_trade_date": trade_dates[-1].strftime("%Y-%m-%d"),
        "initial_capital": initial_capital,
        "ending_asset": latest_asset,
        "total_pnl": latest_asset - initial_capital,
        "return_rate": latest_asset / initial_capital - 1.0,
        "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else 0.0,
        "trade_count": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
        "open_position_count": len(positions),
        "cash": cash,
    }
    summary["story"] = _build_simulation_story(summary)
    return {
        "summary": summary,
        "curve": curve,
        "trades": trades,
    }


def build_single_stock_review(
    frame: pd.DataFrame,
    symbol: str,
    selected_date: str | pd.Timestamp,
    cfg: ResearchCfg,
    initial_capital: float = 200_000.0,
) -> dict:
    code = _normalize_symbol(symbol)
    stock = frame[frame["ts_code"].astype(str).str.zfill(6) == code].copy()
    if stock.empty:
        raise RuntimeError(f"no prediction history found for {code}")

    trade_dates = sorted(pd.to_datetime(stock["trade_date"]).drop_duplicates().tolist())
    effective_date, used_nearest = _resolve_trade_date(trade_dates, selected_date)
    row = stock[stock["trade_date"] == effective_date].iloc[-1]
    next_date = _next_trade_date(trade_dates, effective_date)
    if next_date is None:
        raise RuntimeError("selected stock date has no T+1 execution date")

    buy_row = stock[stock["trade_date"] == next_date].iloc[-1]
    buy_price = float(buy_row["open"])
    stop_loss_price = buy_price * (1 + cfg.risk.stop_loss)
    take_profit_price = buy_price * (1 + cfg.risk.take_profit_partial)
    shares = int((initial_capital // buy_price) // 100 * 100)
    amount = shares * buy_price

    exit_reason = "到期卖出"
    exit_date = next_date
    exit_price = float(buy_row["close"])

    future = stock[stock["trade_date"] >= next_date].sort_values("trade_date").head(cfg.backtest.hold_days + 1)
    for _, day in future.iterrows():
        day_date = pd.Timestamp(day["trade_date"])
        if day_date <= next_date:
            continue
        if float(day["low"]) <= stop_loss_price:
            exit_reason = "止损卖出"
            exit_date = day_date
            exit_price = stop_loss_price
            break
        if float(day["high"]) >= take_profit_price:
            exit_reason = "止盈卖出"
            exit_date = day_date
            exit_price = take_profit_price
            break
        exit_date = day_date
        exit_price = float(day["close"])

    pnl = shares * (exit_price - buy_price)
    actual_ret = float(row["fwd_ret_10"]) if pd.notna(row["fwd_ret_10"]) else None
    actual_direction = "预计上涨" if (actual_ret or 0.0) >= 0 else "预计下跌"

    review = {
        "symbol": code,
        "selected_date": str(pd.Timestamp(selected_date).date()),
        "effective_trade_date": effective_date.strftime("%Y-%m-%d"),
        "used_nearest_trade_date": used_nearest,
        "predicted_direction": row["direction"],
        "predicted_ret_10": float(row["expected_ret_10"]),
        "actual_ret_10": actual_ret,
        "prediction_error": float(row["prediction_error"]) if pd.notna(row["prediction_error"]) else None,
        "rank": int(row["stock_rank"]),
        "buyable_today": bool(row["buyable_today"]),
        "buy_date": next_date.strftime("%Y-%m-%d"),
        "buy_price": buy_price,
        "exit_date": exit_date.strftime("%Y-%m-%d"),
        "exit_price": exit_price,
        "shares": shares,
        "amount": amount,
        "pnl": pnl,
        "return_rate": pnl / amount if amount > 0 else 0.0,
        "exit_reason": exit_reason,
        "accuracy_label": "方向判断正确" if row["direction"] == actual_direction else "方向判断偏差",
        "story": "",
        "series": stock.sort_values("trade_date").reset_index(drop=True),
    }
    review["story"] = _build_stock_review_story(review)
    return review


def build_dashboard_payload(
    selected_date: str | pd.Timestamp,
    stock_symbol: str,
    stock_date: str | pd.Timestamp,
    initial_capital: float = 1_000_000.0,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    config_path: str = "configs/research.yaml",
) -> dict:
    cfg = load_config(config_path)
    frame = load_or_build_prediction_cache(report_dir=report_dir, config_path=config_path)
    recommendations = build_recommendation_table(frame, selected_date=selected_date, cfg=cfg, initial_capital=initial_capital)
    simulation = simulate_from_date(frame, start_date=selected_date, cfg=cfg, initial_capital=initial_capital)
    stock_review = build_single_stock_review(frame, symbol=stock_symbol, selected_date=stock_date, cfg=cfg)
    effective_dates = pd.to_datetime(frame["trade_date"])
    return {
        "config": {
            "top_n": cfg.backtest.top_n,
            "hold_days": cfg.backtest.hold_days,
            "stop_loss": cfg.risk.stop_loss,
            "take_profit_partial": cfg.risk.take_profit_partial,
            "max_positions": cfg.risk.max_positions,
        },
        "metadata": {
            "date_min": str(effective_dates.min().date()),
            "date_max": str(effective_dates.max().date()),
            "symbol_count": int(frame["ts_code"].astype(str).nunique()),
            "model_name": str(frame.attrs.get("model_name", "multifactor_top10")),
        },
        "recommendations": recommendations,
        "simulation": simulation,
        "stock_review": {
            **{k: v for k, v in stock_review.items() if k != "series"},
            "series": stock_review["series"],
        },
        "education": {
            "rule_book": _build_rule_book(cfg),
            "steps": _build_user_steps(),
            "risk_focus": [
                "先看最大回撤，再看总收益。能不能承受回撤，比短期多赚几个点更重要。",
                "单只股票只是样本，系统是否稳定要结合整个历史模拟一起看。",
                "预测收益是概率表达，不是保证收益，重点是长期规则是否有效。",
            ],
        },
    }


def export_dashboard_payload(payload: dict, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exportable = payload.copy()
    exportable["simulation"] = {
        **payload["simulation"],
        "curve": payload["simulation"]["curve"].to_dict(orient="records"),
        "trades": payload["simulation"]["trades"].to_dict(orient="records"),
    }
    exportable["stock_review"] = {
        **{k: v for k, v in payload["stock_review"].items() if k != "series"},
        "series": payload["stock_review"]["series"].to_dict(orient="records"),
    }
    path.write_text(json.dumps(exportable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
