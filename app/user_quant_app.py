from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ashare_quant.reporting.user_decision_service import (
    build_dashboard_payload,
    load_or_build_prediction_cache,
)
from ashare_quant.reporting.user_report import render_html_report


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="A股量化交易系统", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --paper: rgba(255,255,255,0.92);
      --line: rgba(214, 211, 204, 0.88);
      --ink: #1f2937;
      --muted: #5b6472;
      --brand: #0f766e;
      --brand2: #1d4ed8;
      --soft: #f8f4ea;
    }
    .stApp {
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.11), transparent 28%),
        radial-gradient(circle at top right, rgba(29,78,216,0.10), transparent 24%),
        linear-gradient(180deg, #f8f2e6 0%, #edf3f8 100%);
    }
    .hero, .panel, .metric-card {
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }
    .hero {
      border-radius: 24px;
      padding: 28px 30px;
      background: linear-gradient(135deg, rgba(17,60,55,0.96), rgba(29,78,216,0.94));
      color: #fff;
      margin-bottom: 16px;
    }
    .hero p {
      margin: 10px 0 0 0;
      color: rgba(255,255,255,0.92);
      line-height: 1.7;
    }
    .panel {
      border-radius: 20px;
      padding: 18px 20px;
      margin-bottom: 16px;
    }
    .metric-card {
      border-radius: 18px;
      padding: 18px;
      min-height: 122px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 0.92rem;
    }
    .metric-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--ink);
      margin-top: 8px;
    }
    .metric-help {
      color: var(--muted);
      margin-top: 6px;
      font-size: 0.88rem;
    }
    .tag {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      margin-right: 8px;
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.18);
      font-size: 0.82rem;
    }
    .note {
      color: var(--muted);
      line-height: 1.75;
      font-size: 0.97rem;
    }
    .rule-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.76);
      height: 100%;
    }
    .rule-card h4 {
      margin: 0 0 8px 0;
    }
    .rule-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="tag">A 股量化</div>
      <div class="tag">普通用户可读</div>
      <h1 style="margin:0;">A股网页量化交易系统</h1>
      <p>你只需要选一个日期，系统就会告诉你当时推荐买哪些股票、预计涨跌、建议买多少、什么时候买、什么时候卖，以及如果后来一直照做，到现在整体是赚了还是亏了。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=True)
def load_prediction_frame() -> pd.DataFrame:
    return load_or_build_prediction_cache(report_dir=REPORTS)


@st.cache_data(show_spinner=True)
def load_payload(selected_date: str, stock_symbol: str, stock_date: str, initial_capital: float) -> dict:
    return build_dashboard_payload(
        selected_date=selected_date,
        stock_symbol=stock_symbol,
        stock_date=stock_date,
        initial_capital=initial_capital,
        report_dir=REPORTS,
    )


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


frame = load_prediction_frame()
trade_dates = sorted(pd.to_datetime(frame["trade_date"]).drop_duplicates().tolist())
default_trade_date = trade_dates[max(len(trade_dates) - 80, 0)].date()
default_stock = str(frame["ts_code"].iloc[0]).zfill(6)

sidebar = st.sidebar
sidebar.header("参数")
selected_date = sidebar.date_input("回测起点", value=default_trade_date, min_value=trade_dates[0].date(), max_value=trade_dates[-1].date())
stock_symbol = sidebar.text_input("复盘股票代码", value=default_stock).strip() or default_stock
stock_date = sidebar.date_input("复盘日期", value=default_trade_date, min_value=trade_dates[0].date(), max_value=trade_dates[-1].date())
initial_capital = float(sidebar.number_input("初始资金", min_value=100000.0, max_value=10000000.0, value=1000000.0, step=100000.0))

sidebar.markdown("---")
sidebar.caption("导出当前查询")
if sidebar.button("生成 HTML 报告和演示页", use_container_width=True):
    with st.spinner("正在生成 HTML 报告..."):
        report_path = render_html_report(
            selected_date=str(selected_date),
            stock_symbol=stock_symbol,
            stock_date=str(stock_date),
            initial_capital=initial_capital,
            report_dir=REPORTS,
        )
    demo_path = report_path.with_name("user_friendly_quant_demo.html")
    sidebar.success("已生成")
    sidebar.code(str(report_path))
    sidebar.code(str(demo_path))

payload = load_payload(str(selected_date), stock_symbol, str(stock_date), initial_capital)
recommendations = payload["recommendations"]
simulation = payload["simulation"]
stock_review = payload["stock_review"]
education = payload["education"]
summary = simulation["summary"]
curve = simulation["curve"].copy()
curve["trade_date"] = pd.to_datetime(curve["trade_date"])
stock_series = stock_review["series"].copy()
stock_series["trade_date"] = pd.to_datetime(stock_series["trade_date"])

st.markdown('<div class="panel">', unsafe_allow_html=True)
intro_left, intro_right = st.columns([1.7, 1])
with intro_left:
    st.subheader("这套系统会帮你回答什么")
    st.markdown(
        """
        - 当天该买什么
        - 预计上涨还是下跌
        - 建议买多少
        - 什么时候买、什么时候卖
        - 如果后来一直按规则执行，到现在到底赚了还是亏了
        """
    )
with intro_right:
    st.subheader("数据范围")
    st.markdown(
        f"""
        <div class="note">
        覆盖时间: {payload['metadata']['date_min']} 到 {payload['metadata']['date_max']}<br/>
        股票数量: {payload['metadata']['symbol_count']}<br/>
        打分模型: {payload['metadata']['model_name']}
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("总收益", f"{summary['total_pnl']:,.0f}", f"收益率 {summary['return_rate'] * 100:.2f}%")
with metric_cols[1]:
    metric_card("最大回撤", f"{summary['max_drawdown'] * 100:.2f}%", f"交易次数 {summary['trade_count']}")
with metric_cols[2]:
    metric_card("胜率", f"{summary['win_rate'] * 100:.2f}%", f"最终资产 {summary['ending_asset']:,.0f}")
with metric_cols[3]:
    metric_card(
        "当天推荐数",
        str(recommendations["summary"]["candidate_count"]),
        f"资金使用率 {recommendations['summary']['capital_usage_rate'] * 100:.2f}%",
    )

tab1, tab2, tab3, tab4 = st.tabs(["当天建议", "历史模拟", "单股复盘", "新手说明"])

with tab1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("当天应该买什么")
    if recommendations["used_nearest_trade_date"]:
        st.info(f"你选的是 {recommendations['selected_date']}，系统实际使用最近一个交易日 {recommendations['effective_trade_date']}。")
    st.caption(f"T+1 实际买入日: {recommendations['buy_date']}")
    st.markdown(f"<p class='note'>{recommendations['summary']['explanation']}</p>", unsafe_allow_html=True)

    rows = pd.DataFrame(recommendations["rows"])
    if rows.empty:
        st.warning("这一天没有满足规则的买入候选。")
    else:
        show = rows[
            [
                "ts_code",
                "direction",
                "rank",
                "conviction_badge",
                "risk_badge",
                "predicted_ret_10",
                "planned_buy_price",
                "suggested_shares",
                "suggested_amount",
                "expected_profit",
                "max_possible_loss",
                "reason",
            ]
        ].copy()
        show = show.rename(
            columns={
                "ts_code": "股票",
                "direction": "判断",
                "rank": "排名",
                "conviction_badge": "优先级",
                "risk_badge": "风险",
                "predicted_ret_10": "预测10日涨跌",
                "planned_buy_price": "T+1买价",
                "suggested_shares": "建议股数",
                "suggested_amount": "建议金额",
                "expected_profit": "预计可赚",
                "max_possible_loss": "最大可能亏",
                "reason": "为什么选它",
            }
        )
        show["预测10日涨跌"] = show["预测10日涨跌"].map(lambda v: f"{v * 100:.2f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)

        chart_df = rows[["ts_code", "predicted_ret_10", "actual_ret_10"]].copy().head(8)
        chart_df = chart_df.melt(id_vars="ts_code", var_name="类型", value_name="收益率")
        chart_df["类型"] = chart_df["类型"].map({"predicted_ret_10": "预测", "actual_ret_10": "真实"})
        fig = px.bar(chart_df, x="ts_code", y="收益率", color="类型", barmode="group", title="推荐股票: 预测收益 vs 真实收益")
        fig.update_layout(height=420)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("如果从那天开始一直照做，最后会怎样")
    st.markdown(f"<p class='note'>{summary['story']}</p>", unsafe_allow_html=True)
    fig_curve = px.line(curve, x="trade_date", y="total_asset", title="总资产变化")
    fig_curve.update_layout(height=420)
    st.plotly_chart(fig_curve, use_container_width=True)

    fig_dd = px.area(curve, x="trade_date", y="drawdown", title="回撤过程")
    fig_dd.update_layout(height=280)
    fig_dd.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_dd, use_container_width=True)

    trades = simulation["trades"].copy()
    if trades.empty:
        st.info("当前日期范围内还没有完整成交。")
    else:
        show_trades = trades.copy()
        show_trades["realized_return"] = show_trades["realized_return"].map(lambda v: f"{v * 100:.2f}%")
        st.dataframe(show_trades, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(f"单只股票复盘: {stock_review['symbol']}")
    st.markdown(f"<p class='note'>{stock_review['story']}</p>", unsafe_allow_html=True)
    review_cols = st.columns(6)
    review_cols[0].metric("预测判断", stock_review["predicted_direction"])
    review_cols[1].metric("预测10日涨跌", f"{stock_review['predicted_ret_10'] * 100:.2f}%")
    review_cols[2].metric("真实10日涨跌", f"{stock_review['actual_ret_10'] * 100:.2f}%" if stock_review["actual_ret_10"] is not None else "-")
    review_cols[3].metric("预测误差", f"{stock_review['prediction_error'] * 100:.2f}%" if stock_review["prediction_error"] is not None else "-")
    review_cols[4].metric("模拟盈亏", f"{stock_review['pnl']:,.0f}")
    review_cols[5].metric("结果标签", stock_review["accuracy_label"])

    chart = px.line(stock_series, x="trade_date", y=["close", "predicted_close_10", "actual_close_10"], title="预测走势和真实走势对比")
    chart.update_layout(height=460)
    st.plotly_chart(chart, use_container_width=True)

    stock_table = stock_series[["trade_date", "close", "expected_ret_10", "fwd_ret_10", "stock_rank", "buyable_today"]].tail(40).copy()
    stock_table = stock_table.rename(
        columns={
            "trade_date": "日期",
            "close": "收盘价",
            "expected_ret_10": "预测10日涨跌",
            "fwd_ret_10": "真实10日涨跌",
            "stock_rank": "排名",
            "buyable_today": "可买",
        }
    )
    for col in ["预测10日涨跌", "真实10日涨跌"]:
        stock_table[col] = stock_table[col].map(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "-")
    st.dataframe(stock_table, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("普通用户怎么上手")
    step_cols = st.columns(2)
    for idx, step in enumerate(education["steps"]):
        with step_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="rule-card">
                  <h4>{idx + 1}. {step['title']}</h4>
                  <p>{step['detail']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 交易规则")
    rule_cols = st.columns(2)
    for idx, rule in enumerate(education["rule_book"]):
        with rule_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="rule-card">
                  <h4>{rule['title']}</h4>
                  <p>{rule['detail']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 风险提示")
    for item in education["risk_focus"]:
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)
