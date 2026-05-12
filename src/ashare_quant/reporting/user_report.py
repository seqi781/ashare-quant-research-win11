from __future__ import annotations

from pathlib import Path
import base64
import io

import matplotlib.pyplot as plt
import pandas as pd

from ashare_quant.reporting.user_decision_service import build_dashboard_payload, export_dashboard_payload


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _encode_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_equity(curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    x = pd.to_datetime(curve["trade_date"])
    ax.plot(x, curve["total_asset"], color="#0f766e", linewidth=2.4)
    ax.fill_between(x, curve["total_asset"], curve["total_asset"].min(), color="#99f6e4", alpha=0.18)
    ax.set_title("Backtest Equity Curve")
    ax.set_ylabel("Total Asset")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return _encode_fig(fig)


def _plot_drawdown(curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    x = pd.to_datetime(curve["trade_date"])
    ax.fill_between(x, curve["drawdown"], 0, color="#fca5a5", alpha=0.6)
    ax.plot(x, curve["drawdown"], color="#b91c1c", linewidth=1.8)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return _encode_fig(fig)


def _plot_recommendations(rows: list[dict]) -> str:
    df = pd.DataFrame(rows).head(8)
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    if df.empty:
        ax.text(0.5, 0.5, "No recommendations for this day", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return _encode_fig(fig)
    df["predicted_ret_10"] = pd.to_numeric(df["predicted_ret_10"], errors="coerce")
    if "actual_ret_10" in df.columns:
        df["actual_ret_10"] = pd.to_numeric(df["actual_ret_10"], errors="coerce")
    x = range(len(df))
    ax.bar(x, df["predicted_ret_10"], color="#1d4ed8", alpha=0.88, label="Predicted 10D Return")
    if "actual_ret_10" in df.columns and df["actual_ret_10"].notna().any():
        ax.bar(x, df["actual_ret_10"], color="#fb923c", alpha=0.6, label="Actual 10D Return")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["ts_code"], rotation=18)
    ax.set_title("Recommendations: Predicted vs Actual")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    return _encode_fig(fig)


def _plot_stock_review(series: pd.DataFrame, symbol: str, selected_date: str) -> str:
    stock = series.copy()
    stock["trade_date"] = pd.to_datetime(stock["trade_date"])
    fig, ax1 = plt.subplots(figsize=(10.8, 4.8))
    ax1.plot(stock["trade_date"], stock["close"], color="#111827", linewidth=1.8, label="Close")
    ax1.set_ylabel("Close")
    ax2 = ax1.twinx()
    ax2.plot(stock["trade_date"], stock["expected_ret_10"], color="#0f766e", linewidth=1.6, label="Predicted 10D Return")
    ax2.plot(stock["trade_date"], stock["fwd_ret_10"], color="#ea580c", linewidth=1.6, label="Actual 10D Return")
    mark = stock[stock["trade_date"] == pd.Timestamp(selected_date)]
    if not mark.empty:
        ax1.scatter(mark["trade_date"], mark["close"], color="#2563eb", s=76, zorder=5)
    ax1.set_title(f"{symbol} Single-Stock Review")
    ax1.grid(alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.autofmt_xdate()
    return _encode_fig(fig)


def _build_metric_cards(summary: dict, recommendation_summary: dict) -> str:
    cards = [
        ("总收益", f"{summary['total_pnl']:,.0f}", f"收益率 {_fmt_pct(summary['return_rate'])}"),
        ("最大回撤", _fmt_pct(summary["max_drawdown"]), f"交易次数 {summary['trade_count']}"),
        ("胜率", _fmt_pct(summary["win_rate"]), f"最终资产 {summary['ending_asset']:,.0f}"),
        (
            "当天建议",
            str(recommendation_summary["candidate_count"]),
            f"资金使用率 {_fmt_pct(recommendation_summary['capital_usage_rate'])}",
        ),
    ]
    return "\n".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>'
        for label, value, help_text in cards
    )


def _build_recommendation_rows(rows: list[dict]) -> str:
    if not rows:
        return '<tr><td colspan="9">这一天没有满足规则的买入候选。</td></tr>'
    return "\n".join(
        (
            "<tr>"
            f"<td>{row['ts_code']}</td>"
            f"<td>{row['direction']}</td>"
            f"<td>{row['buy_date']}</td>"
            f"<td>{row['suggested_shares']}</td>"
            f"<td>{row['suggested_amount']:,.0f}</td>"
            f"<td>{_fmt_pct(row['predicted_ret_10'])}</td>"
            f"<td>{row['expected_profit']:,.0f}</td>"
            f"<td>{row['max_possible_loss']:,.0f}</td>"
            f"<td>{row['conviction_badge']}</td>"
            "</tr>"
        )
        for row in rows
    )


def _build_trade_rows(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '<tr><td colspan="6">当前日期范围内还没有完整成交。</td></tr>'
    return "\n".join(
        (
            "<tr>"
            f"<td>{row['ts_code']}</td>"
            f"<td>{row['buy_date']}</td>"
            f"<td>{row['sell_date']}</td>"
            f"<td>{row['exit_reason']}</td>"
            f"<td>{row['pnl']:,.0f}</td>"
            f"<td>{_fmt_pct(row['realized_return'])}</td>"
            "</tr>"
        )
        for row in trades.tail(12).to_dict(orient="records")
    )


def _build_bullets(items: list[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


def _build_rule_cards(rule_book: list[dict]) -> str:
    return "\n".join(
        f'<div class="mini-card"><h4>{item["title"]}</h4><p>{item["detail"]}</p></div>'
        for item in rule_book
    )


def _build_step_cards(steps: list[dict]) -> str:
    return "\n".join(
        f'<div class="mini-card"><h4>{idx}. {item["title"]}</h4><p>{item["detail"]}</p></div>'
        for idx, item in enumerate(steps, start=1)
    )


def _build_showcase_shell(title: str, body: str, subtitle: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg1: #f7f1e6;
      --bg2: #ecf4fa;
      --paper: rgba(255,255,255,0.92);
      --ink: #1e293b;
      --muted: #5b6472;
      --line: #ddd6ca;
      --brand: #0f766e;
      --brand2: #1d4ed8;
      --shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(29,78,216,0.11), transparent 25%),
        linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 56px; }}
    .hero {{
      background: linear-gradient(135deg, #113c37 0%, #1d4ed8 100%);
      color: #fff;
      border-radius: 28px;
      padding: 34px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{ margin: 0; font-size: 38px; }}
    .hero p {{ margin: 12px 0 0; line-height: 1.75; color: rgba(255,255,255,0.92); }}
    .tag {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      margin-right: 8px;
      font-size: 13px;
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(255,255,255,0.10);
    }}
    .section {{ margin-top: 18px; }}
    .card {{
      background: var(--paper);
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 16px; }}
    .grid4 {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; }}
    .metric {{
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .metric .label {{ color: var(--muted); font-size: 14px; }}
    .metric .value {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
    .metric .help {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .lead {{ font-size: 18px; line-height: 1.8; }}
    .muted {{ color: var(--muted); }}
    h2, h3 {{ margin-top: 0; }}
    ul {{ margin: 10px 0 0; padding-left: 18px; }}
    li {{ line-height: 1.8; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ece7de; vertical-align: top; font-size: 14px; }}
    th {{ background: #edf6ff; }}
    img {{ width: 100%; border-radius: 16px; border: 1px solid var(--line); background: #fff; }}
    code {{ background: #f3f6fb; padding: 2px 6px; border-radius: 6px; }}
    .footer {{ margin-top: 20px; color: var(--muted); font-size: 13px; text-align: right; }}
    @media (max-width: 980px) {{
      .grid2, .grid3, .grid4 {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="tag">A股量化</div>
      <div class="tag">静态 HTML</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </section>
    {body}
    <div class="footer">Generated by ashare-quant-reporting suite</div>
  </div>
</body>
</html>
"""


def _render_briefing_document(payload: dict) -> str:
    summary = payload["simulation"]["summary"]
    recommendation_summary = payload["recommendations"]["summary"]
    rows = payload["recommendations"]["rows"][:5]
    metadata = payload.get("metadata", {})
    summary_story = summary.get("story", "系统按既定规则完成了从选股到回测的整条演示链路。")
    recommendation_explanation = recommendation_summary.get("explanation", "系统根据当前日期输出了当日推荐和风险估计。")
    curve = payload["simulation"]["curve"]
    stock_review = payload["stock_review"]
    recommendation_img = _plot_recommendations(payload["recommendations"]["rows"])
    equity_img = _plot_equity(curve)
    stock_img = _plot_stock_review(stock_review["series"], stock_review["symbol"], stock_review["effective_trade_date"])
    top_rows = "\n".join(
        (
            "<tr>"
            f"<td>{row['ts_code']}</td>"
            f"<td>{row.get('rank', '-')}</td>"
            f"<td>{row['direction']}</td>"
            f"<td>{_fmt_pct(row['predicted_ret_10'])}</td>"
            f"<td>{row['suggested_amount']:,.0f}</td>"
            f"<td>{row.get('reason', '-')}</td>"
            "</tr>"
        )
        for row in rows
    )
    body = f"""
    <section class="section grid4">
      <div class="metric"><div class="label">累计收益</div><div class="value">{_fmt_pct(summary['return_rate'])}</div><div class="help">总盈亏 {summary['total_pnl']:,.0f}</div></div>
      <div class="metric"><div class="label">最大回撤</div><div class="value">{_fmt_pct(summary['max_drawdown'])}</div><div class="help">风险承受关键指标</div></div>
      <div class="metric"><div class="label">交易次数</div><div class="value">{summary['trade_count']}</div><div class="help">统计截止 {summary['latest_trade_date']}</div></div>
      <div class="metric"><div class="label">胜率</div><div class="value">{_fmt_pct(summary['win_rate'])}</div><div class="help">完成交易中的盈利占比</div></div>
    </section>

    <section class="section grid2">
      <div class="card">
        <h2>项目定位</h2>
        <p class="lead">这是一个面向普通用户的 A 股网页量化交易系统。用户只选日期，系统就能告诉他当时应该买什么、怎么买、什么时候卖，以及后来如果一直照做，到今天是赚还是亏。</p>
        <ul>
          <li>把复杂量化因子、选股规则和回测结果翻译成用户看得懂的决策页面。</li>
          <li>同时给出推荐、风险、仓位、模拟结果和单股复盘，避免“只有信号没有解释”。</li>
          <li>支持网页交互和静态 HTML 两种交付形式，便于演示、汇报和归档。</li>
        </ul>
          </div>
          <div class="card">
            <h2>汇报结论</h2>
            <p class="lead">{summary_story}</p>
            <p class="muted">{recommendation_explanation}</p>
            <ul>
              <li>起始日期：{summary['selected_start_date']}</li>
              <li>实际回测起点：{summary['effective_start_trade_date']}</li>
              <li>数据截止：{summary['latest_trade_date']}</li>
              <li>当前模型：{metadata.get('model_name', '-')}</li>
            </ul>
          </div>
        </section>

        <section class="section card">
          <h2>系统输出给用户的核心价值</h2>
          <div class="grid3">
            <div class="metric"><div class="label">当天推荐数</div><div class="value">{recommendation_summary['candidate_count']}</div><div class="help">覆盖买什么</div></div>
            <div class="metric"><div class="label">资金使用率</div><div class="value">{_fmt_pct(recommendation_summary['capital_usage_rate'])}</div><div class="help">覆盖买多少</div></div>
            <div class="metric"><div class="label">预计收益</div><div class="value">{float(recommendation_summary.get('expected_profit_total', 0.0)):,.0f}</div><div class="help">覆盖可能赚多少</div></div>
          </div>
      <ul>
        <li>推荐页：回答“当天买哪些股票、预计涨跌、建议买多少、什么时候按规则卖”。</li>
        <li>历史模拟页：回答“如果当时按系统建议操作，到现在一共赚了还是亏了”。</li>
        <li>单股复盘页：回答“某只股票在某一天的预测是否准确，误差有多大”。</li>
      </ul>
    </section>

    <section class="section card">
      <h2>示例推荐股票</h2>
      <table>
        <thead><tr><th>股票</th><th>排名</th><th>判断</th><th>预测10日涨跌</th><th>建议金额</th><th>入选原因</th></tr></thead>
        <tbody>{top_rows}</tbody>
      </table>
    </section>

    <section class="section card">
      <h2>真实结果截图</h2>
      <div class="grid3">
        <div>
          <h3>推荐结果截图</h3>
          <img src="data:image/png;base64,{recommendation_img}" alt="real-recommendation-screenshot" />
        </div>
        <div>
          <h3>收益曲线截图</h3>
          <img src="data:image/png;base64,{equity_img}" alt="real-equity-screenshot" />
        </div>
        <div>
          <h3>单股复盘截图</h3>
          <img src="data:image/png;base64,{stock_img}" alt="real-stock-review-screenshot" />
        </div>
      </div>
      <p class="muted">以上截图均来自当前项目真实计算结果生成，不是示意图。</p>
    </section>

    <section class="section grid2">
      <div class="card">
        <h2>适合汇报对象</h2>
        <ul>
          <li>业务负责人：看收益、回撤、用户价值是否足够清晰。</li>
          <li>产品负责人：看用户路径是否从“看不懂量化”变成“看得懂买卖”。</li>
          <li>技术负责人：看系统是否可复现、可验证、可导出。</li>
        </ul>
      </div>
      <div class="card">
        <h2>本次交付物</h2>
        <ul>
          <li><code>user_friendly_quant_report.html</code>：结果报告版。</li>
          <li><code>user_friendly_quant_demo.html</code>：产品演示版。</li>
          <li><code>ashare_quant_briefing.html</code>：汇报摘要版。</li>
          <li><code>ashare_quant_technical_doc.html</code>：技术文档版。</li>
          <li><code>ashare_quant_demo_script.html</code>：现场演示讲稿版。</li>
        </ul>
      </div>
    </section>
    """
    return _build_showcase_shell("A股量化交易系统汇报版", body, "用于项目汇报、阶段总结和业务沟通的高层展示页面。")


def _render_technical_document(payload: dict) -> str:
    config = payload.get("config", {})
    metadata = payload.get("metadata", {})
    summary = payload["simulation"]["summary"]
    technologies = [
        "Python 3.11",
        "Pandas / NumPy",
        "PyArrow Parquet",
        "AkShare / Tushare",
        "Scikit-learn / XGBoost",
        "Streamlit",
        "Matplotlib / Plotly",
        "Pytest",
        "WSL + SSH",
    ]
    tech_list = "".join(f"<li>{item}</li>" for item in technologies)
    body = f"""
    <section class="section card">
      <h2>项目概述</h2>
      <p class="lead">本项目是一个 A 股量化研究与用户决策展示系统。上游负责数据准备、因子计算、打分和回测，下游负责把模型结果转换为普通用户可以理解的网页和静态 HTML 报告。</p>
      <ul>
          <li>数据范围：{metadata.get('date_min', '-')} 至 {metadata.get('date_max', '-')}</li>
          <li>股票覆盖：{metadata.get('symbol_count', '-')} 只</li>
          <li>模型标识：{metadata.get('model_name', '-')}</li>
      </ul>
    </section>

    <section class="section grid2">
      <div class="card">
        <h2>技术栈</h2>
        <ul>{tech_list}</ul>
      </div>
      <div class="card">
        <h2>系统分层</h2>
        <ul>
          <li><code>pipeline/</code>：生成策略结果、候选股、滚动回测和预测留档。</li>
          <li><code>reporting/user_decision_service.py</code>：把策略结果翻译成用户看得懂的推荐、回测和复盘 payload。</li>
          <li><code>app/user_quant_app.py</code>：Streamlit 交互网页入口。</li>
          <li><code>reporting/user_report.py</code>：生成静态 HTML 报告、演示页、汇报页和技术文档。</li>
        </ul>
      </div>
      <div class="card">
        <h2>核心规则</h2>
        <ul>
          <li>T+1 买入：当日出信号，下一交易日开盘尝试成交。</li>
          <li>止盈：{_fmt_pct(config.get('take_profit_partial'))}</li>
          <li>止损：{abs(float(config.get('stop_loss', 0.0))) * 100:.2f}%</li>
          <li>最长持有：{config.get('hold_days', '-')} 个交易日</li>
          <li>最大持仓数量：{config.get('max_positions', '-')}</li>
        </ul>
      </div>
    </section>

    <section class="section card">
      <h2>数据流</h2>
      <table>
        <thead><tr><th>阶段</th><th>输入</th><th>处理</th><th>输出</th></tr></thead>
        <tbody>
          <tr><td>行情与过滤</td><td>全市场可买股票 parquet</td><td>ST/停牌/可买过滤、基础因子</td><td>可研究股票池</td></tr>
          <tr><td>多因子打分</td><td>价格、成交额、估值、质量、流动性</td><td>规则分 + ML 预测 + 校准</td><td>股票排名与预期收益</td></tr>
          <tr><td>用户决策层</td><td>打分结果</td><td>推荐表、T+1 计划、仓位分配、单股复盘</td><td>Dashboard payload</td></tr>
          <tr><td>展示层</td><td>Payload</td><td>Streamlit 页面 + 静态 HTML 导出</td><td>网页、报告、演示文档</td></tr>
        </tbody>
      </table>
    </section>

    <section class="section grid2">
      <div class="card">
        <h2>关键文件</h2>
        <ul>
          <li><code>app/user_quant_app.py</code></li>
          <li><code>src/ashare_quant/reporting/user_decision_service.py</code></li>
          <li><code>src/ashare_quant/reporting/user_report.py</code></li>
          <li><code>scripts/run_user_quant_app.sh</code></li>
          <li><code>scripts/generate_user_report.sh</code></li>
        </ul>
      </div>
      <div class="card">
        <h2>验证方式</h2>
        <ul>
          <li><code>uv run pytest -q</code></li>
          <li><code>./scripts/run_user_quant_app.sh</code></li>
          <li><code>./scripts/generate_user_report.sh 2025-01-02 600397 2025-01-02 1000000</code></li>
          <li>检查 <code>reports/</code> 下 HTML 与 JSON 产物。</li>
        </ul>
      </div>
    </section>

    <section class="section card">
      <h2>当前样例运行结果</h2>
      <div class="grid4">
        <div class="metric"><div class="label">累计收益</div><div class="value">{_fmt_pct(summary['return_rate'])}</div><div class="help">回测总盈亏 {summary['total_pnl']:,.0f}</div></div>
        <div class="metric"><div class="label">最大回撤</div><div class="value">{_fmt_pct(summary['max_drawdown'])}</div><div class="help">风险峰值</div></div>
        <div class="metric"><div class="label">交易次数</div><div class="value">{summary['trade_count']}</div><div class="help">历史完成交易</div></div>
        <div class="metric"><div class="label">胜率</div><div class="value">{_fmt_pct(summary['win_rate'])}</div><div class="help">盈利笔数占比</div></div>
      </div>
    </section>
    """
    return _build_showcase_shell("A股量化交易系统技术文档", body, "用于技术评审、交接和部署说明的静态 HTML 技术说明文档。")


def _render_demo_script_document(payload: dict) -> str:
    summary = payload["simulation"]["summary"]
    recommendation_summary = payload["recommendations"]["summary"]
    top_codes = "、".join(row["ts_code"] for row in payload["recommendations"]["rows"][:3]) or "无"
    candidate_count = recommendation_summary.get("candidate_count", len(payload["recommendations"]["rows"]))
    body = f"""
    <section class="section card">
      <h2>演示目标</h2>
      <p class="lead">本页用于现场演示讲解。它不是技术实现页，而是告诉演示人应该先展示什么，再解释什么，最后如何收口。</p>
      <ul>
        <li>目标一：让观众在 1 分钟内理解系统能回答哪些问题。</li>
        <li>目标二：让观众在 3 分钟内看到推荐、回测、复盘三种典型输出。</li>
        <li>目标三：让观众在 5 分钟内理解这个系统为什么比只给买卖信号更容易落地。</li>
      </ul>
    </section>

    <section class="section grid3">
      <div class="metric"><div class="label">推荐股票数</div><div class="value">{candidate_count}</div><div class="help">开场先讲清楚系统给什么答案</div></div>
      <div class="metric"><div class="label">累计收益</div><div class="value">{_fmt_pct(summary['return_rate'])}</div><div class="help">中段展示规则长期结果</div></div>
      <div class="metric"><div class="label">最大回撤</div><div class="value">{_fmt_pct(summary['max_drawdown'])}</div><div class="help">结尾强调风险边界</div></div>
    </section>

    <section class="section card">
      <h2>5 分钟演示脚本</h2>
      <table>
        <thead><tr><th>时间</th><th>展示内容</th><th>讲解重点</th></tr></thead>
        <tbody>
          <tr><td>第 1 分钟</td><td>打开交互网页首页</td><td>说明用户只需要选日期，系统就能输出买什么、怎么买、什么时候卖。</td></tr>
          <tr><td>第 2 分钟</td><td>切到“当天建议”</td><td>展示示例股票 {top_codes}，说明推荐不是黑盒，而是给出收益预期、金额建议和风险说明。</td></tr>
          <tr><td>第 3 分钟</td><td>切到“历史模拟”</td><td>重点讲累计收益 {_fmt_pct(summary['return_rate'])}、最大回撤 {_fmt_pct(summary['max_drawdown'])}、交易次数 {summary['trade_count']}。</td></tr>
          <tr><td>第 4 分钟</td><td>切到“单股复盘”</td><td>说明系统不仅能推荐，还能验证过去某只股票的预测是否准确。</td></tr>
          <tr><td>第 5 分钟</td><td>展示 HTML 报告</td><td>强调支持汇报、归档、发给领导或客户直接阅读，不需要进入代码环境。</td></tr>
        </tbody>
      </table>
    </section>

    <section class="section grid2">
      <div class="card">
        <h2>演示话术建议</h2>
        <ul>
          <li>不要先讲模型细节，先讲用户能得到什么答案。</li>
          <li>先说“看结论”，再说“看风险”，最后才说“为什么这样选”。</li>
          <li>如果观众不懂量化，重点解释 T+1、止盈止损、最大回撤。</li>
        </ul>
      </div>
      <div class="card">
        <h2>演示结束收口</h2>
        <ul>
          <li>系统已从研究结果扩展到用户可理解的决策界面。</li>
          <li>既能交互演示，也能导出静态 HTML 文档用于汇报。</li>
          <li>下一阶段可以继续做更正式的前后端化和权限/数据更新能力。</li>
        </ul>
      </div>
    </section>
    """
    return _build_showcase_shell("A股量化交易系统项目演示稿", body, "用于现场汇报、路演讲解和产品演示的讲稿型 HTML 页面。")


def _render_html_document(payload: dict, title: str, subtitle: str) -> str:
    summary = payload["simulation"]["summary"]
    rows = payload["recommendations"]["rows"]
    recommendation_summary = payload["recommendations"]["summary"]
    stock_review = payload["stock_review"]
    education = payload["education"]
    summary_story = summary.get("story", "系统按固定规则回放了这段时间内的交易表现。")
    stock_story = stock_review.get("story", "这里展示的是单只股票在指定日期附近的预测与真实走势对比。")
    accuracy_label = stock_review.get("accuracy_label", "未提供")

    equity_img = _plot_equity(payload["simulation"]["curve"])
    drawdown_img = _plot_drawdown(payload["simulation"]["curve"])
    recommendation_img = _plot_recommendations(rows)
    stock_img = _plot_stock_review(stock_review["series"], stock_review["symbol"], stock_review["effective_trade_date"])

    stock_rows = _build_recommendation_rows(rows)
    trade_rows = _build_trade_rows(payload["simulation"]["trades"])
    metric_cards = _build_metric_cards(summary, recommendation_summary)
    rule_cards = _build_rule_cards(education["rule_book"])
    step_cards = _build_step_cards(education["steps"])
    risk_bullets = _build_bullets(education["risk_focus"])

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --paper: rgba(255,255,255,0.9);
      --ink: #1b2430;
      --muted: #5a6677;
      --line: #d7d2c8;
      --brand: #0f766e;
      --brand-2: #1d4ed8;
      --warn: #b45309;
      --danger: #b91c1c;
      --shadow: 0 18px 50px rgba(27, 36, 48, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(29,78,216,0.12), transparent 24%),
        linear-gradient(180deg, #f8f3e8 0%, #eef4f8 100%);
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    .hero {{
      background: linear-gradient(135deg, #113c37 0%, #1d4ed8 100%);
      color: white;
      border-radius: 24px;
      padding: 30px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{ margin: 0; font-size: 34px; }}
    .hero p {{ margin: 10px 0 0; line-height: 1.6; }}
    .section {{ margin-top: 18px; }}
    .card {{
      background: var(--paper);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: 20px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .metric-card {{
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(215,210,200,0.8);
      border-radius: 18px;
      padding: 18px;
    }}
    .metric-label {{ color: var(--muted); font-size: 14px; }}
    .metric-value {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
    .metric-help {{ color: var(--muted); margin-top: 6px; font-size: 13px; }}
    .split {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 16px;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
    }}
    .mini-card h4 {{ margin: 0 0 8px; }}
    .mini-card p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .lead {{
      font-size: 18px;
      line-height: 1.75;
      color: #223042;
    }}
    .muted {{ color: var(--muted); }}
    .tag {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e6fffa;
      color: var(--brand);
      font-size: 13px;
      margin-right: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      overflow: hidden;
      border-radius: 16px;
    }}
    th, td {{
      padding: 11px 12px;
      text-align: left;
      font-size: 14px;
      border-bottom: 1px solid #ebe7df;
      vertical-align: top;
    }}
    th {{ background: #edf6ff; }}
    img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid #ddd6ca;
      background: white;
    }}
    ul {{ margin: 10px 0 0; padding-left: 18px; }}
    li {{ line-height: 1.8; }}
    @media (max-width: 980px) {{
      .metric-grid, .split, .mini-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="tag">HTML 报告</div>
      <div class="tag">普通用户可读</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <p>起始日期 {summary['selected_start_date']}，实际回测起点 {summary['effective_start_trade_date']}，统计截止 {summary['latest_trade_date']}。</p>
    </section>

    <section class="metric-grid">{metric_cards}</section>

        <section class="section split">
          <div class="card">
            <h2>一句话结论</h2>
            <p class="lead">{summary_story}</p>
            <p class="muted">{payload['recommendations']['summary']['explanation']}</p>
          </div>
      <div class="card">
        <h2>系统规则</h2>
        <div class="mini-grid">{rule_cards}</div>
      </div>
    </section>

    <section class="section card">
      <h2>当天推荐怎么买</h2>
      <p class="muted">如果你只想知道“那天该买什么、买多少、可能赚多少、最大可能亏多少”，直接看下面这张表。</p>
      <table>
        <thead>
          <tr>
            <th>股票</th><th>判断</th><th>建议买入日</th><th>建议股数</th><th>建议金额</th><th>预测 10 日涨跌</th><th>预计可赚</th><th>最大可能亏</th><th>优先级</th>
          </tr>
        </thead>
        <tbody>{stock_rows}</tbody>
      </table>
    </section>

    <section class="section split">
      <div class="card">
        <h2>策略收益曲线</h2>
        <img src="data:image/png;base64,{equity_img}" alt="equity" />
      </div>
      <div class="card">
        <h2>风险回撤曲线</h2>
        <img src="data:image/png;base64,{drawdown_img}" alt="drawdown" />
      </div>
    </section>

    <section class="section split">
      <div class="card">
        <h2>推荐股票预测 vs 真实结果</h2>
        <img src="data:image/png;base64,{recommendation_img}" alt="recommendations" />
      </div>
      <div class="card">
        <h2>怎么看结果</h2>
        <ul>
          <li>总收益看这套规则长期有没有赚钱能力。</li>
          <li>最大回撤看这套系统最难熬的时候会亏到什么程度。</li>
          <li>胜率不是越高越好，要和单笔盈亏幅度一起看。</li>
          <li>当天推荐不是保证上涨，而是按规则筛出的相对优先候选。</li>
        </ul>
      </div>
    </section>

        <section class="section split">
          <div class="card">
            <h2>单只股票复盘: {stock_review['symbol']}</h2>
            <p class="lead">{stock_story}</p>
            <p class="muted">预测误差 {_fmt_pct(stock_review['prediction_error'])}，信号标签 {accuracy_label}。</p>
            <img src="data:image/png;base64,{stock_img}" alt="stock-review" />
          </div>
      <div class="card">
        <h2>新手怎么用</h2>
        <div class="mini-grid">{step_cards}</div>
      </div>
    </section>

    <section class="section split">
      <div class="card">
        <h2>最近成交记录</h2>
        <table>
          <thead>
            <tr><th>股票</th><th>买入日</th><th>卖出日</th><th>原因</th><th>盈亏</th><th>收益率</th></tr>
          </thead>
          <tbody>{trade_rows}</tbody>
        </table>
      </div>
      <div class="card">
        <h2>风险提示</h2>
        <ul>{risk_bullets}</ul>
      </div>
    </section>
  </div>
</body>
</html>
"""


def render_html_report(
    selected_date: str,
    stock_symbol: str,
    stock_date: str,
    initial_capital: float = 1_000_000.0,
    report_dir: str | Path = "reports",
) -> Path:
    payload = build_dashboard_payload(
        selected_date=selected_date,
        stock_symbol=stock_symbol,
        stock_date=stock_date,
        initial_capital=initial_capital,
        report_dir=report_dir,
    )
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    report_html = _render_html_document(
        payload,
        title="A股网页量化交易系统报告",
        subtitle="这份报告直接回答普通用户最关心的问题：买什么、怎么买、什么时候卖、后来有没有赚到钱、风险有多大。",
    )
    demo_html = _render_html_document(
        payload,
        title="A股网页量化交易系统演示页",
        subtitle="这是可直接展示给用户的静态演示版本，用来说明系统如何工作、如何阅读结果，以及如何验证单股预测。",
    )

    html_path = report_root / "user_friendly_quant_report.html"
    demo_path = report_root / "user_friendly_quant_demo.html"
    briefing_path = report_root / "ashare_quant_briefing.html"
    technical_path = report_root / "ashare_quant_technical_doc.html"
    demo_script_path = report_root / "ashare_quant_demo_script.html"
    html_path.write_text(report_html, encoding="utf-8")
    demo_path.write_text(demo_html, encoding="utf-8")
    briefing_path.write_text(_render_briefing_document(payload), encoding="utf-8")
    technical_path.write_text(_render_technical_document(payload), encoding="utf-8")
    demo_script_path.write_text(_render_demo_script_document(payload), encoding="utf-8")
    export_dashboard_payload(payload, report_root / "user_friendly_quant_report.json")
    return html_path
