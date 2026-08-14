"""FinMosaic: a results-only financial information workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from src.app_data import AppDataBundle, load_app_data, selected_asset_statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS_SCHEMA_VERSION = "2026-08-13-portfolio-sentiment-v1"

FUND_COLORS = {
    "Combined Equal Weight": "#1485A3",
    "Combined Minimum Variance": "#168F78",
    "Combined Maximum Sharpe": "#D99100",
    "Sentiment-Tilted Equal Weight": "#C74763",
}
METHOD_COLORS = {
    "equal_weight": "#1485A3",
    "min_variance": "#168F78",
    "max_sharpe": "#D99100",
    "sentiment_tilt": "#C74763",
}
ASSET_COLORS = ["#1485A3", "#C74763", "#168F78", "#D99100", "#725E9C", "#576574"]
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

WATCHLIST_PRESETS = {
    "Cross-market sample": {
        "equities": ["NVDA", "V", "WMT", "XOM"],
        "crypto": ["BTC-USD", "ETH-USD"],
        "sectors": ["Tech", "Financials", "Consumer", "Energy"],
    },
    "Defensive-sector sample": {
        "equities": ["KO", "MRK", "SO", "WMT"],
        "crypto": [],
        "sectors": ["Consumer", "Healthcare", "Utilities"],
    },
    "Digital-economy sample": {
        "equities": ["ADBE", "CMCSA", "EA", "NVDA"],
        "crypto": ["BTC-USD", "ETH-USD"],
        "sectors": ["Comm", "Consumer", "Tech"],
    },
}


st.set_page_config(
    page_title="FinMosaic",
    page_icon="FM",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_results(cache_version: str) -> AppDataBundle:
    """Load validated outputs using an explicit artifact-schema cache key."""
    if cache_version != RESULTS_SCHEMA_VERSION:
        raise ValueError(f"unsupported results schema version: {cache_version}")
    return load_app_data(RESULTS)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17212b;
            --muted: #627286;
            --line: #dce3e8;
            --paper: #ffffff;
            --wash: #f4f7f8;
            --teal: #1485a3;
            --coral: #c74763;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] { background: #f2f5f6; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] h1 { font-size: 1.45rem; letter-spacing: 0; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0 !important; }
        h1 { font-size: 2rem !important; margin-bottom: .2rem !important; }
        h2 { font-size: 1.35rem !important; margin-top: 1.5rem !important; }
        h3 { font-size: 1.05rem !important; }
        [data-testid="stMetric"] { border-top: 2px solid var(--teal); padding-top: .55rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        .fm-kicker {
            color: var(--teal); font-size: .76rem; font-weight: 700; letter-spacing: .08em;
        }
        .fm-subtitle {
            color: var(--muted); font-size: .96rem; max-width: 72rem; margin-bottom: 1rem;
        }
        .fm-note {
            border-left: 3px solid var(--teal); padding: .25rem 0 .25rem .8rem;
            color: var(--muted);
        }
        .fm-footer {
            border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem;
            margin-top: 2rem; padding-top: .8rem;
        }
        .stDataFrame { border: 1px solid var(--line); }
        div[data-baseweb="tab-list"] { gap: 1.2rem; }
        button[data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_heading(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="fm-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="fm-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def base_figure(fig: go.Figure, *, y_title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=34, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color="#17212b", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, rangeslider=dict(visible=True, thickness=0.06)),
        yaxis=dict(title=y_title, gridcolor="#dce3e8", zerolinecolor="#aebbc5"),
    )
    return fig


def percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1%}"


def donut_figure(
    frame: pd.DataFrame,
    *,
    names: str,
    values: str,
    colors: list[str],
) -> go.Figure:
    """Build one compact labelled composition chart with no analytical transformation."""
    fig = go.Figure(
        go.Pie(
            labels=frame[names],
            values=frame[values],
            hole=0.58,
            sort=False,
            textinfo="percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
            marker=dict(colors=colors, line=dict(color="white", width=2)),
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="white",
        font=dict(family="Arial", color="#17212b", size=11),
        legend=dict(orientation="h", yanchor="top", y=-0.03, xanchor="center", x=0.5),
    )
    return fig


def fund_view(
    data: AppDataBundle,
    selected_fund: str,
    universe: str,
) -> None:
    page_heading(
        "FUNDS",
        "Compare systematic funds by investable universe",
        "Walk-forward out-of-sample histories with monthly rebalancing, next-observation "
        "effectiveness, zero transaction costs and a zero risk-free rate.",
    )
    all_metrics = data["fund_metrics"].copy()
    all_returns = data["fund_returns"].sort_values(["fund", "date"])
    all_holdings = data["latest_holdings"].copy()

    st.caption(f"Comparison universe: {universe}. Change it in the sidebar.")
    metrics = all_metrics.loc[all_metrics["universe"].eq(universe)].copy()
    method_order = {"equal_weight": 0, "min_variance": 1, "max_sharpe": 2, "sentiment_tilt": 3}
    metrics = metrics.sort_values(
        "method", key=lambda values: values.map(method_order)
    ).reset_index(drop=True)
    returns = all_returns.loc[all_returns["universe"].eq(universe)].copy()
    holdings = all_holdings.loc[all_holdings["universe"].eq(universe)].copy()
    color_map = dict(zip(metrics["fund"], metrics["method"].map(METHOD_COLORS), strict=True))

    selected = metrics.loc[metrics["fund"].eq(selected_fund)].iloc[0]
    cols = st.columns(5)
    cols[0].metric("Annualised return", percent(selected["annualized_return"]))
    cols[1].metric("Annualised volatility", percent(selected["annualized_volatility"]))
    cols[2].metric("Sharpe ratio", f"{selected['sharpe_ratio']:.2f}")
    cols[3].metric("Maximum drawdown", percent(selected["maximum_drawdown"]))
    cols[4].metric("Terminal value of $1", f"${selected['growth_of_1']:.2f}")

    growth_tab, drawdown_tab, risk_tab = st.tabs(["Growth of $1", "Drawdown", "Return and risk"])
    with growth_tab:
        fig = px.line(
            returns,
            x="date",
            y="growth_of_1",
            color="fund",
            color_discrete_map=color_map,
            labels={"fund": "Fund"},
        )
        fig.update_traces(line_width=2)
        base_figure(fig, y_title="Value of $1 (US dollars)")
        fig.update_yaxes(tickprefix="$", tickformat=".2f")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with drawdown_tab:
        fig = px.line(
            returns,
            x="date",
            y="drawdown",
            color="fund",
            color_discrete_map=color_map,
            labels={"fund": "Fund"},
        )
        fig.update_traces(line_width=2)
        base_figure(fig, y_title="Drawdown from prior peak")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with risk_tab:
        fig = px.scatter(
            metrics,
            x="annualized_volatility",
            y="annualized_return",
            color="fund",
            size="growth_of_1",
            color_discrete_map=color_map,
            hover_data={"sharpe_ratio": ":.2f", "growth_of_1": ":.2f"},
            labels={"fund": "Fund"},
        )
        fig.update_traces(marker=dict(sizemin=12, line=dict(color="white", width=1.5)))
        fig.update_layout(
            height=430,
            margin=dict(l=15, r=15, t=34, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.02, yanchor="bottom"),
            xaxis=dict(title="Annualised volatility", tickformat=".0%", gridcolor="#dce3e8"),
            yaxis=dict(title="Annualised return", tickformat=".0%", gridcolor="#dce3e8"),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Latest holdings")
        fund_holdings = holdings.loc[holdings["fund"].eq(selected_fund)].sort_values(
            "weight", ascending=False
        )
        holding_display = fund_holdings[
            ["rank", "ticker", "asset_class", "holding_group", "weight"]
        ]
        st.dataframe(
            holding_display,
            width="stretch",
            hide_index=True,
            column_config={
                "rank": "Rank",
                "ticker": "Ticker",
                "asset_class": "Asset type",
                "holding_group": "Sector / sleeve",
                "weight": st.column_config.NumberColumn("Latest weight", format="percent"),
            },
        )
    with right:
        st.subheader("Method assumptions")
        window_unit = "calendar days" if universe == "Crypto-only" else "trading days"
        st.markdown(
            f"**Estimation window:** {int(selected['estimation_window'])} {window_unit}  \n"
            f"**Rebalancing:** {selected['rebalance_frequency']}  \n"
            f"**Timing:** {selected['rebalance_convention']}  \n"
            f"**Constraints:** {selected['constraints']}  \n"
            f"**Transaction costs:** {selected['transaction_costs']:.0%}  \n"
            f"**Risk-free rate:** {selected['risk_free_rate']:.0%}"
        )
        if selected["method"] == "sentiment_tilt":
            st.caption(
                "The fixed equity-sector tilt uses only lagged VADER sentiment. Missing signals "
                "receive a multiplier of one; the crypto sleeve is unchanged."
            )

    st.subheader("Portfolio news context")
    portfolio_sentiment = data["portfolio_sentiment"].loc[
        data["portfolio_sentiment"]["fund"].eq(selected_fund)
    ].copy()
    if portfolio_sentiment.empty:
        st.info(
            "No eligible saved portfolio news context is available for this fund. Missing "
            "sentiment is not interpreted as neutral."
        )
    elif portfolio_sentiment["total_equity_weight"].max() <= 0:
        st.info(
            "This fund has no equity sleeve, so company-headline sector sentiment is not "
            "applicable. No zero or neutral score is assigned."
        )
    elif not portfolio_sentiment["portfolio_lagged_sentiment"].notna().any():
        st.info(
            "The fund has equity exposure, but no eligible lagged sector-news signal is "
            "available for its saved dates. Missing sentiment is not assigned zero."
        )
    else:
        monthly_context = (
            portfolio_sentiment.assign(
                date=portfolio_sentiment["date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("date", as_index=False)
            .agg(
                portfolio_lagged_sentiment=("portfolio_lagged_sentiment", "mean"),
                sentiment_weight_coverage=("sentiment_weight_coverage", "mean"),
            )
        )
        context = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.68, 0.32],
        )
        context.add_trace(
            go.Scatter(
                x=monthly_context["date"],
                y=monthly_context["portfolio_lagged_sentiment"],
                mode="lines",
                line=dict(color="#1485A3", width=2),
                connectgaps=False,
                hovertemplate="%{x|%b %Y}<br>Sentiment: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        context.add_trace(
            go.Scatter(
                x=monthly_context["date"],
                y=monthly_context["sentiment_weight_coverage"],
                mode="lines",
                line=dict(color="#168F78", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(22, 143, 120, 0.12)",
                hovertemplate="%{x|%b %Y}<br>Coverage: %{y:.1%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        context.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=12, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
        )
        context.update_yaxes(
            title="Monthly mean<br>VADER score",
            range=[-1.05, 1.05],
            gridcolor="#dce3e8",
            row=1,
            col=1,
        )
        context.update_yaxes(
            title="Coverage",
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor="#dce3e8",
            row=2,
            col=1,
        )
        st.plotly_chart(context, width="stretch", config=PLOT_CONFIG)
        latest_context = portfolio_sentiment.sort_values("date").iloc[-1]
        if pd.isna(latest_context["portfolio_lagged_sentiment"]):
            st.info(
                f"No eligible equity-news signal is available on "
                f"{latest_context['date']:%d %b %Y}; the saved value remains missing."
            )
        st.caption(
            "Descriptive equity-news context using saved monthly target sector weights and "
            "one-trading-day-lagged sector VADER scores. Missing-sector weight is excluded, not "
            f"set to neutral. Latest equity-weight coverage: "
            f"{latest_context['sentiment_weight_coverage']:.1%}."
        )

    st.subheader("Allocation workspace")
    allocation = pd.DataFrame(
        {
            "Fund": metrics["fund"],
            "Allocation (%)": 100.0 / len(metrics),
        }
    )
    edited = st.data_editor(
        allocation,
        width="stretch",
        hide_index=True,
        disabled=["Fund"],
        column_config={
            "Allocation (%)": st.column_config.NumberColumn(
                "Allocation (%)", min_value=0.0, max_value=100.0, step=5.0, format="%.1f%%"
            )
        },
        key=f"fund_allocation_{universe}",
    )
    total = float(edited["Allocation (%)"].sum())
    if not np.isclose(total, 100.0):
        st.warning(f"Allocation totals {total:.1f}%. Set the entries to 100% to view exposure.")
    elif total > 0:
        allocation_map = edited.set_index("Fund")["Allocation (%)"] / 100.0
        blended = holdings.assign(
            allocated_weight=holdings["weight"] * holdings["fund"].map(allocation_map)
        )
        exposure = (
            blended.groupby(["asset_class", "holding_group"], observed=True)["allocated_weight"]
            .sum()
            .reset_index()
            .sort_values("allocated_weight", ascending=False)
        )
        fig = px.bar(
            exposure,
            x="holding_group",
            y="allocated_weight",
            color="asset_class",
            color_discrete_map={"equity": "#1485A3", "crypto": "#D99100"},
            labels={"holding_group": "Sector / sleeve", "asset_class": "Asset type"},
        )
        fig.update_layout(
            height=340,
            margin=dict(l=15, r=15, t=28, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.02, yanchor="bottom"),
            xaxis_title=None,
            yaxis=dict(title="Allocated weight", tickformat=".0%", gridcolor="#dce3e8"),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption(
            "Exposure combines your allocation with each fund's latest saved holdings. It does "
            "not create a new backtest or forecast."
        )


def watchlist_view(data: AppDataBundle, equities: list[str], crypto: list[str]) -> None:
    page_heading(
        "PERSONAL WORKSPACE",
        "Track selected assets on their native calendars",
        "Equity histories use observed equity trading days (252-day annualisation). Crypto "
        "histories retain all calendar dates, including weekends (365-day annualisation).",
    )
    performance = data["asset_performance"].sort_values(["ticker", "date"])
    tickers = equities + crypto
    sample = performance.loc[performance["ticker"].isin(tickers)].copy()
    min_date = performance["date"].min().date()
    max_date = performance["date"].max().date()
    start, end = st.date_input(
        "Historical date range",
        value=(pd.Timestamp("2021-01-01").date(), max_date),
        min_value=min_date,
        max_value=max_date,
        key="asset_dates",
    )
    sample = sample.loc[sample["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if sample.empty:
        st.info("No saved observations fall inside this watchlist and date range.")
        return

    chart_rows = []
    for ticker, group in sample.groupby("ticker", observed=True):
        group = group.sort_values("date").copy()
        first_return = group["daily_return"].dropna().iloc[0]
        starting_value = group["growth_of_1"].iloc[0] / (1.0 + first_return)
        group["range_growth"] = group["growth_of_1"] / starting_value
        chart_rows.append(group)
    chart_data = pd.concat(chart_rows, ignore_index=True)
    fig = px.line(
        chart_data,
        x="date",
        y="range_growth",
        color="ticker",
        line_dash="asset_type",
        color_discrete_sequence=ASSET_COLORS,
        labels={"ticker": "Ticker", "asset_type": "Asset type"},
    )
    fig.update_traces(line_width=1.8)
    base_figure(fig, y_title="Precomputed growth index, rebased to $1")
    fig.update_yaxes(tickprefix="$", tickformat=".2f")
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    stats = selected_asset_statistics(sample)
    display = stats[
        [
            "ticker",
            "asset_type",
            "observations",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "maximum_drawdown",
            "best_day",
            "worst_day",
            "calendar",
        ]
    ]
    st.subheader("Historical risk summary")
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "asset_type": "Asset type",
            "observations": "Return observations",
            "total_return": st.column_config.NumberColumn("Range return", format="percent"),
            "annualized_return": st.column_config.NumberColumn(
                "Annualised return", format="percent"
            ),
            "annualized_volatility": st.column_config.NumberColumn(
                "Annualised volatility", format="percent"
            ),
            "maximum_drawdown": st.column_config.NumberColumn("Maximum drawdown", format="percent"),
            "best_day": st.column_config.NumberColumn("Best native day", format="percent"),
            "worst_day": st.column_config.NumberColumn("Worst native day", format="percent"),
            "calendar": "Saved calendar",
        },
    )
    st.markdown(
        '<div class="fm-note">Historical statistics use the selected range and the saved '
        "native-calendar daily returns. They are descriptive, not forecasts.</div>",
        unsafe_allow_html=True,
    )


def sentiment_line(
    frame: pd.DataFrame,
    *,
    group_column: str,
    score_column: str,
    color_map: dict[str, str] | None = None,
    height: int = 440,
) -> go.Figure:
    fig = go.Figure()
    colors = color_map or {}
    for position, (name, group) in enumerate(frame.groupby(group_column, observed=True)):
        color = colors.get(name, ASSET_COLORS[position % len(ASSET_COLORS)])
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group[score_column],
                mode="lines+markers",
                name=name,
                connectgaps=False,
                line=dict(color=color, width=1.1),
                marker=dict(color=color, size=4),
                customdata=group[["headline_count"]],
                hovertemplate=(
                    "%{x|%d %b %Y}<br>Score: %{y:.3f}<br>"
                    "Headlines: %{customdata[0]}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    base_figure(fig, y_title="VADER compound score (-1 to +1)", height=height)
    fig.update_yaxes(range=[-1.05, 1.05])
    return fig


def sentiment_display_frame(
    frame: pd.DataFrame,
    *,
    group_column: str,
    score_column: str,
    coverage_column: str,
    resolution: str,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Prepare monthly or short daily views from a saved daily sentiment grid."""
    if resolution == "Monthly":
        monthly = frame.assign(date=frame["date"].dt.to_period("M").dt.to_timestamp())
        return (
            monthly.groupby([group_column, "date"], observed=True, as_index=False)
            .agg(
                **{
                    score_column: (score_column, "mean"),
                    "headline_count": ("headline_count", "sum"),
                    coverage_column: (coverage_column, "mean"),
                }
            )
            .sort_values([group_column, "date"])
        )

    days = 30 if resolution == "Daily: last 30 days" else 90
    cutoff = end_date - pd.Timedelta(days=days - 1)
    return frame.loc[frame["date"].between(cutoff, end_date)].copy()


def headline_volume_figure(frame: pd.DataFrame, *, monthly: bool) -> go.Figure:
    """Show saved headline volume at the selected display frequency."""
    volume = frame.groupby("date", as_index=False)["headline_count"].sum()
    fig = px.bar(volume, x="date", y="headline_count", color_discrete_sequence=["#1485A3"])
    fig.update_layout(
        height=245,
        margin=dict(l=15, r=15, t=28, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(text=f"{'Monthly' if monthly else 'Daily'} headline volume", font_size=15),
        xaxis_title=None,
        yaxis=dict(title="Headlines", gridcolor="#dce3e8"),
        showlegend=False,
    )
    return fig


def my_mosaic_view(
    data: AppDataBundle,
    equities: list[str],
    crypto: list[str],
    sectors: list[str],
    selected_fund: str,
) -> None:
    """Render one coherent workspace from the session's shared selections."""
    page_heading(
        "MY MOSAIC",
        "Your selected markets in one historical view",
        "A compact workspace assembled from saved fund, asset and company-news outputs. "
        "Selections persist across every FinMosaic workspace.",
    )
    metrics = data["fund_metrics"]
    selected_metrics = metrics.loc[metrics["fund"].eq(selected_fund)].iloc[0]
    holdings = data["latest_holdings"].loc[data["latest_holdings"]["fund"].eq(selected_fund)].copy()

    summary = st.columns([0.7, 0.7, 0.7, 1.9])
    summary[0].metric("Equities", len(equities))
    summary[1].metric("Crypto assets", len(crypto))
    summary[2].metric("Sectors", len(sectors))
    summary[3].markdown(
        f"**Selected fund**  \n{selected_fund}  \n{selected_metrics['universe']} universe"
    )
    st.markdown(
        '<div class="fm-note">All returns, risk measures, sentiment scores and holdings are '
        "historical. This workspace does not assess suitability or provide investment advice."
        "</div>",
        unsafe_allow_html=True,
    )

    composition_left, exposure_right = st.columns(2, gap="large")
    with composition_left:
        st.subheader("Watchlist composition")
        composition = pd.DataFrame(
            {
                "Asset type": ["Equities", "Crypto assets"],
                "Selected": [len(equities), len(crypto)],
            }
        )
        if composition["Selected"].sum() == 0:
            st.info("Select at least one equity or crypto asset in the sidebar.")
        else:
            composition = composition.loc[composition["Selected"].gt(0)]
            st.plotly_chart(
                donut_figure(
                    composition,
                    names="Asset type",
                    values="Selected",
                    colors=["#1485A3", "#D99100"],
                ),
                width="stretch",
                config=PLOT_CONFIG,
            )
            composition["Share"] = composition["Selected"] / composition["Selected"].sum()
            st.dataframe(
                composition,
                width="stretch",
                hide_index=True,
                column_config={
                    "Asset type": "Asset type",
                    "Selected": "Selected assets",
                    "Share": st.column_config.NumberColumn("Watchlist share", format="percent"),
                },
            )
    with exposure_right:
        st.subheader("Selected fund exposure")
        exposure = (
            holdings.groupby("holding_group", observed=True, as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
        )
        st.plotly_chart(
            donut_figure(
                exposure,
                names="holding_group",
                values="weight",
                colors=[
                    "#1485A3",
                    "#168F78",
                    "#D99100",
                    "#C74763",
                    "#725E9C",
                    "#576574",
                    "#73A6AD",
                    "#9A7B4F",
                    "#8B6F8E",
                    "#3F7D65",
                    "#A85454",
                ],
            ),
            width="stretch",
            config=PLOT_CONFIG,
        )
        st.dataframe(
            exposure,
            width="stretch",
            hide_index=True,
            column_config={
                "holding_group": "Equity sector / sleeve",
                "weight": st.column_config.NumberColumn("Latest weight", format="percent"),
            },
        )

    st.subheader("Selected-asset history")
    tickers = equities + crypto
    performance = (
        data["asset_performance"].loc[data["asset_performance"]["ticker"].isin(tickers)].copy()
    )
    if performance.empty:
        st.info("Select at least one asset to view saved growth and risk history.")
    else:
        history_rows = []
        for ticker, group in performance.groupby("ticker", observed=True):
            group = group.sort_values("date").copy()
            group["selected_growth"] = group["growth_of_1"] / group["growth_of_1"].iloc[0]
            history_rows.append(group)
        history = pd.concat(history_rows, ignore_index=True)
        growth = px.line(
            history,
            x="date",
            y="selected_growth",
            color="ticker",
            line_dash="asset_type",
            color_discrete_sequence=ASSET_COLORS,
            labels={"ticker": "Ticker", "asset_type": "Asset type"},
        )
        growth.update_traces(line_width=1.7)
        base_figure(growth, y_title="Saved growth index, rebased to $1", height=350)
        growth.update_yaxes(tickprefix="$", tickformat=".2f")
        st.plotly_chart(growth, width="stretch", config=PLOT_CONFIG)
        risk = selected_asset_statistics(performance)[
            [
                "ticker",
                "asset_type",
                "annualized_return",
                "annualized_volatility",
                "maximum_drawdown",
                "calendar",
            ]
        ]
        st.dataframe(
            risk,
            width="stretch",
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "asset_type": "Asset type",
                "annualized_return": st.column_config.NumberColumn(
                    "Historical annualised return", format="percent"
                ),
                "annualized_volatility": st.column_config.NumberColumn(
                    "Historical annualised volatility", format="percent"
                ),
                "maximum_drawdown": st.column_config.NumberColumn(
                    "Historical maximum drawdown", format="percent"
                ),
                "calendar": "Saved calendar",
            },
        )

    sentiment_left, fund_right = st.columns([1.15, 0.85], gap="large")
    with sentiment_left:
        st.subheader("Selected-sector news context")
        sector_history = data["sector_sentiment"].loc[
            data["sector_sentiment"]["sector"].isin(sectors)
        ]
        if sector_history.empty:
            st.info("Select at least one sector to view saved VADER sentiment.")
        else:
            st.plotly_chart(
                sentiment_line(
                    sector_history,
                    group_column="sector",
                    score_column="sector_sentiment",
                    height=330,
                ),
                width="stretch",
                config=PLOT_CONFIG,
            )
            sector_summary = (
                sector_history.sort_values(["sector", "date"])
                .groupby("sector", observed=True, as_index=False)
                .tail(1)
                .loc[
                    :, ["sector", "date", "sector_sentiment", "headline_count", "coverage_ratio"]
                ]
                .sort_values("sector")
            )
            st.dataframe(
                sector_summary,
                width="stretch",
                hide_index=True,
                column_config={
                    "sector": "Sector",
                    "date": st.column_config.DateColumn("Latest saved date"),
                    "sector_sentiment": st.column_config.NumberColumn(
                        "VADER sentiment", format="%.3f"
                    ),
                    "headline_count": st.column_config.NumberColumn(
                        "Headlines", format="%d"
                    ),
                    "coverage_ratio": st.column_config.NumberColumn(
                        "Ticker coverage", format="percent"
                    ),
                },
            )
            st.caption(
                "Missing sector-days remain no-news observations; they are not converted to "
                "neutral sentiment."
            )
    with fund_right:
        st.subheader("Selected fund history")
        fund_metrics = st.columns(2)
        fund_metrics[0].metric("Annualised return", percent(selected_metrics["annualized_return"]))
        fund_metrics[1].metric(
            "Annualised volatility", percent(selected_metrics["annualized_volatility"])
        )
        fund_metrics[0].metric("Sharpe ratio", f"{selected_metrics['sharpe_ratio']:.2f}")
        fund_metrics[1].metric("Maximum drawdown", percent(selected_metrics["maximum_drawdown"]))
        latest_date = pd.to_datetime(holdings["effective_date"]).max().date()
        st.caption(
            f"Out-of-sample: {pd.to_datetime(selected_metrics['start_date']).date()} to "
            f"{pd.to_datetime(selected_metrics['end_date']).date()}. Latest target: {latest_date}."
        )
        st.dataframe(
            holdings.sort_values("rank").head(8)[["rank", "ticker", "holding_group", "weight"]],
            width="stretch",
            hide_index=True,
            column_config={
                "rank": "Rank",
                "ticker": "Ticker",
                "holding_group": "Sector / sleeve",
                "weight": st.column_config.NumberColumn("Latest weight", format="percent"),
            },
        )
        st.caption("Historical backtest metrics and latest saved target holdings only.")


def news_view(data: AppDataBundle, equities: list[str], sectors: list[str]) -> None:
    page_heading(
        "NEWS INTELLIGENCE",
        "Separate company signals from sector context",
        "Scores are based on original company-headline text. A zero score with observed news is "
        "neutral; a missing score means no news was observed for that ticker-day or sector-day.",
    )
    company = data["company_sentiment"].sort_values(["ticker", "date"])
    sector = data["sector_sentiment"].sort_values(["sector", "date"])
    min_date = company["date"].min().date()
    max_date = company["date"].max().date()
    start, end = st.date_input(
        "News date range",
        value=(pd.Timestamp("2022-01-01").date(), max_date),
        min_value=min_date,
        max_value=max_date,
        key="news_dates",
    )
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    resolution = st.selectbox(
        "Sentiment resolution",
        ["Monthly", "Daily: last 30 days", "Daily: last 90 days"],
        help=(
            "Monthly means use observed daily scores only. Daily views show a short window "
            "ending on the selected range end date."
        ),
        key="news_resolution",
    )

    company_tab, sector_tab = st.tabs(["Company VADER", "Sector VADER"])
    with company_tab:
        selected = company.loc[
            company["ticker"].isin(equities) & company["date"].between(start_ts, end_ts)
        ].copy()
        if selected.empty:
            st.info("Select at least one equity with saved observations.")
        else:
            display = sentiment_display_frame(
                selected,
                group_column="ticker",
                score_column="vader_company_sentiment",
                coverage_column="news_coverage",
                resolution=resolution,
                end_date=end_ts,
            )
            fig = sentiment_line(
                display,
                group_column="ticker",
                score_column="vader_company_sentiment",
            )
            if resolution == "Monthly":
                fig.update_yaxes(title="Monthly mean VADER score (-1 to +1)")
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
            st.plotly_chart(
                headline_volume_figure(display, monthly=resolution == "Monthly"),
                width="stretch",
                config=PLOT_CONFIG,
            )
            coverage = (
                selected.groupby(["ticker", "sector"], observed=True)
                .agg(
                    trading_days=("date", "size"),
                    observed_news_days=("news_observed", "sum"),
                    genuine_neutral_days=(
                        "vader_company_sentiment",
                        lambda values: int(values.eq(0).sum()),
                    ),
                    headlines=("headline_count", "sum"),
                )
                .reset_index()
            )
            coverage["no_news_days"] = coverage["trading_days"] - coverage["observed_news_days"]
            coverage["news_coverage"] = coverage["observed_news_days"] / coverage["trading_days"]
            st.dataframe(
                coverage,
                width="stretch",
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "sector": "Sector",
                    "trading_days": "Trading days",
                    "observed_news_days": "Days with news",
                    "no_news_days": "No-news days",
                    "genuine_neutral_days": "Observed neutral-score days",
                    "headlines": "Headlines",
                    "news_coverage": st.column_config.NumberColumn(
                        "News coverage", format="percent"
                    ),
                },
            )
            st.subheader("Recent saved headline examples")
            headline_examples = data["sentiment_disagreements"].loc[
                data["sentiment_disagreements"]["ticker"].isin(equities),
                ["trading_date", "ticker", "title", "vader_compound"],
            ].copy()
            headline_examples["headline_count"] = 1
            headline_examples = headline_examples.sort_values(
                ["trading_date", "ticker"], ascending=[False, True]
            ).head(10)
            if headline_examples.empty:
                st.info(
                    "No headline examples are available in the saved results for this selection."
                )
            else:
                st.dataframe(
                    headline_examples,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "trading_date": st.column_config.DateColumn("Date"),
                        "ticker": "Ticker",
                        "title": st.column_config.TextColumn("Headline", width="large"),
                        "vader_compound": st.column_config.NumberColumn(
                            "VADER score", format="%.3f"
                        ),
                        "headline_count": st.column_config.NumberColumn(
                            "Headline count", format="%d"
                        ),
                    },
                )
            st.caption(
                "This compact table uses the existing saved headline-example output; it is not "
                "a complete or live headline feed."
            )
    with sector_tab:
        selected = sector.loc[
            sector["sector"].isin(sectors) & sector["date"].between(start_ts, end_ts)
        ].copy()
        if selected.empty:
            st.info("Select at least one sector with saved observations.")
        else:
            display = sentiment_display_frame(
                selected,
                group_column="sector",
                score_column="sector_sentiment",
                coverage_column="coverage_ratio",
                resolution=resolution,
                end_date=end_ts,
            )
            fig = sentiment_line(
                display,
                group_column="sector",
                score_column="sector_sentiment",
            )
            if resolution == "Monthly":
                fig.update_yaxes(title="Monthly mean VADER score (-1 to +1)")
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
            st.plotly_chart(
                headline_volume_figure(display, monthly=resolution == "Monthly"),
                width="stretch",
                config=PLOT_CONFIG,
            )
            coverage = (
                selected.groupby("sector", observed=True)
                .agg(
                    trading_days=("date", "size"),
                    observed_sentiment_days=("sector_sentiment", "count"),
                    genuine_neutral_days=(
                        "sector_sentiment",
                        lambda values: int(values.eq(0).sum()),
                    ),
                    headlines=("headline_count", "sum"),
                    mean_ticker_coverage=("coverage_ratio", "mean"),
                )
                .reset_index()
            )
            coverage["no_news_days"] = (
                coverage["trading_days"] - coverage["observed_sentiment_days"]
            )
            st.dataframe(
                coverage,
                width="stretch",
                hide_index=True,
                column_config={
                    "sector": "Sector",
                    "trading_days": "Trading days",
                    "observed_sentiment_days": "Days with news",
                    "no_news_days": "No-news days",
                    "genuine_neutral_days": "Observed neutral-score days",
                    "headlines": "Headlines",
                    "mean_ticker_coverage": st.column_config.NumberColumn(
                        "Mean ticker coverage", format="percent"
                    ),
                },
            )
            st.caption(
                "Sector scores equal-weight observed ticker-day scores. Missing sectors remain "
                "missing; they are not assigned zero. The portfolio fusion uses only the saved "
                "one-equity-trading-day-lagged signal."
            )


def robustness_view(data: AppDataBundle) -> None:
    page_heading(
        "METHODOLOGY",
        "VADER and financial-dictionary robustness",
        "A read-only comparison of two precomputed headline-scoring methods. This view evaluates "
        "measurement sensitivity; neither method is presented as a trading recommendation.",
    )
    distributions = data["sentiment_distributions"].copy()
    comparison = data["sentiment_model_comparison"].copy()
    disagreements = data["sentiment_disagreements"].copy()
    coverage = data["lm_coverage"].copy()

    distribution_tab, sector_tab, examples_tab, definitions_tab = st.tabs(
        ["Score distributions", "Sector comparison", "Disagreement examples", "Definitions"]
    )
    with distribution_tab:
        st.image(
            RESULTS / "figures" / "sentiment_score_distributions_vader_vs_lm.png",
            width="stretch",
        )
        st.dataframe(
            distributions,
            width="stretch",
            hide_index=True,
            column_config={
                "model": "Method",
                "headline_count": "Headlines",
                "mean_score": st.column_config.NumberColumn("Mean score", format="%.3f"),
                "score_std": st.column_config.NumberColumn(
                    "Score standard deviation", format="%.3f"
                ),
                "negative_ratio": st.column_config.NumberColumn("Negative", format="percent"),
                "neutral_ratio": st.column_config.NumberColumn("Neutral", format="percent"),
                "positive_ratio": st.column_config.NumberColumn("Positive", format="percent"),
            },
        )
    with sector_tab:
        fig = px.bar(
            comparison.sort_values("pearson_correlation"),
            x="pearson_correlation",
            y="sector",
            orientation="h",
            color="classification_disagreement_ratio",
            color_continuous_scale=[[0, "#dce3e8"], [1, "#C74763"]],
            labels={
                "sector": "Sector",
                "pearson_correlation": "Pearson correlation",
                "classification_disagreement_ratio": "Classification disagreement",
            },
        )
        fig.update_layout(
            height=460,
            margin=dict(l=15, r=15, t=30, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(range=[-1, 1], gridcolor="#dce3e8"),
            yaxis_title=None,
            coloraxis_colorbar=dict(tickformat=".0%", title="Disagreement"),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "sector": "Sector",
                "common_observed_days": "Common observed days",
                "pearson_correlation": st.column_config.NumberColumn(
                    "Pearson correlation", format="%.3f"
                ),
                "spearman_correlation": st.column_config.NumberColumn(
                    "Spearman correlation", format="%.3f"
                ),
                "classification_agreement_ratio": st.column_config.NumberColumn(
                    "Agreement", format="percent"
                ),
                "classification_disagreement_ratio": st.column_config.NumberColumn(
                    "Disagreement", format="percent"
                ),
                "opposite_sign_ratio": st.column_config.NumberColumn(
                    "Opposite sign", format="percent"
                ),
                "mean_absolute_score_difference": st.column_config.NumberColumn(
                    "Mean absolute difference", format="%.3f"
                ),
            },
        )
    with examples_tab:
        columns = [
            "trading_date",
            "ticker",
            "sector",
            "title",
            "vader_compound",
            "lm_tone",
            "lm_positive_terms",
            "lm_negative_terms",
            "lm_matched_token_ratio",
        ]
        st.dataframe(
            disagreements[columns],
            width="stretch",
            hide_index=True,
            column_config={
                "trading_date": st.column_config.DateColumn("Trading date"),
                "ticker": "Ticker",
                "sector": "Sector",
                "title": st.column_config.TextColumn("Original headline", width="large"),
                "vader_compound": st.column_config.NumberColumn("VADER", format="%.3f"),
                "lm_tone": st.column_config.NumberColumn("LM tone", format="%.3f"),
                "lm_positive_terms": "LM positive terms",
                "lm_negative_terms": "LM negative terms",
                "lm_matched_token_ratio": st.column_config.NumberColumn(
                    "Matched-token coverage", format="percent"
                ),
            },
        )
        st.caption("Examples are selected for interpretability from the saved comparison output.")
    with definitions_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Coverage by sector")
            st.dataframe(
                coverage,
                width="stretch",
                hide_index=True,
                column_config={
                    "sector": "Sector",
                    "headline_count": "Headlines",
                    "matched_headline_ratio": st.column_config.NumberColumn(
                        "Headlines with an LM match", format="percent"
                    ),
                    "matched_token_ratio": st.column_config.NumberColumn(
                        "Matched-token coverage", format="percent"
                    ),
                    "company_news_coverage_ratio": st.column_config.NumberColumn(
                        "Company news coverage", format="percent"
                    ),
                    "sector_news_coverage_ratio": st.column_config.NumberColumn(
                        "Sector news coverage", format="percent"
                    ),
                },
            )
        with right:
            st.subheader("Fixed methodology")
            vader = data["vader_methodology"].iloc[0]
            lm = data["lm_methodology"].iloc[0]
            st.markdown(
                f"**VADER**  \nOriginal headline text; {vader['no_news_policy']}.  \n\n"
                f"**Loughran-McDonald**  \nDictionary version date: {lm['version_date']}.  \n"
                f"Score: {lm['scoring_formula']}  \nNo-news policy: {lm['no_news_policy']}."
            )
            st.caption(
                "Both methods use the same cleaned headlines, ticker-day and sector aggregation, "
                "missing-data policy, and one-equity-trading-day lag. No lexicon, threshold, tilt, "
                "or portfolio rule was tuned on out-of-sample performance."
            )


def sidebar_controls(
    data: AppDataBundle,
) -> tuple[str, list[str], list[str], list[str], str, str]:
    st.sidebar.title("FinMosaic")
    st.sidebar.caption("Personalised financial information")
    views = ["My Mosaic", "Funds", "Watchlist", "News", "Methodology"]
    initial = st.query_params.get("view", "My Mosaic")
    if initial not in views:
        initial = "My Mosaic"
    view = st.sidebar.radio("Workspace", views, index=views.index(initial))
    st.query_params["view"] = view

    performance = data["asset_performance"]
    equity_options = sorted(
        performance.loc[performance["asset_type"].eq("Equity"), "ticker"].unique()
    )
    crypto_options = sorted(
        performance.loc[performance["asset_type"].eq("Crypto"), "ticker"].unique()
    )
    sector_options = sorted(data["sector_sentiment"]["sector"].unique())

    defaults = WATCHLIST_PRESETS["Cross-market sample"]
    st.session_state.setdefault("selected_equities", defaults["equities"])
    st.session_state.setdefault("selected_crypto", defaults["crypto"])
    st.session_state.setdefault("selected_sectors", defaults["sectors"])

    st.sidebar.divider()
    st.sidebar.subheader("Selected fund")
    universe = st.sidebar.selectbox(
        "Fund universe",
        ["Combined", "Equity-only", "Crypto-only"],
        key="selected_universe",
    )
    fund_options = (
        data["fund_metrics"]
        .loc[data["fund_metrics"]["universe"].eq(universe)]
        .sort_values(
            "method",
            key=lambda values: values.map(
                {"equal_weight": 0, "min_variance": 1, "max_sharpe": 2, "sentiment_tilt": 3}
            ),
        )["fund"]
        .tolist()
    )
    if st.session_state.get("selected_fund") not in fund_options:
        st.session_state["selected_fund"] = fund_options[0]
    selected_fund = st.sidebar.selectbox(
        "Fund",
        fund_options,
        key="selected_fund",
    )
    st.sidebar.markdown(f"**Current fund**  \n{universe} · {selected_fund}")

    st.sidebar.divider()
    st.sidebar.subheader("Watchlist")
    preset = st.sidebar.selectbox(
        "Optional preset",
        ["Manual", *WATCHLIST_PRESETS],
        help="Presets only populate the watchlist controls and are not recommendations.",
        key="watchlist_preset",
    )
    if st.sidebar.button("Apply preset", disabled=preset == "Manual", width="stretch"):
        selected_preset = WATCHLIST_PRESETS[preset]
        st.session_state["selected_equities"] = selected_preset["equities"]
        st.session_state["selected_crypto"] = selected_preset["crypto"]
        st.session_state["selected_sectors"] = selected_preset["sectors"]
    st.sidebar.caption("Presets are interface shortcuts, not investment recommendations.")
    equities = st.sidebar.multiselect(
        "Equities",
        equity_options,
        key="selected_equities",
    )
    crypto = st.sidebar.multiselect(
        "Crypto assets",
        crypto_options,
        key="selected_crypto",
    )
    sectors = st.sidebar.multiselect(
        "Sectors",
        sector_options,
        key="selected_sectors",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Saved outputs through 29 December 2023 for equities and 31 December 2023 for crypto."
    )
    return view, equities, crypto, sectors, selected_fund, universe


def main() -> None:
    inject_style()
    try:
        data = cached_results(RESULTS_SCHEMA_VERSION)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Precomputed results could not be loaded: {error}")
        st.stop()

    view, equities, crypto, sectors, selected_fund, universe = sidebar_controls(data)
    if view == "My Mosaic":
        my_mosaic_view(data, equities, crypto, sectors, selected_fund)
    elif view == "Funds":
        fund_view(data, selected_fund, universe)
    elif view == "Watchlist":
        watchlist_view(data, equities, crypto)
    elif view == "News":
        news_view(data, equities, sectors)
    else:
        robustness_view(data)

    st.markdown(
        '<div class="fm-footer">Historical financial information and model diagnostics only. '
        "FinMosaic does not assess suitability, recommend securities, or provide "
        "investment advice. "
        "Source: hosted course price and company-headline data; author calculations.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
