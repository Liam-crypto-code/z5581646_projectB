"""Report-ready FinMosaic figures for Part B portfolio and sentiment outputs."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BACKGROUND = "#FFFFFF"
TEXT = "#17212B"
MUTED = "#647180"
GRID = "#DCE2E8"
ACCENT = "#F4C542"
NEWS = "#6F4AA8"
NEWS_LIGHT = "#C7B7DF"
BASE_FUND = "#087EA4"
TILTED_FUND = "#C4475D"

FUND_STYLES = {
    "equal_weight": ("Combined Equal Weight", "#087EA4"),
    "min_variance": ("Combined Minimum Variance", "#118C72"),
    "max_sharpe": ("Combined Maximum Sharpe", "#D99400"),
    "sentiment_tilt": ("Sentiment-Tilted Equal Weight", "#C4475D"),
}

SECTOR_COLORS = (
    "#087EA4",
    "#D99400",
    "#118C72",
    "#C4475D",
    "#6F4AA8",
    "#527A3D",
    "#B65C36",
    "#3B6F8A",
    "#9B6A97",
    "#69747C",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )


def _save(fig: plt.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)


def _format_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.day} {timestamp:%b %Y}"


def plot_sector_sentiment_index(index: pd.DataFrame, path: pathlib.Path) -> None:
    """Plot the one-trading-day-lagged daily sector index in small multiples."""
    _style()
    sectors = sorted(index["sector"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(9.2, 10.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.09, hspace=0.38, wspace=0.18)
    fig.text(
        0.09,
        0.97,
        "FINMOSAIC  /  STANDALONE SENTIMENT INDEX",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.93,
        "Sector sentiment varied substantially through time",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.895,
        "VADER compound score; equal-weight ticker aggregation; one equity-trading-day lag",
        color=MUTED,
        fontsize=9,
    )
    for axis, sector, color in zip(axes.flat, sectors, SECTOR_COLORS, strict=True):
        sample = index.loc[index["sector"] == sector]
        axis.axhline(0, color=GRID, linewidth=0.8)
        axis.plot(
            sample["date"],
            sample["lagged_sector_sentiment"],
            color=color,
            linewidth=0.75,
            alpha=0.9,
        )
        axis.set_title(sector, loc="left", pad=4)
        axis.set_ylim(-1, 1)
        axis.set_yticks([-0.5, 0, 0.5])
        axis.grid(axis="y", color=GRID, linewidth=0.55, linestyle=(0, (2, 2)))
        axis.tick_params(length=0)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.spines["bottom"].set_color(GRID)
    fig.text(0.02, 0.50, "Lagged VADER compound score", rotation=90, va="center", color=TEXT)
    fig.text(
        0.09,
        0.035,
        "Source: Hosted course headline data; VADER and author calculations. "
        "Sample: 2 Jan 2020-29 Dec 2023. Missing segments denote no source-day headlines.",
        color=MUTED,
        fontsize=7,
    )
    _save(fig, path)


def plot_sentiment_coverage(summary: pd.DataFrame, path: pathlib.Path) -> None:
    """Compare the share of equity trading days with observed sector sentiment."""
    _style()
    sample = summary.sort_values("observed_day_ratio")
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.76, bottom=0.18)
    fig.text(0.09, 0.95, "FINMOSAIC  /  SENTIMENT COVERAGE", color=MUTED, fontsize=8, weight="bold")
    fig.text(
        0.09,
        0.87,
        "News availability differed across equity sectors",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Share of equity trading days with at least one covered ticker; "
        "no-news days are not neutral",
        color=MUTED,
        fontsize=9,
    )
    colors = [NEWS_LIGHT] * len(sample)
    colors[-1] = NEWS
    bars = axis.barh(sample["sector"], sample["observed_day_ratio"], color=colors, height=0.65)
    axis.set_xlabel("Trading days with observed sentiment")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axis.grid(axis="x", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0)
    axis.spines["bottom"].set_color(GRID)
    for bar, value in zip(bars, sample["observed_day_ratio"], strict=True):
        axis.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            color=TEXT,
            fontsize=8,
        )
    fig.text(
        0.09,
        0.06,
        "Source: Hosted course headline data; author calculations. "
        "Sample: 2 Jan 2020-29 Dec 2023. Coverage is measured before the trading-day lag.",
        color=MUTED,
        fontsize=7,
    )
    _save(fig, path)


def plot_fusion_comparison(
    base_daily: pd.DataFrame,
    tilted_daily: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Compare growth and drawdown on the identical out-of-sample dates."""
    _style()
    if not base_daily.index.equals(tilted_daily.index):
        raise ValueError("base and tilted funds must use identical dates")
    fig, (growth_axis, drawdown_axis) = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.72, bottom=0.13, hspace=0.32)
    fig.text(0.09, 0.95, "FINMOSAIC  /  SENTIMENT FUSION", color=MUTED, fontsize=8, weight="bold")
    fig.text(
        0.09,
        0.87,
        "Fixed sentiment tilt versus Combined Equal Weight",
        color=TEXT,
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Monthly targets; 50% tilt strength; crypto targets unchanged; zero transaction costs",
        color=MUTED,
        fontsize=9,
    )

    series = (
        (base_daily, "Combined Equal Weight", BASE_FUND),
        (tilted_daily, "Sentiment-Tilted Equal Weight", TILTED_FUND),
    )
    for frame, label, color in series:
        growth_axis.plot(frame.index, frame["growth_of_1"], label=label, color=color, linewidth=1.5)
        drawdown_axis.plot(frame.index, frame["drawdown"], label=label, color=color, linewidth=1.2)
    growth_axis.set_ylabel("Value of $1")
    growth_axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:.2f}"))
    growth_axis.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    drawdown_axis.set_xlabel("")
    drawdown_axis.xaxis.set_major_locator(mdates.YearLocator())
    drawdown_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in (growth_axis, drawdown_axis):
        axis.grid(axis="y", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
        axis.tick_params(length=0)
        axis.spines["bottom"].set_color(GRID)
    fig.text(
        0.09,
        0.045,
        "Source: Hosted course price and headline data; author calculations. "
        "Out-of-sample period: 5 Jan 2021-29 Dec 2023.",
        color=MUTED,
        fontsize=7,
    )
    _save(fig, path)


def _fund_sample_note(frame: pd.DataFrame) -> str:
    start = _format_date(pd.to_datetime(frame["date"]).min())
    end = _format_date(pd.to_datetime(frame["date"]).max())
    return f"Out-of-sample period: {start}-{end}."


def plot_all_fund_growth(fund_returns: pd.DataFrame, path: pathlib.Path) -> None:
    """Compare growth of one dollar for all four investable funds."""
    _style()
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.76, bottom=0.18)
    fig.text(0.09, 0.95, "FINMOSAIC  /  FUND COMPARISON", color=MUTED, fontsize=8, weight="bold")
    fig.text(
        0.09,
        0.87,
        "Four funds produced distinct out-of-sample paths",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Growth of $1 from simple daily portfolio returns; no transaction costs",
        color=MUTED,
        fontsize=9,
    )
    for method, (label, color) in FUND_STYLES.items():
        sample = fund_returns[fund_returns["method"] == method].sort_values("date")
        axis.plot(sample["date"], sample["growth_of_1"], label=label, color=color, linewidth=1.45)
    axis.axhline(1.0, color=GRID, linewidth=0.8)
    axis.set_ylabel("Value of $1 investment")
    axis.set_xlabel("Equity trading date")
    axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:.2f}"))
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.09,
        0.055,
        "Source: Hosted course price and headline data; author calculations. "
        + _fund_sample_note(fund_returns),
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.96, 0.055, "Units: US dollars", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_all_fund_drawdowns(fund_returns: pd.DataFrame, path: pathlib.Path) -> None:
    """Compare drawdowns from running peaks for all funds."""
    _style()
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.68, bottom=0.18)
    fig.text(
        0.09, 0.95, "FINMOSAIC  /  DRAWDOWN COMPARISON", color=MUTED, fontsize=8, weight="bold"
    )
    fig.text(
        0.09,
        0.87,
        "Minimum Variance limited the deepest loss",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Percentage decline from each fund's previous growth-of-$1 peak",
        color=MUTED,
        fontsize=9,
    )
    for method, (label, color) in FUND_STYLES.items():
        sample = fund_returns[fund_returns["method"] == method].sort_values("date")
        axis.plot(sample["date"], sample["drawdown"], label=label, color=color, linewidth=1.3)
    axis.axhline(0, color=GRID, linewidth=0.8)
    axis.set_ylabel("Drawdown from prior peak")
    axis.set_xlabel("Equity trading date")
    axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
    )
    fig.text(
        0.09,
        0.055,
        "Source: Hosted course price and headline data; author calculations. "
        + _fund_sample_note(fund_returns),
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.96, 0.055, "Units: percent", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_sharpe_ratios(metrics: pd.DataFrame, path: pathlib.Path) -> None:
    """Compare zero-risk-free-rate annualized Sharpe ratios."""
    _style()
    ordered = metrics.sort_values("sharpe_ratio")
    colors = [FUND_STYLES[method][1] for method in ordered["method"]]
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.76, bottom=0.18)
    fig.text(
        0.09,
        0.95,
        "FINMOSAIC  /  RISK-ADJUSTED PERFORMANCE",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09, 0.87, "Maximum Sharpe led the fund comparison", color=TEXT, fontsize=16, weight="bold"
    )
    fig.text(
        0.09,
        0.815,
        "Annualized mean daily return divided by volatility; risk-free rate assumed zero",
        color=MUTED,
        fontsize=9,
    )
    bars = axis.barh(ordered["fund"], ordered["sharpe_ratio"], color=colors, height=0.60)
    axis.set_xlabel("Annualized Sharpe ratio")
    axis.set_ylabel("")
    axis.set_xlim(0, max(1.0, ordered["sharpe_ratio"].max() * 1.18))
    axis.grid(axis="x", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0)
    for bar, value in zip(bars, ordered["sharpe_ratio"], strict=True):
        axis.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            color=TEXT,
        )
    start = _format_date(pd.to_datetime(metrics["start_date"]).min())
    end = _format_date(pd.to_datetime(metrics["end_date"]).max())
    fig.text(
        0.09,
        0.055,
        "Source: Hosted course price and headline data; author calculations. "
        f"Out-of-sample period: {start}-{end}.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.95, 0.055, "Units: ratio", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_weight_group_history(history: pd.DataFrame, path: pathlib.Path) -> None:
    """Show monthly target weights as aligned sector-plus-crypto heatmaps."""
    _style()
    methods = list(FUND_STYLES)
    groups = ["Crypto", *sorted(set(history["allocation_group"]).difference({"Crypto"}))]
    dates = pd.DatetimeIndex(sorted(history["effective_date"].unique()))
    maximum = max(0.30, float(history["target_weight"].max()))
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 11.2), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.84, bottom=0.09, hspace=0.30)
    fig.text(
        0.09, 0.97, "FINMOSAIC  /  MONTHLY TARGET WEIGHTS", color=MUTED, fontsize=8, weight="bold"
    )
    fig.text(
        0.09,
        0.93,
        "Dynamic methods departed from Equal Weight in different ways",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.89,
        "Ten equity sectors plus unchanged or optimized crypto sleeve; "
        "first effective date each month",
        color=MUTED,
        fontsize=9,
    )
    image = None
    for axis, method in zip(axes, methods, strict=True):
        sample = history[history["method"] == method]
        matrix = (
            sample.pivot(index="allocation_group", columns="effective_date", values="target_weight")
            .reindex(index=groups, columns=dates)
            .fillna(0.0)
        )
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="YlGnBu",
            vmin=0,
            vmax=maximum,
        )
        axis.set_yticks(np.arange(len(groups)), labels=groups)
        axis.set_title(FUND_STYLES[method][0], loc="left", pad=5)
        axis.tick_params(length=0)
    tick_positions = np.unique(
        np.linspace(0, len(dates) - 1, min(4, len(dates)), dtype=int)
    ).tolist()
    tick_labels = [dates[position].strftime("%b %Y") for position in tick_positions]
    axes[-1].set_xticks(tick_positions, labels=tick_labels)
    axes[-1].set_xlabel("Monthly target effective date")
    if image is not None:
        colorbar_axis = fig.add_axes([0.90, 0.18, 0.018, 0.55])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Target portfolio weight")
        colorbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.text(
        0.09,
        0.035,
        "Source: results/data/fund_weights.csv; author calculations. "
        "Sample: monthly targets effective 5 Jan 2021-4 Dec 2023. Units: percent of portfolio.",
        color=MUTED,
        fontsize=7,
    )
    _save(fig, path)


def plot_fact_sheet(
    daily: pd.DataFrame,
    metrics: pd.Series,
    holdings: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Render one report-ready fund fact sheet from saved metrics and targets."""
    _style()
    method = str(metrics["method"])
    _, color = FUND_STYLES[method]
    fig = plt.figure(figsize=(9.2, 8.5))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.96,
        top=0.61,
        bottom=0.18,
        width_ratios=(1.65, 1.0),
        hspace=0.42,
        wspace=0.34,
    )
    growth_axis = fig.add_subplot(grid[0, 0])
    drawdown_axis = fig.add_subplot(grid[1, 0], sharex=growth_axis)
    holdings_axis = fig.add_subplot(grid[:, 1])

    fig.text(0.09, 0.96, "FINMOSAIC  /  FUND FACT SHEET", color=MUTED, fontsize=8, weight="bold")
    fig.text(0.09, 0.90, str(metrics["fund"]), color=TEXT, fontsize=17, weight="bold")
    start = _format_date(metrics["start_date"])
    end = _format_date(metrics["end_date"])
    latest = _format_date(holdings["effective_date"].iloc[0])
    fig.text(
        0.09,
        0.855,
        f"Out-of-sample performance: {start}-{end}  |  Latest target holdings: {latest}",
        color=MUTED,
        fontsize=9,
    )

    metric_items = (
        ("ANNUAL RETURN", f"{metrics['annualized_return']:.1%}"),
        ("VOLATILITY", f"{metrics['annualized_volatility']:.1%}"),
        ("SHARPE", f"{metrics['sharpe_ratio']:.3f}"),
        ("MAX DRAWDOWN", f"{metrics['maximum_drawdown']:.1%}"),
        ("GROWTH OF $1", f"${metrics['growth_of_1']:.3f}"),
    )
    for x, (label, value) in zip(np.linspace(0.09, 0.82, 5), metric_items, strict=True):
        fig.text(x, 0.77, label, color=MUTED, fontsize=7, weight="bold")
        fig.text(x, 0.72, value, color=TEXT, fontsize=14, weight="bold")

    growth_axis.plot(daily.index, daily["growth_of_1"], color=color, linewidth=1.4)
    growth_axis.set_title("Growth of $1", loc="left")
    growth_axis.set_ylabel("US dollars")
    growth_axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:.2f}"))
    drawdown_axis.plot(daily.index, daily["drawdown"], color=color, linewidth=1.2)
    drawdown_axis.set_title("Drawdown", loc="left")
    drawdown_axis.set_ylabel("Percent")
    drawdown_axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    drawdown_axis.xaxis.set_major_locator(mdates.YearLocator())
    drawdown_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in (growth_axis, drawdown_axis):
        axis.grid(axis="y", color=GRID, linewidth=0.55, linestyle=(0, (2, 2)))
        axis.tick_params(length=0)

    top = holdings.nsmallest(10, "rank").sort_values("weight")
    holdings_axis.barh(top["ticker"], top["weight"], color=color, alpha=0.85, height=0.62)
    holdings_axis.set_title("Ten largest target holdings", loc="left")
    holdings_axis.set_xlabel("Portfolio target weight")
    holdings_axis.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    holdings_axis.grid(axis="x", color=GRID, linewidth=0.55, linestyle=(0, (2, 2)))
    holdings_axis.tick_params(length=0)

    assumptions = (
        f"Method: {method.replace('_', ' ').title()}  |  Estimation: "
        f"{int(metrics['estimation_window'])} trading days  |  "
        f"Rebalance: {metrics['rebalance_frequency']}, {metrics['rebalance_convention']}"
    )
    fig.text(0.09, 0.115, assumptions, color=TEXT, fontsize=7.5)
    fig.text(
        0.09,
        0.082,
        f"Constraints: {metrics['constraints']}  |  "
        f"Risk-free rate: {metrics['risk_free_rate']:.0%}  |  "
        f"Transaction costs: {metrics['transaction_costs']:.0%}",
        color=MUTED,
        fontsize=7.2,
    )
    parameter_note = "Parameters: fixed ex ante; no out-of-sample tuning"
    if method == "sentiment_tilt":
        parameter_note += (
            f"  |  Tilt strength: {metrics['tilt_strength']:.2f}  |  "
            f"Missing signal: {metrics['missing_signal_policy']}"
        )
    fig.text(0.09, 0.052, parameter_note, color=MUTED, fontsize=7.2)
    fig.text(
        0.09,
        0.020,
        "Source: results/data/fund_returns.csv, results/data/fund_weights.csv and "
        "results/tables/performance_metrics.csv; author calculations. Not investment advice.",
        color=MUTED,
        fontsize=7,
    )
    _save(fig, path)


def plot_sentiment_model_distributions(
    comparison: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Compare headline-level VADER and historical LM score distributions."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.74, bottom=0.22, wspace=0.20)
    fig.text(
        0.09,
        0.95,
        "FINMOSAIC  /  FINANCIAL-DICTIONARY ROBUSTNESS",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.87,
        "The finance dictionary changed headline-score shape",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Original headline text; fixed model-native scoring; no portfolio use or OOS tuning",
        color=MUTED,
        fontsize=9,
    )
    bins = np.linspace(-1, 1, 51)
    specifications = (
        ("VADER compound", comparison["vader_compound"], NEWS),
        ("Loughran-McDonald 2018 tone", comparison["lm_tone"], "#118C72"),
    )
    for axis, (title, values, color) in zip(axes, specifications, strict=True):
        weights = np.full(len(values), 100 / len(values))
        axis.hist(values, bins=bins, weights=weights, color=color, alpha=0.88)
        axis.axvline(0, color=TEXT, linewidth=0.8)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Headline sentiment score")
        axis.set_xlim(-1, 1)
        axis.grid(axis="y", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
        axis.tick_params(length=0)
    axes[0].set_ylabel("Share of headlines per bin (%)")
    start = _format_date(comparison["trading_date"].min())
    end = _format_date(comparison["trading_date"].max())
    fig.text(
        0.09,
        0.070,
        "Source: Hosted course headlines; NLTK VADER; official Loughran-McDonald "
        "2018 word lists; author calculations.\n"
        f"Sample: headline dates aligned over {start}-{end}.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(
        0.97,
        0.035,
        "Units: percent; scores [-1, 1]",
        color=MUTED,
        fontsize=7,
        ha="right",
    )
    _save(fig, path)


def plot_sector_sentiment_model_comparison(
    sector_comparison: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Show daily-score correlation and model-native disagreement by sector."""
    _style()
    ordered = sector_comparison.sort_values("pearson_correlation")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.4), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.73, bottom=0.19, wspace=0.12)
    fig.text(
        0.09,
        0.95,
        "FINMOSAIC  /  SECTOR MODEL COMPARISON",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.87,
        "VADER and financial tone were related but not interchangeable",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Same observed sector-days; equal-weight ticker aggregation; contemporaneous scores",
        color=MUTED,
        fontsize=9,
    )
    axes[0].barh(y, ordered["pearson_correlation"], color=NEWS, height=0.62)
    axes[0].set_yticks(y, labels=ordered["sector"])
    axes[0].set_xlabel("Pearson correlation")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Daily score association", loc="left")
    axes[1].barh(
        y,
        ordered["classification_disagreement_ratio"],
        color="#C4475D",
        height=0.62,
    )
    axes[1].set_xlabel("Classification disagreement")
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[1].set_xlim(0, max(0.60, ordered["classification_disagreement_ratio"].max() * 1.12))
    axes[1].set_title("Model-native sign/neutral classes", loc="left")
    for axis in axes:
        axis.grid(axis="x", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
        axis.tick_params(length=0)
    fig.text(
        0.09,
        0.065,
        "Source: results/data/sector_sentiment_index.csv and "
        "results/data/lm_sector_sentiment_index.csv; author calculations.\n"
        "Sample: 2 Jan 2020-29 Dec 2023. VADER thresholds: +/-0.05; LM threshold: zero.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.97, 0.030, "Units: ratio", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_lm_coverage(coverage: pd.DataFrame, path: pathlib.Path) -> None:
    """Show news availability and LM sentiment-token matching by sector."""
    _style()
    ordered = coverage.sort_values("matched_headline_ratio")
    y = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(9.2, 6.4))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.73, bottom=0.18)
    fig.text(
        0.09,
        0.95,
        "FINMOSAIC  /  FINANCIAL-LEXICON COVERAGE",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.87,
        "Most news days were observed, but sentiment words were sparse",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.815,
        "Coverage is disclosed separately from tone; unmatched headlines remain observed zero tone",
        color=MUTED,
        fontsize=9,
    )
    height = 0.23
    axis.barh(
        y + height,
        ordered["sector_news_coverage_ratio"],
        height=height,
        label="Sector days with news",
        color=BASE_FUND,
    )
    axis.barh(
        y,
        ordered["matched_headline_ratio"],
        height=height,
        label="Headlines with LM match",
        color="#118C72",
    )
    axis.barh(
        y - height,
        ordered["matched_token_ratio"],
        height=height,
        label="Tokens matched to LM",
        color=ACCENT,
    )
    axis.set_yticks(y, labels=ordered["sector"])
    axis.set_xlabel("Coverage ratio")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axis.grid(axis="x", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0)
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=3,
        fontsize=7.5,
    )
    fig.text(
        0.09,
        0.060,
        "Source: Hosted course headlines and official Loughran-McDonald 2018 sentiment "
        "word lists; author calculations. Sample: 2 Jan 2020-29 Dec 2023.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.96, 0.060, "Units: percent", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_company_vader_sentiment_heatmap(
    company_index: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Show same-day company VADER tone while displaying no-news cells as missing."""
    _style()
    ordered = (
        company_index[["sector", "ticker"]]
        .drop_duplicates()
        .sort_values(["sector", "ticker"])
    )
    row_index = pd.MultiIndex.from_frame(ordered[["sector", "ticker"]])
    dates = pd.DatetimeIndex(sorted(company_index["date"].unique()))
    matrix = (
        company_index.pivot(
            index=["sector", "ticker"], columns="date", values="vader_company_sentiment"
        )
        .reindex(index=row_index, columns=dates)
    )
    colour_map = plt.get_cmap("RdBu").copy()
    colour_map.set_bad("#EEF1F4")

    fig, axis = plt.subplots(figsize=(9.2, 11.4))
    fig.subplots_adjust(left=0.17, right=0.89, top=0.85, bottom=0.115)
    fig.text(
        0.09,
        0.97,
        "FINMOSAIC  /  COMPANY NEWS SENTIMENT",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.93,
        "Company tone and news availability varied across the equity universe",
        color=TEXT,
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.09,
        0.89,
        "Same-day VADER ticker sentiment; grey cells are no-news days, not neutral scores",
        color=MUTED,
        fontsize=9,
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix.to_numpy()),
        aspect="auto",
        interpolation="nearest",
        cmap=colour_map,
        vmin=-1,
        vmax=1,
    )
    axis.set_yticks(np.arange(len(ordered)), labels=ordered["ticker"])
    axis.set_ylabel("Equity ticker (grouped by sector)")
    tick_positions = np.unique(np.linspace(0, len(dates) - 1, 5, dtype=int))
    tick_labels = [dates[position].strftime("%b %Y") for position in tick_positions]
    axis.set_xticks(tick_positions, labels=tick_labels)
    axis.set_xlabel("Equity trading date")
    axis.tick_params(length=0, labelsize=6.5)

    sector_counts = ordered.groupby("sector", sort=True)["ticker"].size()
    boundary = 0
    for count in sector_counts.iloc[:-1]:
        boundary += count
        axis.axhline(boundary - 0.5, color=BACKGROUND, linewidth=1.0)
    colorbar_axis = fig.add_axes([0.91, 0.22, 0.018, 0.50])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Same-day VADER compound score")
    fig.text(
        0.09,
        0.045,
        "Source: Hosted course headlines; NLTK VADER; author calculations.\n"
        f"Sample: {_format_date(dates.min())}-{_format_date(dates.max())}. "
        "Equal-weight headline aggregation within ticker-day.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.93, 0.018, "Units: score [-1, 1]", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_company_news_coverage(
    ticker_summary: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Compare the share of equity trading days with observed news by ticker."""
    _style()
    ordered = ticker_summary.sort_values(
        ["sector", "observed_news_ratio", "ticker"], ascending=[True, True, True]
    ).reset_index(drop=True)
    sectors = sorted(ordered["sector"].unique())
    colour_lookup = dict(zip(sectors, SECTOR_COLORS, strict=True))
    colors = ordered["sector"].map(colour_lookup)
    labels = ordered["ticker"] + "  " + ordered["sector"]

    fig, axis = plt.subplots(figsize=(9.2, 11.5))
    fig.subplots_adjust(left=0.23, right=0.96, top=0.84, bottom=0.09)
    fig.text(
        0.09,
        0.97,
        "FINMOSAIC  /  COMPANY NEWS COVERAGE",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.93,
        "News coverage was broad but uneven across companies",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.89,
        "Observed ticker-days divided by the complete equity trading calendar",
        color=MUTED,
        fontsize=9,
    )
    y = np.arange(len(ordered))
    axis.barh(y, ordered["observed_news_ratio"], color=colors, height=0.70)
    axis.set_yticks(y, labels=labels)
    axis.set_xlabel("Trading days with at least one headline")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axis.grid(axis="x", color=GRID, linewidth=0.6, linestyle=(0, (2, 2)))
    axis.tick_params(length=0, labelsize=6.5)
    start = _format_date(ticker_summary["start_date"].min())
    end = _format_date(ticker_summary["end_date"].max())
    fig.text(
        0.09,
        0.035,
        "Source: Hosted course headlines; author calculations. "
        f"Sample: {start}-{end}. Complete grid: 50 tickers by 1,006 trading days.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.96, 0.035, "Units: percent", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)


def plot_monthly_company_watchlist_sentiment(
    company_index: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    """Plot monthly observed-news VADER tone for a transparent eight-stock watchlist."""
    observed = company_index[company_index["news_observed"]].copy()
    sector_rank = (
        observed.groupby("sector", observed=True)["headline_count"]
        .sum()
        .sort_values(ascending=False)
    )
    selected_sectors = sector_rank.head(8).index
    coverage = (
        company_index[company_index["sector"].isin(selected_sectors)]
        .groupby(["sector", "ticker"], observed=True, as_index=False)
        .agg(observed_news_ratio=("news_coverage", "mean"))
        .sort_values(
            ["sector", "observed_news_ratio", "ticker"],
            ascending=[True, False, True],
        )
    )
    watchlist = coverage.groupby("sector", observed=True, sort=True).head(1)
    watchlist = watchlist.sort_values("sector").reset_index(drop=True)
    selected_tickers = watchlist["ticker"].tolist()

    watchlist_daily = observed[observed["ticker"].isin(selected_tickers)].copy()
    watchlist_daily["month"] = watchlist_daily["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        watchlist_daily.groupby(
            ["sector", "ticker", "month"], observed=True, as_index=False
        )["vader_company_sentiment"]
        .mean()
        .rename(columns={"vader_company_sentiment": "monthly_sentiment"})
    )
    months = pd.date_range(
        company_index["date"].min().to_period("M").to_timestamp(),
        company_index["date"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    row_index = pd.MultiIndex.from_frame(watchlist[["sector", "ticker"]])
    matrix = (
        monthly.pivot(
            index=["sector", "ticker"], columns="month", values="monthly_sentiment"
        )
        .reindex(index=row_index, columns=months)
    )

    colour_map = plt.get_cmap("RdBu").copy()
    colour_map.set_bad("#D8DDE3")
    fig, axis = plt.subplots(figsize=(9.2, 5.9))
    fig.subplots_adjust(left=0.20, right=0.89, top=0.70, bottom=0.21)
    fig.text(
        0.09,
        0.95,
        "FINMOSAIC  /  MONTHLY COMPANY SENTIMENT WATCHLIST",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.09,
        0.87,
        "Monthly news tone across eight representative equities",
        color=TEXT,
        fontsize=16,
        weight="bold",
    )
    ticker_text = ", ".join(selected_tickers)
    fig.text(
        0.09,
        0.805,
        f"Watchlist: {ticker_text}. Highest-coverage ticker from each of the eight "
        "largest sectors by headline count.",
        color=MUTED,
        fontsize=8.5,
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix.to_numpy()),
        aspect="auto",
        interpolation="nearest",
        cmap=colour_map,
        vmin=-0.35,
        vmax=0.35,
    )
    labels = [f"{row.ticker}  |  {row.sector}" for row in watchlist.itertuples()]
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    year_positions = [position for position, month in enumerate(months) if month.month == 1]
    year_labels = [months[position].strftime("%Y") for position in year_positions]
    axis.set_xticks(year_positions, labels=year_labels)
    axis.set_xlabel("Calendar month")
    axis.set_ylabel("Watchlist equity and sector")
    axis.tick_params(length=0)
    colorbar_axis = fig.add_axes([0.91, 0.28, 0.018, 0.32])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Monthly mean VADER score")
    colorbar.ax.text(
        0.5,
        -0.10,
        "Grey = no news",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    fig.text(
        0.09,
        0.070,
        "Source: results/data/company_sentiment_index.csv; hosted course headlines; "
        "NLTK VADER; author calculations.\n"
        f"Sample: {months.min():%b %Y}-{months.max():%b %Y}. "
        "Monthly means use observed ticker-days only; no-news months remain missing.",
        color=MUTED,
        fontsize=7,
    )
    fig.text(0.93, 0.035, "Units: mean VADER score", color=MUTED, fontsize=7, ha="right")
    _save(fig, path)
