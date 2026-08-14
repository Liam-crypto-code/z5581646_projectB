# Prompt log - Sentiment index and fixed fusion test

## What I wanted

I wanted a descriptive equity-news sentiment index with clear missing-data
treatment, followed by one fixed and transparent sentiment-fusion test. I did not
want a tuned signal, financial advice or a second backtest designed after viewing
results.

## Prompt(s)

> Please now implement the standalone Station 3 sentiment-index component only.
> Use ... VADER to score the original headline text. Aggregate scores first to
> ticker-day level, then build a daily sector sentiment index by equal-weighting
> ticker-day scores within each sector ... Apply a one-equity-trading-day lag ...

> I confirm the sentiment methodology decisions: score each original headline ...
> keep no-news observations missing ... use only the one-trading-day-lagged signal
> ... use standard VADER without a finance-specific lexicon extension.

> Please now implement the sentiment-fusion component only. Use Combined Equal
> Weight as the base fund ... keep the cryptocurrency sleeve unchanged ... apply a
> fixed and pre-specified sector sentiment tilt ...

## What the assistant produced

It created the VADER headline, ticker-day and sector-day pipeline with coverage
fields, one-trading-day lagged signals, supporting tables and figures. It then
created a Sentiment-Tilted Combined Equal Weight fund using a fixed 0.50 sector
tilt within the equity sleeve, leaving crypto targets unchanged.

## What was wrong or risky

No-news days could easily be mistaken for neutral scores, and weekend/Monday news
could be used too early. There was also a temptation to choose the tilt strength
after inspecting performance. The resulting fusion did not improve the base fund:
it reduced annualised return and Sharpe slightly, despite a small drawdown
improvement.

## What I changed and why

I explicitly chose observed-only aggregation and missing coverage rather than
imputing neutral sentiment. I fixed the tilt strength before examining the
out-of-sample result and retained the negative result in the outputs and report.
I required identical dates for base and fused funds, long-only fully-invested
targets, unchanged crypto exposure and lag-safety tests.
