# AGENTS.md

## Project

This is my FINS5545 FinTech Project Part B.

The product is called FinMosaic. It is a personalised financial intelligence
platform where users can build a tailored market view around selected funds,
assets, sectors and news. It helps users compare information and systematic
fund results; it does not provide personal financial advice.

Part B covers:

- Station 3: portfolio construction, out-of-sample backtesting, sentiment
  analysis and sentiment-fund fusion;
- Station 4: a deployed Streamlit app that lets users compare funds, view fact
  sheets, set an allocation and explore sentiment analytics.

## Reuse of Part A

Reuse my verified Part A data foundation from `../z5581646_projectA` where
appropriate. The Part A work includes cleaned equity, crypto and headline data,
daily returns, calendar alignment and an aligned daily headline text panel.

Do not copy raw data files into this project. Load source data only through
`src/data_access.py`. Commit only small derived outputs needed by the app under
`results/data/`.

## Folder rules

- Put data and modelling code in `src/`.
- Put runnable end-to-end scripts in `scripts/`.
- Save report tables in `results/tables/`.
- Save report figures in `results/figures/`.
- Save app-readable derived outputs in `results/data/`.
- Save the editable report and PDF in `report/`.
- Save prompt logs and AI notes in `ai/`.
- Keep `streamlit_app.py` as the app entry point at the project root.

## Backtest rules

- Use adjusted-close simple daily returns.
- Calculate returns on each asset's native calendar before any alignment.
- Construct combined portfolios on the equity trading calendar.
- Use a walk-forward out-of-sample backtest only.
- At every rebalance date, estimate weights using information available before
  that date only. Never use future returns, sentiment or prices.
- State the estimation window, rebalance frequency, constraints and start date
  of the live out-of-sample period.
- Use the correct annualisation convention: 252 trading days for equity-based
  portfolios and 365 calendar days for crypto-only portfolios.
- Report annualised return, annualised volatility, Sharpe ratio, maximum
  drawdown, growth of $1 and portfolio weights.
- Include at least two optimisation methods for the combined equity-crypto fund.
- Check that portfolio weights change across methods and rebalances.

## Sentiment and fusion rules

- Use the aligned headline text panel from Part A.
- Preserve original headline text for VADER or other sentiment scoring.
- Build a daily sector sentiment index using equal-weight ticker aggregation.
- Lag the sentiment signal by at least one equity trading day before it can
  affect a portfolio decision.
- Clearly document how days without headlines are handled.
- Compare the base fund with the sentiment-augmented fund using the same
  out-of-sample evaluation period.
- A negative fusion result is valid and must be reported honestly.

## App rules

- The deployed Streamlit app must read precomputed outputs from `results/`.
- The app must not rerun backtests or download VADER resources during use.
- Let users compare funds, inspect a fact sheet, view holdings and metrics,
  set an allocation, and explore sector sentiment.
- Keep the interface clear and informative. Do not present investment advice or
  claim that any fund is suitable for a specific individual.

## Verification and AI use

Before accepting AI-generated work, I will:

1. Read and understand the code and methodology.
2. Run the relevant tests and end-to-end script.
3. Check that all metrics are reproducible from saved outputs.
4. Check for look-ahead bias in returns, weights and sentiment features.
5. Inspect dates, rebalance timing, missing values and portfolio weights.
6. Check figures, tables and fact sheets against `PROJECT_BRIEF.md`.
7. Verify every number and citation used in the report.
8. Rewrite report analysis and economic interpretation in my own words.
9. Record important prompts, AI mistakes, corrections and my decisions in `ai/`.
