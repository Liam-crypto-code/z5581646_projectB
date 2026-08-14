# Prompt log - Walk-forward fund backtests

## What I wanted

I wanted transparent out-of-sample funds for FinMosaic: a combined Equal Weight
benchmark, Minimum Variance and Maximum Sharpe. The rules had to be fixed before
examining out-of-sample performance and had to avoid look-ahead bias.

## Prompt(s)

> Please now begin Station 3 portfolio work only. Implement one transparent
> walk-forward out-of-sample backtest engine for the combined equity-crypto panel.
> Start with an equal-weight benchmark and one minimum-variance method ...
> Ensure weights at each rebalance use information available no later than the
> previous trading day, and that the portfolio earns returns only after weights
> are formed.

> Please add a Combined Maximum Sharpe fund as a higher-risk comparison fund ...
> Apply a 10% maximum weight per asset ... Do not tune the cap or method using
> out-of-sample performance.

> Please extend ... to add two new fund universes: Equity-only ... and
> Crypto-only ... using their correct native calendars and annualisation.

## What the assistant produced

It created a shared monthly walk-forward engine with long-only, fully invested
weights, an estimation window, next-observation effectiveness and weight drift
between rebalances. It produced ten fund histories: four combined funds, three
equity-only funds and three crypto-only funds. It saved returns, weights,
performance metrics and fact-sheet inputs.

## What was wrong or risky

Two problems were found during review. First, the initial Minimum Variance
optimiser scaling was not suitable for the full 60-asset covariance matrix.
Second, calculating returns from fixed target weights every day would have implied
daily rebalancing, even though the stated policy was monthly.

## What I changed and why

I accepted a covariance-scale adjustment and required the portfolio to reset to
target weights only at monthly rebalances, then allow holdings to drift. I required
tests for formation-date timing, non-negative weights, weights summing to one, the
10% Maximum Sharpe cap and calendar/annualisation differences. I retained the
crypto Maximum Sharpe result even though it equals Crypto Equal Weight: with ten
assets and a 10% cap, that equality is a mechanical constraint outcome rather than
a claimed optimisation success.
