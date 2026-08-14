# AI Notes - Project B

## 1. Project design and boundaries

I defined FinMosaic as a personalised financial-information and fund-comparison
application for self-directed investors. It is not an investment-advice,
forecasting or automated-trading product. I decided that users should be able to
compare equity-only, crypto-only and combined funds, build a selected market view,
and use company- and sector-level news context alongside historical fund results.

I also set the main analytical boundaries before implementation. The project had
to reuse the verified Part A data foundation, calculate crypto returns on the
native calendar before alignment, use adjusted-close returns, keep no-news
observations distinct from neutral sentiment, and avoid look-ahead in every
portfolio or sentiment decision. I kept the Part B implementation separate from
raw-data processing in the deployed app: Streamlit reads saved `results/` outputs
only.

## 2. Task decomposition and instructions

I did not ask the assistant to complete the whole project in one prompt. I
divided the work into stages and gave each stage a scope and stopping rule:

1. Reuse and test the Part A cleaning, return and headline-alignment foundation.
2. Build a common walk-forward portfolio engine before adding portfolio methods.
3. Add Equal Weight, Minimum Variance and Maximum Sharpe funds with stated
   constraints and rebalance timing.
4. Build the VADER sector index, then a fixed lagged sentiment-fusion test.
5. Create required figures, fact sheets and a separate Loughran-McDonald
   robustness comparison without changing the original VADER/fusion result.
6. Build the results-only Streamlit application after output schemas were stable.
7. Prepare the report, AI workflow pack, GitHub repository and deployment only
   after the analytical outputs were complete.

My task prompts specified practical rules, including a previous-observation
formation rule, next-observation effectiveness, long-only fully invested weights,
a pre-specified 10% Maximum Sharpe asset cap, no tuning using out-of-sample
performance, and clear treatment of missing news.

## 3. Verification, questioning and correction

I treated assistant output as work to review, rather than as a result to accept
automatically. I checked saved CSV outputs, figures, sample headline alignment,
tests, the Streamlit interface and the course hand-in checker. I asked for
targeted changes when I identified a problem.

Important corrections and decisions included:

- The original Minimum Variance numerical scaling was changed because it was not
  suitable for the full covariance matrix.
- The backtest was changed so target weights are reset only at monthly
  rebalances and drift between them. Calculating every day from unchanged target
  weights would have implied daily rebalancing.
- I required the one-trading-day sentiment lag and retained no-news as missing,
  rather than converting unavailable news to a zero score.
- I requested corrections to legends and annotations that obscured data, and
  replaced a dense daily company-sentiment default with a clearer monthly
  observed-only display.
- I identified and requested repairs for the `sector_summary` and
  `portfolio_sentiment` AppData errors. The latter also required cache versioning
  after the schema changed.
- I kept the fixed sentiment-fusion result even though it did not improve the
  base fund. I did not tune the tilt after seeing that result.
- I retained Loughran-McDonald as a robustness comparison, not as a proven
  replacement for VADER.

## 4. Integration and final responsibility

I decided which outputs belonged in the main report and which belonged in the
appendix. I reviewed the product claims, financial interpretation, limitations
and recommendations, and I checked that the report links to the public GitHub
repository and deployed Streamlit app.

Before submission, I will confirm that every reported number traces to a saved
table, figure or reproducible calculation; that every external reference is a
real source I have opened; and that the final ZIP contains the report, code,
derived results and AI workflow evidence. The final interpretation and submission
remain my responsibility.
