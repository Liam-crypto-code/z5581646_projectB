# Prompt log - Robustness comparison and report exhibits

## What I wanted

I wanted every required Part B exhibit to exist and be readable in a report. I
also wanted to test whether a finance-specific dictionary changed the headline
sentiment picture, without replacing the existing VADER index or tuning a new fund.

## Prompt(s)

> Please audit the current Part B outputs against the required exhibits and output
> files in PROJECT_BRIEF.md, Section 5. Do not modify any files yet.

> Please implement the missing Part B presentation outputs only ... a Growth of
> $1 figure, drawdown comparison, weights-over-time figure, Sharpe figure and
> report-ready fact sheets ...

> Before extending the Streamlit app, implement a financial-dictionary robustness
> comparison using the Loughran-McDonald dictionary alongside the existing VADER
> baseline. Do not replace or modify the existing VADER sentiment index, fusion
> rule, or portfolio outputs.

## What the assistant produced

It generated self-contained fund comparison figures, fact sheets, presentation
tables and captions from saved CSV artifacts. It also built a separate historical
Loughran-McDonald comparison with company and sector outputs, coverage diagnostics,
score distributions and disagreement examples.

## What was wrong or risky

The first exhibit audit found that data existed but a cross-method Growth of $1
figure, a weights-over-time figure and a Sharpe/return-risk figure still needed to
be rendered. Some early legends overlapped plotted lines; this could make a correct
figure unreadable. A financial dictionary could not be used in the backtest unless
its historical version and provenance were verified.

## What I changed and why

I asked for the missing outputs to be produced from existing saved data only, so
the presentation work could not change any portfolio result. I requested specific
legend fixes after visually checking the figures. I kept Loughran-McDonald as a
read-only robustness comparison using a version available before the 2021-2023
out-of-sample period; it did not replace VADER or create another tuned fusion fund.
