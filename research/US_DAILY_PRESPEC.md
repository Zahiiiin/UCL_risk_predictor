# US sub-study with NRC daily power data: what was decided BEFORE running it (21 November 2026)

This note is written before the forecasting test is run. The results are reported whatever they show.

## What was already done before this note (data checks only)
- 21 yearly NRC Power Reactor Status files (2005 to 2025) were read: 764,898 daily values, 107 PRIS units
  after name matching (Palisades, Vogtle 3 and Vogtle 4 are not in the Part II panel).
- Check against PRIS: the yearly loss computed from the daily power (100 minus mean daily power) agrees with
  PRIS (100 minus EAF) with correlation 0.94 and median absolute difference 0.9 points (2,073 unit-years).
- Same-year correlations of four descriptors with PRIS UCL and PUF were looked at (0.11 to 0.57). This is
  disclosed because it was seen before the design below was fixed. No forecasting result was seen.

## Question
Do descriptors of a reactor's past DAILY power history improve the one-year-ahead UCL forecast beyond the
annual PRIS inputs used in Part II? (This is the "event-level data" direction recommended by the paper.)

## Daily descriptors (from US_NRC_Daily.py, per unit and calendar year)
Shutdown = run of consecutive days at 0% power. Short shutdown = 14 days or fewer (most refuelling outages
are longer, so short shutdowns are more often unplanned; this is an assumption, not a label).
Reduction = run of days with power above 0 and below 90%.

Inputs added to set C (the DAILY set), each computed over the usable daily years among t-2..t (mean) and,
where marked, for year t alone:
- number of short shutdowns (mean of 3 years; and year t)
- energy loss in short shutdowns, % of year (mean of 3 years; and year t)
- energy loss at partial power, % of year (mean of 3 years; and year t)
- number of power reductions (mean of 3 years)
- days in long shutdowns (mean of 3 years)
- longest shutdown in days (year t)

## Rows and design
- Part II panel rows of US reactors whose origin year t is 2007 or later (so that t-2 is at least 2005).
- Rolling origin on US rows only: test target years 2011 to 2025; training = all US rows with an
  earlier target year (from 2008). Same RF settings as Part II.
- Compared methods on exactly the same rows:
  1. RF, set C, trained on US rows (reference model)
  2. RF, set C + daily descriptors, trained on US rows (the test)
  3. Median of last 5 years (rule)
  4. The stored Part II global forecast (RF set C, 15-year window, all countries) for these rows
  5. Ridge, set C and set C + daily (secondary, linear check)
- Secondary: probability of UCL above 6% with an RF classifier (same settings as analysis A), set C against
  set C + daily, Brier score.

## Scores and reading rule
MAE (main), class accuracy, Brier score for the secondary event. 95% intervals of differences from 1000
resamples of whole reactors (seed 42). The daily descriptors are said to help only if the interval of the
MAE difference (model 2 minus model 1) lies entirely below zero. Permutation importance of the daily
group is reported for description only.

## Not decided in advance
Nothing else.
