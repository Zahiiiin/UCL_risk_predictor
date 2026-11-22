# Two added analyses: what was decided BEFORE running them (21 November 2026)

This note is written before either analysis is run. The results are reported whatever they show.

## A. Probability of a bad year

**Question.** Can the probability that a reactor has a bad next year be forecast better than simple rules?

**Events.** E6: next-year UCL above 6% (the High limit of the paper). E20: next-year UCL above 20%.

**Rows and design.** The same forecasting panel, rows and rolling origin as Part II of the paper (target years
2010 to 2025, 6,196 scored rows). Primary training window: latest 15 years. Secondary: all earlier years.
Input sets A and C.

**Learners (fixed settings, not tuned).**
- Logistic regression: median imputation, standardisation, L2 penalty, C = 1.
- Random forest classifier: 300 trees, minimum leaf size 5, 50% of inputs per split, no class weights.
- XGBoost classifier: the settings of the Part II regressor (300 trees, learning rate 0.05, depth 4,
  subsample 0.8, column sample 0.8, minimum child weight 5), logistic objective.

**Rules (no learning).**
- Climatology: the event rate among the training rows.
- Own frequency: share of the reactor's usable years among the last five above the limit, shrunk towards
  climatology with one pseudo-year: (k + p_clim) / (n + 1).
- Calibrated median rule: a one-input logistic regression on log(1 + median of the last five years),
  fitted on the training rows. This turns the strongest point-forecast rule into a probability.

**Scores.** Brier score (main), Brier skill score against climatology, ROC AUC, log loss, a ten-bin
reliability diagram and the expected calibration error. Decision view: the event rate among the 10% of
reactor-years with the highest forecast probability.

**Uncertainty and reading rule.** Differences in Brier score between each learner and the calibrated median
rule and the own-frequency rule, with 95% intervals from 1000 resamples of whole reactors (seed 42).
A learner is called better only if the interval of the Brier difference lies below zero.

## B. Mixed-effects statistical benchmark

**Question.** Does machine learning beat a standard statistical reliability model?

**Model.** Linear mixed model for log(1 + next-year UCL) with fixed effects log(1 + median of the last five
years), log(1 + last usable value), log(1 + mean of the last ten years), share of the last five years above 6%,
number of usable years in the last five, and log(1 + world mean UCL at t); a random intercept for each country
and a random intercept for each reactor within its country. Fitted by restricted maximum likelihood
(statsmodels MixedLM) on the training rows of each target year, latest 15 years.

**Forecast.** Fixed part plus the estimated (best linear unbiased) country and reactor effects where the
country or reactor appears in the training rows; zero otherwise. Back-transformed and limited to 0 to 100%.

**Scores and reading rule.** As in Part II: MAE, class accuracy, intervals of differences from 1000 resamples
of whole reactors. Compared with the five-year median and the random forest with set C.

## Not decided in advance
Nothing else. If a model fails to converge in a year, that is reported, and the fixed-effects prediction is used
for that year.
