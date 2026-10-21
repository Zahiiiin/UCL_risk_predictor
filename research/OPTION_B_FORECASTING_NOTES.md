# Option B: forecasting each reactor's next-year UCL (notes in plain language)

These notes explain what was built, what the results say, and what they do not say. All numbers come from `research/expected_results/forecast/FORECAST_RESULTS.md`, which is produced by `research/UCL_Forecast_Report.py` from the stored result files.

## 1. The question

The paper puts each reactor into one class (Low, Medium, High) using its 2018 to 2024 average UCL. Option B asks something closer to real use:

"At the end of year t, how much unplanned capability loss will this reactor have in year t+1?"

The answer is a number in percent, and it can also be turned into the same three classes (below 2, 2 to 6, above 6).

## 2. The table

Every row is one reactor at one year-end. The columns describe only what was known by then: its own recent UCL values, its planned and external losses, the world average and its country average, and its age and size. The last column is the UCL of the next year, the thing to predict.

The table has 17,548 rows from 603 reactors in 35 countries. It includes reactors that were later shut down, for the years in which they ran. If only today's operating reactors were used, the reactors that failed would be missing and the results would look too good.

The one rule that protects the results: no feature may use data from after the year-end. A check recomputed 400 random rows from scratch with only the allowed years and found 0 differences.

## 3. The test

The shuffled cross-validation of the paper is not right for time. Here the past is replayed instead. For each target year from 2010 to 2025, the models learn only from earlier years and then predict that year. Think of a forecaster who updates every January.

All predictions are collected and scored together: 6,196 reactor-years.

## 4. What was compared

Simple rules with no learning: repeat last year, average of the last 5 years, median of the last 5 years, country average, world average.

Three learners, all untuned with fixed settings: Ridge (a linear model), random forest and XGBoost.

Because UCL has fallen over the decades, the learners were run with three training windows (all earlier years, the latest 15, the latest 10). The window was not chosen using the test years, and all three are reported.

## 5. Results in plain words

Machine learning clearly beats "same as last year". The mean absolute error (MAE) of the random forest is about 3.6 percentage points against 4.56 for repeating last year, about 21 percent lower, and it wins in every one of the 16 test years.

The best simple rule is the median of the last 5 years (MAE 3.92). The random forest beats it by about 0.3 points (95 percent interval roughly -0.45 to -0.21 for the 15-year window) and wins in 16 of 16 years. This is a modest but steady gain.

The gain is in the size of the error, not in the three-class label. Class accuracy is about 65.9 percent for the forest and 65.1 percent for the median rule, and the difference is not distinguishable from zero (+0.7 points, interval -0.2 to +1.7).

More inputs help only a little. For the random forest, MAE goes from 3.67 (own history only) to 3.60 (adding planned losses and world and country context).

Design facts do not help. On the 387 reactors of the paper, adding reactor type and construction time leaves the random forest at 3.10 either way. This agrees with the paper.

New sites are almost as easy as known sites. New countries are harder: the forest with all context inputs goes from 3.60 to 3.78, and the linear model with context collapses to 4.05. The random forest and XGBoost with only the reactor's own history lose less (3.67 to 3.75 and 3.68 to 3.76) and still beat both repeating last year and the median rule on MAE. The linear model with own history only does not clearly beat the median rule in new countries. The reason for the linear model's collapse is not tested; one guess is that the country-size input acts like a country label.

The unusually high French values of 2022 account for only about 3 percent of the forest's total error, so the results are not driven by that year.

## 6. What the results do not show

Twelve learner configurations were compared and the best one is quoted, so its numbers are a little optimistic. The pattern is the same across models and windows, which is reassuring, but no correction for the number of comparisons was made.

Rows are scored only when the target year is a usable year (a published value, not a full planned shutdown, not the first grid year). A real forecaster would not know in advance that a reactor will run.

The sites are guessed from unit names. The 2025 values were in the October 2026 download, but I have not checked whether the IAEA treats them as final.

The model forecasts a number, not a cause. It does not say why a reactor will lose capability.

## 7. How to repeat everything

```
python3 research/UCL_Forecast_Panel.py
python3 research/UCL_Forecast_Benchmark.py --tag main_expanding
python3 research/UCL_Forecast_Benchmark.py --window 15 --tag main_window15
python3 research/UCL_Forecast_Benchmark.py --window 10 --tag main_window10
python3 research/UCL_Forecast_Benchmark.py --grouped country --window 15 --sets A C --tag grouped-country_window15
python3 research/UCL_Forecast_Benchmark.py --grouped site --window 15 --sets A C --tag grouped-site_window15
python3 research/UCL_Forecast_Benchmark.py --subset 387 --window 15 --sets D E --tag subset387_window15
python3 research/UCL_Forecast_Report.py
```

The whole set takes well under an hour on two processor cores.

## 8. Checks that were done

The panel passes the leakage check. The context features were recomputed independently. Persistence and median-of-5 errors were recomputed straight from the raw PRIS file with separate code and match exactly (6,196 rows, 4.563 and 3.917). Re-running two test years reproduces the stored predictions to four decimals.
