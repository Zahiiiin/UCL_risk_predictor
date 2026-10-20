# Data dictionary

File: `UCL_Dataset_v6_PRIS.xlsx` (plus the flat copy `UCL_Dataset_flat.csv`)

**What this is.** One row per nuclear power reactor (387 reactors in 31 countries). It supports the paper 'Machine Learning Classification of Unplanned Capability Loss Risk in Nuclear Power Plants: Evidence from IAEA RDS-2 Data' and the web app UCL Risk Predictor. Design columns come from IAEA RDS-2 and PRIS reactor details. The label and the history columns come from the annual performance values that the IAEA PRIS Data Browser publishes for every reactor.

**Target.** UCL_Value is the mean of the annual unplanned capability loss factor (UCL) published in PRIS for 2018 to 2024, in percent. Classes: Low below 2 percent, Medium 2 to 6 percent, High above 6 percent. Classes in this file: Low 218, Medium 91, High 78. The class limits were chosen by the authors; 0 reactors lie within 0.005 of a limit.

**Which years count.** A year is skipped when PRIS has no UCL value for it, when the reactor was completely shut down for planned reasons that year (UCF = 0 and UCL = 0), and in the year of first grid connection (a partial year). A reactor needs at least 3 usable years in 2018 to 2024; 20 reactors of the earlier 407 do not have them and are listed in the EXCLUDED sheet. UCL_Years_Used shows how many years went into each average. The PRIS_YEARLY sheet shows every year and whether it was used.

**History.** HISTORY holds the mean UCL and the mean UCF for 2010 to 2013 and for 2014 to 2017 (at least 2 usable years per block, same skipping rules). All of these years lie before the label window. 324 reactors have both UCL blocks (Model2_Subset = 1); these form the Model 2 subset.

**Which sheets the scripts read.** Only FINAL_DATASET and HISTORY. The other sheets document where the numbers come from.

**Not an input.** UCL_Value, UCL_Class and UCL_Years_Used describe the label window and are never model inputs.

**Population.** Reactors listed as operational at the end of 2024 in RDS-2 (2025 edition), minus nine that were left out (see EXCLUDED), minus the 20 that lack enough usable years. Reactors that were permanently shut down before the end of 2024 are not included, so the data describe the current fleet.

**Checks.** The PRIS annual values reproduce figures that IAEA prints: the 2013 to 2017 UCF of RDS-2 2018 Table 14 for 356 reactors (median absolute difference 0.03 points) and the 2021 to 2023 planned capability loss by reactor type and size of RDS-2 2024 Table 18 (mean absolute difference 0.39 points over six groups). tools/audit_dataset.py re-computes the label and the history columns from the raw download and checks the other columns; its last report is data/AUDIT_REPORT.txt.

**Note on RDS-2 Table 14.** Version 5 of this dataset used the Table 14 columns headed 2010-2020 (2021 edition) and 2014-2024 (2025 edition). Compared with the annual PRIS values, those columns match averages over the whole life of each reactor up to 2020 and 2024 (mean absolute difference 0.60 and 0.45 points) and not the stated periods (5.23 and 5.27 points). They were therefore not used here.

## Where the numbers come from

| Quantity | Source |
|---|---|
| UCL_Value, UCL_Years_Used, history columns, PRIS_YEARLY | IAEA PRIS Data Browser, Performance indicators report (https://pris.iaea.org), annual UCL, UCF, PUF, UUF, XUF and EAF per reactor and year; downloaded October 2026. The unfiltered download is in data/source/PRIS_annual_values_raw.csv. |
| Net_Capacity_MW, Thermal_Capacity_MW, Capacity_Category, Thermal_Efficiency_Pct | IAEA RDS-2, 2025 edition, Table 14, and PRIS reactor details |
| First_Grid_Year, Const_Start_Year, Construction_Duration, Reactor_Age, Operational_Years | IAEA RDS-2 (2018 and 2025 editions, Table 14) and PRIS reactor details; the last three are computed from the first two |
| Country, Type, NSSS_Supplier, Reactor_Model | IAEA PRIS reactor details and RDS-2 Table 14 |

## Sheets

| Sheet | Rows | What it holds |
|---|---|---|
| `README` | - | The same information as this file, inside the workbook. |
| `FINAL_DATASET` | 387 | The table the scripts read. One row per reactor, with the label. |
| `HISTORY` | 387 | Mean UCL and UCF for 2010 to 2013 and 2014 to 2017 per reactor, joined to FINAL_DATASET by Reactor_Name. |
| `PRIS_YEARLY` | 5481 | Every reactor-year from 2010 to 2024 that the numbers above are built from, with a flag for whether it was used and why not. |
| `EXCLUDED` | 29 | Reactors that are not in FINAL_DATASET and the reason. |

## `FINAL_DATASET` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Reactor name. Unique. Join key. Spelling follows earlier versions of this dataset; PRIS_Name gives the PRIS spelling. |
| `Country` | Country name as listed by PRIS, in capitals. |
| `Type` | Reactor type: PWR, BWR, PHWR, GCR, LWGR or FBR. |
| `Net_Capacity_MW` | Net electrical capacity in MW. |
| `Capacity_Category` | '<600 MWe' or '>=600 MWe'. |
| `First_Grid_Year` | Year of first grid connection. |
| `Reactor_Age` | Age in years in 2024 (2024 minus First_Grid_Year). |
| `Operational_Years` | Years since first grid connection up to 2024, at most 10. Kept from version 5 as a maturity feature; its cap of 10 is no longer the label window. |
| `UCL_Value` | Mean annual unplanned capability loss, 2018 to 2024, percent, rounded to 2 decimals. The numeric label. |
| `UCL_Class` | Low, Medium or High. This is the label the models learn. |
| `NSSS_Supplier` | Supplier of the nuclear steam supply system, as listed by PRIS. |
| `Reactor_Model` | Design model name, as listed by PRIS. |
| `Construction_Duration` | Years from the start of construction to first grid connection. |
| `Thermal_Efficiency_Pct` | Net electrical capacity divided by thermal capacity, in percent. The scripts use a constant 33.0 instead unless they are told to use this column. |
| `UCL_Years_Used` | Number of usable years in 2018 to 2024 behind UCL_Value (3 to 7). |
| `PRIS_Name` | Name of the reactor in the PRIS table. Differs from Reactor_Name for ZAPORIZHZHIA-1 to 6, KHMELNYTSKYI-1 and 2, and COMANCHE PEAK-2. |

## `HISTORY` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Join key. |
| `UCL_2010_2013, UCL_2014_2017` | Mean annual UCL in percent for the block. Blank if fewer than 2 usable years. |
| `UCF_2010_2013, UCF_2014_2017` | Mean annual unit capability factor in percent for the block. Same rules. |
| `Years_2010_2013, Years_2014_2017` | Number of usable years in the block. |
| `Model2_Subset` | 1 if both UCL blocks are available, otherwise 0. |

## `PRIS_YEARLY` columns

| Column | Meaning |
|---|---|
| `Reactor_Name, PRIS_Name, Year` | Reactor, its PRIS spelling and the calendar year. |
| `Window` | Which block the year belongs to. |
| `UCL, UCF, PUF, UUF, XUF, EAF` | PRIS annual factors in percent: unplanned capability loss, unit capability factor, planned unavailability, unplanned unavailability, external unavailability, energy availability factor. Blank where PRIS has no value. |
| `Used` | yes if the year went into the averages, otherwise no. |
| `Reason_not_used` | No UCL value published; Fully shut down (UCF = 0 and UCL = 0); or Year of first grid connection. |

## `EXCLUDED` columns

| Column | Meaning |
|---|---|
| `Reactor_Name, Country, Type, First_Grid_Year` | Identification (blank for the nine reactors that were never in the population). |
| `Usable_Years_2018_2024` | Usable years found, for reactors that were in the earlier 407. |
| `Reason` | Why the reactor is not in FINAL_DATASET. |

## `UCL_Dataset_flat.csv`

One row per labelled reactor with the design columns, the history columns and the label in one table (23 columns), for readers who prefer a plain text file. It is for reading and checking. The scripts read the workbook, not the CSV.

## Features built by the code

Supplier group, model family, age group, operational category, country encoding and the other model inputs are computed by the scripts (`ucl_core.py` and the files in `research/`) from the columns above. They are not stored here.

## Changes from the previous release

Version 6 replaces the label of version 5. Version 5 estimated unplanned loss as 100 - UCF - PCL, with PCL a planned-loss average for the reactor's type and size class, and it agreed with the real PRIS unplanned loss for only 58 percent of reactors. Version 6 uses the PRIS annual UCL directly. The UCF history columns, PCL_REFERENCE and RAW_DATA of version 5 are retired. Design columns are unchanged from the corrected version 5, which fixed grid year and construction duration of Tianwan-1 and -2, net capacity of Tarapur-3 and -4, Daya Bay-1, Dukovany-2 to 4, Ringhals-4 and Forsmark-1, and thermal capacity of Rostov-2 against the printed RDS-2 tables.

## Terms of use

The numbers come from IAEA publications and the IAEA PRIS database, which remain the property of the IAEA; please cite them together with the paper. This workbook is a derived research dataset, shared so that readers can check the paper's results. Check the IAEA terms of use before reusing the underlying data for any other purpose. The MIT licence in the repository root covers the code only.
