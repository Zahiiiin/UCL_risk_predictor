# Data dictionary

File: `UCL_Dataset_v5_longitudinal.xlsx` (plus the flat copy `UCL_Dataset_flat.csv`)

**What this is.** One row per nuclear power reactor (407 reactors in 31 countries). It supports the paper 'Machine Learning Classification of Unplanned Capability Loss Risk in Nuclear Power Plants: Evidence from IAEA RDS-2 Data' and the web app UCL Risk Predictor. It was compiled by hand from IAEA publications and the IAEA PRIS database.

**Target.** UCL = 100 - UCF - PCL, where UCF is the unit capability factor (2014 to 2024 average) and PCL is a planned capability loss reference value for the reactor's type and size class. A negative result is set to 0. Classes: Low below 2 percent, Medium 2 to 6 percent, High above 6 percent. Classes: Low 231, Medium 58, High 118.

**Zero values.** UCL_Value is 0 for 191 reactors: 190 where 100 - UCF - PCL is negative and 1 (Tarapur-4) where it is exactly 0. All of them are in the Low class. The number before negative values are set to 0 is kept in the UCL_Computed column of RAW_DATA and of the flat CSV.

**Which sheets the scripts read.** Only FINAL_DATASET and UCL_LONGITUDAL. The other sheets document where the numbers come from.

**Model 2 subset.** Model 2 needs both history values (UCF_2013_2017 and UCF_2010_2020). They exist for 368 of the 407 reactors. The rest were commissioned after 2017 or lack an earlier value.

**Not an input.** UCF_2014_2024 defines the target. It is stored for traceability and is never used as a model input.

## Where the numbers come from

| Quantity | Source |
|---|---|
| UCF 2014 to 2024 | IAEA, Nuclear Power Reactors in the World (Reference Data Series No. 2, RDS-2), 2025 edition, Table 14 |
| UCF 2013 to 2017 | RDS-2, 2018 edition, Table 14 |
| UCF 2010 to 2020 | RDS-2, 2021 edition, Table 14 |
| PCL reference values | RDS-2, 2018, 2021 and 2024 editions, Table 18 (by reactor type and size class) |
| Reactor name, country, type, capacity, grid year, supplier, model, construction start, thermal capacity | IAEA PRIS, https://pris.iaea.org |

## Sheets

| Sheet | Rows | What it holds |
|---|---|---|
| `README` | - | The same information as this file, inside the workbook. |
| `FINAL_DATASET` | 407 | The table the scripts read. One row per reactor. |
| `UCL_LONGITUDAL` | 407 | Three UCF averages per reactor, joined to FINAL_DATASET by Reactor_Name. The sheet name keeps its original spelling because the scripts look for it. |
| `PCL_REFERENCE` | 10 | The planned capability loss values by reactor type and size class. |
| `RAW_DATA` | 407 | The hand-entered source table with the UCL arithmetic shown. Values only. Not read by the scripts. |

## `FINAL_DATASET` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Reactor name as listed by PRIS. Unique. Join key. |
| `Country` | Country name as listed by PRIS, in capitals. 31 countries. |
| `Type` | Reactor type: PWR, BWR, PHWR, GCR, LWGR or FBR. |
| `Net_Capacity_MW` | Net electrical capacity in MW. |
| `Capacity_Category` | '<600 MWe' or '>=600 MWe'. |
| `First_Grid_Year` | Year of first grid connection. |
| `Reactor_Age` | Age in years in 2024 (2024 minus First_Grid_Year). |
| `Operational_Years` | Years of operation inside the 2014 to 2024 window, at most 10. |
| `UCL_Value` | Unplanned capability loss in percent, after setting negative values to 0. |
| `UCL_Class` | Low, Medium or High. This is the label the models learn. |
| `NSSS_Supplier` | Supplier of the nuclear steam supply system, as listed by PRIS. |
| `Reactor_Model` | Design model name, as listed by PRIS. |
| `Construction_Duration` | Years from the start of construction to first grid connection. |
| `Thermal_Efficiency_Pct` | Net electrical capacity divided by thermal capacity, in percent. The scripts use a constant 33.0 instead unless they are told to use this column. |

## `UCL_LONGITUDAL` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Join key. |
| `UCF_2013_2017` | Average unit capability factor, percent (history period 1). Blank if the reactor has no value in that edition. The note under Table 18 of that edition says 2017 data were not available for eight Indian units (Kaiga 1 to 4, Madras 1 and 2, Tarapur 3 and 4). Used by Model 2. |
| `UCF_2010_2020` | Average unit capability factor, percent (history period 2). Blank if not available, and also blank for a unit that had not yet been in commercial operation (the 2021 edition prints 0 for such units). Zeros that remain are real: the Japanese units that were shut down throughout 2013 to 2017. Used by Model 2. |
| `UCF_2014_2024` | Average unit capability factor, percent. Defines the target, so it is never a model input. |

## `PCL_REFERENCE` columns

| Column | Meaning |
|---|---|
| `Type` | Reactor type. |
| `Capacity_Category` | '<600 MWe', '>=600 MWe', or 'ALL' (GCR and FBR have one row for both size classes). |
| `PCL_RDS2_2018_ed` | Planned capability loss factor, percent, from Table 18 of the 2018 edition. |
| `PCL_RDS2_2021_ed` | Same, from Table 18 of the 2021 edition. |
| `PCL_RDS2_2024_ed` | Same, from Table 18 of the 2024 edition. Blank where no unit of that group operated. |
| `Editions_Averaged` | How many of the three values were available and averaged. |
| `PCL_Average` | Mean of the available values. This is the PCL subtracted in the UCL formula. |

## `RAW_DATA` columns

| Column | Meaning |
|---|---|
| `Reactor_Name ... Reactor_Model` | Same meaning as in FINAL_DATASET. |
| `UCF_2014_2024` | Unit capability factor, percent (RDS-2, 2025 edition, Table 14). |
| `PCL_Average` | PCL for the reactor's type and size class (see PCL_REFERENCE). |
| `UCL_Computed` | 100 - UCF_2014_2024 - PCL_Average, before negative values are set to 0. |
| `Book_Year` | RDS-2 edition year the UCF value was read from. |
| `Const_Start_Year` | Year construction started. |
| `Thermal_Capacity_MW` | Thermal capacity in MW. Used only to compute Thermal_Efficiency_Pct. |

## `UCL_Dataset_flat.csv`

One row per reactor with every column above in one table (21 columns), for readers who prefer a plain text file. It is for reading and checking. The scripts read the workbook, not the CSV.

## Features built by the code

Supplier group, model family, age group, operational category, country encoding and the other model inputs are computed by the scripts (`ucl_core.py` and the files in `research/`) from the columns above. They are not stored here.

## Reactors and exclusions

Nine further reactors were left out: Shidao Bay-1 (HTGR), Akademik Lomonosov-1 and -2 (floating units), Bilibino-2, -3 and -4 (very small Arctic units), and Zhangzhou-1, Flamanville-3 and Mochovce-3 (UCF reported as zero because only first-year commissioning data existed). Section 3 of the paper gives the reasons.

## Corrections in this version

The first release of this dataset was checked value by value against the printed RDS-2 tables (2018, 2021, 2024 and 2025 editions). 24 reactors had values that did not match and were corrected as listed below. Thermal_Efficiency_Pct was recomputed for the nine reactors whose capacity changed. Nothing else differs from the first release. The target (UCL_Value and UCL_Class) did not change for any reactor. Models trained on the first release give slightly different numbers, so use the results in the repository (research/expected_results) that belong to this version.

| Reactor | Change |
|---|---|
| TIANWAN-1 | First_Grid_Year 2018 to 2006; Construction_Duration 19 to 7; Reactor_Age 6 to 18; Operational_Years 6 to 10. Grid connection 2006-5 (RDS-2 2018 ed., Table 14). |
| TIANWAN-2 | First_Grid_Year 2019 to 2007; Construction_Duration 19 to 7; Reactor_Age 5 to 17; Operational_Years 5 to 10. Grid connection 2007-5 (RDS-2 2018 ed., Table 14). |
| SALEM-1 | UCF_2010_2020 72.2 to 73.2. RDS-2 2021 ed., Table 14. |
| SAEUL-1 | UCF_2010_2020 82.6 to 76.5. RDS-2 2021 ed., Table 14 (listed as SHIN-KORI-3). |
| SAEUL-2 | UCF_2010_2020 blank to 82.6. RDS-2 2021 ed., Table 14 (listed as SHIN-KORI-4). |
| TRICASTIN-2 | UCF_2013_2017 75.5 to 77.5. RDS-2 2018 ed., Table 14. |
| MADRAS-2 | UCF_2013_2017 92.3 to 77.9. RDS-2 2018 ed., Table 14 (92.3 is the MADRAS-1 value). |
| QINSHAN 2-2 | UCF_2013_2017 blank to 88.3. RDS-2 2018 ed., Table 14. |
| SANMEN-1 | UCF_2013_2017 93.1 to blank. No value printed in the 2018 edition; the old number was the 2014-2024 value. |
| HAIYANG-1 | UCF_2013_2017 92.6 to blank. No value printed in the 2018 edition; the old number was the 2014-2024 value. |
| PALO VERDE-3 | UCF_2010_2020 86.5 to blank. This unit is not printed in Table 14 of the 2021 edition, so the old number has no source. |
| BELARUSIAN-1 | UCF_2010_2020 0 to blank. The 2021 edition prints 0 for a unit without a full year of operation. |
| LENINGRAD 2-2 | UCF_2010_2020 0 to blank. The 2021 edition prints 0 for a unit without a full year of operation. |
| BARAKAH-1 | UCF_2010_2020 0 to blank. The 2021 edition prints 0 for a unit without a full year of operation. |
| TAISHAN-2 | UCF_2010_2020 0 to blank. The 2021 edition prints 0 for a unit not yet in commercial operation. |
| TARAPUR-3 | Net_Capacity_MW 482 to 490. RDS-2 2025 ed., Table 14. |
| TARAPUR-4 | Net_Capacity_MW 378 to 490. RDS-2 2025 ed., Table 14. |
| DAYA BAY-1 | Net_Capacity_MW 986 to 944. RDS-2 2025 ed., Table 14. |
| DUKOVANY-2 | Net_Capacity_MW 479 to 482. RDS-2 2025 ed., Table 14. |
| DUKOVANY-3 | Net_Capacity_MW 479 to 475. RDS-2 2025 ed., Table 14. |
| DUKOVANY-4 | Net_Capacity_MW 479 to 471. RDS-2 2025 ed., Table 14. |
| RINGHALS-4 | Net_Capacity_MW 1134 to 1130. RDS-2 2025 ed., Table 14. |
| FORSMARK-1 | Net_Capacity_MW 1103 to 1104. RDS-2 2025 ed., Table 14. |
| ROSTOV-2 | Thermal_Capacity_MW 3000 to 3200. RDS-2 2025 ed., Table 14. |

## Terms of use

The numbers come from IAEA publications and the IAEA PRIS database, which remain the property of the IAEA. This workbook is a derived research dataset, shared so that readers can check the paper's results. Please cite the IAEA sources above together with the paper, and check the IAEA terms of use before reusing the underlying data for any other purpose. The MIT licence in the repository root covers the code only.
