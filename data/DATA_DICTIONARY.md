# Data dictionary

File: `UCL_Dataset_v5_longitudinal.xlsx`

The workbook holds one row per nuclear power reactor. It was compiled by hand from
IAEA publications and the IAEA PRIS database. It is the only data file the scripts need.

## Where the numbers come from

| Quantity | Source |
|---|---|
| Unit capability factor, UCF (2014 to 2024 average) | IAEA, Nuclear Power Reactors in the World (Reference Data Series No. 2, RDS-2), 2025 edition, Table 14 |
| UCF average 2013 to 2017 | RDS-2, 2018 edition, Table 14 |
| UCF average 2010 to 2020 | RDS-2, 2021 edition, Table 14 |
| Planned capability loss factor, PCL (reference values by reactor type and size) | RDS-2, 2018, 2021 and 2024 editions, Table 18 |
| Reactor name, country, type, capacity, grid dates, supplier, model, construction dates, thermal capacity | IAEA PRIS (https://pris.iaea.org) |

The target is computed per reactor as **UCL = 100 - UCF - PCL**, with negative results set to 0.
Classes: **Low** below 2 %, **Medium** from 2 % to 6 %, **High** above 6 %.
The full derivation is in Section 3 of the paper.

## Sheets

| Sheet | Rows | What it holds |
|---|---|---|
| `FINAL_DATASET` | 407 reactors (plus a few note rows below) | The cleaned table the scripts read. |
| `UCL_LONGITUDAL` | 407 | The three UCF averages per reactor (the sheet name keeps its original spelling). |
| `RAW_DATA` | 413 | The hand-entered source table, including thermal capacity. Not read by the scripts. |
| `PCL_REFERENCE` | 10 | The PCL reference values by reactor type and size class. |

## `FINAL_DATASET` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Reactor name as listed by PRIS. Used to join the two sheets. |
| `Country` | Country name as listed by PRIS. The scripts trim spaces and use capitals, which gives 31 countries. |
| `Type` | PWR, BWR, PHWR, GCR, LWGR or FBR. |
| `Net_Capacity_MW` | Net electrical capacity in MW. |
| `Capacity_Category` | `<600 MWe` or `>=600 MWe`. |
| `First_Grid_Year` | Year of first grid connection. |
| `Reactor_Age` | Age in years in 2024. |
| `Operational_Years` | Years of operation inside the 2014 to 2024 window, at most 10. |
| `UCL_Value` | Unplanned capability loss in percent (the target before it is cut into classes). |
| `UCL_Class` | Low, Medium or High. This is the label the models learn. |
| `NSSS_Supplier` | Supplier of the nuclear steam supply system, as listed by PRIS. |
| `Reactor_Model` | Design model name, as listed by PRIS. |
| `Construction_Duration` | Years from the start of construction to first grid connection. |
| `Thermal_Efficiency_Pct` | Net electrical capacity divided by thermal capacity, in percent. The scripts do not use it unless asked (see the README, "Known limitations"). |
| `NSSS_Group`, `Model_Family`, `Age_Group`, `Op_Category`, `Reactor_Age_Sq` | Helper columns kept for convenience. The scripts rebuild these from the columns above and ignore the stored copies. |
| `UCL_Reliable` | A flag kept from data construction (1 for 216 reactors, 0 for 191). The scripts do not use it. |

## `UCL_LONGITUDAL` columns

| Column | Meaning |
|---|---|
| `Reactor_Name` | Join key. |
| `UCF_2013_2017` | Average unit capability factor, 2013 to 2017 (history period 1). Used by Model 2. |
| `UCF_2010_2020` | Average unit capability factor, 2010 to 2020 (history period 2). Used by Model 2. |
| `UCF_2014_2024` | Average unit capability factor, 2014 to 2024. **This defines the target, so it is never used as an input.** The scripts do not read it. |

## Reactors and exclusions

407 reactors are labelled. Nine further reactors were left out: Shidao Bay-1 (HTGR),
Akademik Lomonosov-1 and -2 (floating units), Bilibino-2, -3 and -4 (very small Arctic
units), and Zhangzhou-1, Flamanville-3 and Mochovce-3 (UCF reported as zero because only
first-year commissioning data existed). Section 3 of the paper gives the reasons. Model 2 needs both
history values, which exist for 370 of the 407 reactors. The other 37 were commissioned
after 2017 and are not in the 2018 edition.

## Terms of use

The numbers come from IAEA publications and the IAEA PRIS database, which remain the
property of the IAEA. This workbook is a derived research dataset, shared so that readers
can check the paper's results. Please cite the IAEA sources above together with the paper,
and check the IAEA terms of use before reusing the underlying data for any other purpose.
The MIT licence in the repository root covers the code only.
