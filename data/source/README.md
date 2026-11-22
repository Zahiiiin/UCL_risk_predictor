# Raw PRIS download

`PRIS_annual_values_raw.csv` is the annual performance table of the IAEA Power Reactor Information System (PRIS), as downloaded in October 2026 from the PRIS Data Browser (https://pris.iaea.org, report "Performance indicators", all years, one country at a time). Nothing was changed except the column order. It holds 20,600 rows, one per reactor and year (1969 to 2025), for 649 units in 35 countries.

| Column | Meaning |
|---|---|
| `Country`, `Unit`, `Year` | Country, reactor name as PRIS spells it, calendar year. |
| `UCL` | Unplanned capability loss factor, percent. |
| `UCF` | Unit capability factor, percent. |
| `PUF`, `UUF`, `XUF` | Planned, unplanned and external unavailability factors, percent. |
| `EAF`, `FLR`, `LF`, `OF` | Energy availability factor, forced loss rate, load factor, operation factor, percent. |
| `MW`, `MWh` | Reference unit power in MW and electricity supplied in MWh. |

Empty cells mean PRIS publishes no value. In every row where values exist, UCF + PUF + UUF = 100 and UCL = UUF; `tools/audit_dataset.py` checks both.

The dataset in `data/` is built from this file by `tools/build_dataset_v6.py` and checked by `tools/audit_dataset.py`. PRIS data are the property of the IAEA; cite them whenever you use this file, and check the IAEA terms of use before reusing it for any other purpose.
