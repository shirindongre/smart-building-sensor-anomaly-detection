# Dataset notes

## meta_open.csv
- Unique building identifier: `uid`. One row per building.
- 507 buildings.
- Building types include: Primary/Secondary School, University/College, Office, Commercial Real Estate,
  Corporate Office, Business Services, etc.
- `timezone` is the local timezone used for calendar features (e.g. America/New_York, Europe/London,
  America/Phoenix - Phoenix has no daylight saving).

## Buildings used in this project (src/config.py -> BUILDINGS)
| uid | timezone | data period (local) |
|---|---|---|
| Office_Annika | America/Phoenix | 2015-01-01 .. 2015-12-31 |
| Office_Cristina | America/New_York | 2015-01-01 .. 2015-12-31 |
| PrimClass_Jolie | Europe/London | 2014-12-01 .. 2015-11-30 |
| UnivDorm_Mitch (dormitory, 6,929 sqm) | America/Chicago | 2014-05-01 .. 2015-04-30 |
| UnivLab_Brenna (college laboratory, 8,103 sqm) | America/Los_Angeles | 2012-01-01 .. 2012-12-31 |

Each has about one year of hourly data, so the split policy trains on all of it (`scripts/splitting.py`).
Office_Jesus and PrimClass_Jaylin were used earlier and removed from the list (same types as buildings kept).

## How representative are the chosen buildings?
Dataset (507 buildings) by type: Office 156, Primary/Secondary Classroom 105, College Laboratory 95,
College Classroom 81, Dormitory 70. By timezone: New York 151, London 143, Phoenix 96, Chicago 75, Los Angeles 22,
Zurich 12, Singapore 5, Denver 3. Median floor area 6,193 sqm (10th-90th percentile 1,550-16,686).
Offices and schools behave differently from labs and dormitories: median weekend/weekday load ratio 0.43 (schools),
0.76 (offices), 0.92 (labs), 0.98 (dormitories); labs and dorms have a steady, almost round-the-clock load.
The five chosen buildings cover every type above except College Classroom, and five of the eight timezones
(New York, London, Phoenix, Chicago, Los Angeles). None is above about 8,600 sqm. Results should be
described as covering these building types, not "all 507 buildings".
Data quality of the whole table is good: 497 of 507 buildings have no gap longer than 6 h and no flat-lined stretch
longer than 24 h. Data periods differ by building (2012, 2014-15), which does not matter for per-building models.
Next candidates to add: UnivClass_Nicholas (college classroom, Chicago), Office_Emerald (large, New York, 44,245 sqm).

## Candidate buildings noted earlier (not used yet)
PrimClass_Uma, PrimClass_Umar, UnivDorm_Una, PrimClass_Uriah, PrimClass_Ulysses, UnivLab_Bethany,
Office_Maryann, UnivClass_Boyd, Office_Myron, Office_Mick, Office_Moises, Office_Mason, UnivLab_Margret
(earlier notes starred Uma, Maryann, Mick, Margret - meaning not recorded).
