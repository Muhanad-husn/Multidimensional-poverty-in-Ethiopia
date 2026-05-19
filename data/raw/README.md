# `data/raw/` — DHS Ethiopia 2019 microdata

**These files are NOT committed to git.** The DHS license forbids redistribution of
raw microdata. Only this README is tracked (see the `.gitignore` exception). Anyone
reproducing the analysis must obtain the files themselves via the workflow below.

## Required files

Each DHS recode downloads as a ZIP that unpacks into its own folder. The analysis
reads the Stata `.DTA` file inside each:

| Folder | File | Recode | Used for |
|--------|------|--------|----------|
| `ETPR81DT/` | `ETPR81FL.DTA` | PR — Household Member | Most OPHI indicators (schooling, attendance, members) |
| `ETHR81DT/` | `ETHR81FL.DTA` | HR — Household | Living-standards indicators (water, sanitation, fuel, electricity, housing, assets) |
| `ETIR81DT/` | `ETIR81FL.DTA` | IR — Individual (women) | Woman's nutrition (BMI < 18.5) |
| `ETBR81DT/` | `ETBR81FL.DTA` | BR — Births | Child-mortality indicator |
| `ETKR81DT/` | `ETKR81FL.DTA` | KR — Children's | Child nutrition (stunting / underweight, under-5 anthropometry) |

`ETFW81DT/` (Fieldworker recode) may also be present from the download bundle; it is
**not used** by the MPI methodology.

The `81` in each filename is the DHS version code for the Ethiopia 2019 wave; the
recode letters (`PR`/`HR`/`IR`/`BR`/`KR`) and the `.DTA` (Stata) format are what matter.

## Access workflow

1. Register a research project at <https://dhsprogram.com/data/> (free; requires a
   short description of the intended use).
2. Request access to the **Ethiopia 2019 Standard/Mini DHS** dataset.
3. Once approved, download the PR, HR, IR, BR, and KR recodes in **Stata (`.DTA`)
   format**.
4. Unzip each archive and place the extracted folders directly under `data/raw/`.
5. Record the access date below.

## Provenance

- Source: DHS Program — Ethiopia Mini Demographic and Health Survey 2019
- Survey design: stratified, multistage cluster sample
  (weights `v005`/1e6 · strata `v022`/`v023` · clusters/PSUs `v021`)
- Access date: _fill in — YYYY-MM-DD_
