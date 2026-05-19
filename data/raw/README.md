# `data/raw/` — DHS Ethiopia 2019 microdata

**These files are NOT committed to git.** The DHS license forbids redistribution of
raw microdata. Only this README is tracked (see the `.gitignore` exception). Anyone
reproducing the analysis must obtain the files themselves via the workflow below.

## Required files

Each DHS recode downloads as a ZIP that unpacks into its own folder. The analysis
reads the Stata `.DTA` file inside each:

| Folder | File | Recode | Used by the MPI engine for |
|--------|------|--------|----------------------------|
| `ETHR81DT/` | `ETHR81FL.DTA` | HR — Household | Living-standards indicators (water, sanitation, fuel, electricity, housing, assets) + design columns (`hv001`/`hv021`/`hv022`/`hv005`/`hv024`) + ML features |
| `ETPR81DT/` | `ETPR81FL.DTA` | PR — Household Member | Schooling, school attendance, **child anthropometry (`hc70`/`hc72`)**, **woman BMI (`ha40`)** — both nutrition sources live on PR per Decision #3; PR aggregates also feed the ML feature matrix |
| `ETBR81DT/` | `ETBR81FL.DTA` | BR — Births | Child-mortality indicator (classic OPHI rule: `b5 == 0`) |
| `ETIR81DT/` | `ETIR81FL.DTA` | IR — Individual (women) | **Not used** by the MPI engine — nutrition is sourced from PR (`ha40`) instead. Listed because the DHS bundle ships it; safe to omit if disk space matters |
| `ETKR81DT/` | `ETKR81FL.DTA` | KR — Children's | **Not used** by the MPI engine — child nutrition is sourced from PR (`hc70`/`hc72`) instead. Listed for completeness; safe to omit |

`ETFW81DT/` (Fieldworker recode) may also be present from the download bundle; it is
**not used** by the MPI methodology.

**Minimal set to run the analysis:** `HR`, `PR`, `BR`. `IR` and `KR` are required
DHS downloads in the original plan but the engine does not load them — see
implementation plan Decision #3 (Session 3).

The `81` in each filename is the DHS version code for the Ethiopia 2019 wave; the
recode letters (`HR`/`PR`/`BR`/`IR`/`KR`) and the `.DTA` (Stata) format are what matter.

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
