# Eco-efficiency analysis code

This repository contains the core machine-learning, scenario-screening, and reporting code used in the manuscript. The baseline `efficiency` labels are calculated separately in MATLAB and are read by the Python workflow as fixed inputs.

> **Current revision status (2026-07-17):** `data.xlsx` contains the corrected effective irrigated area (`EIA`), carbon emissions from agricultural activities (`CEA`), and all 720 refreshed MATLAB global-frontier Super-SBM labels. The main notebook, independent predictor-ablation script, descriptive-results script, and five-year alternative-frontier analysis have been rerun successfully. The refreshed outputs have been synchronized to the clean manuscript, marked manuscript, response letter, Online Resource 1, and Online Resource 2. The formal files are stored in `../../05_final_resubmission_package/`.

## Files

- `eco_efficiency_model_pipeline.ipynb`  
  Main notebook for model training, benchmarking, prediction, and scenario-output export.

- `eco_efficiency_results.py`
  Revised results-story script for descriptive tables, trend figures, heatmaps, inequality metrics, rank persistence, and Section 4.3 perturbation interpretation assets.

- `revision_minimal_robustness.py`
  Revision-stage predictor-ablation analysis.

- `postprocess_revision_outputs.py`
  Post-processing utility for canonical panel-validation aggregation.

- `super_sbm_rolling_frontier.py`
  Independently validated Python implementation of the non-oriented
  undesirable-output Super-SBM under VRS. It first reconstructs all 720
  MATLAB pooled-frontier labels and only then computes the centered five-year
  local-window frontier (target year +/-2, truncated at the sample endpoints).

- `Global_Un_Super_SBM_VRS_Efficiency_Gt.csv`
  Authoritative MATLAB export containing the refreshed 720 `EG_V(t)` scores keyed by province ID and year.

- `Global_Un_Super_SBM_VRS_Info.txt`
  MATLAB model-settings record for the refreshed global-frontier run.

- `requirements.txt`  
  Python dependencies required to run the workflow.

## Input data

The main workflow reads the base dataset from:

```python
./data.xlsx
```

The intended authoritative `efficiency` column consists of 720 precomputed Super-SBM labels calculated in MATLAB from the 30-province, 2000-2023 panel. The required MATLAB settings are: panel data; original model; non-oriented; undesirable output included; super-efficiency enabled; variable returns to scale (VRS); pooled global frontier; and equal weights. The model uses eight inputs (`Qx = 0.125` for each input), one desirable output (`Qy = 0.5`), and one undesirable output (`Qb = 0.5`). The independent Python solver is a validation and alternative-frontier tool; it does not overwrite the MATLAB labels.

### Irrigation-data correction record

The stable code entry point remains `data.xlsx`; no script path needs to change. On 2026-07-16, the official National Bureau of Statistics workbook was checked against the panel and the following limited corrections were made:

- `EIA`, 2005 and 2006: the two annual blocks had been transposed. The 60 province-year values were restored to their official same-year values.
- `CEA`, 2005 and 2006: no change was required because the stored carbon totals already used the correct same-year irrigation activity.
- `CEA`, 2021: 30 values were recalculated with the 2021 `EIA` activity instead of the 2020 activity. All other activity data and coefficients were held fixed.
- `efficiency`: no value was changed during the initial EIA/CEA correction. The user subsequently reran MATLAB under the documented pooled global-frontier settings, and the 720 exported `EG_V(t)` values were matched by `ID-Year` and written to `efficiency` on 2026-07-16.

Local traceability archives (retained in the revision workspace and excluded from the public Git repository):

- `data_before_eia_correction_20260716.xlsx`: exact pre-correction input archive; SHA-256 `DFBB39ED8204DCF22FFB0FBF2183C85485771C9BDCA18D2583AFCAA985B4B687`.
- `effective_irrigated_area_NBS_2000_2024.xlsx`: archived official source workbook; SHA-256 `7853A25BA83F2916721138E06DF7E32E177B11F11B095EA58789A0A37CE60F16`.
- `data_before_efficiency_refresh_20260716.xlsx`: corrected EIA/CEA input immediately before the MATLAB-label refresh; SHA-256 `236C495A86BFA2A274B2669FBC28C60EF47D5F414BE6329C1B730AC0FD54E40C`.

Public-release traceability files:

- `Global_Un_Super_SBM_VRS_Efficiency_Gt.csv`: MATLAB efficiency export; SHA-256 `587DEE4789C3706CC61AAB31E8ECD3F1E62A0A72E2EF254DD0776954C09032A5`.
- `Global_Un_Super_SBM_VRS_Info.txt`: MATLAB parameter record; SHA-256 `63A206A97554BA58DBC430E6C5E99F6FE667FB000A20843B05DA08C96C75DFE3`.
- `data.xlsx`: corrected canonical input with refreshed MATLAB labels; SHA-256 `8E85425D6624629237E8223DE29F4E9ECE47086B8FAEB97F155C046B222D489D`.

The refresh passed the following checks: 720 unique `ID-Year` keys with complete 30-province, 2000-2023 coverage; no missing or non-positive scores; no changes outside the `efficiency` column; and no workbook formula-error values. Compared with the preceding labels, 103 cells changed above `1e-12`, Pearson correlation was 0.99946, Spearman correlation was 0.99998, and the count of scores at or above one changed from 53 to 52. These are diagnostic comparisons only; all manuscript-facing results must still be regenerated.

The revised results script also requires the scenario export generated by the main notebook:

```python
./tables/Scenario_analysis_full_2023.xlsx
```

Please keep these files in the expected working directory, or revise the paths in the code if needed.

## Outputs

The workflow writes figures and tables to:

- `fig/`
- `tables/`
- `results_story/`

These directories mainly store generated outputs and are excluded from version control by `.gitignore`.

The formal revision package stores selected supplementary Excel outputs separately as Online Resource workbooks. The repository is kept focused on the code, input data, and scripts needed to reproduce those outputs.

## Recommended tracked files

The repository should normally track only the core research assets:

- `data.xlsx`
- `Global_Un_Super_SBM_VRS_Efficiency_Gt.csv`
- `Global_Un_Super_SBM_VRS_Info.txt`
- `eco_efficiency_model_pipeline.ipynb`
- `eco_efficiency_results.py`
- `revision_minimal_robustness.py`
- `postprocess_revision_outputs.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

## Suggested run order

1. Completed: recalculate all 720 `efficiency` labels in MATLAB and match the exported scores to `data.xlsx` by `ID-Year`.
2. Completed: run `eco_efficiency_model_pipeline.ipynb` from beginning to end and refresh the main modelling, scenario, ablation, and panel-validation outputs.
3. Completed: run `revision_minimal_robustness.py` and refresh `tables/Surrogate_predictor_ablation_revision.xlsx`.
4. Completed as part of the main notebook: run the panel-aware validation code and apply `postprocess_revision_outputs.py` to produce the canonical aggregation.
5. Completed: run `eco_efficiency_results.py` for the revised descriptive outputs and Section 4.3 interpretation tables and figures.
6. Completed: run `super_sbm_rolling_frontier.py` to refresh the alternative-frontier
   robustness output. The script stops before computing
   the rolling frontier unless its pooled-frontier reconstruction passes the
   documented numerical validation gate.
7. Completed: cross-check all refreshed outputs and synchronize the clean manuscript, marked manuscript, Online Resources, figures, and response letter as one controlled update.

### Latest main-notebook run

The corrected-data run completed successfully on 2026-07-16 (approximately 71 minutes). Key diagnostics were: repeated-CV XGBoost `R2 = 0.905`; random held-out XGBoost `R2 = 0.946`; 11-split rolling time-forward TSLR-MLP mean `R2 = 0.719`; and province-block XGBoost `R2 = 0.679`. Mean 2023 diagnostic changes were 0.0613 for P1, 0.0636 for P2, and 0.0990 for P3. These values have been cross-checked against the formal manuscript and Online Resources.

The predictor-ablation refresh completed on 2026-07-17. Random five-fold mean `R2` values were 0.899 for the full set, 0.895 without aggregate CEA, and 0.689 without CEA and its six activity components. Corresponding 11-split time-forward values were 0.127, 0.116, and 0.010. The evidence still supports the same bounded interpretation: aggregate CEA alone adds limited incremental approximation, while the broader production/accounting information set matters; these are predictive diagnostics, not causal mechanisms.

The descriptive-results refresh also completed on 2026-07-17. The revised 2023 mean diagnostic changes are 0.0613 for P1, 0.0636 for P2, and 0.0990 for P3; the corresponding medians are 0.0586, 0.0566, and 0.0936. All three scenarios cover 30 provinces with no missing or duplicate province-scenario rows. The mean 5-year and 10-year rank-persistence correlations are 0.9092 and 0.8290. All `results_story` tables and figures were regenerated, including the heatmap with English province names, and synchronized to the formal revision package.

The five-year alternative-frontier refresh completed on 2026-07-17. The pooled-frontier reconstruction matched the 52 MATLAB frontier observations and passed the numerical gate (`N = 720`, maximum absolute error `2.84e-5`, Spearman `1.000000`). The centered five-year window was feasible for all 720 observations. Overall pooled-versus-window Spearman was `0.567537`; mean and median within-year Spearman were `0.891157` and `0.881869`; mean and median absolute score differences were `0.294103` and `0.264279`; and the median within-year absolute rank difference was `2`. Both `tables/rolling_5yr_frontier_robustness_revision.csv` and `tables/Alternative_frontier_robustness_revision.xlsx` were refreshed. The bounded interpretation is unchanged: absolute score levels are frontier-sensitive, whereas within-year ordering is more stable.

## Reproducibility

Random seeds and model settings are fixed in the code where applicable. The corrected input now contains refreshed MATLAB labels and is ready for the machine-learning and reporting stages. The MATLAB `efficiency` column remains the baseline result. The Python Super-SBM implementation is retained as an independently checked alternative-frontier tool and must pass the pooled-label validation gate against these refreshed labels before its five-year local-window output is used. Direct DEA perturbation, CEA-exclusion, and emission-factor-sensitivity results are not generated by this script.

Windows runtime note: in the current local environment, TensorFlow failed to load when the long project path and MATLAB/Anaconda DLL paths were active. The successful run used an isolated short-path staging directory with the same `data.xlsx`, a clean process-local `PATH`, UTF-8 console output, and XGBoost imported before TensorFlow. The formal notebook code and input workbook were not modified by this compatibility workaround.
