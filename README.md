# Carbon-constrained agricultural eco-efficiency in China

This repository contains the data, code, and computational records used in the study **“Carbon-constrained agricultural eco-efficiency indicators for monitoring and directional screening in China.”**

The analysis covers 30 mainland Chinese provinces from 2000 to 2023. It combines a non-oriented undesirable-output Super-SBM model with machine-learning surrogate models to support efficiency monitoring, validation-aware model comparison, and standardized directional screening.

## Analytical scope

The repository supports four connected tasks:

1. compute pooled global-frontier Super-SBM scores and use them as the efficiency labels;
2. compare machine-learning models for approximating those labels;
3. evaluate model behavior under random, rolling time-forward, province-block, and predictor-ablation designs;
4. examine standardized input-output perturbations and alternative-frontier sensitivity.

The machine-learning layer is a surrogate for DEA-generated labels. Its outputs are intended for monitoring, diagnostic comparison, and directional screening. They are not causal estimates, calibrated policy simulations, or evidence of policy effectiveness.

## Repository contents

| Path | Purpose |
|---|---|
| `data.xlsx` | Canonical 30-province, 2000-2023 analysis panel, including the pooled global-frontier Super-SBM score in `efficiency`. |
| `eco_efficiency_model_pipeline.ipynb` | Main workflow for model comparison, validation, surrogate construction, and directional-screening outputs. |
| `predictor_ablation_analysis.py` | Predictor-group ablation under random and rolling time-forward evaluation. |
| `panel_validation_postprocess.py` | Canonical aggregation of panel-aware validation outputs. It is called by the main notebook. |
| `eco_efficiency_results.py` | Descriptive indicators, figures, inequality and rank-persistence summaries, and province-level screening profiles. |
| `super_sbm_rolling_frontier.py` | Python implementation of the pooled global common frontier and centered five-year local-window frontier, including CSV and Excel summary exports. |
| `requirements.txt` | Python dependencies. |

Generated figures and workbooks are written to `fig/`, `tables/`, and `results_story/`. These directories are excluded from version control because they can be regenerated from the tracked inputs and code. Frozen copies of the detailed workbooks used in the article are supplied separately as Online Resource 2; they are not mirrored in the GitHub repository.

The principal generated workbooks include:

| Output | Contents |
|---|---|
| `tables/cv_summary.xlsx` | Repeated cross-validation summaries. |
| `tables/test_results.xlsx` | Held-out random-test performance. |
| `tables/Panel_dependence_robustness_checks.xlsx` | Random, rolling time-forward, and province-block validation details. |
| `tables/Panel_dependence_robustness_checks_reaggregated.xlsx` | Canonical model-name aggregation across panel-validation outputs. |
| `tables/Surrogate_predictor_ablation_revision.xlsx` | Predictor-group ablation summaries and split-level results. |
| `tables/Scenario_analysis_full_2023.xlsx` | Province-level outputs for the three standardized 2023 perturbations. |
| `tables/Alternative_frontier_robustness_revision.xlsx` | Pooled global-frontier and five-year local-window sensitivity results. |

## Data structure

`data.xlsx` contains 720 province-year observations. The principal analytical fields are:

| Field | Definition | Unit / coding |
|---|---|---|
| `ID` | Province identifier | 1-30 |
| `Year` | Observation year | 2000-2023 |
| `TPAM` | Total power of agricultural machinery | 10,000 kW |
| `EIA` | Effective irrigated area | 1,000 ha |
| `CS` | Crops sown | 1,000 ha |
| `AFA` | Agricultural fertilizer application | 10,000 tonnes |
| `PU` | Pesticide use | 10,000 tonnes |
| `ADY` | Agricultural diesel use | 10,000 tonnes |
| `PFU` | Plastic film use | 10,000 tonnes |
| `NRP` | Primary-industry employment; legacy code name for manuscript variable `PIE` | 10,000 persons |
| `GAO` | Constant-price gross output value of crop farming, base year 2000 | 100 million yuan |
| `CEA` | Crop-related agricultural carbon emissions | 10,000 tonnes C |
| `efficiency` | Global-frontier Super-SBM score | dimensionless |

The carbon-emission indicator is constructed from fertilizer, pesticide, plastic-film, diesel, irrigation, and sown-area activities. Detailed definitions, coefficients, and source documentation are provided in the associated article and Online Resources.

## Data provenance

- `GAO` is obtained from the National Bureau of Statistics of China. It is already expressed as the constant-price gross output value of crop farming with 2000 as the base year; the code does not apply an additional deflator.
- Other agricultural-activity series are compiled from official statistical sources, including the National Bureau of Statistics of China and the *China Rural Statistical Yearbook*.
- Primary-industry employment is obtained from [Data Pipixia](https://ppmandata.net/). The analysis workbook retains the legacy field name `NRP`, while the associated article uses the more accurate abbreviation `PIE`.
- `CEA` is an accounting variable calculated from the six documented crop-related activity groups rather than a directly observed emissions series.

`data.xlsx` is the canonical analysis panel rather than a collection of unprocessed source workbooks. Full source citations, coefficient references, measurement boundaries, and unit-conversion rules are documented in the associated article and Online Resources.

## Super-SBM score construction

The baseline `efficiency` scores in `data.xlsx` use the following specification:

- panel data;
- original, non-oriented model;
- undesirable output included;
- super-efficiency enabled;
- variable returns to scale (VRS);
- pooled global frontier;
- equal weights across eight inputs, one desirable output, and one undesirable output.

The eight input weights are `0.125` each. The desirable-output and undesirable-output weights are `0.5` each. All pooled global-frontier and five-year local-window Super-SBM scores are computed using the Python implementation developed for this study. The pooled global common frontier produces the main `efficiency` labels, and the same model specification is applied to target-year +/-2 local reference sets for the sensitivity analysis.

## Environment

The reference environment used for the reported results ran Python `3.12.3`. Exact package versions are pinned in `requirements.txt` to make the neural-network and XGBoost execution paths explicit.

Create an isolated Python environment and install the declared dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction workflow

Run commands from the repository root.

1. Confirm that `data.xlsx` is present.
2. Open `eco_efficiency_model_pipeline.ipynb` and run the notebook from beginning to end. This produces the model-comparison, validation, ablation, and scenario workbooks used by the downstream scripts.
3. Run the standalone predictor-ablation analysis:

   ```bash
   python predictor_ablation_analysis.py
   ```

   This standalone diagnostic uses `StandardScaler` fitted within each training fold and a monotone XGBoost specification with `n_estimators=500`, `max_depth=3`, and `learning_rate=0.03`. These settings are intentionally reported separately from the main-pipeline XGBoost specification.

4. Generate descriptive tables, figures, rank-persistence measures, and province profiles:

   ```bash
   python eco_efficiency_results.py
   ```

5. Compute the pooled global-frontier scores and centered five-year local-window sensitivity results:

   ```bash
   python super_sbm_rolling_frontier.py
   ```

   This command writes both `tables/rolling_5yr_frontier_robustness_revision.csv` and `tables/Alternative_frontier_robustness_revision.xlsx`.

`eco_efficiency_results.py` requires `tables/Scenario_analysis_full_2023.xlsx`, which is produced by the main notebook. Before the local-window calculation, the frontier script checks that the stored pooled output remains numerically consistent with the documented Python specification. This is an internal integrity check within the same implementation, not a comparison with external DEA software or an independently generated benchmark.

Some generated workbooks retain filenames ending in `_revision`. These stable filenames match the corresponding supplementary workbooks and do not represent a separate model specification.

## Validation design

The repository distinguishes among:

- repeated cross-validation and random held-out evaluation;
- rolling time-forward evaluation;
- province-block evaluation;
- predictor-group ablation;
- pooled versus five-year local-window frontier sensitivity;
- standardized perturbation diagnostics generated by the fitted VWLB surrogate.

Results from these designs are not directly interchangeable because they use different samples, split rules, model sets, and aggregation procedures. All target scores are constructed ex post from the complete pooled 2000-2023 frontier before the surrogate holdout splits. The rolling chronological and province-block designs therefore diagnose temporal and spatial distribution shifts in fixed pooled-frontier labels; they are not fully prospective frontier construction or strict leave-province-out DEA validation.

## Reproducibility notes

- Random seeds and model settings are fixed in the code where applicable.
- `ID` and `Year` form the province-year key used throughout the panel and output files.
- The main notebook imports `fix_panel_reaggregation` from `panel_validation_postprocess.py` to harmonize model names and aggregate panel-validation outputs.
- The XGBoost surrogate uses the documented monotonic directions for the ten predictors.
- The 2023 scenario workbook is generated with the fitted validation-weighted linear blend (VWLB), which combines monotone XGBoost and TSLR-MLP from the main random holdout pipeline. Each reported `Delta` is the VWLB prediction for the perturbed profile minus the VWLB prediction for the observed profile.
- The `efficiency` column in `data.xlsx` contains the pooled global-frontier scores computed by the Python Super-SBM implementation; the same implementation is used for five-year local-window sensitivity analysis.
- Local paths, author-specific directories, and manuscript-management files are not required by the workflow.

## Interpretation boundary

The directional scenarios are standardized diagnostics. A positive model response indicates a response of the fitted VWLB under the specified input-output change; it does not establish implementation feasibility, behavioral response, welfare effects, or causal policy impact. Feature ablation and model-response analysis describe predictive dependence on the available information set rather than independent real-world mechanisms.

## Citation

Please cite the associated article when bibliographic details become available. The final citation will be added after publication.
