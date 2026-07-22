# Carbon-constrained agricultural eco-efficiency in China

This repository contains the data, code, and computational records used in the study **“Carbon-constrained agricultural eco-efficiency indicators for monitoring and directional screening in China.”**

The analysis covers 30 mainland Chinese provinces from 2000 to 2023. It combines a non-oriented undesirable-output Super-SBM model with machine-learning surrogate models to support efficiency monitoring, validation-aware model comparison, and standardized directional screening.

## Analytical scope

The repository supports four connected tasks:

1. use pooled global-frontier Super-SBM scores as the reference efficiency labels;
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
| `super_sbm_rolling_frontier.py` | Independent Python reconstruction check and centered five-year local-window frontier analysis. |
| `requirements.txt` | Python dependencies. |

Generated figures and workbooks are written to `fig/`, `tables/`, and `results_story/`. These directories are excluded from version control because they can be regenerated from the tracked inputs and code.

The principal generated workbooks include:

| Output | Contents |
|---|---|
| `tables/cv_summary.xlsx` | Repeated cross-validation summaries. |
| `tables/test_results.xlsx` | Held-out random-test performance. |
| `tables/Panel_dependence_robustness_checks.xlsx` | Random, rolling time-forward, and province-block validation details. |
| `tables/Panel_dependence_robustness_checks_reaggregated.xlsx` | Canonical model-name aggregation across panel-validation outputs. |
| `tables/Surrogate_predictor_ablation_revision.xlsx` | Predictor-group ablation summaries and split-level results. |
| `tables/Scenario_analysis_full_2023.xlsx` | Province-level outputs for the three standardized 2023 perturbations. |
| `tables/Alternative_frontier_robustness_revision.xlsx` | Pooled-frontier reconstruction and five-year local-window sensitivity results. |

## Data structure

`data.xlsx` contains 720 province-year observations. The principal analytical fields are:

| Field | Definition | Unit / coding |
|---|---|---|
| `ID` | Province identifier | 1-30 |
| `Year` | Observation year | 2000-2023 |
| `TPAM` | Total power of agricultural machinery | million kW |
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

## Super-SBM score construction and verification

The baseline `efficiency` scores in `data.xlsx` use the following specification:

- panel data;
- original, non-oriented model;
- undesirable output included;
- super-efficiency enabled;
- variable returns to scale (VRS);
- pooled global frontier;
- equal weights across eight inputs, one desirable output, and one undesirable output.

The eight input weights are `0.125` each. The desirable-output and undesirable-output weights are `0.5` each. The Python frontier implementation reconstructs the pooled scores from the panel variables in `data.xlsx`, compares them with the `efficiency` reference column, and applies the same specification to the alternative-frontier sensitivity analysis. It does not overwrite the reference scores in `data.xlsx`.

## Environment

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

4. Generate descriptive tables, figures, rank-persistence measures, and province profiles:

   ```bash
   python eco_efficiency_results.py
   ```

5. Validate the Python pooled-frontier reconstruction and compute the centered five-year local-window sensitivity results:

   ```bash
   python super_sbm_rolling_frontier.py
   ```

`eco_efficiency_results.py` requires `tables/Scenario_analysis_full_2023.xlsx`, which is produced by the main notebook. The frontier script stops before the local-window calculation if its pooled-frontier reconstruction does not pass the built-in validation gate.

Some generated workbooks retain filenames ending in `_revision`. These stable filenames are preserved solely to maintain exact links with the archived Online Resources; they do not represent a separate model specification or development branch.

## Validation design

The repository distinguishes among:

- repeated cross-validation and random held-out evaluation;
- rolling time-forward evaluation;
- province-block evaluation;
- predictor-group ablation;
- pooled versus five-year local-window frontier sensitivity;
- perturbation validation and emission-factor sensitivity implemented in the main workflow.

Results from these designs are not directly interchangeable because they use different samples, split rules, model sets, and aggregation procedures. In particular, strong random-split performance should not be interpreted as equivalent to future-period or unseen-province transfer performance.

## Reproducibility notes

- Random seeds and model settings are fixed in the code where applicable.
- `ID` and `Year` form the province-year key used throughout the panel and output files.
- The main notebook imports `fix_panel_reaggregation` from `panel_validation_postprocess.py` to harmonize model names and aggregate panel-validation outputs.
- The XGBoost surrogate uses the documented monotonic directions for the ten predictors.
- The `efficiency` column in `data.xlsx` contains the reference scores; the Python Super-SBM implementation is used for reconstruction checks and alternative-frontier sensitivity.
- Local paths, author-specific directories, and manuscript-management files are not required by the workflow.

## Interpretation boundary

The directional scenarios are standardized diagnostics. A positive model response indicates consistency with the imposed input-output direction under the fitted surrogate; it does not establish implementation feasibility, behavioral response, welfare effects, or causal policy impact. Feature ablation and model-response analysis describe predictive dependence on the available information set rather than independent real-world mechanisms.

## Citation

Please cite the associated article when bibliographic details become available. The final citation will be added after publication.
