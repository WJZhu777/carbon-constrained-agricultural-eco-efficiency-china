# Eco-efficiency analysis code

This repository contains the code used to generate the benchmark modelling results and the results-story figures and tables for the Ecological Informatics submission.

## Files

- `eco_efficiency_model_pipeline.py`  
  Main analysis script for model training, benchmarking, and prediction.

- `eco_efficiency_model_pipeline.ipynb`  
  Notebook version of the main analysis workflow.

- `eco_efficiency_results.py`  
  Script for descriptive tables, trend figures, heatmaps, inequality metrics, rank persistence, and scenario-based result summaries.

- `requirements.txt`  
  Python dependencies required to run the scripts.

## Input data

Both scripts read the dataset from:

```python
./data.xlsx
```

Please keep the data file in the same working directory as the script, or revise the path in the code if needed.

## Outputs

The scripts write figures and tables to the output directories defined in the code:

- `fig/`
- `tables/`
- `results_story/`

These directories contain generated outputs and are excluded from version control by `.gitignore`.

## Recommended tracked files

The repository should normally track only the core research assets:

- `data.xlsx`
- `eco_efficiency_model_pipeline.py`
- `eco_efficiency_model_pipeline.ipynb`
- `eco_efficiency_results.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

## Suggested run order

1. Run `eco_efficiency_model_pipeline.py` for the main modelling workflow.
2. Run `eco_efficiency_results.py` for descriptive outputs and scenario-based summaries.

## Reproducibility

Random seeds and model settings are fixed in the scripts. Running the code under the same software environment and with the same input data should reproduce the same numerical outputs.
