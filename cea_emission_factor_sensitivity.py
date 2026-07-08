from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from revision_minimal_robustness import (
    CEA_ACCOUNTING_COEFS_DATA_UNITS,
    DATA_PATH,
    ID_COL,
    TABLES_DIR,
    YEAR_COL,
    Y_COL,
    load_modeling_data,
    super_sbm_score_one,
)


OUT_PATH = TABLES_DIR / "CEA_emission_factor_sensitivity_revision.xlsx"


SOURCE_LABELS = {
    "AFA": "Fertilizer coefficient",
    "PU": "Pesticide coefficient",
    "PFU": "Plastic-film coefficient",
    "ADY": "Diesel coefficient",
    "EIA": "Irrigation coefficient",
    "CS": "Plowing / crop-sown-area coefficient",
}


GROUPS = {
    "material_inputs": ["AFA", "PU", "PFU", "ADY"],
    "land_water_activity": ["EIA", "CS"],
}


def _safe_pearson(a, b) -> float:
    s1 = pd.Series(a, dtype="float64")
    s2 = pd.Series(b, dtype="float64")
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return np.nan
    return float(s1[mask].corr(s2[mask]))


def _safe_spearman(a, b) -> float:
    s1 = pd.Series(a, dtype="float64")
    s2 = pd.Series(b, dtype="float64")
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return np.nan
    return float(s1[mask].rank(method="average").corr(s2[mask].rank(method="average")))


def make_scenarios(delta: float = 0.10) -> pd.DataFrame:
    rows = []
    for source in ["AFA", "PU", "PFU", "ADY", "EIA", "CS"]:
        for sign, direction in [(-1.0, "minus"), (1.0, "plus")]:
            pct = sign * delta
            rows.append(
                {
                    "scenario_key": f"{source}_{direction}{int(delta * 100)}",
                    "scenario_type": "single_coefficient",
                    "perturbed_sources": source,
                    "delta_pct": pct,
                    "description": f"{SOURCE_LABELS[source]} {pct:+.0%}; all other emission factors unchanged.",
                    "rationale": "Single-source emission-factor uncertainty check.",
                }
            )
    for group_name, sources in GROUPS.items():
        for sign, direction in [(-1.0, "minus"), (1.0, "plus")]:
            pct = sign * delta
            rows.append(
                {
                    "scenario_key": f"{group_name}_{direction}{int(delta * 100)}",
                    "scenario_type": "grouped_coefficients",
                    "perturbed_sources": ", ".join(sources),
                    "delta_pct": pct,
                    "description": f"{group_name} coefficients {pct:+.0%}; all other emission factors unchanged.",
                    "rationale": "Grouped relative-emission-factor uncertainty check.",
                }
            )
    return pd.DataFrame(rows)


def perturb_cea(df: pd.DataFrame, perturbed_sources: list[str], delta_pct: float) -> pd.Series:
    cea = df["CEA"].astype(float).copy()
    for source in perturbed_sources:
        cea = cea + df[source].astype(float) * CEA_ACCOUNTING_COEFS_DATA_UNITS[source] * float(delta_pct)
    if (cea <= 0).any():
        bad_n = int((cea <= 0).sum())
        raise ValueError(f"Perturbed CEA has {bad_n} non-positive values.")
    return cea


def solve_scores_for_scenario(df_base: pd.DataFrame, scenario: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [s.strip() for s in str(scenario["perturbed_sources"]).split(",")]
    delta_pct = float(scenario["delta_pct"])
    df_alt = df_base.copy()
    df_alt["CEA"] = perturb_cea(df_base, sources, delta_pct)

    rows = []
    errors = []
    for row_pos, row in df_alt.reset_index(drop=True).iterrows():
        try:
            score = super_sbm_score_one(df_alt, int(row_pos))
            status = "ok"
            error = ""
        except Exception as exc:
            score = np.nan
            status = "failed"
            error = str(exc)
            errors.append(
                {
                    "scenario_key": scenario["scenario_key"],
                    "ID": int(row[ID_COL]),
                    "Province": row["Province"],
                    "Year": int(row[YEAR_COL]),
                    "Error": error,
                }
            )

        baseline_score = float(df_base.loc[row_pos, Y_COL])
        baseline_cea = float(df_base.loc[row_pos, "CEA"])
        sensitivity_cea = float(df_alt.loc[row_pos, "CEA"])
        rows.append(
            {
                "scenario_key": scenario["scenario_key"],
                "scenario_type": scenario["scenario_type"],
                "perturbed_sources": scenario["perturbed_sources"],
                "delta_pct": delta_pct,
                "ID": int(row[ID_COL]),
                "Province": row["Province"],
                "Year": int(row[YEAR_COL]),
                "baseline_score": baseline_score,
                "sensitivity_score": score,
                "score_diff": score - baseline_score if np.isfinite(score) else np.nan,
                "abs_score_diff": abs(score - baseline_score) if np.isfinite(score) else np.nan,
                "baseline_CEA": baseline_cea,
                "sensitivity_CEA": sensitivity_cea,
                "CEA_diff": sensitivity_cea - baseline_cea,
                "CEA_rel_diff_pct": (sensitivity_cea - baseline_cea) / baseline_cea * 100.0,
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(errors)


def add_common_ranks(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["score_pair_valid"] = out["sensitivity_score"].notna()
    out["rank_baseline_year_common"] = np.nan
    out["rank_sensitivity_year_common"] = np.nan
    for (_, _), g in out[out["score_pair_valid"]].groupby(["scenario_key", "Year"]):
        idx = g.index
        out.loc[idx, "rank_baseline_year_common"] = g["baseline_score"].rank(
            method="average", ascending=False
        )
        out.loc[idx, "rank_sensitivity_year_common"] = g["sensitivity_score"].rank(
            method="average", ascending=False
        )
    out["rank_diff_year_common"] = (
        out["rank_sensitivity_year_common"] - out["rank_baseline_year_common"]
    )
    out["abs_rank_diff_year_common"] = out["rank_diff_year_common"].abs()
    return out


def summarize_scores(detail: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, scenario in scenarios.iterrows():
        g = detail[detail["scenario_key"] == scenario["scenario_key"]]
        ge = g[g["score_pair_valid"]].copy()
        rows.append(
            {
                "scenario_key": scenario["scenario_key"],
                "scenario_type": scenario["scenario_type"],
                "perturbed_sources": scenario["perturbed_sources"],
                "delta_pct": float(scenario["delta_pct"]),
                "N_total": int(len(g)),
                "N_effective": int(len(ge)),
                "N_failed": int(len(g) - len(ge)),
                "mean_CEA_rel_diff_pct": float(ge["CEA_rel_diff_pct"].mean()) if len(ge) else np.nan,
                "max_abs_CEA_rel_diff_pct": float(ge["CEA_rel_diff_pct"].abs().max()) if len(ge) else np.nan,
                "pearson_score": _safe_pearson(ge["baseline_score"], ge["sensitivity_score"]),
                "spearman_score": _safe_spearman(ge["baseline_score"], ge["sensitivity_score"]),
                "mean_abs_score_diff": float(ge["abs_score_diff"].mean()) if len(ge) else np.nan,
                "median_abs_score_diff": float(ge["abs_score_diff"].median()) if len(ge) else np.nan,
                "max_abs_score_diff": float(ge["abs_score_diff"].max()) if len(ge) else np.nan,
                "mean_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].mean()) if len(ge) else np.nan,
                "median_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].median()) if len(ge) else np.nan,
                "max_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].max()) if len(ge) else np.nan,
                "share_abs_rank_diff_le_1": float((ge["abs_rank_diff_year_common"] <= 1).mean()) if len(ge) else np.nan,
                "share_abs_rank_diff_le_3": float((ge["abs_rank_diff_year_common"] <= 3).mean()) if len(ge) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_score_diff", "mean_abs_rank_diff_year_common"], ascending=[False, False]
    )


def summarize_by_year(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_key, year), g in detail.groupby(["scenario_key", "Year"]):
        ge = g[g["score_pair_valid"]].copy()
        rows.append(
            {
                "scenario_key": scenario_key,
                "Year": int(year),
                "N_total": int(len(g)),
                "N_effective": int(len(ge)),
                "N_failed": int(len(g) - len(ge)),
                "pearson_score": _safe_pearson(ge["baseline_score"], ge["sensitivity_score"]),
                "spearman_score": _safe_spearman(ge["baseline_score"], ge["sensitivity_score"]),
                "mean_abs_score_diff": float(ge["abs_score_diff"].mean()) if len(ge) else np.nan,
                "median_abs_score_diff": float(ge["abs_score_diff"].median()) if len(ge) else np.nan,
                "mean_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].mean()) if len(ge) else np.nan,
                "median_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].median()) if len(ge) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_key", "Year"]).reset_index(drop=True)


def summarize_province_longrun(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (scenario_key, province), g in detail.groupby(["scenario_key", "Province"]):
        ge = g[g["score_pair_valid"]].copy()
        rows.append(
            {
                "scenario_key": scenario_key,
                "Province": province,
                "N_total_years": int(len(g)),
                "N_effective_years": int(len(ge)),
                "baseline_longrun_mean": float(ge["baseline_score"].mean()) if len(ge) else np.nan,
                "sensitivity_longrun_mean": float(ge["sensitivity_score"].mean()) if len(ge) else np.nan,
                "mean_abs_score_diff": float(ge["abs_score_diff"].mean()) if len(ge) else np.nan,
            }
        )
    prov = pd.DataFrame(rows)
    prov["rank_baseline_longrun_common"] = np.nan
    prov["rank_sensitivity_longrun_common"] = np.nan
    for scenario_key, g in prov.dropna(subset=["sensitivity_longrun_mean"]).groupby("scenario_key"):
        idx = g.index
        prov.loc[idx, "rank_baseline_longrun_common"] = g["baseline_longrun_mean"].rank(
            method="average", ascending=False
        )
        prov.loc[idx, "rank_sensitivity_longrun_common"] = g["sensitivity_longrun_mean"].rank(
            method="average", ascending=False
        )
    prov["rank_diff_longrun_common"] = (
        prov["rank_sensitivity_longrun_common"] - prov["rank_baseline_longrun_common"]
    )
    prov["abs_rank_diff_longrun_common"] = prov["rank_diff_longrun_common"].abs()

    summary_rows = []
    for scenario_key, g in prov.groupby("scenario_key"):
        ge = g.dropna(subset=["sensitivity_longrun_mean"]).copy()
        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "N_provinces": int(len(ge)),
                "spearman_longrun_mean": _safe_spearman(
                    ge["baseline_longrun_mean"], ge["sensitivity_longrun_mean"]
                ),
                "mean_abs_longrun_score_diff": float(
                    (ge["sensitivity_longrun_mean"] - ge["baseline_longrun_mean"]).abs().mean()
                )
                if len(ge)
                else np.nan,
                "median_abs_longrun_score_diff": float(
                    (ge["sensitivity_longrun_mean"] - ge["baseline_longrun_mean"]).abs().median()
                )
                if len(ge)
                else np.nan,
                "mean_abs_rank_diff_longrun_common": float(ge["abs_rank_diff_longrun_common"].mean())
                if len(ge)
                else np.nan,
                "median_abs_rank_diff_longrun_common": float(ge["abs_rank_diff_longrun_common"].median())
                if len(ge)
                else np.nan,
                "max_abs_rank_diff_longrun_common": float(ge["abs_rank_diff_longrun_common"].max())
                if len(ge)
                else np.nan,
            }
        )
    prov_summary = pd.DataFrame(summary_rows)
    return prov, prov_summary


def make_readme() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Notes": [
                "CEA emission-factor sensitivity for reviewer comments on emission-factor uncertainty.",
                "No new statistical data are introduced; activity data are from data.xlsx.",
                "Each scenario perturbs one coefficient or one source group by +/-10% relative to the Li et al. (2011)-based coefficient values.",
                "Perturbed CEA is computed as current CEA plus the coefficient-delta contribution, so baseline CEA is not replaced by a newly rounded reconstruction.",
                "For each perturbation, the pooled global Super-SBM scores are recomputed and compared with the revised baseline efficiency scores stored in data.xlsx.",
                "Uniformly scaling all emission factors is not reported because it mostly changes the undesirable-output measurement unit; relative source-specific perturbations are more informative.",
                "This is an uncertainty/sensitivity diagnostic, not evidence about causal policy effects or implementation feasibility.",
            ]
        }
    )


def run_cea_emission_factor_sensitivity() -> Path:
    TABLES_DIR.mkdir(exist_ok=True)
    df = load_modeling_data()
    scenarios = make_scenarios(delta=0.10)

    detail_frames = []
    error_frames = []
    print(f"[CEA sensitivity] loaded {len(df)} rows from {DATA_PATH}")
    for i, scenario in scenarios.iterrows():
        print(
            f"[CEA sensitivity] {i + 1}/{len(scenarios)} {scenario['scenario_key']} "
            f"({scenario['perturbed_sources']} {scenario['delta_pct']:+.0%})"
        )
        detail, errors = solve_scores_for_scenario(df, scenario)
        detail_frames.append(detail)
        if not errors.empty:
            error_frames.append(errors)

    detail = pd.concat(detail_frames, ignore_index=True)
    detail = add_common_ranks(detail)
    errors = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(columns=["scenario_key", "ID", "Province", "Year", "Error"])
    )
    summary = summarize_scores(detail, scenarios)
    year_summary = summarize_by_year(detail)
    province_longrun, province_longrun_summary = summarize_province_longrun(detail)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="scenario_summary", index=False)
        year_summary.to_excel(writer, sheet_name="year_summary", index=False)
        province_longrun_summary.to_excel(writer, sheet_name="province_longrun_summary", index=False)
        province_longrun.to_excel(writer, sheet_name="province_longrun", index=False)
        detail.to_excel(writer, sheet_name="score_detail", index=False)
        scenarios.to_excel(writer, sheet_name="coefficient_scenarios", index=False)
        errors.to_excel(writer, sheet_name="errors", index=False)
        make_readme().to_excel(writer, sheet_name="ReadMe", index=False)
    return OUT_PATH


def main() -> None:
    out = run_cea_emission_factor_sensitivity()
    print(f"[CEA sensitivity] saved {out}")


if __name__ == "__main__":
    main()
