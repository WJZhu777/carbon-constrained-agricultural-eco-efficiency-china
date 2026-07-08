from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "tables"


def _safe_corr(a, b, method: str = "pearson") -> float:
    s1 = pd.Series(a, dtype="float64")
    s2 = pd.Series(b, dtype="float64")
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return np.nan
    return float(s1[mask].corr(s2[mask], method=method))


def _canonical_model(name: str) -> str:
    name = str(name).strip()
    if name.startswith("Ensemble("):
        return "VWLB"
    if name.startswith("Stacking["):
        return "OOF-Stacking"
    if name == "VanillaMLP":
        return "MLP"
    return name


def _extract_w_xgb(name: str) -> float:
    match = re.search(r"Ensemble\(XGB([0-9]*\.?[0-9]+)\+TSLR-MLP", str(name))
    return float(match.group(1)) if match else np.nan


def _extract_meta(name: str) -> str | float:
    match = re.search(r"^Stacking\[(.+?)\]$", str(name).strip())
    return match.group(1) if match else np.nan


def _std_or_zero(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def fix_cea_effective_n(tables_dir: Path = TABLES_DIR, year_focus: int = 2023) -> Path:
    """Correct CEA-exclusion summaries to use paired effective samples."""
    path = tables_dir / "CEA_exclusion_robustness.xlsx"
    sheets = pd.read_excel(path, sheet_name=None)
    out = sheets["all_scores_pooled"].copy()

    out["score_pair_valid"] = out["score_baseline"].notna() & out["score_no_cea"].notna()
    out["rank_baseline_year_all"] = out.groupby("Year")["score_baseline"].rank(
        method="average", ascending=False
    )

    out["rank_baseline_year"] = np.nan
    out["rank_no_cea_year"] = np.nan
    valid = out["score_pair_valid"]
    out.loc[valid, "rank_baseline_year"] = out.loc[valid].groupby("Year")[
        "score_baseline"
    ].rank(method="average", ascending=False)
    out.loc[valid, "rank_no_cea_year"] = out.loc[valid].groupby("Year")[
        "score_no_cea"
    ].rank(method="average", ascending=False)
    out["rank_diff_year"] = out["rank_no_cea_year"] - out["rank_baseline_year"]
    out["abs_rank_diff_year"] = out["rank_diff_year"].abs()

    year_rows = []
    for yr, g in out.groupby("Year"):
        ge = g[g["score_pair_valid"]].copy()
        year_rows.append(
            {
                "Year": int(yr),
                "N_effective": int(len(ge)),
                "N_total": int(len(g)),
                "N_missing_no_cea": int(len(g) - len(ge)),
                "pearson_score": _safe_corr(ge["score_baseline"], ge["score_no_cea"]),
                "spearman_score": _safe_corr(
                    ge["score_baseline"], ge["score_no_cea"], method="spearman"
                ),
                "spearman_rank_common": _safe_corr(
                    ge["rank_baseline_year"], ge["rank_no_cea_year"], method="pearson"
                ),
                "mean_abs_score_diff": float(ge["abs_score_diff"].mean()),
                "median_abs_score_diff": float(ge["abs_score_diff"].median()),
                "mean_abs_rank_diff_common": float(ge["abs_rank_diff_year"].mean()),
                "median_abs_rank_diff_common": float(ge["abs_rank_diff_year"].median()),
            }
        )
    year_summary = pd.DataFrame(year_rows).sort_values("Year").reset_index(drop=True)

    prov_rows = []
    for province, g in out.groupby("Province"):
        ge = g[g["score_pair_valid"]].copy()
        prov_rows.append(
            {
                "Province": province,
                "N_effective_years": int(len(ge)),
                "N_total_years": int(len(g)),
                "N_missing_no_cea": int(len(g) - len(ge)),
                "longrun_mean_baseline_all": float(g["score_baseline"].mean()),
                "longrun_mean_baseline": float(ge["score_baseline"].mean())
                if len(ge)
                else np.nan,
                "longrun_mean_no_cea": float(ge["score_no_cea"].mean())
                if len(ge)
                else np.nan,
                "mean_abs_score_diff": float(ge["abs_score_diff"].mean())
                if len(ge)
                else np.nan,
            }
        )
    prov_long = pd.DataFrame(prov_rows)
    valid_prov = prov_long["longrun_mean_no_cea"].notna()
    prov_long["rank_baseline_longrun_all"] = prov_long["longrun_mean_baseline_all"].rank(
        method="average", ascending=False
    )
    prov_long["rank_baseline_longrun"] = np.nan
    prov_long["rank_no_cea_longrun"] = np.nan
    prov_long.loc[valid_prov, "rank_baseline_longrun"] = prov_long.loc[
        valid_prov, "longrun_mean_baseline"
    ].rank(method="average", ascending=False)
    prov_long.loc[valid_prov, "rank_no_cea_longrun"] = prov_long.loc[
        valid_prov, "longrun_mean_no_cea"
    ].rank(method="average", ascending=False)
    prov_long["rank_diff_longrun"] = (
        prov_long["rank_no_cea_longrun"] - prov_long["rank_baseline_longrun"]
    )
    prov_long["abs_rank_diff_longrun"] = prov_long["rank_diff_longrun"].abs()
    prov_long = prov_long.sort_values(
        ["abs_rank_diff_longrun", "mean_abs_score_diff"], ascending=[False, False]
    ).reset_index(drop=True)

    focus_2023 = (
        out[out["Year"] == year_focus]
        .copy()
        .sort_values(["score_pair_valid", "abs_rank_diff_year", "abs_score_diff"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    focus_eff = focus_2023[focus_2023["score_pair_valid"]].copy()
    out_eff = out[out["score_pair_valid"]].copy()
    prov_eff = prov_long[prov_long["longrun_mean_no_cea"].notna()].copy()

    overall_summary = pd.DataFrame(
        [
            {
                "comparison_level": "all_province_year_rows",
                "N_effective": int(len(out_eff)),
                "N_total": int(len(out)),
                "N_missing_no_cea": int(len(out) - len(out_eff)),
                "pearson_score": _safe_corr(out_eff["score_baseline"], out_eff["score_no_cea"]),
                "spearman_score": _safe_corr(
                    out_eff["score_baseline"], out_eff["score_no_cea"], method="spearman"
                ),
                "spearman_rank_common": np.nan,
                "mean_abs_score_diff": float(out_eff["abs_score_diff"].mean()),
                "median_abs_score_diff": float(out_eff["abs_score_diff"].median()),
                "mean_abs_rank_diff_common": np.nan,
                "median_abs_rank_diff_common": np.nan,
            },
            {
                "comparison_level": f"year_{year_focus}",
                "N_effective": int(len(focus_eff)),
                "N_total": int(len(focus_2023)),
                "N_missing_no_cea": int(len(focus_2023) - len(focus_eff)),
                "pearson_score": _safe_corr(focus_eff["score_baseline"], focus_eff["score_no_cea"]),
                "spearman_score": _safe_corr(
                    focus_eff["score_baseline"], focus_eff["score_no_cea"], method="spearman"
                ),
                "spearman_rank_common": _safe_corr(
                    focus_eff["rank_baseline_year"],
                    focus_eff["rank_no_cea_year"],
                    method="pearson",
                ),
                "mean_abs_score_diff": float(focus_eff["abs_score_diff"].mean()),
                "median_abs_score_diff": float(focus_eff["abs_score_diff"].median()),
                "mean_abs_rank_diff_common": float(focus_eff["abs_rank_diff_year"].mean()),
                "median_abs_rank_diff_common": float(focus_eff["abs_rank_diff_year"].median()),
            },
            {
                "comparison_level": "province_longrun_mean",
                "N_effective": int(len(prov_eff)),
                "N_total": int(len(prov_long)),
                "N_missing_no_cea": int(len(prov_long) - len(prov_eff)),
                "pearson_score": _safe_corr(
                    prov_eff["longrun_mean_baseline"], prov_eff["longrun_mean_no_cea"]
                ),
                "spearman_score": _safe_corr(
                    prov_eff["longrun_mean_baseline"],
                    prov_eff["longrun_mean_no_cea"],
                    method="spearman",
                ),
                "spearman_rank_common": _safe_corr(
                    prov_eff["rank_baseline_longrun"],
                    prov_eff["rank_no_cea_longrun"],
                    method="pearson",
                ),
                "mean_abs_score_diff": float(
                    (prov_eff["longrun_mean_no_cea"] - prov_eff["longrun_mean_baseline"]).abs().mean()
                ),
                "median_abs_score_diff": float(
                    (prov_eff["longrun_mean_no_cea"] - prov_eff["longrun_mean_baseline"]).abs().median()
                ),
                "mean_abs_rank_diff_common": float(prov_eff["abs_rank_diff_longrun"].mean()),
                "median_abs_rank_diff_common": float(prov_eff["abs_rank_diff_longrun"].median()),
            },
        ]
    )

    readme = pd.DataFrame(
        {
            "Notes": [
                "N_effective counts paired rows with both score_baseline and score_no_cea available.",
                "N_total is the full comparison universe before excluding infeasible no-CEA scores.",
                "Rank differences are computed within the common effective subset for each year/province summary.",
                "rank_baseline_year_all and rank_baseline_longrun_all are retained only for audit context.",
            ]
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="all_scores_pooled", index=False)
        year_summary.to_excel(writer, sheet_name="year_summary", index=False)
        prov_long.to_excel(writer, sheet_name="province_longrun", index=False)
        focus_2023.to_excel(writer, sheet_name=f"focus_{year_focus}_rank_shift", index=False)
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)

    return path


def fix_panel_reaggregation(tables_dir: Path = TABLES_DIR) -> Path:
    """Canonicalize all panel robustness detail sheets and reaggregate them together."""
    src = tables_dir / "Panel_dependence_robustness_checks.xlsx"
    out_path = tables_dir / "Panel_dependence_robustness_checks_reaggregated.xlsx"
    sheets = pd.read_excel(src, sheet_name=None)

    frames = []
    sheet_groups = [
        ("random_stratified", ["random_stratified_long"]),
        ("time_forward", ["time_forward_rolling_long", "time_forward_each_split"]),
        ("province_block", ["province_block_long"]),
    ]
    for _, candidate_names in sheet_groups:
        candidates = []
        for sheet_name in candidate_names:
            if sheet_name not in sheets:
                continue
            df = sheets[sheet_name].copy()
            if df.empty:
                continue
            df = df.rename(columns={"R²": "R2", "R^2": "R2"})
            required = {"SplitType", "Model", "MSE", "MAE", "R2"}
            if not required.issubset(df.columns):
                continue
            valid_n = df[["MSE", "MAE", "R2"]].notna().all(axis=1).sum()
            candidates.append((int(valid_n), sheet_name, df))
        if not candidates:
            continue
        _, sheet_name, df = max(candidates, key=lambda item: item[0])
        df["source_sheet"] = sheet_name
        frames.append(df)
    if not frames:
        raise ValueError("No panel robustness detail sheets were found for reaggregation.")

    detail = pd.concat(frames, ignore_index=True, sort=False)
    detail = detail.rename(columns={"R²": "R2", "R^2": "R2"})
    for col in ["MSE", "MAE", "R2"]:
        detail[col] = pd.to_numeric(detail[col], errors="coerce")
    detail["Model_raw"] = detail["Model"].astype(str).str.strip()
    detail["Model_canon"] = detail["Model_raw"].apply(_canonical_model)
    detail["VWLB_w_xgb"] = detail["Model_raw"].apply(_extract_w_xgb)
    detail["Stack_meta"] = detail["Model_raw"].apply(_extract_meta)
    detail = detail.dropna(subset=["MSE", "MAE", "R2"]).reset_index(drop=True)

    agg = (
        detail.groupby(["SplitType", "Model_canon"], as_index=False)
        .agg(
            n_splits=("MSE", "size"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", _std_or_zero),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", _std_or_zero),
            R2_mean=("R2", "mean"),
            R2_std=("R2", _std_or_zero),
        )
        .sort_values(["SplitType", "R2_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    df_w = detail[detail["Model_raw"].str.startswith("Ensemble(")].copy()
    if df_w.empty:
        w_stats = pd.DataFrame(columns=["SplitType", "n", "mean", "std", "min", "max"])
    else:
        w_stats = (
            df_w.groupby("SplitType")["VWLB_w_xgb"]
            .agg(n="count", mean="mean", std=_std_or_zero, min="min", max="max")
            .reset_index()
        )

    stack_mask = detail["Model_raw"].str.startswith("Stacking[")
    df_s = detail.loc[stack_mask].copy()
    if df_s.empty:
        meta_freq = pd.DataFrame(columns=["SplitType", "Stack_meta", "count"])
    else:
        meta_freq = (
            df_s.groupby(["SplitType", "Stack_meta"])
            .size()
            .reset_index(name="count")
            .sort_values(["SplitType", "count"], ascending=[True, False])
        )

    sort_cols = ["SplitType", "SplitID"] if "SplitID" in detail.columns else ["SplitType"]
    best_idx = detail.groupby(sort_cols)["R2"].idxmax()
    best_by_split = detail.loc[
        best_idx,
        [
            c
            for c in [
                "SplitType",
                "SplitID",
                "TrainEnd",
                "ValYears",
                "TestYears",
                "Model_raw",
                "Model_canon",
                "MSE",
                "MAE",
                "R2",
            ]
            if c in detail.columns
        ],
    ].sort_values(sort_cols).reset_index(drop=True)

    readme = pd.DataFrame(
        {
            "Notes": [
                "source_all_detail combines random_stratified_long, time_forward_rolling_long, and province_block_long.",
                "Model_canon maps Ensemble(XGB...+TSLR-MLP...) to VWLB and Stacking[...] to OOF-Stacking.",
                "agg_canonical_mean_std is the preferred panel robustness summary for manuscript-level reporting.",
                "best_by_split reports the highest-R2 model within each split.",
            ]
        }
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="source_all_detail", index=False)
        agg.to_excel(writer, sheet_name="agg_canonical_mean_std", index=False)
        w_stats.to_excel(writer, sheet_name="vwlb_weight_stats", index=False)
        meta_freq.to_excel(writer, sheet_name="stack_meta_frequency", index=False)
        best_by_split.to_excel(writer, sheet_name="best_by_split", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)

    return out_path


def main() -> None:
    cea_path = fix_cea_effective_n()
    panel_path = fix_panel_reaggregation()
    print(f"[postprocess] updated {cea_path}")
    print(f"[postprocess] updated {panel_path}")


if __name__ == "__main__":
    main()
