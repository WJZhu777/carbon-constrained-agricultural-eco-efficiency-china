from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "tables"


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
    panel_path = fix_panel_reaggregation()
    print(f"[postprocess] updated {panel_path}")


if __name__ == "__main__":
    main()
