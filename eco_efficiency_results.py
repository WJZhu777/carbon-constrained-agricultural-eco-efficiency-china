# -*- coding: utf-8 -*-
"""
Generate descriptive figures, summary tables, and directional-perturbation assets
for the revised ESI manuscript.

This version extends Section 4.3 with a province-level interpretation layer so that
results can move beyond ranking and toward management-oriented explanation.

Key additions for Section 4.3:
- Province story profile table
- Representative case selection table aligned with the revised manuscript
  (Jilin + Gansu + Ningxia by default)
- Representative case response-profile figure

CRITICAL LOGIC FOR SECTION 4.3
------------------------------
This script does NOT retrain a new surrogate for Figure 4 / Table 5.
Instead, it reads the full-coverage 2023 scenario output exported from the
main manuscript pipeline, so that Section 4.3 uses the SAME final surrogate
(e.g., VWLB) as Section 3.2.

Required upstream input for Section 4.3:
    ./tables/Scenario_analysis_full_2023.xlsx

Expected columns in that file (sheet = scenario_outputs):
    ID, Province, Year, SuperSBM_observed, yhat_base,
    yhat_S1_input10,     delta_S1_input10,
    yhat_S2_allsource10, delta_S2_allsource10,
    yhat_S3_upgrade_mix, delta_S3_upgrade_mix

What this script produces for Section 4.3:
    results_story/figures/Fig_4_3_perturbation_distribution_<year>.tif
    results_story/figures/Fig_4_3_quadrant_typology_P1_10pct_<year>.tif
    results_story/figures/Fig_4_3_representative_case_profiles_<year>.tif
    results_story/tables/Table_4_3_perturbation_pred_long_<year>.csv
    results_story/tables/Table_4_3_perturbation_summary_<year>.xlsx
    results_story/tables/Table_4_3_top_bottom_10pct_<year>.xlsx
    results_story/tables/Table_4_3_province_ranking_<year>.xlsx
    results_story/tables/Table_4_3_typology_P1_10pct_<year>.xlsx
    results_story/tables/Table_4_3_province_story_profile_<year>.xlsx
    results_story/tables/Table_4_3_representative_cases_<year>.xlsx

Default data input:
    ./data.xlsx (Sheet1)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


DATA_PATH = "./data.xlsx"
SHEET_NAME = "Sheet1"
SCENARIO_INPUT_PATH = "./tables/Scenario_analysis_full_2023.xlsx"

ID_COL = "ID"
T_COL = "Year"
Y_COL = "efficiency"

FEATURES = ["TPAM", "EIA", "CS", "AFA", "PU", "ADY", "PFU", "NRP", "GAO", "CEA"]
STORY_FEATURES = ["AFA", "PU", "ADY", "PFU", "EIA", "CS", "GAO", "CEA", "NRP", "TPAM"]

PROVINCES_BY_ID = [
    "Beijing",
    "Tianjin",
    "Hebei",
    "Shanxi",
    "Inner Mongolia",
    "Liaoning",
    "Jilin",
    "Heilongjiang",
    "Shanghai",
    "Jiangsu",
    "Zhejiang",
    "Anhui",
    "Fujian",
    "Jiangxi",
    "Shandong",
    "Henan",
    "Hubei",
    "Hunan",
    "Guangdong",
    "Guangxi",
    "Hainan",
    "Chongqing",
    "Sichuan",
    "Guizhou",
    "Yunnan",
    "Shaanxi",
    "Gansu",
    "Qinghai",
    "Ningxia",
    "Xinjiang",
]
PROVINCE_NAME_BY_ID = {i + 1: name for i, name in enumerate(PROVINCES_BY_ID)}

OUT_DIR = "results_story"
SEED = 42
PERTURBATION_YEAR = None

SAVE_DPI = 600
FIG_FMT = "tif"
HEATMAP_CMAP = "cividis"
USE_ROBUST_HEATMAP = False
ROBUST_Q = (0.02, 0.98)

# Representative-case mode for Section 4.3 discussion.
# "fixed_manuscript_cases" uses the revised manuscript trio:
# Jilin + Gansu + Ningxia.
# "auto" falls back to data-driven selection.
REPRESENTATIVE_CASE_MODE = "fixed_manuscript_cases"
FORCED_REPRESENTATIVE_CASES = {
    "Priority follow-up case": "Jilin",
    "Direction-sensitive case": "Gansu",
    "Low-return case": "Ningxia",
}


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def set_mpl_style() -> None:
    plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 150,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_path: str) -> None:
    fig.savefig(out_path, format="tiff", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def ensure_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return {"out": out_dir, "fig": fig_dir, "tab": tab_dir}


def check_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def theil_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x <= 0):
        return np.nan
    mu = x.mean()
    return np.mean((x / mu) * np.log(x / mu))


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    ra = a.rank(method="average")
    rb = b.rank(method="average")
    return ra.corr(rb, method="pearson")


def _get_heatmap_limits(data: np.ndarray) -> Tuple[float, float]:
    if USE_ROBUST_HEATMAP:
        vmin = float(np.nanquantile(data, ROBUST_Q[0]))
        vmax = float(np.nanquantile(data, ROBUST_Q[1]))
        if np.isfinite(vmin) and np.isfinite(vmax) and not np.isclose(vmin, vmax):
            return vmin, vmax
    return float(np.nanmin(data)), float(np.nanmax(data))


def _province_labels_for_ids(ids: List[int]) -> List[str]:
    labels = []
    for province_id in ids:
        try:
            key = int(province_id)
        except (TypeError, ValueError):
            labels.append(str(province_id))
            continue
        labels.append(PROVINCE_NAME_BY_ID.get(key, str(province_id)))
    return labels


def _pretty_perturbation_label(name: str) -> str:
    label_map = {
        "P1_10pct": "P1 (10%)",
        "P2_10pct": "P2 (10%)",
        "P3_hybrid": "P3 (hybrid)",
    }
    return label_map.get(name, str(name))


def _short_scenario_name(name: str) -> str:
    label_map = {
        "delta_P1": "P1",
        "delta_P2": "P2",
        "delta_P3": "P3",
    }
    return label_map.get(name, name)


def make_indicator_assets(df: pd.DataFrame, dirs: Dict[str, str]) -> None:
    check_cols(df, [ID_COL, T_COL, Y_COL])

    d = df[[ID_COL, T_COL, Y_COL]].dropna().copy()
    d[T_COL] = d[T_COL].astype(int)

    desc_all = d[Y_COL].describe(percentiles=[0.25, 0.5, 0.75]).to_frame("overall")
    desc_by_year = d.groupby(T_COL)[Y_COL].describe(percentiles=[0.25, 0.5, 0.75])
    with pd.ExcelWriter(os.path.join(dirs["tab"], "Table_4_1_descriptive_stats.xlsx")) as writer:
        desc_all.to_excel(writer, sheet_name="overall")
        desc_by_year.to_excel(writer, sheet_name="by_year")

    agg = d.groupby(T_COL)[Y_COL].agg(
        mean="mean",
        median="median",
        q25=lambda x: np.quantile(x, 0.25),
        q75=lambda x: np.quantile(x, 0.75),
        n="count",
    ).reset_index()

    x = agg[T_COL].to_numpy()
    mean = agg["mean"].to_numpy()
    median = agg["median"].to_numpy()
    q25 = agg["q25"].to_numpy()
    q75 = agg["q75"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(x, mean, label="Mean")
    ax.plot(x, median, label="Median")
    ax.fill_between(x, q25, q75, alpha=0.20, label="IQR (25–75%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Eco-efficiency (Super-SBM score)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_1_trend_mean_median_IQR.{FIG_FMT}"))

    pivot = d.pivot_table(index=ID_COL, columns=T_COL, values=Y_COL, aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    data = pivot.values.astype(float)
    vmin, vmax = _get_heatmap_limits(data)
    n_rows, n_cols = data.shape
    years = pivot.columns.astype(int).to_list()
    provinces = _province_labels_for_ids(pivot.index.to_list())

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=HEATMAP_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )

    max_labels = 15
    step = max(1, int(np.ceil(n_cols / max_labels)))
    xticks = np.arange(0, n_cols, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(years[i]) for i in xticks], rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(provinces, fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Eco-efficiency (Super-SBM score)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.subplots_adjust(left=0.28, bottom=0.15, right=0.96, top=0.92)
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_1_heatmap_province_year.{FIG_FMT}"))

    ineq = d.groupby(T_COL)[Y_COL].apply(
        lambda x: pd.Series(
            {
                "Gini": gini(x.values),
                "TheilT": theil_t(x.values),
                "N": x.size,
                "Mean": float(np.mean(x.values)),
                "SD": float(np.std(x.values, ddof=1)) if x.size > 1 else np.nan,
                "Min": float(np.min(x.values)),
                "Q25": float(np.quantile(x.values, 0.25)),
                "Median": float(np.quantile(x.values, 0.50)),
                "Q75": float(np.quantile(x.values, 0.75)),
                "Max": float(np.max(x.values)),
                "IQR": float(np.quantile(x.values, 0.75) - np.quantile(x.values, 0.25)),
            }
        )
    ).reset_index()
    ineq.to_excel(os.path.join(dirs["tab"], "Table_4_2_inequality_by_year.xlsx"), index=False)

    years_sorted = sorted(d[T_COL].unique())
    pers = []
    for gap in [5, 10]:
        for y in years_sorted:
            y2 = y + gap
            if y2 not in years_sorted:
                continue
            a = d[d[T_COL] == y].set_index(ID_COL)[Y_COL]
            b = d[d[T_COL] == y2].set_index(ID_COL)[Y_COL]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            pers.append(
                {
                    "gap": gap,
                    "year": y,
                    "year2": y2,
                    "spearman_rank_corr": spearman_rank_corr(a.loc[common], b.loc[common]),
                    "n": len(common),
                }
            )
    pd.DataFrame(pers).to_excel(os.path.join(dirs["tab"], "Table_4_2_rank_persistence.xlsx"), index=False)


def _read_scenario_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Section 4.3 scenario input is missing:\n"
            f"  {path}\n\n"
            "Generate this file from the main manuscript pipeline using the SAME final surrogate "
            "(e.g., VWLB) used in Section 3.2, with ALL 2023 provinces included."
        )
    df = pd.read_excel(path, sheet_name="scenario_outputs")
    expected = [
        "ID", "Province", "Year", "SuperSBM_observed", "yhat_base",
        "yhat_S1_input10", "delta_S1_input10",
        "yhat_S2_allsource10", "delta_S2_allsource10",
        "yhat_S3_upgrade_mix", "delta_S3_upgrade_mix",
    ]
    check_cols(df, expected)
    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    return df


def _scenario_to_long(df_scn: pd.DataFrame, year: int) -> pd.DataFrame:
    base = df_scn[df_scn["Year"] == year].copy()
    if base.empty:
        raise ValueError(f"No scenario rows found for year={year} in {SCENARIO_INPUT_PATH}")

    rows = []
    mapping = [
        ("P1_10pct", "yhat_S1_input10", "delta_S1_input10"),
        ("P2_10pct", "yhat_S2_allsource10", "delta_S2_allsource10"),
        ("P3_hybrid", "yhat_S3_upgrade_mix", "delta_S3_upgrade_mix"),
    ]
    for scenario_name, yhat_col, delta_col in mapping:
        tmp = base[["ID", "Province", "Year", "SuperSBM_observed", "yhat_base", yhat_col, delta_col]].copy()
        tmp = tmp.rename(columns={yhat_col: "y_pred_scn_mean", delta_col: "delta_mean"})
        tmp["scenario"] = scenario_name
        tmp["y_pred_base_mean"] = tmp["yhat_base"]
        tmp["y_true"] = tmp["SuperSBM_observed"]
        tmp = tmp[["ID", "Province", "Year", "scenario", "y_pred_base_mean", "y_pred_scn_mean", "delta_mean", "y_true"]]
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out["scenario"] = pd.Categorical(
        out["scenario"], categories=["P1_10pct", "P2_10pct", "P3_hybrid"], ordered=True
    )
    out = out.sort_values(["scenario", "Province"]).reset_index(drop=True)
    return out


def _make_perturbation_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in long_df.groupby("scenario", sort=False):
        delta = g["delta_mean"].astype(float).values
        rows.append(
            {
                "scenario": str(name),
                "scenario_pretty": _pretty_perturbation_label(str(name)),
                "n_provinces": int(len(delta)),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "min_delta": float(np.min(delta)),
                "max_delta": float(np.max(delta)),
                "share_delta_gt_0": float(np.mean(delta > 0)),
                "share_delta_ge_0": float(np.mean(delta >= 0)),
            }
        )
    return pd.DataFrame(rows)


def _make_top_bottom_10pct(long_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    frames = []
    for scenario in ["P1_10pct", "P2_10pct", "P3_hybrid"]:
        g = long_df[long_df["scenario"] == scenario].copy()
        g = g[["Province", "delta_mean"]].sort_values("delta_mean", ascending=False).reset_index(drop=True)

        top = g.head(top_n).copy()
        top["group"] = "Top"

        bottom = g.tail(top_n).copy().sort_values("delta_mean", ascending=True).reset_index(drop=True)
        bottom["group"] = "Bottom"

        tb = pd.concat([top, bottom], ignore_index=True)
        tb.insert(0, "scenario", scenario)
        tb.insert(1, "scenario_pretty", _pretty_perturbation_label(scenario))
        frames.append(tb)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"delta_mean": "delta"})
    return out[["scenario", "scenario_pretty", "group", "Province", "delta"]]


def _build_typology(df: pd.DataFrame, long_df: pd.DataFrame, df_scn: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    key = "P1_10pct"
    resp = long_df[long_df["scenario"] == key].set_index("ID")["delta_mean"]
    base_level = df.groupby(ID_COL)[Y_COL].mean()

    typ = pd.DataFrame({"base_level": base_level, "response": resp}).dropna()
    base_cut = float(typ["base_level"].median())
    resp_cut = float(typ["response"].median())

    typ["quadrant"] = np.select(
        [
            (typ["base_level"] >= base_cut) & (typ["response"] >= resp_cut),
            (typ["base_level"] >= base_cut) & (typ["response"] < resp_cut),
            (typ["base_level"] < base_cut) & (typ["response"] >= resp_cut),
            (typ["base_level"] < base_cut) & (typ["response"] < resp_cut),
        ],
        [
            "High base / High response",
            "High base / Low response",
            "Low base / High response",
            "Low base / Low response",
        ],
        default="NA",
    )

    id_to_prov = df_scn[["ID", "Province"]].drop_duplicates().set_index("ID")["Province"].to_dict()
    typ = typ.reset_index().rename(columns={"index": "ID"})
    typ["Province"] = typ["ID"].map(id_to_prov)
    typ = typ[["ID", "Province", "base_level", "response", "quadrant"]]
    return typ, base_cut, resp_cut


def _make_story_profile(
    df: pd.DataFrame,
    long_df: pd.DataFrame,
    typ_df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    current = df[df[T_COL] == year].copy()
    check_cols(current, [ID_COL] + FEATURES + [Y_COL])

    current = current[[ID_COL] + FEATURES + [Y_COL]].copy()
    current = current.rename(columns={Y_COL: "efficiency_current_year"})

    for col in STORY_FEATURES:
        current[f"{col}_pct"] = current[col].rank(pct=True, method="average")

    current["material_input_burden_pct"] = current[["AFA_pct", "PU_pct", "ADY_pct", "PFU_pct"]].mean(axis=1)
    current["all_source_activity_pct"] = current[["AFA_pct", "PU_pct", "ADY_pct", "PFU_pct", "EIA_pct", "CS_pct"]].mean(axis=1)

    long_run = (
        df.groupby(ID_COL)[Y_COL]
        .mean()
        .rename("long_run_mean_efficiency")
        .reset_index()
    )

    wide = (
        long_df.pivot_table(index=["ID", "Province"], columns="scenario", values="delta_mean", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={
            "P1_10pct": "delta_P1",
            "P2_10pct": "delta_P2",
            "P3_hybrid": "delta_P3",
        })
    )

    base_info = (
        long_df[["ID", "y_true", "y_pred_base_mean"]]
        .drop_duplicates(subset=["ID"])
        .rename(columns={"y_true": "y_true_2023", "y_pred_base_mean": "yhat_base_2023"})
    )

    wide = wide.merge(base_info, on="ID", how="left")

    for delta_col in ["delta_P1", "delta_P2", "delta_P3"]:
        wide[f"rank_{delta_col}"] = wide[delta_col].rank(ascending=False, method="min")

    wide["mean_delta_3scn"] = wide[["delta_P1", "delta_P2", "delta_P3"]].mean(axis=1)
    wide["max_delta"] = wide[["delta_P1", "delta_P2", "delta_P3"]].max(axis=1)
    wide["min_delta"] = wide[["delta_P1", "delta_P2", "delta_P3"]].min(axis=1)
    wide["response_spread"] = wide["max_delta"] - wide["min_delta"]
    wide["P2_minus_P1"] = wide["delta_P2"] - wide["delta_P1"]
    wide["P3_minus_P1"] = wide["delta_P3"] - wide["delta_P1"]
    wide["top5_count"] = (
        (wide["rank_delta_P1"] <= 5).astype(int)
        + (wide["rank_delta_P2"] <= 5).astype(int)
        + (wide["rank_delta_P3"] <= 5).astype(int)
    )

    dom = wide[["delta_P1", "delta_P2", "delta_P3"]].idxmax(axis=1)
    wide["dominant_scenario_raw"] = dom
    wide["dominant_scenario"] = dom.map(_short_scenario_name)

    out = (
        wide.merge(long_run, on="ID", how="left")
        .merge(current, on="ID", how="left")
        .merge(typ_df[["ID", "quadrant"]].rename(columns={"quadrant": "quadrant_P1"}), on="ID", how="left")
    )

    order_cols = [
        "ID", "Province", "quadrant_P1",
        "long_run_mean_efficiency", "efficiency_current_year", "y_true_2023", "yhat_base_2023",
        "delta_P1", "delta_P2", "delta_P3",
        "rank_delta_P1", "rank_delta_P2", "rank_delta_P3",
        "mean_delta_3scn", "max_delta", "min_delta", "response_spread",
        "P2_minus_P1", "P3_minus_P1", "top5_count", "dominant_scenario",
        "AFA", "PU", "ADY", "PFU", "EIA", "CS", "GAO", "CEA", "NRP", "TPAM",
        "AFA_pct", "PU_pct", "ADY_pct", "PFU_pct", "EIA_pct", "CS_pct", "GAO_pct", "CEA_pct", "NRP_pct", "TPAM_pct",
        "material_input_burden_pct", "all_source_activity_pct",
    ]
    keep = [c for c in order_cols if c in out.columns]
    out = out[keep].sort_values("mean_delta_3scn", ascending=False).reset_index(drop=True)
    return out


def _selection_reason_priority(row: pd.Series) -> str:
    return (
        f"Low-base/high-response under P1; top5_count={int(row['top5_count'])}; "
        f"mean_delta_3scn={row['mean_delta_3scn']:.6f}; min_delta={row['min_delta']:.6f}."
    )


def _selection_reason_direction(row: pd.Series) -> str:
    return (
        f"Direction-sensitive profile; dominant={row['dominant_scenario']}; "
        f"response_spread={row['response_spread']:.6f}; P2_minus_P1={row['P2_minus_P1']:.6f}; "
        f"P3_minus_P1={row['P3_minus_P1']:.6f}."
    )


def _selection_reason_forced(row: pd.Series, case_type: str) -> str:
    return (
        f"Fixed manuscript case for discussion ({case_type}); quadrant_P1={row['quadrant_P1']}; "
        f"dominant={row['dominant_scenario']}; mean_delta_3scn={row['mean_delta_3scn']:.6f}; "
        f"response_spread={row['response_spread']:.6f}."
    )


def _selection_reason_low(row: pd.Series) -> str:
    return (
        f"Low-base/low-response under P1; mean_delta_3scn={row['mean_delta_3scn']:.6f}; "
        f"max_delta={row['max_delta']:.6f}."
    )


def _select_representative_cases(profile: pd.DataFrame) -> pd.DataFrame:
    prof = profile.copy()

    case_order = ["Priority follow-up case", "Direction-sensitive case", "Low-return case"]

    if REPRESENTATIVE_CASE_MODE == "fixed_manuscript_cases":
        frames = []
        missing = []
        for case_type in case_order:
            province = FORCED_REPRESENTATIVE_CASES.get(case_type)
            g = prof[prof["Province"].astype(str) == str(province)].copy()
            if g.empty:
                missing.append(str(province))
                continue
            g = g.head(1).copy()
            g["case_type"] = case_type
            g["selection_reason"] = g.apply(lambda r: _selection_reason_forced(r, case_type), axis=1)
            frames.append(g)

        if missing:
            raise ValueError(
                "The following forced representative provinces were not found in the story-profile table: "
                f"{missing}. Check province names and scenario input coverage."
            )

        out = pd.concat(frames, ignore_index=True)
        out["case_type"] = pd.Categorical(out["case_type"], categories=case_order, ordered=True)
        out = out.sort_values("case_type").reset_index(drop=True)
        return out

    # Auto mode fallback
    case1 = (
        prof[prof["quadrant_P1"] == "Low base / High response"]
        .sort_values(["top5_count", "min_delta", "mean_delta_3scn"], ascending=[False, False, False])
        .head(1)
        .copy()
    )
    case1["case_type"] = "Priority follow-up case"
    case1["selection_reason"] = case1.apply(_selection_reason_priority, axis=1)

    used = set(case1["Province"].astype(str).tolist())

    cand2 = prof[~prof["Province"].astype(str).isin(used)].copy()
    pref2 = cand2[cand2["dominant_scenario"] != "P1"]
    if pref2.empty:
        pref2 = cand2.copy()
    case2 = (
        pref2.sort_values(["response_spread", "mean_delta_3scn"], ascending=[False, False])
        .head(1)
        .copy()
    )
    case2["case_type"] = "Direction-sensitive case"
    case2["selection_reason"] = case2.apply(_selection_reason_direction, axis=1)

    used |= set(case2["Province"].astype(str).tolist())

    case3 = (
        prof[
            (prof["quadrant_P1"] == "Low base / Low response")
            & (~prof["Province"].astype(str).isin(used))
        ]
        .sort_values(["mean_delta_3scn", "max_delta"], ascending=[True, True])
        .head(1)
        .copy()
    )
    case3["case_type"] = "Low-return case"
    case3["selection_reason"] = case3.apply(_selection_reason_low, axis=1)

    out = pd.concat([case1, case2, case3], ignore_index=True)
    out["case_type"] = pd.Categorical(out["case_type"], categories=case_order, ordered=True)
    out = out.sort_values("case_type").reset_index(drop=True)
    return out


def _plot_representative_case_profiles(rep_df: pd.DataFrame, dirs: Dict[str, str], year: int) -> None:
    if rep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x_labels = ["P1 (10%)", "P2 (10%)", "P3 (hybrid)"]
    x = np.arange(len(x_labels))

    for _, row in rep_df.iterrows():
        y = [row["delta_P1"], row["delta_P2"], row["delta_P3"]]
        ax.plot(x, y, marker="o", label=f"{row['Province']} — {row['case_type']}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Predicted Δ eco-efficiency")
    ax.set_xlabel("Perturbation scenario")
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_3_representative_case_profiles_{year}.{FIG_FMT}"))


def make_perturbation_assets_from_pipeline(df: pd.DataFrame, dirs: Dict[str, str]) -> None:
    check_cols(df, [ID_COL, T_COL, Y_COL])

    years = sorted(df[T_COL].astype(int).unique())
    year = int(years[-1]) if PERTURBATION_YEAR is None else int(PERTURBATION_YEAR)

    df_scn = _read_scenario_input(SCENARIO_INPUT_PATH)
    long_df = _scenario_to_long(df_scn, year=year)

    n_prov = long_df["Province"].nunique()
    if n_prov < 20:
        raise ValueError(
            f"Scenario file appears to cover only {n_prov} provinces, which is too small for Figure 4 / Table 5. "
            "Use the full-coverage 2023 scenario export from the main pipeline, not the 4-province demo or the 12-province recomputation subset."
        )

    long_df.to_csv(os.path.join(dirs["tab"], f"Table_4_3_perturbation_pred_long_{year}.csv"), index=False)

    summary = _make_perturbation_summary(long_df)
    summary.to_excel(os.path.join(dirs["tab"], f"Table_4_3_perturbation_summary_{year}.xlsx"), index=False)

    top_bottom = _make_top_bottom_10pct(long_df, top_n=5)
    top_bottom.to_excel(os.path.join(dirs["tab"], f"Table_4_3_top_bottom_10pct_{year}.xlsx"), index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    dmin = float(np.nanmin(long_df["delta_mean"].values))
    dmax = float(np.nanmax(long_df["delta_mean"].values))
    if np.isclose(dmin, dmax):
        dmin -= 1e-6
        dmax += 1e-6
    bins = np.linspace(dmin, dmax, 28)

    plot_order = ["P1_10pct", "P2_10pct", "P3_hybrid"]
    for name in plot_order:
        g = long_df[long_df["scenario"] == name]
        ax.hist(
            g["delta_mean"].values,
            bins=bins,
            alpha=0.25,
            label=_pretty_perturbation_label(name),
            histtype="stepfilled",
            edgecolor="none",
        )

    ax.set_xlabel("Δ eco-efficiency (perturbation − baseline)")
    ax.set_ylabel("Count (provinces)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_3_perturbation_distribution_{year}.{FIG_FMT}"))

    with pd.ExcelWriter(os.path.join(dirs["tab"], f"Table_4_3_province_ranking_{year}.xlsx")) as writer:
        for name in plot_order:
            g = long_df[long_df["scenario"] == name][["Province", "delta_mean"]].copy()
            g = g.sort_values("delta_mean", ascending=False).reset_index(drop=True)
            g.to_excel(writer, sheet_name=name[:31], index=False)

    typ_df, base_cut, resp_cut = _build_typology(df, long_df, df_scn)
    typ_df.to_excel(os.path.join(dirs["tab"], f"Table_4_3_typology_P1_10pct_{year}.xlsx"), index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for q, g in typ_df.groupby("quadrant"):
        ax.scatter(g["base_level"], g["response"], alpha=0.85, label=q)

    ax.axvline(base_cut, linewidth=0.9, linestyle="--")
    ax.axhline(resp_cut, linewidth=0.9, linestyle="--")
    ax.set_xlabel("Long-run mean eco-efficiency")
    ax.set_ylabel("Response under P1 (10%)")
    ax.grid(linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_3_quadrant_typology_P1_10pct_{year}.{FIG_FMT}"))

    story_profile = _make_story_profile(df=df, long_df=long_df, typ_df=typ_df, year=year)
    story_profile.to_excel(os.path.join(dirs["tab"], f"Table_4_3_province_story_profile_{year}.xlsx"), index=False)

    representative_cases = _select_representative_cases(story_profile)
    representative_cases.to_excel(os.path.join(dirs["tab"], f"Table_4_3_representative_cases_{year}.xlsx"), index=False)

    _plot_representative_case_profiles(representative_cases, dirs=dirs, year=year)


def write_readme(dirs: Dict[str, str]) -> None:
    txt = f"""\
Output directory: {dirs['out']}

Section 4.1:
  - figures/Fig_4_1_trend_mean_median_IQR.tif
  - figures/Fig_4_1_heatmap_province_year.tif
  - tables/Table_4_1_descriptive_stats.xlsx

Section 4.2:
  - tables/Table_4_2_inequality_by_year.xlsx
  - tables/Table_4_2_rank_persistence.xlsx

Section 4.3:
  Input required:
    - {SCENARIO_INPUT_PATH}

  Core screening outputs:
    - figures/Fig_4_3_perturbation_distribution_<year>.tif
    - figures/Fig_4_3_quadrant_typology_P1_10pct_<year>.tif
    - tables/Table_4_3_perturbation_pred_long_<year>.csv
    - tables/Table_4_3_perturbation_summary_<year>.xlsx
    - tables/Table_4_3_top_bottom_10pct_<year>.xlsx
    - tables/Table_4_3_province_ranking_<year>.xlsx
    - tables/Table_4_3_typology_P1_10pct_<year>.xlsx

  Interpretation-layer outputs:
    - figures/Fig_4_3_representative_case_profiles_<year>.tif
    - tables/Table_4_3_province_story_profile_<year>.xlsx
    - tables/Table_4_3_representative_cases_<year>.xlsx

Notes:
- Section 4.3 does not retrain a new surrogate locally.
- Figure 4 / Table 5 are constructed from the same full-coverage scenario file exported from the main manuscript pipeline.
- The added story-profile layer is for management-oriented interpretation only and does not change the underlying surrogate outputs.
"""
    with open(os.path.join(dirs["out"], "README_results_story.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def main() -> None:
    set_seed(SEED)
    set_mpl_style()
    dirs = ensure_dirs(OUT_DIR)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}. Place the script and data file in the same directory.")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    check_cols(df, [ID_COL, T_COL, Y_COL] + FEATURES)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ID_COL, T_COL, Y_COL] + FEATURES).copy()
    df[T_COL] = df[T_COL].astype(int)

    make_indicator_assets(df, dirs)
    make_perturbation_assets_from_pipeline(df, dirs)

    write_readme(dirs)
    print(f"[OK] Results written to: {dirs['out']}")


if __name__ == "__main__":
    main()
