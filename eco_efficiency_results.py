# -*- coding: utf-8 -*-
"""
Generate descriptive figures, summary tables, and scenario-analysis assets
for the Ecological Informatics submission.

This file preserves the original calculations and file outputs.
Only comments, messages, and path settings were cleaned up.

Default input file: ./data.xlsx (Sheet1)
"""

import json
import os
import warnings
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


DATA_PATH = "./data.xlsx"
SHEET_NAME = "Sheet1"

ID_COL = "Province"
T_COL = "Year"
Y_COL = "efficiency"

FEATURES = ["TPAM", "EIA", "CS", "AFA", "PU", "ADY", "PFU", "NRP", "GAO", "CEA"]
CEA_ACTIVITY_VARS = ["AFA", "PU", "PFU", "ADY", "EIA", "CS"]

OUT_DIR = "results_story"
N_MODELS = 15
SEED = 42
SCENARIO_RS = (0.05, 0.10)
SCENARIO_YEAR = None

SAVE_DPI = 600
FIG_FMT = "tif"
HEATMAP_CMAP = "cividis"
ROBUST_Q = (0.02, 0.98)


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
    vmin = float(np.nanquantile(data, ROBUST_Q[0]))
    vmax = float(np.nanquantile(data, ROBUST_Q[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))

    n_rows, n_cols = data.shape
    years = pivot.columns.astype(int).to_list()
    provinces = pivot.index.to_list()

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
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
    ax.set_yticklabels(provinces)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Eco-efficiency (Super-SBM score)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.subplots_adjust(left=0.24, bottom=0.16, right=0.96, top=0.90)
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_1_heatmap_province_year.{FIG_FMT}"))

    ineq = d.groupby(T_COL)[Y_COL].apply(
        lambda x: pd.Series({"Gini": gini(x.values), "TheilT": theil_t(x.values), "N": x.size})
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


def build_models(seed: int, n_models: int) -> List[HistGradientBoostingRegressor]:
    mono = [(1 if f == "GAO" else -1) for f in FEATURES]
    models = []
    for k in range(n_models):
        rs = seed + 1000 + k
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=250,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            monotonic_cst=mono,
            random_state=rs,
        )
        models.append(model)
    return models


def fit_models(df: pd.DataFrame, models: List[HistGradientBoostingRegressor]) -> List[HistGradientBoostingRegressor]:
    X = df[FEATURES].astype(float)
    y = df[Y_COL].astype(float)
    for model in models:
        model.fit(X, y)
    return models


def predict_ens(models: List[HistGradientBoostingRegressor], X: pd.DataFrame) -> np.ndarray:
    preds = [model.predict(X) for model in models]
    return np.vstack(preds)


def fit_cea_reconstructor(df: pd.DataFrame) -> Dict[str, float]:
    check_cols(df, CEA_ACTIVITY_VARS + ["CEA"])
    X = df[CEA_ACTIVITY_VARS].astype(float).values
    y = df["CEA"].astype(float).values
    lr = LinearRegression(fit_intercept=False, positive=True)
    lr.fit(X, y)
    yhat = lr.predict(X)
    r2 = r2_score(y, yhat)
    coef = {v: float(c) for v, c in zip(CEA_ACTIVITY_VARS, lr.coef_)}
    coef["__r2__"] = float(r2)
    return coef


def recompute_cea(df: pd.DataFrame, coef_map: Dict[str, float]) -> pd.DataFrame:
    d = df.copy()
    ce = np.zeros(len(d), dtype=float)
    for v in CEA_ACTIVITY_VARS:
        ce += d[v].astype(float).values * float(coef_map[v])
    d["CEA"] = ce
    return d


def apply_scenario(base: pd.DataFrame, scenario: str, r: float, coef_map: Dict[str, float]) -> pd.DataFrame:
    d = base.copy()

    if scenario == "S1":
        for col in ["AFA", "PU", "PFU", "ADY", "TPAM", "NRP"]:
            d[col] = d[col] * (1 - r)
        d = recompute_cea(d, coef_map)
    elif scenario == "S2":
        d["GAO"] = d["GAO"] * (1 + r)
    elif scenario == "S3":
        for col in ["AFA", "PU", "PFU", "ADY"]:
            d[col] = d[col] * (1 - r)
        d["GAO"] = d["GAO"] * (1 + 0.6 * r)
        d = recompute_cea(d, coef_map)
    else:
        raise ValueError("scenario must be one of {S1, S2, S3}")

    check_cols(d, FEATURES)
    return d


def _pretty_scenario_label(name: str) -> str:
    try:
        scn, pct = name.split("_")
        pct_num = pct.replace("pct", "")
        return f"{scn} ({pct_num}%)"
    except Exception:
        return name


def make_scenario_assets(
    df: pd.DataFrame,
    models: List[HistGradientBoostingRegressor],
    dirs: Dict[str, str],
) -> None:
    check_cols(df, [ID_COL, T_COL] + FEATURES)

    years = sorted(df[T_COL].astype(int).unique())
    year = int(years[-1]) if SCENARIO_YEAR is None else int(SCENARIO_YEAR)

    base = df[df[T_COL].astype(int) == year].copy()
    if base.empty:
        raise ValueError(f"No data found for year {year}. Available years: {years}")

    coef_map = fit_cea_reconstructor(df)
    r2 = coef_map.get("__r2__", np.nan)
    coef_map2 = {k: v for k, v in coef_map.items() if k != "__r2__"}

    with open(os.path.join(dirs["tab"], "CEA_internal_reconstructor.json"), "w", encoding="utf-8") as f:
        json.dump({"r2": float(r2), **coef_map2}, f, ensure_ascii=False, indent=2)

    if r2 < 0.99:
        warnings.warn(
            f"CEA linear reconstructor fit has R2={r2:.4f}. Check whether CEA is defined as a linear accounting identity."
        )

    X_base = base[FEATURES].astype(float)
    pred_base = predict_ens(models, X_base)
    base_mean = pred_base.mean(axis=0)

    rows = []
    for scn in ["S1", "S2", "S3"]:
        for r in SCENARIO_RS:
            d_scn = apply_scenario(base, scn, r, coef_map2)
            X_scn = d_scn[FEATURES].astype(float)

            pred_scn = predict_ens(models, X_scn)
            delta = pred_scn - pred_base

            tmp = pd.DataFrame(
                {
                    ID_COL: base[ID_COL].values,
                    T_COL: base[T_COL].values,
                    "scenario": f"{scn}_{int(r * 100)}pct",
                    "y_pred_base_mean": base_mean,
                    "y_pred_scn_mean": pred_scn.mean(axis=0),
                    "delta_mean": delta.mean(axis=0),
                    "delta_p025": np.quantile(delta, 0.025, axis=0),
                    "delta_p975": np.quantile(delta, 0.975, axis=0),
                    "y_true": base[Y_COL].values,
                }
            )
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(dirs["tab"], f"Table_4_3_scenario_pred_long_{year}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    dmin = float(np.nanmin(out["delta_mean"].values))
    dmax = float(np.nanmax(out["delta_mean"].values))
    bins = np.linspace(dmin, dmax, 28)

    desired_order = ["S1_10pct", "S2_10pct", "S3_10pct", "S1_5pct", "S2_5pct", "S3_5pct"]
    available = set(out["scenario"].unique().tolist())
    plot_order = [s for s in desired_order if s in available] + [s for s in sorted(available) if s not in desired_order]

    for name in plot_order:
        g = out[out["scenario"] == name]
        ax.hist(
            g["delta_mean"].values,
            bins=bins,
            alpha=0.25,
            label=_pretty_scenario_label(name),
            histtype="stepfilled",
            edgecolor="none",
        )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_3_delta_distribution_{year}.{FIG_FMT}"))

    with pd.ExcelWriter(os.path.join(dirs["tab"], f"Table_4_3_province_ranking_{year}.xlsx")) as writer:
        for name, g in out.groupby("scenario"):
            rank = g.set_index(ID_COL)[["delta_mean", "delta_p025", "delta_p975"]].sort_values(
                "delta_mean",
                ascending=False,
            )
            rank.to_excel(writer, sheet_name=name[:31])

    key = "S1_10pct" if "S1_10pct" in out["scenario"].unique() else sorted(out["scenario"].unique())[0]
    resp = out[out["scenario"] == key].set_index(ID_COL)["delta_mean"]
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

    typ.to_excel(os.path.join(dirs["tab"], f"Table_4_3_typology_{key}_{year}.xlsx"))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for q, g in typ.groupby("quadrant"):
        ax.scatter(g["base_level"], g["response"], alpha=0.85, label=q)

    ax.axvline(base_cut, linewidth=0.9, linestyle="--")
    ax.axhline(resp_cut, linewidth=0.9, linestyle="--")
    ax.grid(linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["fig"], f"Fig_4_3_quadrant_typology_{key}_{year}.{FIG_FMT}"))


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
  - figures/Fig_4_3_delta_distribution_<year>.tif
  - figures/Fig_4_3_quadrant_typology_S1_10pct_<year>.tif
  - tables/Table_4_3_scenario_pred_long_<year>.csv
  - tables/Table_4_3_province_ranking_<year>.xlsx
  - tables/Table_4_3_typology_S1_10pct_<year>.xlsx

Notes:
- tables/CEA_internal_reconstructor.json is used only to recompute CEA consistently under scenario perturbations.
- It is not a paper-level emissions coefficient file.
- Figures follow the original journal-style output settings.
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

    models = build_models(SEED, N_MODELS)
    fit_models(df, models)
    make_scenario_assets(df, models, dirs)

    write_readme(dirs)
    print(f"[OK] Results written to: {dirs['out']}")


if __name__ == "__main__":
    main()
