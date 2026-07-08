from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.xlsx"
TABLES_DIR = ROOT / "tables"

ID_COL = "ID"
YEAR_COL = "Year"
Y_COL = "efficiency"

FEATURES = ["TPAM", "EIA", "CS", "AFA", "PU", "ADY", "PFU", "NRP", "GAO", "CEA"]
INPUT_COLS = ["TPAM", "EIA", "CS", "AFA", "PU", "ADY", "PFU", "NRP"]
GOOD_COLS = ["GAO"]
BAD_COLS = ["CEA"]
CEA_COMPONENT_COLS = ["AFA", "PU", "ADY", "PFU", "EIA", "CS"]

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

MONOTONE_BY_FEATURE = {
    "TPAM": -1,
    "EIA": -1,
    "CS": -1,
    "AFA": -1,
    "PU": -1,
    "ADY": -1,
    "PFU": -1,
    "NRP": -1,
    "GAO": 1,
    "CEA": -1,
}

CEA_ACCOUNTING_COEFS_DATA_UNITS = {
    # Data units follow the modeling table:
    # AFA, PU, ADY, PFU: 10,000 tons; EIA, CS: 1,000 hectares.
    # Target CEA unit: 10,000 tons C.
    "AFA": 0.8956,
    "PU": 4.9341,
    "PFU": 5.18,
    "ADY": 0.5927,
    "EIA": 20.476 * 1000.0 / 1e7,
    "CS": 312.60 * 10.0 / 1e7,
}

SCENARIOS = {
    "S1_input10": {
        "label": "S1: -10% (AFA, PU, ADY, PFU; CEA updated)",
        "shock_inputs": {"AFA": 0.90, "PU": 0.90, "ADY": 0.90, "PFU": 0.90},
        "shock_outputs": {},
        "update_cea": True,
    },
    "S2_allsource10": {
        "label": "S2: -10% (AFA, PU, ADY, PFU, EIA, CS; CEA updated)",
        "shock_inputs": {
            "AFA": 0.90,
            "PU": 0.90,
            "ADY": 0.90,
            "PFU": 0.90,
            "EIA": 0.90,
            "CS": 0.90,
        },
        "shock_outputs": {},
        "update_cea": True,
    },
    "S3_upgrade_mix": {
        "label": "S3: +5% (GAO) & -10% (AFA, PU, ADY, PFU; CEA updated)",
        "shock_inputs": {"AFA": 0.90, "PU": 0.90, "ADY": 0.90, "PFU": 0.90},
        "shock_outputs": {"GAO": 1.05},
        "update_cea": True,
    },
}


def load_modeling_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
    need = [ID_COL, YEAR_COL, Y_COL] + FEATURES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data.xlsx: {missing}")
    df = df[need].copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Province"] = df[ID_COL].astype(int).map(
        lambda x: PROVINCES_BY_ID[x - 1] if 1 <= x <= len(PROVINCES_BY_ID) else f"ID_{x}"
    )
    df["row_key"] = np.arange(len(df), dtype=int)
    return df.dropna(subset=need).sort_values([YEAR_COL, ID_COL]).reset_index(drop=True)


def safe_corr(a, b, method: str = "pearson") -> float:
    s1 = pd.Series(a, dtype="float64")
    s2 = pd.Series(b, dtype="float64")
    mask = s1.notna() & s2.notna()
    if mask.sum() < 3:
        return np.nan
    return float(s1[mask].corr(s2[mask], method=method))


def metrics_row(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_xgb(features: list[str], random_state: int = 42) -> XGBRegressor:
    constraints = tuple(MONOTONE_BY_FEATURE[f] for f in features)
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        monotone_constraints=constraints,
    )


def fit_predict_xgb(df: pd.DataFrame, features: list[str], train_idx, test_idx, seed: int) -> np.ndarray:
    x_train = df.loc[train_idx, features].to_numpy(dtype=float)
    y_train = df.loc[train_idx, Y_COL].to_numpy(dtype=float)
    x_test = df.loc[test_idx, features].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = make_xgb(features, random_state=seed)
    model.fit(x_train_s, y_train)
    return model.predict(x_test_s)


def run_surrogate_predictor_ablation(df: pd.DataFrame) -> Path:
    out_path = TABLES_DIR / "Surrogate_predictor_ablation_revision.xlsx"
    feature_sets = {
        "full_predictors": FEATURES,
        "without_CEA": [c for c in FEATURES if c != "CEA"],
        "without_CEA_and_CEA_components": [
            c for c in FEATURES if c != "CEA" and c not in CEA_COMPONENT_COLS
        ],
    }

    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for feature_set, feats in feature_sets.items():
        for fold, (tr_pos, te_pos) in enumerate(kf.split(df), start=1):
            pred = fit_predict_xgb(df, feats, df.index[tr_pos], df.index[te_pos], seed=4200 + fold)
            m = metrics_row(df.loc[df.index[te_pos], Y_COL], pred)
            rows.append(
                {
                    "scheme": "random_5fold",
                    "split_id": fold,
                    "feature_set": feature_set,
                    "n_features": len(feats),
                    "train_n": int(len(tr_pos)),
                    "test_n": int(len(te_pos)),
                    "train_year_max": np.nan,
                    "test_years": "mixed",
                    **m,
                }
            )

    years = sorted(int(y) for y in df[YEAR_COL].unique())
    min_year, max_year = min(years), max(years)
    split_id = 0
    for train_end in range(min_year + 9, max_year - 3):
        val_year = train_end + 1
        test_years = [train_end + 2, train_end + 3, train_end + 4]
        if not all(y in years for y in test_years):
            continue
        tr_idx = df.index[df[YEAR_COL] <= train_end]
        te_idx = df.index[df[YEAR_COL].isin(test_years)]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        split_id += 1
        for feature_set, feats in feature_sets.items():
            pred = fit_predict_xgb(df, feats, tr_idx, te_idx, seed=5200 + split_id)
            m = metrics_row(df.loc[te_idx, Y_COL], pred)
            rows.append(
                {
                    "scheme": "time_forward_rolling",
                    "split_id": split_id,
                    "feature_set": feature_set,
                    "n_features": len(feats),
                    "train_n": int(len(tr_idx)),
                    "test_n": int(len(te_idx)),
                    "train_year_max": int(train_end),
                    "validation_year_gap": int(val_year),
                    "test_years": ",".join(str(y) for y in test_years),
                    **m,
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["scheme", "feature_set"], as_index=False)
        .agg(
            n_splits=("split_id", "count"),
            n_features=("n_features", "first"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .sort_values(["scheme", "R2_mean"], ascending=[True, False])
    )
    readme = pd.DataFrame(
        {
            "Notes": [
                "Reviewer-driven surrogate ablation for circularity concerns.",
                "Model is monotone-constrained XGBoost using the same predictor signs as the main pipeline.",
                "without_CEA removes only the aggregate carbon-emission variable.",
                "without_CEA_and_CEA_components removes CEA plus AFA, PU, ADY, PFU, EIA, and CS.",
                "The analysis is diagnostic: it evaluates label approximation under reduced predictor sets, not causal mechanisms.",
            ]
        }
    )
    fs = pd.DataFrame(
        [{"feature_set": k, "features": ", ".join(v), "n_features": len(v)} for k, v in feature_sets.items()]
    )
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="split_detail", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        fs.to_excel(writer, sheet_name="feature_sets", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)
    return out_path


def _safe_pos(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    return np.where(x <= eps, eps, x)


def _solve_linprog(c, A_eq, b_eq, bounds, msg_prefix="LP"):
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"{msg_prefix} failed: status={res.status}, msg={res.message}")
    return res


def _sbm_undesirable_vrs_standard(x0, y0, b0, X, Y, B, eps=1e-9):
    x0 = _safe_pos(x0, eps)
    y0 = _safe_pos(y0, eps)
    b0 = _safe_pos(b0, eps)
    m, s, h = len(x0), len(y0), len(b0)
    n = X.shape[1]
    dim = n + m + s + h + 1
    idx_l = slice(0, n)
    idx_sm = slice(n, n + m)
    idx_sp = slice(n + m, n + m + s)
    idx_sb = slice(n + m + s, n + m + s + h)
    idx_t = dim - 1
    c = np.zeros(dim)
    c[idx_t] = 1.0
    c[idx_sm] = -(1.0 / m) * (1.0 / x0)
    A_eq, b_eq = [], []
    for i in range(m):
        row = np.zeros(dim)
        row[idx_l] = X[i, :]
        row[n + i] = 1.0
        row[idx_t] = -x0[i]
        A_eq.append(row)
        b_eq.append(0.0)
    for r in range(s):
        row = np.zeros(dim)
        row[idx_l] = Y[r, :]
        row[n + m + r] = -1.0
        row[idx_t] = -y0[r]
        A_eq.append(row)
        b_eq.append(0.0)
    for k in range(h):
        row = np.zeros(dim)
        row[idx_l] = B[k, :]
        row[n + m + s + k] = 1.0
        row[idx_t] = -b0[k]
        A_eq.append(row)
        b_eq.append(0.0)
    row = np.zeros(dim)
    row[idx_l] = 1.0
    row[idx_t] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)
    row = np.zeros(dim)
    row[idx_t] = 1.0
    row[idx_sp] = 1.0 / (s + h) * (1.0 / y0)
    row[idx_sb] = 1.0 / (s + h) * (1.0 / b0)
    A_eq.append(row)
    b_eq.append(1.0)
    return float(
        _solve_linprog(c, np.array(A_eq), np.array(b_eq), [(0, None)] * dim, "SBM-STD").fun
    )


def _sbm_undesirable_vrs_super(x0, y0, b0, X_wo, Y_wo, B_wo, eps=1e-9):
    x0 = _safe_pos(x0, eps)
    y0 = _safe_pos(y0, eps)
    b0 = _safe_pos(b0, eps)
    m, s, h = len(x0), len(y0), len(b0)
    n = X_wo.shape[1]
    dim = n + m + s + h + 1
    idx_l = slice(0, n)
    idx_sm = slice(n, n + m)
    idx_sp = slice(n + m, n + m + s)
    idx_sb = slice(n + m + s, n + m + s + h)
    idx_t = dim - 1
    c = np.zeros(dim)
    c[idx_t] = 1.0
    c[idx_sm] = +(1.0 / m) * (1.0 / x0)
    A_eq, b_eq = [], []
    for i in range(m):
        row = np.zeros(dim)
        row[idx_l] = X_wo[i, :]
        row[n + i] = 1.0
        row[idx_t] = -x0[i]
        A_eq.append(row)
        b_eq.append(0.0)
    for r in range(s):
        row = np.zeros(dim)
        row[idx_l] = Y_wo[r, :]
        row[n + m + r] = -1.0
        row[idx_t] = -y0[r]
        A_eq.append(row)
        b_eq.append(0.0)
    for k in range(h):
        row = np.zeros(dim)
        row[idx_l] = B_wo[k, :]
        row[n + m + s + k] = 1.0
        row[idx_t] = -b0[k]
        A_eq.append(row)
        b_eq.append(0.0)
    row = np.zeros(dim)
    row[idx_l] = 1.0
    row[idx_t] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)
    row = np.zeros(dim)
    row[idx_t] = 1.0
    row[idx_sp] = -(1.0 / (s + h)) * (1.0 / y0)
    row[idx_sb] = -(1.0 / (s + h)) * (1.0 / b0)
    A_eq.append(row)
    b_eq.append(1.0)
    return float(
        _solve_linprog(c, np.array(A_eq), np.array(b_eq), [(0, None)] * dim, "SBM-SUPER").fun
    )


def super_sbm_score_one(reference: pd.DataFrame, row_pos: int, tol=1e-6, eps=1e-9) -> float:
    ref = reference.reset_index(drop=True)
    X = ref[INPUT_COLS].to_numpy(dtype=float).T
    Y = ref[GOOD_COLS].to_numpy(dtype=float).T
    B = ref[BAD_COLS].to_numpy(dtype=float).T
    x0, y0, b0 = X[:, row_pos], Y[:, row_pos], B[:, row_pos]
    rho = _sbm_undesirable_vrs_standard(x0, y0, b0, X, Y, B, eps=eps)
    if rho < 1.0 - tol:
        return rho
    mask = np.ones(X.shape[1], dtype=bool)
    mask[row_pos] = False
    return _sbm_undesirable_vrs_super(x0, y0, b0, X[:, mask], Y[:, mask], B[:, mask], eps=eps)


def run_alternative_frontier_robustness(df: pd.DataFrame) -> Path:
    out_path = TABLES_DIR / "Alternative_frontier_robustness_revision.xlsx"
    rows, errors = [], []
    years = sorted(int(y) for y in df[YEAR_COL].unique())
    year_min, year_max = min(years), max(years)

    for frontier_type in ["year_specific", "rolling_5yr_window"]:
        print(f"[frontier] computing {frontier_type}")
        for n_done, (_, target) in enumerate(df.iterrows(), start=1):
            year = int(target[YEAR_COL])
            if frontier_type == "year_specific":
                ref = df[df[YEAR_COL] == year].copy()
            else:
                lo, hi = max(year_min, year - 2), min(year_max, year + 2)
                ref = df[(df[YEAR_COL] >= lo) & (df[YEAR_COL] <= hi)].copy()
            ref = ref.reset_index(drop=True)
            hit = ref.index[ref["row_key"] == int(target["row_key"])].tolist()
            if not hit:
                raise RuntimeError("Target row was not found in its reference frontier.")
            try:
                alt = super_sbm_score_one(ref, int(hit[0]))
                status, err = "ok", ""
            except Exception as exc:
                alt = np.nan
                status, err = "failed", str(exc)
                errors.append(
                    {
                        "frontier_type": frontier_type,
                        "ID": int(target[ID_COL]),
                        "Province": target["Province"],
                        "Year": year,
                        "Error": err,
                    }
                )
            rows.append(
                {
                    "frontier_type": frontier_type,
                    "ID": int(target[ID_COL]),
                    "Province": target["Province"],
                    "Year": year,
                    "pooled_score": float(target[Y_COL]),
                    "alternative_score": alt,
                    "alt_minus_pooled": alt - float(target[Y_COL]) if np.isfinite(alt) else np.nan,
                    "abs_diff": abs(alt - float(target[Y_COL])) if np.isfinite(alt) else np.nan,
                    "reference_n": int(len(ref)),
                    "status": status,
                }
            )
            if n_done % 120 == 0:
                print(f"[frontier] {frontier_type}: {n_done}/{len(df)} rows")

    detail = pd.DataFrame(rows)
    valid = detail["alternative_score"].notna()
    detail["rank_pooled_year_common"] = np.nan
    detail["rank_alt_year_common"] = np.nan
    for (frontier, year), idx in detail[valid].groupby(["frontier_type", "Year"]).groups.items():
        gidx = list(idx)
        detail.loc[gidx, "rank_pooled_year_common"] = detail.loc[gidx, "pooled_score"].rank(
            ascending=False, method="average"
        )
        detail.loc[gidx, "rank_alt_year_common"] = detail.loc[gidx, "alternative_score"].rank(
            ascending=False, method="average"
        )
    detail["rank_diff_year_common"] = detail["rank_alt_year_common"] - detail["rank_pooled_year_common"]
    detail["abs_rank_diff_year_common"] = detail["rank_diff_year_common"].abs()

    overall_rows = []
    for frontier, g in detail.groupby("frontier_type"):
        ge = g[g["alternative_score"].notna()]
        overall_rows.append(
            {
                "frontier_type": frontier,
                "N_total": int(len(g)),
                "N_effective": int(len(ge)),
                "N_failed": int(len(g) - len(ge)),
                "pearson_score": safe_corr(ge["pooled_score"], ge["alternative_score"]),
                "spearman_score": safe_corr(ge["pooled_score"], ge["alternative_score"], "spearman"),
                "mean_abs_diff": float(ge["abs_diff"].mean()) if len(ge) else np.nan,
                "median_abs_diff": float(ge["abs_diff"].median()) if len(ge) else np.nan,
                "mean_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].mean())
                if len(ge)
                else np.nan,
                "median_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].median())
                if len(ge)
                else np.nan,
            }
        )
    overall = pd.DataFrame(overall_rows)

    year_rows = []
    for (frontier, year), g in detail.groupby(["frontier_type", "Year"]):
        ge = g[g["alternative_score"].notna()]
        year_rows.append(
            {
                "frontier_type": frontier,
                "Year": int(year),
                "N_total": int(len(g)),
                "N_effective": int(len(ge)),
                "N_failed": int(len(g) - len(ge)),
                "pearson_score": safe_corr(ge["pooled_score"], ge["alternative_score"]),
                "spearman_score": safe_corr(ge["pooled_score"], ge["alternative_score"], "spearman"),
                "mean_abs_diff": float(ge["abs_diff"].mean()) if len(ge) else np.nan,
                "mean_abs_rank_diff_year_common": float(ge["abs_rank_diff_year_common"].mean())
                if len(ge)
                else np.nan,
            }
        )
    year_summary = pd.DataFrame(year_rows)

    province_rows = []
    for (frontier, province), g in detail.groupby(["frontier_type", "Province"]):
        ge = g[g["alternative_score"].notna()]
        province_rows.append(
            {
                "frontier_type": frontier,
                "Province": province,
                "N_total_years": int(len(g)),
                "N_effective_years": int(len(ge)),
                "pooled_longrun_mean": float(ge["pooled_score"].mean()) if len(ge) else np.nan,
                "alternative_longrun_mean": float(ge["alternative_score"].mean()) if len(ge) else np.nan,
                "mean_abs_diff": float(ge["abs_diff"].mean()) if len(ge) else np.nan,
            }
        )
    province_long = pd.DataFrame(province_rows)
    for frontier, idx in province_long.dropna(subset=["alternative_longrun_mean"]).groupby("frontier_type").groups.items():
        gidx = list(idx)
        province_long.loc[gidx, "rank_pooled_longrun_common"] = province_long.loc[
            gidx, "pooled_longrun_mean"
        ].rank(ascending=False, method="average")
        province_long.loc[gidx, "rank_alt_longrun_common"] = province_long.loc[
            gidx, "alternative_longrun_mean"
        ].rank(ascending=False, method="average")
    province_long["rank_diff_longrun_common"] = (
        province_long["rank_alt_longrun_common"] - province_long["rank_pooled_longrun_common"]
    )
    province_long["abs_rank_diff_longrun_common"] = province_long["rank_diff_longrun_common"].abs()

    readme = pd.DataFrame(
        {
            "Notes": [
                "Alternative-frontier robustness for reviewer comments on pooled global frontier choice.",
                "year_specific uses each year's 30 province frontier.",
                "rolling_5yr_window uses target year +/- 2 years, truncated at sample endpoints.",
                "Scores are compared with the revised pooled-frontier Super-SBM labels stored in data.xlsx.",
                "Failed super-SBM solves are retained in the errors sheet and excluded from paired statistics.",
            ]
        }
    )
    err_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["frontier_type", "ID", "Province", "Year", "Error"])
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="score_detail", index=False)
        overall.to_excel(writer, sheet_name="overall_summary", index=False)
        year_summary.to_excel(writer, sheet_name="year_summary", index=False)
        province_long.to_excel(writer, sheet_name="province_longrun", index=False)
        err_df.to_excel(writer, sheet_name="errors", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)
    return out_path


def cea_accounting_value(df: pd.DataFrame) -> pd.Series:
    out = np.zeros(len(df), dtype=float)
    for col, coef in CEA_ACCOUNTING_COEFS_DATA_UNITS.items():
        out += pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) * float(coef)
    return pd.Series(out, index=df.index, dtype=float)


def update_cea_from_accounting_delta(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    out = after.copy()
    delta = cea_accounting_value(after) - cea_accounting_value(before)
    out["CEA"] = pd.to_numeric(before["CEA"], errors="coerce").to_numpy(dtype=float) + delta.to_numpy(dtype=float)
    out["CEA"] = np.maximum(out["CEA"].to_numpy(dtype=float), 0.0)
    return out


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    spec = SCENARIOS[scenario]
    before = df.copy()
    after = df.copy()
    for col, mult in spec["shock_inputs"].items():
        after[col] = pd.to_numeric(after[col], errors="coerce") * float(mult)
    for col, mult in spec["shock_outputs"].items():
        after[col] = pd.to_numeric(after[col], errors="coerce") * float(mult)
    if spec["update_cea"]:
        after = update_cea_from_accounting_delta(before, after)
    return after


def select_validation_targets(df: pd.DataFrame, year: int, max_targets: int = 15) -> pd.DataFrame:
    g = df[(df[YEAR_COL] == year) & (df[Y_COL] < 1.0)].sort_values(Y_COL).copy()
    g["eligible_subfrontier_n_year"] = int(len(g))
    if len(g) <= max_targets:
        g["selected_by"] = "all_subfrontier"
        return g
    pos = np.linspace(0, len(g) - 1, max_targets)
    idx = sorted(set(int(round(p)) for p in pos))
    while len(idx) < max_targets:
        for candidate in range(len(g)):
            if candidate not in idx:
                idx.append(candidate)
                if len(idx) == max_targets:
                    break
    out = g.iloc[sorted(idx)].copy()
    out["selected_by"] = f"efficiency_quantile_grid_max{max_targets}"
    return out


def train_leave_year_out_xgb(df: pd.DataFrame, holdout_year: int):
    train = df[df[YEAR_COL] != holdout_year].copy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].to_numpy(dtype=float))
    y_train = train[Y_COL].to_numpy(dtype=float)
    model = make_xgb(FEATURES, random_state=6200 + int(holdout_year))
    model.fit(x_train, y_train)
    return scaler, model


def predict_with_model(df_rows: pd.DataFrame, scaler: StandardScaler, model: XGBRegressor) -> np.ndarray:
    x = scaler.transform(df_rows[FEATURES].to_numpy(dtype=float))
    return model.predict(x)


def run_expanded_perturbation_validation(
    df: pd.DataFrame,
    validation_years: tuple[int, ...] = (2005, 2010, 2015, 2020, 2023),
    max_targets_per_year: int = 15,
) -> Path:
    out_path = TABLES_DIR / "Expanded_perturbation_validation_revision.xlsx"
    df_pool = df[[ID_COL, "Province", YEAR_COL, Y_COL] + INPUT_COLS + GOOD_COLS + BAD_COLS + ["row_key"]].copy()
    rows, errors, selection_rows = [], [], []

    for year in validation_years:
        print(f"[perturbation] year={year}")
        targets = select_validation_targets(df_pool, year, max_targets=max_targets_per_year)
        selection_rows.append(targets[[ID_COL, "Province", YEAR_COL, Y_COL, "eligible_subfrontier_n_year", "selected_by"]])
        scaler, model = train_leave_year_out_xgb(df, holdout_year=year)

        for _, target in targets.iterrows():
            hit = df_pool.index[df_pool["row_key"] == int(target["row_key"])].tolist()
            if not hit:
                raise RuntimeError("Target row was not found in pooled reference data.")
            idx = int(hit[0])
            base_row = df_pool.loc[[idx]].copy()
            sur_base = float(predict_with_model(base_row, scaler, model)[0])
            try:
                dea_base = super_sbm_score_one(df_pool, idx)
                base_error = ""
            except Exception as exc:
                dea_base = np.nan
                base_error = str(exc)
                errors.append(
                    {
                        "Province": target["Province"],
                        "Year": year,
                        "Scenario": "BASE",
                        "Stage": "baseline_solve_pooled",
                        "Error": base_error,
                    }
                )

            for scenario in SCENARIOS:
                sur_scn_row = apply_scenario(base_row, scenario)
                sur_scn = float(predict_with_model(sur_scn_row, scaler, model)[0])
                sur_delta = sur_scn - sur_base
                try:
                    df_cf = df_pool.copy()
                    df_cf.loc[idx, :] = apply_scenario(df_cf.loc[[idx], :], scenario).iloc[0]
                    dea_scn = super_sbm_score_one(df_cf, idx)
                    dea_delta = dea_scn - dea_base if np.isfinite(dea_base) else np.nan
                    err = ""
                    status = "ok"
                except Exception as exc:
                    dea_scn = np.nan
                    dea_delta = np.nan
                    err = str(exc)
                    status = "failed"
                    errors.append(
                        {
                            "Province": target["Province"],
                            "Year": year,
                            "Scenario": scenario,
                            "Stage": "scenario_solve_pooled",
                            "Error": err,
                        }
                    )

                rows.append(
                    {
                        "Province": target["Province"],
                        "ID": int(target[ID_COL]),
                        "Year": year,
                        "Scenario": scenario,
                        "scenario_label": SCENARIOS[scenario]["label"],
                        "SuperSBM_observed": float(target[Y_COL]),
                        "DEA_base_pooled": dea_base,
                        "DEA_scenario_pooled": dea_scn,
                        "DEA_delta": dea_delta,
                        "SUR_base_leave_year_out": sur_base,
                        "SUR_scenario_leave_year_out": sur_scn,
                        "SUR_delta": sur_delta,
                        "abs_delta_error": abs(dea_delta - sur_delta) if np.isfinite(dea_delta) else np.nan,
                        "sign_agree": sign_agreement(dea_delta, sur_delta) if np.isfinite(dea_delta) else np.nan,
                        "status": status,
                    }
                )

    detail = pd.DataFrame(rows)
    selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary_rows = []
    for scenario, g in detail.groupby("Scenario"):
        ge = g.dropna(subset=["DEA_delta", "SUR_delta"])
        summary_rows.append(validation_summary_row(("all_years", scenario), ge))
    by_year_rows = []
    for (year, scenario), g in detail.groupby(["Year", "Scenario"]):
        ge = g.dropna(subset=["DEA_delta", "SUR_delta"])
        by_year_rows.append(validation_summary_row((year, scenario), ge))
    summary = pd.DataFrame(summary_rows)
    by_year = pd.DataFrame(by_year_rows)
    err_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["Province", "Year", "Scenario", "Stage", "Error"])
    readme = pd.DataFrame(
        {
            "Notes": [
                "Expanded perturbation validation for reviewer comments on scenario-screening credibility.",
                "Validation years: 2005, 2010, 2015, 2020, and 2023.",
                "Surrogate deltas are predicted by monotone XGBoost trained leave-year-out for each validation year.",
                "DEA deltas are recomputed against the same pooled full-sample frontier after perturbing only the target province-year row.",
                "Targets are sub-frontier observations (efficiency < 1); when many exist, a deterministic efficiency-quantile grid selects up to 15 provinces per year.",
                "This remains a diagnostic consistency check, not a policy feasibility or causal-effect validation.",
            ]
        }
    )
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="validation_detail", index=False)
        summary.to_excel(writer, sheet_name="summary_all_years", index=False)
        by_year.to_excel(writer, sheet_name="summary_by_year", index=False)
        selection.to_excel(writer, sheet_name="target_selection", index=False)
        err_df.to_excel(writer, sheet_name="errors", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)
    return out_path


def sign_agreement(a: float, b: float, tol: float = 1e-12) -> float:
    if abs(float(a)) <= tol and abs(float(b)) <= tol:
        return 1.0
    return 1.0 if np.sign(float(a)) == np.sign(float(b)) else 0.0


def validation_summary_row(keys, ge: pd.DataFrame) -> dict:
    prefix, scenario = keys
    if ge.empty:
        return {
            "group": prefix,
            "Scenario": scenario,
            "N": 0,
            "sign_agreement": np.nan,
            "MAE_delta": np.nan,
            "RMSE_delta": np.nan,
            "pearson_delta": np.nan,
            "spearman_delta": np.nan,
            "mean_DEA_delta": np.nan,
            "mean_SUR_delta": np.nan,
        }
    mse = mean_squared_error(ge["DEA_delta"], ge["SUR_delta"])
    return {
        "group": prefix,
        "Scenario": scenario,
        "N": int(len(ge)),
        "sign_agreement": float(ge["sign_agree"].mean()),
        "MAE_delta": float(ge["abs_delta_error"].mean()),
        "RMSE_delta": float(np.sqrt(mse)),
        "pearson_delta": safe_corr(ge["DEA_delta"], ge["SUR_delta"]),
        "spearman_delta": safe_corr(ge["DEA_delta"], ge["SUR_delta"], "spearman"),
        "mean_DEA_delta": float(ge["DEA_delta"].mean()),
        "mean_SUR_delta": float(ge["SUR_delta"].mean()),
    }


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    df = load_modeling_data()
    print(f"[revision robustness] loaded {len(df)} rows from {DATA_PATH}")
    paths = [
        run_surrogate_predictor_ablation(df),
        run_alternative_frontier_robustness(df),
        run_expanded_perturbation_validation(df),
    ]
    for path in paths:
        print(f"[revision robustness] saved {path}")


if __name__ == "__main__":
    main()
