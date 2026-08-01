"""Validate and apply the undesirable-output Super-SBM under VRS.

The implementation follows the documented pooled-frontier settings:
non-oriented, undesirable-output Super-SBM, variable returns to scale,
and equal weights across eight inputs and the two output components.

The script first reconstructs the pooled global-frontier scores and compares
all 720 values with the reference ``efficiency`` column. It computes the five-year
local-window frontier only when that validation gate passes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data.xlsx"
DEFAULT_OUTPUT = ROOT / "tables" / "rolling_5yr_frontier_robustness_revision.csv"
DEFAULT_WORKBOOK = ROOT / "tables" / "Alternative_frontier_robustness_revision.xlsx"

ID_COL = "ID"
YEAR_COL = "Year"
LABEL_COL = "efficiency"
INPUT_COLS = ["TPAM", "EIA", "CS", "AFA", "PU", "ADY", "PFU", "NRP"]
GOOD_OUTPUT_COLS = ["GAO"]
BAD_OUTPUT_COLS = ["CEA"]


class DEAError(RuntimeError):
    """Raised when a DEA linear program is infeasible or fails numerically."""


@dataclass(frozen=True)
class ValidationResult:
    n: int
    n_frontier_reference: int
    n_frontier_reconstructed: int
    max_abs_error: float
    mean_abs_error: float
    pearson: float
    spearman: float
    n_within_1e_6: int


def _solve_lp(
    objective: np.ndarray,
    *,
    a_eq: list[np.ndarray],
    b_eq: list[float],
    a_ub: list[np.ndarray] | None = None,
    b_ub: list[float] | None = None,
    name: str,
) -> float:
    result = linprog(
        objective,
        A_ub=None if a_ub is None else np.asarray(a_ub),
        b_ub=None if b_ub is None else np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=[(0.0, None)] * len(objective),
        method="highs",
    )
    if not result.success or result.fun is None or not np.isfinite(result.fun):
        raise DEAError(f"{name} failed: {result.message}")
    return float(result.fun)


def _standard_sbm(
    x0: np.ndarray,
    y0: np.ndarray,
    b0: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    b_ref: np.ndarray,
) -> float:
    """Return the non-oriented undesirable-output SBM score in (0, 1]."""
    m, s, h = len(x0), len(y0), len(b0)
    n = x_ref.shape[1]
    i_lambda = slice(0, n)
    i_x = slice(n, n + m)
    i_y = slice(n + m, n + m + s)
    i_b = slice(n + m + s, n + m + s + h)
    i_t = n + m + s + h
    dimension = i_t + 1

    qx = np.full(m, 1.0 / m)
    qy = np.full(s, 1.0 / (s + h))
    qb = np.full(h, 1.0 / (s + h))

    objective = np.zeros(dimension)
    objective[i_t] = 1.0
    objective[i_x] = -qx / x0

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    for i in range(m):
        row = np.zeros(dimension)
        row[i_lambda] = x_ref[i]
        row[n + i] = 1.0
        row[i_t] = -x0[i]
        a_eq.append(row)
        b_eq.append(0.0)

    for r in range(s):
        row = np.zeros(dimension)
        row[i_lambda] = y_ref[r]
        row[n + m + r] = -1.0
        row[i_t] = -y0[r]
        a_eq.append(row)
        b_eq.append(0.0)

    for r in range(h):
        row = np.zeros(dimension)
        row[i_lambda] = b_ref[r]
        row[n + m + s + r] = 1.0
        row[i_t] = -b0[r]
        a_eq.append(row)
        b_eq.append(0.0)

    # VRS after the Charnes-Cooper transformation: sum(Lambda) = t.
    row = np.zeros(dimension)
    row[i_lambda] = 1.0
    row[i_t] = -1.0
    a_eq.append(row)
    b_eq.append(0.0)

    # Normalize the denominator of the fractional SBM objective to one.
    row = np.zeros(dimension)
    row[i_t] = 1.0
    row[i_y] = qy / y0
    row[i_b] = qb / b0
    a_eq.append(row)
    b_eq.append(1.0)

    return _solve_lp(objective, a_eq=a_eq, b_eq=b_eq, name="standard SBM")


def _super_sbm(
    x0: np.ndarray,
    y0: np.ndarray,
    b0: np.ndarray,
    x_ref_without_target: np.ndarray,
    y_ref_without_target: np.ndarray,
    b_ref_without_target: np.ndarray,
) -> float:
    """Return Tone's Super-SBM score for an SBM-efficient observation.

    The target is removed from the reference set. Relative to the target, the
    projected comparison point may use more inputs, produce less desirable
    output, and produce more undesirable output. This direction is required
    for a feasible and monotone undesirable-output super-efficiency measure.
    """
    m, s, h = len(x0), len(y0), len(b0)
    n = x_ref_without_target.shape[1]
    i_lambda = slice(0, n)
    i_x = slice(n, n + m)
    i_y = slice(n + m, n + m + s)
    i_b = slice(n + m + s, n + m + s + h)
    i_t = n + m + s + h
    dimension = i_t + 1

    qx = np.full(m, 1.0 / m)
    qy = np.full(s, 1.0 / (s + h))
    qb = np.full(h, 1.0 / (s + h))

    objective = np.zeros(dimension)
    objective[i_t] = 1.0
    objective[i_x] = qx / x0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # X*Lambda <= x0*t + Sx.
    for i in range(m):
        row = np.zeros(dimension)
        row[i_lambda] = x_ref_without_target[i]
        row[n + i] = -1.0
        row[i_t] = -x0[i]
        a_ub.append(row)
        b_ub.append(0.0)

    # Y*Lambda >= y0*t - Sy.
    for r in range(s):
        row = np.zeros(dimension)
        row[i_lambda] = -y_ref_without_target[r]
        row[n + m + r] = -1.0
        row[i_t] = y0[r]
        a_ub.append(row)
        b_ub.append(0.0)

    # B*Lambda <= b0*t + Sb.
    for r in range(h):
        row = np.zeros(dimension)
        row[i_lambda] = b_ref_without_target[r]
        row[n + m + s + r] = -1.0
        row[i_t] = -b0[r]
        a_ub.append(row)
        b_ub.append(0.0)

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    row = np.zeros(dimension)
    row[i_lambda] = 1.0
    row[i_t] = -1.0
    a_eq.append(row)
    b_eq.append(0.0)

    # Normalize 1 - desirable-output slack - bad-output slack to one.
    row = np.zeros(dimension)
    row[i_t] = 1.0
    row[i_y] = -qy / y0
    row[i_b] = -qb / b0
    a_eq.append(row)
    b_eq.append(1.0)

    return _solve_lp(
        objective,
        a_eq=a_eq,
        b_eq=b_eq,
        a_ub=a_ub,
        b_ub=b_ub,
        name="Super-SBM",
    )


def _score_target(reference: pd.DataFrame, target_index: int, *, tol: float = 1e-7) -> float:
    x_ref = reference[INPUT_COLS].to_numpy(dtype=float).T
    y_ref = reference[GOOD_OUTPUT_COLS].to_numpy(dtype=float).T
    b_ref = reference[BAD_OUTPUT_COLS].to_numpy(dtype=float).T
    x0 = x_ref[:, target_index]
    y0 = y_ref[:, target_index]
    b0 = b_ref[:, target_index]

    score = _standard_sbm(x0, y0, b0, x_ref, y_ref, b_ref)
    if score < 1.0 - tol:
        return score

    keep = np.ones(reference.shape[0], dtype=bool)
    keep[target_index] = False
    return _super_sbm(x0, y0, b0, x_ref[:, keep], y_ref[:, keep], b_ref[:, keep])


def _validate_data(df: pd.DataFrame) -> None:
    required = [ID_COL, YEAR_COL, *INPUT_COLS, *GOOD_OUTPUT_COLS, *BAD_OUTPUT_COLS, LABEL_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[[ID_COL, YEAR_COL]].duplicated().any():
        raise ValueError("ID-Year pairs must be unique.")
    if df[required].isna().any().any():
        raise ValueError("DEA inputs, outputs, identifiers, and labels must not contain missing values.")
    measure_cols = [*INPUT_COLS, *GOOD_OUTPUT_COLS, *BAD_OUTPUT_COLS]
    if (df[measure_cols] <= 0).any().any():
        raise ValueError("This implementation requires strictly positive inputs and outputs.")


def validate_pooled_frontier(df: pd.DataFrame) -> tuple[np.ndarray, ValidationResult]:
    reference = df.reset_index(drop=True)
    reconstructed = np.asarray([_score_target(reference, i) for i in range(len(reference))])
    reference_scores = reference[LABEL_COL].to_numpy(dtype=float)
    errors = np.abs(reconstructed - reference_scores)

    result = ValidationResult(
        n=len(reference),
        n_frontier_reference=int(np.sum(reference_scores >= 1.0)),
        n_frontier_reconstructed=int(np.sum(reconstructed >= 1.0)),
        max_abs_error=float(errors.max()),
        mean_abs_error=float(errors.mean()),
        pearson=float(pearsonr(reconstructed, reference_scores).statistic),
        spearman=float(spearmanr(reconstructed, reference_scores).statistic),
        n_within_1e_6=int(np.sum(errors <= 1e-6)),
    )
    return reconstructed, result


def _validation_passes(result: ValidationResult) -> bool:
    return (
        result.n == 720
        and result.n_frontier_reference == result.n_frontier_reconstructed
        and result.max_abs_error <= 5e-5
        and result.spearman >= 0.999999
    )


def compute_centered_five_year_frontier(
    df: pd.DataFrame,
    reconstructed_pooled: np.ndarray,
) -> pd.DataFrame:
    """Compute a target-year +/-2 local frontier, truncated at sample endpoints."""
    work = df.reset_index(drop=True).copy()
    work["_row_key"] = np.arange(len(work))
    year_min = int(work[YEAR_COL].min())
    year_max = int(work[YEAR_COL].max())
    rows: list[dict[str, float | int | str]] = []

    for row_position, target in work.iterrows():
        year = int(target[YEAR_COL])
        start_year = max(year_min, year - 2)
        end_year = min(year_max, year + 2)
        reference = work.loc[
            work[YEAR_COL].between(start_year, end_year),
            [*df.columns, "_row_key"],
        ].reset_index(drop=True)
        hits = np.flatnonzero(reference["_row_key"].to_numpy() == row_position)
        if len(hits) != 1:
            raise RuntimeError("Target observation was not found exactly once in its rolling reference set.")

        rolling_score = _score_target(reference, int(hits[0]))
        rows.append(
            {
                ID_COL: int(target[ID_COL]),
                YEAR_COL: year,
                "pooled_reference_score": float(target[LABEL_COL]),
                "pooled_reconstructed_score": float(reconstructed_pooled[row_position]),
                "pooled_validation_abs_error": float(
                    abs(reconstructed_pooled[row_position] - float(target[LABEL_COL]))
                ),
                "rolling_5yr_score": rolling_score,
                "rolling_minus_pooled": rolling_score - float(target[LABEL_COL]),
                "rolling_reference_start": start_year,
                "rolling_reference_end": end_year,
                "rolling_reference_years": end_year - start_year + 1,
                "rolling_reference_n": int(len(reference)),
            }
        )

    detail = pd.DataFrame(rows)
    detail["pooled_rank_within_year"] = detail.groupby(YEAR_COL)["pooled_reference_score"].rank(
        ascending=False, method="average"
    )
    detail["rolling_rank_within_year"] = detail.groupby(YEAR_COL)["rolling_5yr_score"].rank(
        ascending=False, method="average"
    )
    detail["absolute_rank_difference"] = (
        detail["rolling_rank_within_year"] - detail["pooled_rank_within_year"]
    ).abs()
    return detail


def _print_validation(result: ValidationResult) -> None:
    print("Pooled global-frontier validation")
    print(f"  N: {result.n}")
    print(
        "  Frontier observations (reference / reconstructed): "
        f"{result.n_frontier_reference} / {result.n_frontier_reconstructed}"
    )
    print(f"  Mean absolute error: {result.mean_abs_error:.12g}")
    print(f"  Maximum absolute error: {result.max_abs_error:.12g}")
    print(f"  Values within 1e-6: {result.n_within_1e_6}/{result.n}")
    print(f"  Pearson: {result.pearson:.12g}")
    print(f"  Spearman: {result.spearman:.12g}")


def _print_rolling_summary(detail: pd.DataFrame) -> None:
    pooled = detail["pooled_reference_score"]
    rolling = detail["rolling_5yr_score"]
    year_spearman = detail.groupby(YEAR_COL)[["pooled_reference_score", "rolling_5yr_score"]].apply(
        lambda group: spearmanr(group["pooled_reference_score"], group["rolling_5yr_score"]).statistic,
    )
    print("Five-year local-window frontier summary")
    print(f"  Effective N: {rolling.notna().sum()}/{len(detail)}")
    print(f"  Rolling scores at or above one: {(rolling >= 1.0).sum()}")
    print(f"  Pearson with pooled scores: {pearsonr(pooled, rolling).statistic:.6f}")
    print(f"  Spearman with pooled scores: {spearmanr(pooled, rolling).statistic:.6f}")
    print(f"  Mean within-year Spearman: {year_spearman.mean():.6f}")
    print(f"  Median within-year Spearman: {year_spearman.median():.6f}")
    print(f"  Mean absolute score difference: {(rolling - pooled).abs().mean():.6f}")
    print(f"  Median absolute score difference: {(rolling - pooled).abs().median():.6f}")
    print(f"  Median within-year absolute rank difference: {detail['absolute_rank_difference'].median():.3f}")


def _gini(values: pd.Series) -> float:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    if x.size == 0 or np.any(x < 0):
        return float("nan")
    if np.allclose(x, 0.0):
        return 0.0
    cumulative = np.cumsum(x)
    return float((x.size + 1 - 2 * cumulative.sum() / cumulative[-1]) / x.size)


def _theil_t(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0 or np.any(x <= 0):
        return float("nan")
    ratio = x / x.mean()
    return float(np.mean(ratio * np.log(ratio)))


def _build_year_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in detail.groupby(YEAR_COL, sort=True):
        pooled = group["pooled_reference_score"]
        rolling = group["rolling_5yr_score"]
        rows.append(
            {
                "Year": int(year),
                "N": int(len(group)),
                "Reference start": int(group["rolling_reference_start"].iloc[0]),
                "Reference end": int(group["rolling_reference_end"].iloc[0]),
                "Reference years": int(group["rolling_reference_years"].iloc[0]),
                "Reference N": int(group["rolling_reference_n"].iloc[0]),
                "Pooled mean": float(pooled.mean()),
                "Window mean": float(rolling.mean()),
                "Window median": float(rolling.median()),
                "Window Gini": _gini(rolling),
                "Window Theil-T": _theil_t(rolling),
                "Window N (score >= 1)": int((rolling >= 1.0).sum()),
                "Mean absolute difference": float((rolling - pooled).abs().mean()),
                "Pearson": float(pearsonr(pooled, rolling).statistic),
                "Spearman": float(spearmanr(pooled, rolling).statistic),
                "Median absolute rank difference": float(group["absolute_rank_difference"].median()),
            }
        )
    return pd.DataFrame(rows)


def _build_workbook_summary(detail: pd.DataFrame, validation: ValidationResult) -> pd.DataFrame:
    pooled = detail["pooled_reference_score"]
    rolling = detail["rolling_5yr_score"]
    year_spearman = detail.groupby(YEAR_COL).apply(
        lambda group: spearmanr(group["pooled_reference_score"], group["rolling_5yr_score"]).statistic,
        include_groups=False,
    )
    return pd.DataFrame(
        [
            ("Pooled reconstruction", "N", validation.n, "All reference observations reconstructed"),
            (
                "Pooled reconstruction",
                "Frontier observations (reference / reconstructed)",
                f"{validation.n_frontier_reference} / {validation.n_frontier_reconstructed}",
                "Efficient-set classification is identical",
            ),
            ("Pooled reconstruction", "Mean absolute error", validation.mean_abs_error, "Numerical agreement with reference scores"),
            (
                "Pooled reconstruction",
                "Maximum absolute error",
                validation.max_abs_error,
                "Below the pre-specified 5e-5 validation threshold",
            ),
            (
                "Pooled reconstruction",
                "Pearson / Spearman",
                f"{validation.pearson:.12f} / {validation.spearman:.12f}",
                "Score and rank agreement",
            ),
            ("Five-year local window", "Effective N", int(rolling.notna().sum()), "No infeasible observation"),
            (
                "Five-year local window",
                "Scores at or above 1",
                int((rolling >= 1.0).sum()),
                "A smaller local reference set identifies more local-frontier observations",
            ),
            (
                "Five-year local window",
                "Overall Pearson / Spearman",
                f"{pearsonr(pooled, rolling).statistic:.6f} / {spearmanr(pooled, rolling).statistic:.6f}",
                "Overall association mixes level and time effects",
            ),
            (
                "Five-year local window",
                "Mean / median within-year Spearman",
                f"{year_spearman.mean():.6f} / {year_spearman.median():.6f}",
                "Within-year provincial ordering is more stable than pooled level agreement",
            ),
            (
                "Five-year local window",
                "Mean / median absolute score difference",
                f"{(rolling - pooled).abs().mean():.6f} / {(rolling - pooled).abs().median():.6f}",
                "Absolute efficiency levels are sensitive to frontier choice",
            ),
            (
                "Five-year local window",
                "Mean / median absolute rank difference",
                f"{detail['absolute_rank_difference'].mean():.3f} / {detail['absolute_rank_difference'].median():.3f}",
                "Typical within-year rank movement is limited",
            ),
            (
                "Method note",
                "Reference set",
                "Target year +/-2",
                "At sample endpoints, available years are used (3 or 4 years).",
            ),
            (
                "Method note",
                "DEA specification",
                "Non-oriented undesirable-output Super-SBM; VRS; equal weights",
                "Matches the documented pooled-frontier settings.",
            ),
            (
                "Interpretation",
                "Supported statement",
                "Absolute levels are frontier-sensitive; within-year ranks are more stable",
                "Use as a sensitivity result, not as confirmation that frontier choice is immaterial.",
            ),
            (
                "Interpretation",
                "Not estimated",
                "Malmquist or policy effects",
                "The check does not estimate dynamic productivity or causal policy effects.",
            ),
        ],
        columns=["Section", "Metric", "Value", "Interpretation"],
    )


def _export_workbook(detail: pd.DataFrame, validation: ValidationResult, path: Path) -> None:
    year_summary = _build_year_summary(detail)
    summary = _build_workbook_summary(detail, validation)
    readme = pd.DataFrame(
        [
            ("Purpose", "Assess sensitivity of pooled global-frontier Super-SBM scores to a five-year local reference frontier."),
            ("Baseline", "The efficiency column in data.xlsx contains the pooled global-frontier reference scores."),
            (
                "Validation gate",
                "The Python solver must reconstruct all 720 pooled scores, match the 52 frontier observations, achieve Spearman >= 0.999999, and have maximum absolute error <= 5e-5.",
            ),
            (
                "Window definition",
                "For each target province-year, the reference set uses the target year +/-2; at 2000-2001 and 2022-2023 the available endpoint years are used.",
            ),
            ("Inputs", "TPAM, EIA, CS, AFA, PU, ADY, PFU, and PIE (legacy code column NRP)."),
            ("Outputs", "GAO is the desirable output and CEA is the undesirable output."),
            (
                "Model",
                "Original non-oriented undesirable-output Super-SBM under VRS with equal weights (Qx=0.125 each; Qy=0.5; Qb=0.5).",
            ),
            ("Code", "super_sbm_rolling_frontier.py"),
            (
                "Interpretation boundary",
                "This is a reference-frontier sensitivity check. It is not Malmquist analysis, a dynamic productivity decomposition, a policy simulation, or a causal estimate.",
            ),
        ],
        columns=["Item", "Description"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="score_detail", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        year_summary.to_excel(writer, sheet_name="year_summary", index=False)
        readme.to_excel(writer, sheet_name="ReadMe", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook-output", type=Path, default=DEFAULT_WORKBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.data, sheet_name="Sheet1")
    _validate_data(df)

    reconstructed, validation = validate_pooled_frontier(df)
    _print_validation(validation)
    if not _validation_passes(validation):
        raise RuntimeError(
            "The Python solver did not reproduce the pooled-frontier reference scores closely enough; "
            "the five-year frontier was not computed."
        )

    detail = compute_centered_five_year_frontier(df, reconstructed)
    _print_rolling_summary(detail)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Saved: {args.output}")
    _export_workbook(detail, validation, args.workbook_output)
    print(f"Saved: {args.workbook_output}")


if __name__ == "__main__":
    main()
