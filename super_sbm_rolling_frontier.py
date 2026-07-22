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
    year_spearman = detail.groupby(YEAR_COL).apply(
        lambda group: spearmanr(group["pooled_reference_score"], group["rolling_5yr_score"]).statistic,
        include_groups=False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


if __name__ == "__main__":
    main()
