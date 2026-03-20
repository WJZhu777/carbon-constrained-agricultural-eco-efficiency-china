# -*- coding: utf-8 -*-

"""
Reproducible regression benchmark for eco-efficiency prediction.

This script keeps the original modeling workflow intact:
repeated stratified cross-validation, outer validation for early stopping,
an independent test split, ensemble diagnostics, ablation checks, and
panel-robustness checks.

Default input file: ./data.xlsx (Sheet1)
"""
import os, random, warnings, math, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV
import xgboost as xgb
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float32')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path='./data.xlsx', sheet='Sheet1'):
    df = pd.read_excel(file_path, sheet_name=sheet)
    return df

def preprocess_numeric(df, return_panel=False, id_col='ID', year_col=None):
    """
    Convert the input dataframe to numeric arrays.

    By default the function returns only X, y, and input_features so the main
    workflow stays unchanged. When return_panel=True it also returns panel keys
    (ID and year) for robustness checks.
    """
    input_features = ['TPAM', 'EIA', 'CS', 'AFA', 'PU', 'ADY', 'PFU', 'NRP', 'GAO', 'CEA']
    target_feature = 'efficiency'
    cols = input_features + [target_feature]
    df_num = df[cols].dropna().copy()
    X = df_num[input_features].astype(np.float32).values
    y = df_num[target_feature].astype(np.float32).values
    if not return_panel:
        return (X, y, input_features)
    if id_col not in df.columns:
        raise ValueError(f"ID column not found:  '{id_col}', please pass id_col=... explicitly to preprocess_numeric.")
    if year_col is None:
        candidates = ['Year', 'year', 'YEAR', '年份']
        year_col_found = None
        for c in candidates:
            if c in df.columns:
                year_col_found = c
                break
        if year_col_found is None:
            raise ValueError('Year column not found (tried Year/year/YEAR/年份). Please pass year_col=... explicitly to preprocess_numeric.')
        year_col = year_col_found
    if year_col not in df.columns:
        raise ValueError(f"Year column not found: '{year_col}''. Check the column name or parameter setting.")
    idx = df_num.index
    ids = df.loc[idx, id_col].values
    years = df.loc[idx, year_col].values
    return (X, y, input_features, ids, years)

def _make_bins(y, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    if len(edges) <= 2:
        edges = np.unique(np.quantile(y, np.linspace(0, 1, 4)))
    bins = np.digitize(y, edges[1:-1], right=False)
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins

def stratified_train_val_test_split(X, y, test_size=0.2, val_size=0.2, max_bins=10, random_state=42):
    for n_bins in range(max_bins, 2, -1):
        try:
            bins = _make_bins(y, n_bins)
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(sss1.split(X, bins))
            bins_tr = bins[tr_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            tr_sub, va_sub = next(sss2.split(X[tr_idx], bins_tr))
            tr_final = tr_idx[tr_sub]
            va_final = tr_idx[va_sub]
            return (tr_final, va_final, te_idx, n_bins)
        except Exception:
            continue
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    te_size = int(len(y) * test_size)
    va_size = int((len(y) - te_size) * val_size)
    te_idx = idx[:te_size]
    rest = idx[te_size:]
    va_idx = rest[:va_size]
    tr_idx = rest[va_size:]
    return (tr_idx, va_idx, te_idx, None)

def gen_repeated_stratified_folds(y, repeats=2, folds=5, max_bins=10, random_state=42):
    for n_bins in range(max_bins, 2, -1):
        try:
            bins = _make_bins(y, n_bins)
            rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
            for tr, va in rskf.split(np.zeros_like(y), bins):
                yield (tr, va)
            return
        except Exception:
            continue
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
    for tr, va in rkf.split(np.zeros_like(y)):
        yield (tr, va)

def panel_time_forward_rolling_splits(years, *, val_window=1, test_window=1, min_train_years=10, start_val_year=None, end_val_year=None, max_splits=None):
    """
    Rolling time-forward split without future leakage.

    For anchor year v:
        Train: years <= v - 1
        Val:   years in [v, v + val_window - 1]
        Test:  years in [v + val_window, v + val_window + test_window - 1]
    """
    years_arr = np.asarray(years)
    years_num = pd.to_numeric(years_arr, errors='coerce')
    if np.any(pd.isna(years_num)):
        raise ValueError('The year column cannot be fully converted to numeric. Check Year/year/年份.')
    years_num = years_num.astype(int)
    uniq_years = np.sort(np.unique(years_num))
    nT = len(uniq_years)
    need = int(min_train_years) + int(val_window) + int(test_window)
    if nT < need:
        raise ValueError(f'Too few distinct years (nT={nT}); rolling time-forward requires at least {need} unique years: min_train_years={min_train_years}, val_window={val_window}, test_window={test_window}.')
    total_possible = nT - int(min_train_years) - int(val_window) - int(test_window) + 1
    splits = []
    for i in range(int(min_train_years), nT - int(val_window) - int(test_window) + 1):
        val_years = uniq_years[i:i + int(val_window)]
        test_years = uniq_years[i + int(val_window):i + int(val_window) + int(test_window)]
        train_years = uniq_years[:i]
        v = int(val_years[0])
        if start_val_year is not None and v < int(start_val_year):
            continue
        if end_val_year is not None and v > int(end_val_year):
            continue
        tr_idx = np.where(np.isin(years_num, train_years))[0]
        va_idx = np.where(np.isin(years_num, val_years))[0]
        te_idx = np.where(np.isin(years_num, test_years))[0]
        if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_idx) == 0:
            continue
        info = {'train_end': int(train_years[-1]) if len(train_years) else None, 'val_years': val_years.astype(int), 'test_years': test_years.astype(int), 'val_window': int(val_window), 'test_window': int(test_window), 'total_possible': int(total_possible)}
        splits.append((tr_idx, va_idx, te_idx, info))
        if max_splits is not None and len(splits) >= int(max_splits):
            break
    if len(splits) == 0:
        raise ValueError('Rolling time-forward produced no valid split. Years may have been filtered out or dropped by dropna().')
    return splits

def panel_province_block_split(ids, train_frac=0.6, val_frac=0.2, random_state=42):
    """Province-block split based on province IDs."""
    ids_arr = np.asarray(ids)
    uniq_ids = np.unique(ids_arr)
    nI = len(uniq_ids)
    if nI < 3:
        raise ValueError(f'Too few provinces ({nI}) for a province-block Train/Val/Test split.')
    rng = np.random.default_rng(random_state)
    uniq_ids_shuffled = uniq_ids.copy()
    rng.shuffle(uniq_ids_shuffled)
    n_train = max(1, int(round(nI * train_frac)))
    n_val = max(1, int(round(nI * val_frac)))
    if n_train + n_val >= nI:
        n_train = max(1, nI - 2)
        n_val = 1
    train_ids = uniq_ids_shuffled[:n_train]
    val_ids = uniq_ids_shuffled[n_train:n_train + n_val]
    test_ids = uniq_ids_shuffled[n_train + n_val:]
    tr_idx = np.where(np.isin(ids_arr, train_ids))[0]
    va_idx = np.where(np.isin(ids_arr, val_ids))[0]
    te_idx = np.where(np.isin(ids_arr, test_ids))[0]
    info = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    return (tr_idx, va_idx, te_idx, info)

def metrics_dict(y_true, y_pred):
    return {'MSE': float(mean_squared_error(y_true, y_pred)), 'MAE': float(mean_absolute_error(y_true, y_pred)), 'R2': float(r2_score(y_true, y_pred))}

def _apply_sorted_barh(names, values, colors, descending=True):
    idx = np.argsort(values)[::-1] if descending else np.argsort(values)
    names_sorted = [names[i] for i in idx]
    values_sorted = [values[i] for i in idx]
    colors_sorted = [colors[i] for i in idx]
    return (names_sorted, values_sorted, colors_sorted)

def plot_comparison(results_test, y_test, history_dict=None, title_suffix=''):
    """
    Save figures to ./fig/ with stable naming and compact display labels.
    """
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 120, 'savefig.dpi': 300, 'axes.edgecolor': '#000000', 'axes.linewidth': 1.0, 'axes.grid': True, 'grid.color': '#B0B0B0', 'grid.linestyle': '--', 'grid.linewidth': 0.6})
    okabe_ito = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FF0000']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=okabe_ito)
    os.makedirs('fig', exist_ok=True)
    tag = 'test' if 'Test' in title_suffix or 'test' in title_suffix.lower() else 'plot'

    def display_name(name: str) -> str:
        if name == 'VanillaMLP':
            return 'MLP'
        if name == 'TSLR-MLP':
            return 'TSLR-MLP'
        if name.startswith('Ensemble('):
            return 'VWLB'
        if name == 'Stacking[LassoCV]':
            return 'OS[LV]'
        if name.startswith('Stacking['):
            return name.replace('Stacking[', 'OOF-Stack[')
        return name
    model_names = list(results_test.keys())
    disp_names = [display_name(n) for n in model_names]
    colors = (okabe_ito * math.ceil(len(model_names) / len(okabe_ito)))[:len(model_names)]
    for i, n in enumerate(model_names):
        if n.startswith('Stacking['):
            colors[i] = '#FF0000'
        elif n == 'SVM':
            colors[i] = '#0072B2'
    color_map = {model_names[i]: colors[i] for i in range(len(model_names))}
    fig1 = plt.figure(figsize=(9, 7))
    for name in model_names:
        pred = results_test[name]['y_pred']
        plt.scatter(y_test, pred, alpha=0.7, label=display_name(name), s=30, color=color_map[name])
    xymin, xymax = (float(np.min(y_test)), float(np.max(y_test)))
    plt.plot([xymin, xymax], [xymin, xymax], '--', lw=1.5, color='#000000')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend(frameon=False)
    plt.tight_layout()
    fig1.savefig(f'fig/Figure_1_pred_vs_true_{tag}.tiff', bbox_inches='tight')
    plt.close(fig1)
    r2_scores = [results_test[n]['test_metrics']['R2'] for n in model_names]
    order_idx = np.argsort(r2_scores)[::-1]
    top_idx = list(order_idx[:4])
    other_idx = list(order_idx[4:]) if len(order_idx) > 4 else []
    all_preds = np.concatenate([np.asarray(results_test[n]['y_pred']).ravel() for n in model_names])
    all_resid = np.concatenate([y_test - np.asarray(results_test[n]['y_pred']).ravel() for n in model_names])
    x_min, x_max = (float(all_preds.min()), float(all_preds.max()))
    y_min, y_max = (float(all_resid.min()), float(all_resid.max()))
    x_pad = 0.03 * (x_max - x_min + 1e-09)
    y_pad = 0.05 * (y_max - y_min + 1e-09)
    TOP4_STYLE = {'OS[LV]': dict(color='#D55E00', marker='X', s=68, alpha=0.95, edge='k', lw=0.7, z=6), 'VWLB': dict(color='#0072B2', marker='s', s=52, alpha=0.85, edge='white', lw=0.7, z=5), 'Random Forest': dict(color='#CC79A7', marker='o', s=50, alpha=0.85, edge='white', lw=0.7, z=4)}

    def style_for(name, is_highlight=False):
        disp = display_name(name)
        st = TOP4_STYLE.get(disp, None)
        if st is not None:
            st = st.copy()
            if is_highlight:
                st['s'] = st['s'] * 1.15
                st['z'] = max(st['z'], 7)
            return st
        return dict(color=color_map[name], marker='o', s=38, alpha=0.8, edge='white', lw=0.6, z=4)
    fig2, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
    ax_left, ax_right = (axes[0], axes[1])
    highlight_idx = top_idx[0] if top_idx else None
    for i in [i for i in top_idx if i != highlight_idx]:
        name = model_names[i]
        pred = np.asarray(results_test[name]['y_pred']).ravel()
        residuals = y_test - pred
        st = style_for(name, is_highlight=False)
        ax_left.scatter(pred, residuals, alpha=st['alpha'], s=st['s'], zorder=st['z'], label=display_name(name), color=st['color'], marker=st['marker'], edgecolors=st['edge'], linewidths=st['lw'])
    if highlight_idx is not None:
        name = model_names[highlight_idx]
        pred = np.asarray(results_test[name]['y_pred']).ravel()
        residuals = y_test - pred
        st = style_for(name, is_highlight=True)
        ax_left.scatter(pred, residuals, alpha=st['alpha'], s=st['s'], zorder=st['z'], label=display_name(name), color=st['color'], marker=st['marker'], edgecolors=st['edge'], linewidths=st['lw'])
    ax_left.axhline(y=0, color='#000000', linestyle='--', linewidth=1.2, zorder=1)
    ax_left.set_xlabel('Predictions')
    ax_left.set_ylabel('Residuals')
    ax_left.set_title('Residuals (Top-4 by R²)')
    ax_left.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_left.set_ylim(y_min - y_pad, y_max + y_pad)
    handles, labels = ax_left.get_legend_handles_labels()
    if highlight_idx is not None:
        hl = display_name(model_names[highlight_idx])
        order = [labels.index(hl)] + [i for i, l in enumerate(labels) if l != hl]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    ax_left.legend(handles, labels, frameon=False, markerscale=1.3, handletextpad=0.4)
    if other_idx:
        for i in other_idx:
            name = model_names[i]
            pred = np.asarray(results_test[name]['y_pred']).ravel()
            residuals = y_test - pred
            ax_right.scatter(pred, residuals, alpha=0.3, s=24, zorder=2, label=display_name(name), color=color_map[name], marker='o', edgecolors='none')
        ax_right.axhline(y=0, color='#000000', linestyle='--', linewidth=1.2, zorder=1)
        ax_right.set_xlabel('Predictions')
        ax_right.set_title('Residuals (Others)')
        ax_right.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_right.legend(frameon=False, ncol=1, markerscale=1.1, handletextpad=0.4)
    else:
        ax_right.axis('off')
        ax_right.text(0.5, 0.5, 'No other models', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    fig2.savefig(f'fig/Figure_2_residuals_{tag}.tiff', bbox_inches='tight')
    plt.close(fig2)
    fig3 = plt.figure(figsize=(11, 8))
    mse_scores = [results_test[n]['test_metrics']['MSE'] for n in model_names]
    disp_mse, mse_sorted, colors_mse = _apply_sorted_barh(disp_names, mse_scores, colors, descending=True)
    bars = plt.barh(disp_mse, mse_sorted, color=colors_mse)
    plt.xlabel('MSE')
    xmax = max(mse_sorted) if mse_sorted else 1.0
    pad = 0.12
    right_lim = xmax * (1.0 + pad)
    plt.xlim(0, right_lim)
    offset = 0.01 * right_lim
    for b, s in zip(bars, mse_sorted):
        plt.text(min(b.get_width() + offset, right_lim * 0.995), b.get_y() + b.get_height() / 2.0, f'{s:.4f}', va='center', ha='left', fontsize=9)
    plt.tight_layout()
    fig3.savefig(f'fig/Figure_3_MSE_{tag}_desc.tiff', bbox_inches='tight')
    plt.close(fig3)
    fig4 = plt.figure(figsize=(11, 8))
    r2_scores = [results_test[n]['test_metrics']['R2'] for n in model_names]
    disp_r2, r2_sorted, colors_r2 = _apply_sorted_barh(disp_names, r2_scores, colors, descending=True)
    bars = plt.barh(disp_r2, r2_sorted, color=colors_r2)
    plt.xlabel('R²')
    xmax = max(r2_sorted) if r2_sorted else 1.0
    pad = 0.12
    right_lim = xmax * (1.0 + pad)
    plt.xlim(0, right_lim)
    offset = 0.01 * right_lim
    for b, s in zip(bars, r2_sorted):
        plt.text(min(b.get_width() + offset, right_lim * 0.995), b.get_y() + b.get_height() / 2.0, f'{s:.4f}', va='center', ha='left', fontsize=9)
    plt.tight_layout()
    fig4.savefig(f'fig/Figure_4_R2_{tag}_desc.tiff', bbox_inches='tight')
    plt.close(fig4)
    fig5_mae = plt.figure(figsize=(11, 8))
    mae_scores = [results_test[n]['test_metrics']['MAE'] for n in model_names]
    disp_mae, mae_sorted, colors_mae = _apply_sorted_barh(disp_names, mae_scores, colors, descending=True)
    bars = plt.barh(disp_mae, mae_sorted, color=colors_mae)
    plt.xlabel('MAE')
    xmax = max(mae_sorted) if mae_sorted else 1.0
    pad = 0.12
    right_lim = xmax * (1.0 + pad)
    plt.xlim(0, right_lim)
    offset = 0.01 * right_lim
    for b, s in zip(bars, mae_sorted):
        plt.text(min(b.get_width() + offset, right_lim * 0.995), b.get_y() + b.get_height() / 2.0, f'{s:.4f}', va='center', ha='left', fontsize=9)
    plt.tight_layout()
    fig5_mae.savefig(f'fig/Figure_5_MAE_{tag}_desc.tiff', bbox_inches='tight')
    plt.close(fig5_mae)
    if history_dict:
        try:
            fig6 = plt.figure(figsize=(10, 7))
            for name, hist in history_dict.items():
                disp = display_name(name)
                if hist and 'loss' in hist:
                    if len(hist['loss']) > 0:
                        plt.plot(hist['loss'], label=f'{disp}-Train', linestyle='-', alpha=0.85)
                    if 'val_loss' in hist and len(hist['val_loss']) > 0:
                        plt.plot(hist['val_loss'], label=f'{disp}-Val', linestyle='--', alpha=0.85)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Curves')
            plt.legend(frameon=False)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig6.savefig(f'fig/Figure_6_training_curves_{tag}.tiff', bbox_inches='tight')
            plt.close(fig6)
        except Exception as e:
            print(f'Plotting error: {e}')

def bootstrap_mse_diff(y_true, y_pred_A, y_pred_B, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        eA = np.mean((y_true[idx] - y_pred_A[idx]) ** 2)
        eB = np.mean((y_true[idx] - y_pred_B[idx]) ** 2)
        diffs[b] = eA - eB
    diffs.sort()
    ci = (diffs[int(0.025 * B)], diffs[int(0.975 * B)])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return (float(np.mean(diffs)), (float(ci[0]), float(ci[1])), float(p))

def make_figure5_and_table3(y_true, results_test, out_fig='fig/Figure_7_Bootstrap_distribution_of_MSE_differences_(model_minus_monotone_XGBoost_baseline).tiff', out_xlsx='tables/Table_3.xlsx', B=2000, seed=42):
    """
    Build the bootstrap figure and Table 3 for model-vs-baseline MSE differences.
    """
    os.makedirs('fig', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    def display_name(name: str) -> str:
        if name == 'VanillaMLP':
            return 'MLP'
        if name == 'TSLR-MLP':
            return 'TSLR-MLP'
        if name.startswith('Ensemble('):
            return 'VWLB'
        if name == 'Stacking[LassoCV]':
            return 'OS[LV]'
        if name.startswith('Stacking['):
            return name.replace('Stacking[', 'OOF-Stack[')
        return name
    model_names = list(results_test.keys())
    if 'XGBoost' in model_names:
        baseline = 'XGBoost'
    else:
        mse_pairs = [(m, results_test[m]['test_metrics']['MSE']) for m in model_names]
        mse_pairs.sort(key=lambda x: x[1])
        baseline = mse_pairs[0][0]
        print(f"[Warning] 'XGBoost' not found; using the model with the smallest test MSE as baseline: {baseline}")
    y_base = np.asarray(results_test[baseline]['y_pred']).ravel()
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs_dict = {}
    summary_rows = []
    for m in model_names:
        if m == baseline:
            continue
        y_m = np.asarray(results_test[m]['y_pred']).ravel()
        diffs = np.empty(B, dtype=np.float64)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            mse_m = np.mean((y_true[idx] - y_m[idx]) ** 2)
            mse_b = np.mean((y_true[idx] - y_base[idx]) ** 2)
            diffs[b] = mse_m - mse_b
        diffs_dict[display_name(m)] = diffs
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        summary_rows.append({'model': m, 'baseline': baseline, 'mean_mse_diff': float(diffs.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi), 'p(two-sided)': float(p)})
    df_out = pd.DataFrame(summary_rows).sort_values('mean_mse_diff')
    try:
        df_out.to_excel(out_xlsx, index=False)
    except Exception as e:
        print(f'Failed to save {out_xlsx}: {e}')
    fig = plt.figure(figsize=(8, 5))
    for disp_name, diffs in diffs_dict.items():
        plt.hist(diffs, bins=40, histtype='step', density=True, label=disp_name, alpha=0.9)
    plt.axvline(0.0, linestyle='--')
    plt.xlabel('MSE(model) − MSE(baseline)')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f'[Figure 7] saved -> {out_fig}')
    print(f'[Table 3 ] saved -> {out_xlsx}')
    return df_out

def build_bpnn(num_features):
    return Sequential([Dense(512, activation='relu', input_shape=(num_features,)), Dropout(0.4), Dense(256, activation='relu'), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.2), Dense(64, activation='relu'), Dense(32, activation='relu'), Dense(1, activation='linear')])

def build_tslr_mlp_base(num_features):
    """Base MLP architecture used by TSLR-MLP."""
    return Sequential([Dense(256, activation='relu', input_shape=(num_features,)), Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(1, activation='linear')])

def build_mlp(num_features):
    return Sequential([Dense(256, activation='relu', input_shape=(num_features,)), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.3), Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(1, activation='linear')])

def train_dl(model, X_tr, y_tr, X_va, y_va, epochs=150, batch=32, name=''):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-06), EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch, verbose=0, callbacks=cbs)
    print(f'{name} training complete')
    return (hist.history, model)

def train_tslr_mlp(model, X_tr, y_tr, X_va, y_va, epochs=60, batch=32, name='TSLR-MLP'):
    """
    Two-stage training for TSLR-MLP:
    warm-up with a larger learning rate, then refinement with a smaller
    learning rate plus ReduceLROnPlateau and early stopping.
    """
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=50, batch_size=batch, verbose=0)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-06), EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch, verbose=0, callbacks=cbs)
    print(f'{name} training complete')
    return (hist.history, model)

def fit_xgb_compat(est, Xtr, ytr, Xva, yva):
    """Version-safe early stopping for XGBoost."""
    try:
        cb = [xgb.callback.EarlyStopping(rounds=200, save_best=True, maximize=False)]
        est.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], callbacks=cb, verbose=False)
        yva_pred = est.predict(Xva)
        return (est, yva_pred)
    except TypeError:
        pass
    try:
        est.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], early_stopping_rounds=200, verbose=False)
        yva_pred = est.predict(Xva)
        return (est, yva_pred)
    except TypeError:
        pass
    candidates = [200, 300, 400, 600, 800]
    best_mse, best_est, best_pred = (float('inf'), None, None)
    base_params = est.get_params(deep=True)
    for n in candidates:
        est_n = xgb.XGBRegressor(**{**base_params, 'n_estimators': n})
        est_n.fit(Xtr, ytr, verbose=False)
        pred = est_n.predict(Xva)
        mse = mean_squared_error(yva, pred)
        if mse < best_mse:
            best_mse, best_est, best_pred = (mse, est_n, pred)
    return (best_est, best_pred)

def make_xgb_regressor(monotone_constraints=None, random_state=42, n_jobs=-1):
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, objective='reg:squarederror', eval_metric='rmse', random_state=random_state, n_jobs=n_jobs)
    if monotone_constraints is not None:
        params['monotone_constraints'] = monotone_constraints
    return xgb.XGBRegressor(**params)

def make_xgb_meta_regressor(random_state=42, n_jobs=-1):
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.001, reg_lambda=1.0, objective='reg:squarederror', eval_metric='rmse', random_state=random_state, n_jobs=n_jobs)

def build_meta_estimator(meta_name: str):
    if meta_name == 'RidgeCV':
        return RidgeCV(alphas=np.logspace(-4, 2, 30), cv=5)
    if meta_name == 'LassoCV':
        return LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=10000, random_state=42)
    if meta_name == 'ElasticNetCV':
        return ElasticNetCV(l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95, 1.0], alphas=np.logspace(-4, 1, 30), cv=5, max_iter=10000, random_state=42)
    if meta_name == 'XGB-meta':
        return make_xgb_meta_regressor(random_state=42, n_jobs=-1)
    raise ValueError(f'Unknown meta estimator: {meta_name}')

def cross_validate(models_flat_dl, models_ml, X_tr, y_tr, repeats=2, folds=5, max_bins=10):
    F = X_tr.shape[1]
    names_all = list(models_flat_dl.keys()) + list(models_ml.keys())
    fold_metrics = {name: {'R2': [], 'MSE': [], 'MAE': []} for name in names_all}
    splits = list(gen_repeated_stratified_folds(y_tr, repeats=repeats, folds=folds, max_bins=max_bins, random_state=42))
    for k, (tr_idx, va_idx) in enumerate(splits, 1):
        Xtr, Xva = (X_tr[tr_idx], X_tr[va_idx])
        ytr, yva = (y_tr[tr_idx], y_tr[va_idx])
        scaler = RobustScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)
        for name, builder in models_flat_dl.items():
            tf.keras.backend.clear_session()
            model = builder(F)
            if name == 'TSLR-MLP':
                _, model = train_tslr_mlp(model, Xtr_s, ytr, Xva_s, yva, epochs=60, batch=32, name=name)
            else:
                _, model = train_dl(model, Xtr_s, ytr, Xva_s, yva, epochs=120, batch=32, name=name)
            yva_pred = model.predict(Xva_s, verbose=0).flatten()
            m = metrics_dict(yva, yva_pred)
            for k2 in ['R2', 'MSE', 'MAE']:
                fold_metrics[name][k2].append(m[k2])
            del model
            gc.collect()
        for name, builder in models_ml.items():
            est = builder(F)
            if name.startswith('XGBoost'):
                est_fitted, yva_pred = fit_xgb_compat(est, Xtr_s, ytr, Xva_s, yva)
            else:
                est.fit(Xtr_s, ytr)
                yva_pred = est.predict(Xva_s)
            m = metrics_dict(yva, yva_pred)
            for k2 in ['R2', 'MSE', 'MAE']:
                fold_metrics[name][k2].append(m[k2])
        print(f'[CV Fold {k}/{len(splits)}] complete')
    cv_summary = {}
    for name in fold_metrics:
        cv_summary[name] = {'R2': (float(np.mean(fold_metrics[name]['R2'])), float(np.std(fold_metrics[name]['R2']))), 'MSE': (float(np.mean(fold_metrics[name]['MSE'])), float(np.std(fold_metrics[name]['MSE']))), 'MAE': (float(np.mean(fold_metrics[name]['MAE'])), float(np.std(fold_metrics[name]['MAE'])))}
    return cv_summary

def best_weight_by_val(y_val, pred_xgb_val, pred_tslr_val):
    grid = np.linspace(0, 1, 101)
    best_w, best_mse = (0.5, float('inf'))
    for w in grid:
        yhat = w * pred_xgb_val + (1.0 - w) * pred_tslr_val
        mse = mean_squared_error(y_val, yhat)
        if mse < best_mse:
            best_mse, best_w = (mse, w)
    return (float(best_w), float(best_mse))

def gen_stratified_kfold_indices(y, folds=5, max_bins=10, random_state=42):
    for n_bins in range(max_bins, 2, -1):
        try:
            bins = _make_bins(y, n_bins)
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            for tr, va in skf.split(np.zeros_like(y), bins):
                yield (tr, va)
            return
        except Exception:
            continue
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    for tr, va in kf.split(np.zeros_like(y)):
        yield (tr, va)

def stacking_oof_predict(models_flat_dl, models_ml, X_tr, y_tr, X_va, X_te, folds=5, max_bins=10):
    F = X_tr.shape[1]
    base_names = list(models_flat_dl.keys()) + list(models_ml.keys())
    n_tr, n_va, n_te = (len(y_tr), X_va.shape[0], X_te.shape[0])
    oof_meta = {name: np.zeros(n_tr, dtype=np.float32) for name in base_names}
    va_meta = {name: np.zeros(n_va, dtype=np.float32) for name in base_names}
    te_meta = {name: np.zeros(n_te, dtype=np.float32) for name in base_names}
    counts_va = {name: 0 for name in base_names}
    counts_te = {name: 0 for name in base_names}
    for fold_id, (tr_idx, oof_idx) in enumerate(gen_stratified_kfold_indices(y_tr, folds=folds, max_bins=max_bins, random_state=42), 1):
        Xtr, Xoof = (X_tr[tr_idx], X_tr[oof_idx])
        ytr, yoof = (y_tr[tr_idx], y_tr[oof_idx])
        scaler = RobustScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xoof_s = scaler.transform(Xoof)
        Xva_s = scaler.transform(X_va)
        Xte_s = scaler.transform(X_te)
        for name, builder in models_flat_dl.items():
            tf.keras.backend.clear_session()
            model = builder(F)
            if name == 'TSLR-MLP':
                _, model = train_tslr_mlp(model, Xtr_s, ytr, Xoof_s, yoof, epochs=60, batch=32, name=f'{name}-F{fold_id}')
            else:
                _, model = train_dl(model, Xtr_s, ytr, Xoof_s, yoof, epochs=120, batch=32, name=f'{name}-F{fold_id}')
            oof_pred = model.predict(Xoof_s, verbose=0).ravel()
            oof_meta[name][oof_idx] = oof_pred
            va_meta[name] += model.predict(Xva_s, verbose=0).ravel()
            te_meta[name] += model.predict(Xte_s, verbose=0).ravel()
            counts_va[name] += 1
            counts_te[name] += 1
            del model
            gc.collect()
        for name, builder in models_ml.items():
            est = builder(F)
            if name.startswith('XGBoost'):
                est_fitted, _ = fit_xgb_compat(est, Xtr_s, ytr, Xoof_s, yoof)
                est_used = est_fitted
            else:
                est.fit(Xtr_s, ytr)
                est_used = est
            oof_pred = np.asarray(est_used.predict(Xoof_s)).ravel()
            oof_meta[name][oof_idx] = oof_pred
            va_meta[name] += np.asarray(est_used.predict(Xva_s)).ravel()
            te_meta[name] += np.asarray(est_used.predict(Xte_s)).ravel()
            counts_va[name] += 1
            counts_te[name] += 1
    for name in base_names:
        if counts_va[name] > 0:
            va_meta[name] /= counts_va[name]
        if counts_te[name] > 0:
            te_meta[name] /= counts_te[name]
    X_tr_meta = np.vstack([oof_meta[name] for name in base_names]).T
    X_va_meta = np.vstack([va_meta[name] for name in base_names]).T
    X_te_meta = np.vstack([te_meta[name] for name in base_names]).T
    return (base_names, X_tr_meta, X_va_meta, X_te_meta)

def choose_meta_and_predict(base_names, X_tr_meta, y_tr, X_va_meta, y_va, X_te_meta):
    metas = {}
    metas['RidgeCV'] = RidgeCV(alphas=np.logspace(-4, 2, 30), cv=5)
    metas['LassoCV'] = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=10000, random_state=42)
    metas['ElasticNetCV'] = ElasticNetCV(l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95, 1.0], alphas=np.logspace(-4, 1, 30), cv=5, max_iter=10000, random_state=42)
    metas['XGB-meta'] = make_xgb_meta_regressor(random_state=42, n_jobs=-1)
    best_name, best_mse, best_est = (None, float('inf'), None)
    val_preds_cache = {}
    for name, est in metas.items():
        est.fit(X_tr_meta, y_tr)
        y_va_hat = np.asarray(est.predict(X_va_meta)).ravel()
        mse = mean_squared_error(y_va, y_va_hat)
        val_preds_cache[name] = y_va_hat
        print(f'[Stacking] meta-learner {name} | Val MSE = {mse:.6f}')
        if mse < best_mse:
            best_mse, best_name, best_est = (mse, name, est)
    y_te_hat = np.asarray(best_est.predict(X_te_meta)).ravel()
    info = {'meta': best_name, 'val_mse': float(best_mse)}
    if hasattr(best_est, 'coef_'):
        info['coef'] = {bn: float(w) for bn, w in zip(base_names, best_est.coef_)}
        info['intercept'] = float(getattr(best_est, 'intercept_', 0.0))
    return (best_name, val_preds_cache[best_name], y_te_hat, info)

def stacking_naive_predict(models_flat_dl, models_ml, X_tr, y_tr, X_va, y_va, X_te, *, dl_epochs=120, tslr_epochs=60, batch=32):
    F = X_tr.shape[1]
    base_names = list(models_flat_dl.keys()) + list(models_ml.keys())
    scaler = RobustScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xva_s = scaler.transform(X_va)
    Xte_s = scaler.transform(X_te)
    tr_meta = {}
    va_meta = {}
    te_meta = {}
    for name, builder in models_flat_dl.items():
        tf.keras.backend.clear_session()
        model = builder(F)
        if name == 'TSLR-MLP':
            _, model = train_tslr_mlp(model, Xtr_s, y_tr, Xva_s, y_va, epochs=tslr_epochs, batch=batch, name=f'{name}-Naive')
        else:
            _, model = train_dl(model, Xtr_s, y_tr, Xva_s, y_va, epochs=dl_epochs, batch=batch, name=f'{name}-Naive')
        tr_meta[name] = model.predict(Xtr_s, verbose=0).ravel()
        va_meta[name] = model.predict(Xva_s, verbose=0).ravel()
        te_meta[name] = model.predict(Xte_s, verbose=0).ravel()
        del model
        gc.collect()
    for name, builder in models_ml.items():
        est = builder(F)
        if name.startswith('XGBoost'):
            est, _ = fit_xgb_compat(est, Xtr_s, y_tr, Xva_s, y_va)
        else:
            est.fit(Xtr_s, y_tr)
        tr_meta[name] = np.asarray(est.predict(Xtr_s)).ravel()
        va_meta[name] = np.asarray(est.predict(Xva_s)).ravel()
        te_meta[name] = np.asarray(est.predict(Xte_s)).ravel()
    X_tr_meta = np.vstack([tr_meta[n] for n in base_names]).T
    X_va_meta = np.vstack([va_meta[n] for n in base_names]).T
    X_te_meta = np.vstack([te_meta[n] for n in base_names]).T
    return (base_names, X_tr_meta, X_va_meta, X_te_meta)

def export_ablation_and_leakage_checks(out_xlsx_path, metrics_rows, bootstrap_rows):
    os.makedirs(os.path.dirname(out_xlsx_path), exist_ok=True)
    df_metrics = pd.DataFrame(metrics_rows)
    df_boot = pd.DataFrame(bootstrap_rows)
    if not df_metrics.empty:
        df_metrics = df_metrics.sort_values(['Check', 'Role', 'Model']).reset_index(drop=True)
    if not df_boot.empty:
        df_boot = df_boot.sort_values(['Check', 'MeanDiff(A-B)']).reset_index(drop=True)
    with pd.ExcelWriter(out_xlsx_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
        df_boot.to_excel(writer, sheet_name='Bootstrap_MSEdiff', index=False)
        note = pd.DataFrame({'Notes': ['Ablation & leakage-sensitivity checks', 'Bootstrap_MSEdiff: MeanDiff(A-B)=MSE(A)-MSE(B); CI is the 95% percentile interval; p is the two-sided proportion test.', 'In the manuscript, B=2000 paired bootstrap can be reported as a robustness check consistent with the main table.']})
        note.to_excel(writer, sheet_name='ReadMe', index=False)
    print(f'[Ablation] saved -> {out_xlsx_path}')
    return (df_metrics, df_boot)

def run_ablation_and_leakage_sensitivity_checks(*, X_tr, y_tr, X_va, y_va, X_te, y_te, X_tr_s, X_va_s, X_te_s, models_flat_dl, models_ml, results_test, monotone_constraints_str, meta_name_proper, B_boot=2000, seed=42, out_xlsx='tables/Ablation and leakage-sensitivity checks.xlsx', RUN=True):
    if not RUN:
        return (None, None)
    metrics_rows = []
    bootstrap_rows = []

    def add_metrics(check, role, model_name, y_val_pred, y_test_pred):
        mv = metrics_dict(y_va, y_val_pred)
        mt = metrics_dict(y_te, y_test_pred)
        metrics_rows.append({'Check': check, 'Role': role, 'Model': model_name, 'Val_MSE': mv['MSE'], 'Val_MAE': mv['MAE'], 'Val_R2': mv['R2'], 'Test_MSE': mt['MSE'], 'Test_MAE': mt['MAE'], 'Test_R2': mt['R2']})

    def add_boot(check, A_name, yA, B_name, yB):
        mean_diff, ci, p = bootstrap_mse_diff(y_te, np.asarray(yA).ravel(), np.asarray(yB).ravel(), B=B_boot, seed=seed)
        bootstrap_rows.append({'Check': check, 'A': A_name, 'B': B_name, 'MeanDiff(A-B)': mean_diff, 'CI95_low': ci[0], 'CI95_high': ci[1], 'p(two-sided)': p})
    check1 = 'Monotonicity constraint (XGBoost)'
    if 'XGBoost' in results_test and 'y_pred_val' in results_test['XGBoost']:
        y_val_mono = results_test['XGBoost']['y_pred_val']
        y_te_mono = results_test['XGBoost']['y_pred']
        add_metrics(check1, 'Baseline', 'XGBoost (monotone)', y_val_mono, y_te_mono)
    else:
        raise RuntimeError('results_test is missing XGBoost val/test predictions, so ablation cannot be run.')
    xgb_nomono = make_xgb_regressor(monotone_constraints=None, random_state=42, n_jobs=-1)
    xgb_nomono, y_val_nomono = fit_xgb_compat(xgb_nomono, X_tr_s, y_tr, X_va_s, y_va)
    y_te_nomono = xgb_nomono.predict(X_te_s)
    add_metrics(check1, 'Variant', 'XGBoost (no-monotone)', y_val_nomono, y_te_nomono)
    add_boot(check1, 'XGBoost (no-monotone)', y_te_nomono, 'XGBoost (monotone)', y_te_mono)
    check2 = 'VWLB weight selection (tuned vs fixed)'
    ens_keys = [k for k in results_test.keys() if k.startswith('Ensemble(')]
    if len(ens_keys) >= 1 and 'XGBoost' in results_test and ('TSLR-MLP' in results_test):
        ens_key = ens_keys[0]
        w = float(results_test[ens_key].get('info', {}).get('w_xgb', np.nan))
        if np.isnan(w):
            w, _ = best_weight_by_val(y_va, results_test['XGBoost']['y_pred_val'], results_test['TSLR-MLP']['y_pred_val'])
        y_val_tuned = w * results_test['XGBoost']['y_pred_val'] + (1.0 - w) * results_test['TSLR-MLP']['y_pred_val']
        y_te_tuned = w * results_test['XGBoost']['y_pred'] + (1.0 - w) * results_test['TSLR-MLP']['y_pred']
        add_metrics(check2, 'Baseline', f'VWLB tuned (w={w:.2f})', y_val_tuned, y_te_tuned)
        w0 = 0.5
        y_val_fix = w0 * results_test['XGBoost']['y_pred_val'] + (1.0 - w0) * results_test['TSLR-MLP']['y_pred_val']
        y_te_fix = w0 * results_test['XGBoost']['y_pred'] + (1.0 - w0) * results_test['TSLR-MLP']['y_pred']
        add_metrics(check2, 'Variant', 'VWLB fixed (w=0.50)', y_val_fix, y_te_fix)
        add_boot(check2, 'VWLB fixed (w=0.50)', y_te_fix, f'VWLB tuned (w={w:.2f})', y_te_tuned)
    else:
        print('[Ablation] Missing XGBoost/TSLR-MLP/Ensemble(...) for VWLB; skipping Check 2.')
    check3 = 'Stacking leakage-sensitivity (proper OOF vs naive)'
    proper_key = f'Stacking[{meta_name_proper}]'
    if proper_key in results_test and 'y_pred_val' in results_test[proper_key]:
        y_val_proper = results_test[proper_key]['y_pred_val']
        y_te_proper = results_test[proper_key]['y_pred']
        add_metrics(check3, 'Baseline', f'Proper OOF stacking [{meta_name_proper}]', y_val_proper, y_te_proper)
    else:
        print('[Ablation] proper stacking val/test predictions not found in results_test; skipping Check 3 baseline.')
        y_te_proper = None
    base_names_n, X_tr_meta_n, X_va_meta_n, X_te_meta_n = stacking_naive_predict(models_flat_dl=models_flat_dl, models_ml=models_ml, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, dl_epochs=120, tslr_epochs=60, batch=32)
    meta_est = build_meta_estimator(meta_name_proper)
    meta_est.fit(X_tr_meta_n, y_tr)
    y_val_naive = np.asarray(meta_est.predict(X_va_meta_n)).ravel()
    y_te_naive = np.asarray(meta_est.predict(X_te_meta_n)).ravel()
    add_metrics(check3, 'Variant', f'Naive stacking [{meta_name_proper}]', y_val_naive, y_te_naive)
    if y_te_proper is not None:
        add_boot(check3, f'Naive stacking [{meta_name_proper}]', y_te_naive, f'Proper OOF stacking [{meta_name_proper}]', y_te_proper)
    check4 = 'Two-stage LR (TSLR-MLP) necessity'
    if 'TSLR-MLP' in results_test and 'y_pred_val' in results_test['TSLR-MLP']:
        y_val_2stage = results_test['TSLR-MLP']['y_pred_val']
        y_te_2stage = results_test['TSLR-MLP']['y_pred']
        add_metrics(check4, 'Baseline', 'TSLR-MLP (two-stage)', y_val_2stage, y_te_2stage)
    else:
        print('[Ablation] TSLR-MLP val/test predictions are missing in results_test; skipping Check 4 baseline.')
        y_te_2stage = None
    tf.keras.backend.clear_session()
    one_stage = build_tslr_mlp_base(X_tr_s.shape[1])
    _, one_stage = train_dl(one_stage, X_tr_s, y_tr, X_va_s, y_va, epochs=120, batch=32, name='TSLR-MLP(one-stage)')
    y_val_1stage = one_stage.predict(X_va_s, verbose=0).ravel()
    y_te_1stage = one_stage.predict(X_te_s, verbose=0).ravel()
    add_metrics(check4, 'Variant', 'TSLR-MLP (one-stage)', y_val_1stage, y_te_1stage)
    del one_stage
    gc.collect()
    if y_te_2stage is not None:
        add_boot(check4, 'TSLR-MLP (one-stage)', y_te_1stage, 'TSLR-MLP (two-stage)', y_te_2stage)
    return export_ablation_and_leakage_checks(out_xlsx, metrics_rows, bootstrap_rows)

def _train_and_eval_on_split(X, y, tr_idx, va_idx, te_idx, models_flat_dl, models_ml, feats, description=''):
    """
    Reuse the main training pipeline on a given Train/Val/Test split and return
    base-model metrics, validation predictions, VWLB, and OOF stacking outputs.
    """
    X_tr, y_tr = (X[tr_idx], y[tr_idx])
    X_va, y_va = (X[va_idx], y[va_idx])
    X_te, y_te = (X[te_idx], y[te_idx])
    print(f'\n[Panel-robustness] {description} split:')
    print(f'  Train n = {len(y_tr)}, Val n = {len(y_va)}, Test n = {len(y_te)}')
    F = X_tr.shape[1]
    scaler_outer = RobustScaler().fit(X_tr)
    X_tr_s = scaler_outer.transform(X_tr)
    X_va_s = scaler_outer.transform(X_va)
    X_te_s = scaler_outer.transform(X_te)
    results_split = {}
    history_dummy = {}
    for name, builder in models_flat_dl.items():
        tf.keras.backend.clear_session()
        model = builder(F)
        if name == 'TSLR-MLP':
            hist, model = train_tslr_mlp(model, X_tr_s, y_tr, X_va_s, y_va, epochs=60, batch=32, name=f'{name}-{description}')
        else:
            hist, model = train_dl(model, X_tr_s, y_tr, X_va_s, y_va, epochs=120, batch=32, name=f'{name}-{description}')
        y_pred_te = model.predict(X_te_s, verbose=0).ravel()
        y_pred_va = model.predict(X_va_s, verbose=0).ravel()
        results_split[name] = {'y_pred': y_pred_te, 'test_metrics': metrics_dict(y_te, y_pred_te), 'y_pred_val': y_pred_va}
        history_dummy[name] = hist
        del model
        gc.collect()
    xgb_model = models_ml['XGBoost'](F)
    xgb_model, y_pred_va_xgb = fit_xgb_compat(xgb_model, X_tr_s, y_tr, X_va_s, y_va)
    y_pred_te_xgb = xgb_model.predict(X_te_s)
    results_split['XGBoost'] = {'y_pred': y_pred_te_xgb, 'test_metrics': metrics_dict(y_te, y_pred_te_xgb), 'y_pred_val': y_pred_va_xgb}
    rf_model = models_ml['Random Forest'](F)
    rf_model.fit(X_tr_s, y_tr)
    y_pred = rf_model.predict(X_te_s)
    results_split['Random Forest'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': rf_model.predict(X_va_s)}
    svm_model = models_ml['SVM'](F)
    svm_model.fit(X_tr_s, y_tr)
    y_pred = svm_model.predict(X_te_s)
    results_split['SVM'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': svm_model.predict(X_va_s)}
    ridge_model = models_ml['Ridge'](F)
    ridge_model.fit(X_tr_s, y_tr)
    y_pred = ridge_model.predict(X_te_s)
    results_split['Ridge'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': ridge_model.predict(X_va_s)}
    lasso_model = models_ml['Lasso'](F)
    lasso_model.fit(X_tr_s, y_tr)
    y_pred = lasso_model.predict(X_te_s)
    results_split['Lasso'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': lasso_model.predict(X_va_s)}
    if 'XGBoost' in results_split and 'TSLR-MLP' in results_split:
        w, val_mse = best_weight_by_val(y_va, results_split['XGBoost']['y_pred_val'], results_split['TSLR-MLP']['y_pred_val'])
        y_pred_ens = w * results_split['XGBoost']['y_pred'] + (1.0 - w) * results_split['TSLR-MLP']['y_pred']
        results_split[f'Ensemble(XGB{w:.2f}+TSLR-MLP{1 - w:.2f})'] = {'y_pred': y_pred_ens, 'test_metrics': metrics_dict(y_te, y_pred_ens), 'info': {'w_xgb': w, 'val_mse': val_mse}}
        print(f'[{description}][VWLB] Auto-selected Val weight: w_xgb={w:.2f} (Val MSE={val_mse:.6f})')
    base_names, X_tr_meta, X_va_meta, X_te_meta = stacking_oof_predict(models_flat_dl=models_flat_dl, models_ml=models_ml, X_tr=X_tr, y_tr=y_tr, X_va=X_va, X_te=X_te, folds=5, max_bins=10)
    meta_name, y_va_hat_stack, y_te_hat_stack, meta_info = choose_meta_and_predict(base_names, X_tr_meta, y_tr, X_va_meta, y_va, X_te_meta)
    results_split[f'Stacking[{meta_name}]'] = {'y_pred': y_te_hat_stack, 'test_metrics': metrics_dict(y_te, y_te_hat_stack), 'y_pred_val': y_va_hat_stack, 'info': meta_info}
    print(f"[{description}][Stacking] Selected meta-learner: {meta_name} | Val MSE={meta_info['val_mse']:.6f}")
    df_test_split = pd.DataFrame({'Model': list(results_split.keys()), 'MSE': [results_split[m]['test_metrics']['MSE'] for m in results_split], 'MAE': [results_split[m]['test_metrics']['MAE'] for m in results_split], 'R²': [results_split[m]['test_metrics']['R2'] for m in results_split]}).sort_values(by='R²', ascending=False).reset_index(drop=True)
    print(f'\n[Panel-robustness] {description} test-set performance (sorted by R2 descending):')
    print(df_test_split.to_string(index=False))
    return (df_test_split, results_split, meta_name)

def run_panel_dependence_robustness_checks(*, df_raw, models_flat_dl, models_ml, monotone_constraints_str, base_results_random_split, out_xlsx='tables/Panel_dependence_robustness_checks.xlsx', RUN=True, TF_VAL_WINDOW=1, TF_TEST_WINDOW=1, TF_MIN_TRAIN_YEARS=10, TF_START_VAL_YEAR=None, TF_END_VAL_YEAR=None, TF_MAX_SPLITS=None, PB_RANDOM_STATE=42):
    """
    Panel-dependence robustness checks.

    1) Record the random-stratified baseline without retraining.
    2) Refit across rolling time-forward splits and aggregate mean ± std.
    3) Refit once on a province-block split.
    """
    if not RUN:
        return None
    try:
        X_all, y_all, feats, ids_all, years_all = preprocess_numeric(df_raw, return_panel=True, id_col='ID', year_col=None)
    except Exception as e:
        print('\n[Panel-robustness] Failed to extract panel structure from the data:', str(e))
        print('  Check whether the data contain ID and Year/year/YEAR/年份 columns.')
        return None
    os.makedirs('tables', exist_ok=True)
    rows_random_long = []
    for name, res in base_results_random_split.items():
        mt = res['test_metrics']
        rows_random_long.append({'SplitType': 'Random-stratified', 'SplitID': 'base', 'TrainEnd': np.nan, 'ValYears': '', 'TestYears': '', 'Model': name, 'MSE': mt['MSE'], 'MAE': mt['MAE'], 'R2': mt['R2']})
    df_random_long = pd.DataFrame(rows_random_long)
    rows_tf_long = []
    df_tf_each_split_list = []
    df_tf_agg = pd.DataFrame()
    try:
        splits = panel_time_forward_rolling_splits(years_all, val_window=TF_VAL_WINDOW, test_window=TF_TEST_WINDOW, min_train_years=TF_MIN_TRAIN_YEARS, start_val_year=TF_START_VAL_YEAR, end_val_year=TF_END_VAL_YEAR, max_splits=TF_MAX_SPLITS)
        print('\n[Panel-robustness] Rolling time-forward splits:')
        print(f'  val_window={TF_VAL_WINDOW}, test_window={TF_TEST_WINDOW}, min_train_years={TF_MIN_TRAIN_YEARS}, #splits={len(splits)}')
        if TF_START_VAL_YEAR is not None or TF_END_VAL_YEAR is not None:
            print(f'  val_year_range=[{TF_START_VAL_YEAR},{TF_END_VAL_YEAR}]')
        for sid, (tr, va, te, info) in enumerate(splits, start=1):
            desc = f"Time-forward-rolling[{sid}] train<= {info['train_end']} | val={list(info['val_years'])} | test={list(info['test_years'])}"
            df_split, results_split, meta_name_split = _train_and_eval_on_split(X_all, y_all, tr, va, te, models_flat_dl=models_flat_dl, models_ml=models_ml, feats=feats, description=desc)
            df_split = df_split.copy()
            df_split['SplitType'] = 'Time-forward-rolling'
            df_split['SplitID'] = sid
            df_split['TrainEnd'] = info['train_end']
            df_split['ValYears'] = ','.join([str(int(x)) for x in info['val_years']])
            df_split['TestYears'] = ','.join([str(int(x)) for x in info['test_years']])
            df_tf_each_split_list.append(df_split)
            for _, r in df_split.iterrows():
                rows_tf_long.append({'SplitType': 'Time-forward-rolling', 'SplitID': int(sid), 'TrainEnd': int(info['train_end']), 'ValYears': ','.join([str(int(x)) for x in info['val_years']]), 'TestYears': ','.join([str(int(x)) for x in info['test_years']]), 'Model': r['Model'], 'MSE': float(r['MSE']), 'MAE': float(r['MAE']), 'R2': float(r['R²'])})
        df_tf_long = pd.DataFrame(rows_tf_long)
        df_tf_agg = df_tf_long.groupby('Model').agg(n_splits=('SplitID', 'nunique'), MSE_mean=('MSE', 'mean'), MSE_std=('MSE', 'std'), MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'), R2_mean=('R2', 'mean'), R2_std=('R2', 'std')).reset_index().sort_values('R2_mean', ascending=False)
        print('\n[Panel-robustness] Rolling time-forward aggregate (top 10 by R2_mean):')
        print(df_tf_agg.head(10).to_string(index=False))
    except Exception as e:
        print('\n[Panel-robustness] Rolling time-forward failed:', str(e))
        df_tf_long = pd.DataFrame(columns=['SplitType', 'SplitID', 'TrainEnd', 'ValYears', 'TestYears', 'Model', 'MSE', 'MAE', 'R2'])
        df_tf_agg = pd.DataFrame(columns=['Model', 'n_splits', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std'])
    rows_pb_long = []
    try:
        tr_pb, va_pb, te_pb, info_pb = panel_province_block_split(ids_all, train_frac=0.6, val_frac=0.2, random_state=PB_RANDOM_STATE)
        print('\n[Panel-robustness] Province-block partition:')
        print(f"  Train IDs (n={len(info_pb['train_ids'])}) = {list(info_pb['train_ids'])[:12]}{('...' if len(info_pb['train_ids']) > 12 else '')}")
        print(f"  Val   IDs (n={len(info_pb['val_ids'])})   = {list(info_pb['val_ids'])}")
        print(f"  Test  IDs (n={len(info_pb['test_ids'])})  = {list(info_pb['test_ids'])[:12]}{('...' if len(info_pb['test_ids']) > 12 else '')}")
        df_pb, results_pb, meta_name_pb = _train_and_eval_on_split(X_all, y_all, tr_pb, va_pb, te_pb, models_flat_dl=models_flat_dl, models_ml=models_ml, feats=feats, description=f'Province-block(seed={PB_RANDOM_STATE})')
        for _, r in df_pb.iterrows():
            rows_pb_long.append({'SplitType': 'Province-block', 'SplitID': f'seed{PB_RANDOM_STATE}', 'TrainEnd': np.nan, 'ValYears': '', 'TestYears': '', 'Model': r['Model'], 'MSE': float(r['MSE']), 'MAE': float(r['MAE']), 'R2': float(r['R²'])})
        df_pb_long = pd.DataFrame(rows_pb_long)
    except Exception as e:
        print('\n[Panel-robustness] Province-block failed:', str(e))
        df_pb_long = pd.DataFrame(columns=['SplitType', 'SplitID', 'TrainEnd', 'ValYears', 'TestYears', 'Model', 'MSE', 'MAE', 'R2'])

    def _summary_mean_std(df_long, split_type):
        if df_long.empty:
            return pd.DataFrame(columns=['SplitType', 'Model', 'n_splits', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std'])
        g = df_long.groupby('Model').agg(n_splits=('Model', 'count'), MSE_mean=('MSE', 'mean'), MSE_std=('MSE', 'std'), MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'), R2_mean=('R2', 'mean'), R2_std=('R2', 'std')).reset_index()
        g['SplitType'] = split_type
        if g['n_splits'].max() == 1:
            g['MSE_std'] = 0.0
            g['MAE_std'] = 0.0
            g['R2_std'] = 0.0
        return g[['SplitType', 'Model', 'n_splits', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']]
    df_summary = pd.concat([_summary_mean_std(df_random_long, 'Random-stratified'), df_tf_agg.assign(SplitType='Time-forward-rolling')[['SplitType', 'Model', 'n_splits', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']] if not df_tf_agg.empty else pd.DataFrame(columns=['SplitType', 'Model', 'n_splits', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']), _summary_mean_std(df_pb_long, 'Province-block')], ignore_index=True)
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='summary_mean_std', index=False)
        df_random_long.to_excel(writer, sheet_name='random_stratified_long', index=False)
        df_tf_long.to_excel(writer, sheet_name='time_forward_rolling_long', index=False)
        df_tf_agg.to_excel(writer, sheet_name='time_forward_rolling_agg', index=False)
        if len(df_tf_each_split_list) > 0:
            pd.concat(df_tf_each_split_list, ignore_index=True).to_excel(writer, sheet_name='time_forward_each_split', index=False)
        df_pb_long.to_excel(writer, sheet_name='province_block_long', index=False)
        pd.DataFrame({'Notes': ['Panel-dependence robustness checks', 'Rolling time-forward: Train<=v-1, Val=[v..v+VAL_WINDOW-1], Test=[v+VAL_WINDOW..v+VAL_WINDOW+TEST_WINDOW-1].', f'TF params: VAL_WINDOW={TF_VAL_WINDOW}, TEST_WINDOW={TF_TEST_WINDOW}, MIN_TRAIN_YEARS={TF_MIN_TRAIN_YEARS}, val_year_range=[{TF_START_VAL_YEAR},{TF_END_VAL_YEAR}], max_splits={TF_MAX_SPLITS}', f'Province-block: random_state={PB_RANDOM_STATE}, split by ID (no overlap).', 'summary_mean_std: mean±std across splits (Random/PB are single split so std=0).']}).to_excel(writer, sheet_name='ReadMe', index=False)
    print(f'\n[Panel-robustness] All outputs saved to {out_xlsx}')
    return df_summary
PROVINCES_BY_ID = ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia', 'Liaoning', 'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 'Yunnan', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang']

def _attach_province_name(df, id_col='ID', new_col='Province'):
    """Attach province names using the 1..30 ID order in PROVINCES_BY_ID."""
    df = df.copy()
    if id_col not in df.columns:
        return df

    def _map_one(x):
        try:
            xi = int(x)
        except Exception:
            return str(x)
        if 1 <= xi <= len(PROVINCES_BY_ID):
            return PROVINCES_BY_ID[xi - 1]
        return f'ID_{xi}'
    df[new_col] = df[id_col].apply(_map_one)
    return df

def _apply_scenario(df, scenario):
    """Apply a scenario shock to the specified columns of the input dataframe."""
    df = df.copy()
    if scenario == 'S1_input10':
        for col in ['AFA', 'PU', 'ADY', 'PFU']:
            if col in df.columns:
                df[col] = df[col] * 0.9
    elif scenario == 'S2_cea10':
        if 'CEA' in df.columns:
            df['CEA'] = df['CEA'] * 0.9
    elif scenario == 'S3_gao5':
        if 'GAO' in df.columns:
            df['GAO'] = df['GAO'] * 1.05
    else:
        raise ValueError(f'Unknown scenario: {scenario}')
    return df

def _predict_vwlb_from_df(df_rows, feats, scaler, xgb_model, tslr_model, w_xgb):
    """Predict with VWLB: w * XGB + (1 - w) * TSLR-MLP."""
    if tslr_model is None:
        raise ValueError('TSLR-MLP model is None. Make sure tslr_model_keep is preserved in the main training step.')
    X_new = df_rows[feats].astype(np.float32).values
    X_new_s = scaler.transform(X_new)
    y_xgb = xgb_model.predict(X_new_s)
    y_tslr = tslr_model.predict(X_new_s, verbose=0).ravel()
    return w_xgb * y_xgb + (1.0 - w_xgb) * y_tslr

def scenario_analysis_demo(df_raw, feats, scaler, xgb_model, tslr_model, w_xgb, year=2023, provinces=('Anhui', 'Chongqing', 'Shandong', 'Beijing'), out_xlsx='tables/Scenario_analysis_demo_2023.xlsx', out_fig='fig/Figure_10_scenario_analysis.tiff'):
    """
    Generate a compact scenario-analysis demo for tables and figures.

    Baseline: observed rows for the selected provinces in 2023
    S1: AFA/PU/ADY/PFU each -10%
    S2: CEA -10%
    S3: GAO +5%
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    df0 = _attach_province_name(df_raw, id_col='ID', new_col='Province')
    year_col = 'Year' if 'Year' in df0.columns else 'year' if 'year' in df0.columns else None
    if year_col is None:
        raise ValueError('Year column not found (Year/year).')
    df_y = df0[df0[year_col] == year].copy()
    if df_y.empty:
        raise ValueError(f'No data found for year={year}. Check the year range in the dataset.')
    if provinces is None or len(provinces) == 0:
        tmp = df_y[['Province', 'efficiency']].sort_values('efficiency')
        provinces = (tmp.iloc[0]['Province'], tmp.iloc[len(tmp) // 2]['Province'], tmp.iloc[-1]['Province'])
    df_sel = df_y[df_y['Province'].isin(list(provinces))].copy()
    if df_sel.empty:
        raise ValueError(f'Specified provinces={provinces} not found for year={year}. Check the Province mapping or province names.')
    yhat_base = _predict_vwlb_from_df(df_sel, feats, scaler, xgb_model, tslr_model, w_xgb)
    out = df_sel[['ID', 'Province', year_col, 'efficiency']].copy()
    out = out.rename(columns={'efficiency': 'SuperSBM_observed'})
    out['yhat_base'] = yhat_base
    scenarios = {'S1_input10': 'S1: -10% (AFA, PU, ADY, PFU)', 'S2_cea10': 'S2: -10% (CEA)', 'S3_gao5': 'S3: +5% (GAO)'}
    for key, label in scenarios.items():
        dfx = _apply_scenario(df_sel, key)
        yhat = _predict_vwlb_from_df(dfx, feats, scaler, xgb_model, tslr_model, w_xgb)
        out[f'yhat_{key}'] = yhat
        out[f'delta_{key}'] = yhat - yhat_base
    out.to_excel(out_xlsx, index=False)
    print(f'[Scenario] Saved: {out_xlsx}')
    try:
        import matplotlib.pyplot as plt
        plot_df = out[['Province', 'delta_S1_input10', 'delta_S2_cea10', 'delta_S3_gao5']].copy()
        plot_df = plot_df.set_index('Province')
        plot_df = plot_df.rename(columns={'delta_S1_input10': 'S1: inputs -10%', 'delta_S2_cea10': 'S2: CEA -10%', 'delta_S3_gao5': 'S3: GAO +5%'})
        plot_df.index = [s.replace(' ', '\n') for s in plot_df.index]
        plot_df.index = [s.replace(' ', '\n') for s in plot_df.index]
        ax = plot_df.plot(kind='bar', rot=0, figsize=(14, 6))
        ax.set_ylabel('Δ predicted eco-efficiency (VWLB)')
        ax.set_title(f'Scenario-based changes in predicted eco-efficiency (year={year})')
        ax.tick_params(axis='x', labelsize=9)
        plt.tight_layout()
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print('[Scenario] Plotting failed:', str(e))
    return out
from scipy.optimize import linprog

def _safe_pos(x, eps=1e-09):
    """Shift values away from zero before division."""
    x = np.asarray(x, dtype=float)
    x = np.where(x <= eps, eps, x)
    return x

def _solve_linprog(c, A_eq, b_eq, bounds, msg_prefix='LP'):
    """Single LP solver wrapper that raises a clear error on failure."""
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'{msg_prefix} failed: status={res.status}, msg={res.message}')
    return res

def _sbm_undesirable_vrs_standard(x0, y0, b0, X, Y, B, eps=1e-09):
    """
    Standard SBM with undesirable outputs under VRS.
    The Charnes-Cooper transform yields a linear program that minimizes rho.
    """
    x0 = _safe_pos(x0, eps)
    y0 = _safe_pos(y0, eps)
    b0 = _safe_pos(b0, eps)
    m = len(x0)
    s = len(y0)
    h = len(b0)
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
    A_eq = []
    b_eq = []
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
    bounds = [(0, None)] * dim
    res = _solve_linprog(c=c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, msg_prefix='SBM-STD')
    return float(res.fun)

def _sbm_undesirable_vrs_super(x0, y0, b0, X_wo, Y_wo, B_wo, eps=1e-09):
    """
    Leave-one-out Super-SBM with undesirable outputs under VRS.
    The linearized program minimizes sigma.
    """
    x0 = _safe_pos(x0, eps)
    y0 = _safe_pos(y0, eps)
    b0 = _safe_pos(b0, eps)
    m = len(x0)
    s = len(y0)
    h = len(b0)
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
    A_eq = []
    b_eq = []
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
    bounds = [(0, None)] * dim
    res = _solve_linprog(c=c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, msg_prefix='SBM-SUPER')
    return float(res.fun)

def super_sbm_undesirable_vrs_score_one(df_year, dmu_row, input_cols, good_cols, bad_cols, tol=1e-06, eps=1e-09):
    """
    Compute the efficiency score for one DMU within a given year.
    Standard SBM is used first; if rho is near one, super-SBM is then applied.
    """
    dfy = df_year.reset_index(drop=True)
    j0 = int(dmu_row)
    X = dfy[input_cols].to_numpy(dtype=float).T
    Y = dfy[good_cols].to_numpy(dtype=float).T
    B = dfy[bad_cols].to_numpy(dtype=float).T
    x0 = X[:, j0]
    y0 = Y[:, j0]
    b0 = B[:, j0]
    rho = _sbm_undesirable_vrs_standard(x0, y0, b0, X, Y, B, eps=eps)
    if rho < 1.0 - tol:
        return rho
    mask = np.ones(X.shape[1], dtype=bool)
    mask[j0] = False
    X_wo = X[:, mask]
    Y_wo = Y[:, mask]
    B_wo = B[:, mask]
    sigma = _sbm_undesirable_vrs_super(x0, y0, b0, X_wo, Y_wo, B_wo, eps=eps)
    return sigma

def select_provinces_quantile_stratified(df_raw, year=2023, id_col='ID', eff_col='efficiency', n_quantiles=5, n_total=12, seed=42, extra_to_extremes=True):
    """
    Stratified province sampling for the DEA recomputation check based on
    efficiency quantiles. Sampling is deterministic given the seed.
    """
    df0 = _attach_province_name(df_raw, id_col=id_col, new_col='Province')
    year_col = 'Year' if 'Year' in df0.columns else 'year' if 'year' in df0.columns else None
    if year_col is None:
        raise ValueError('Year column not found (Year/year).')
    df_y = df0[df0[year_col] == year].copy()
    if df_y.empty:
        raise ValueError(f'No data found for year={year}.')
    if eff_col not in df_y.columns:
        raise ValueError(f'Efficiency column not found: {eff_col}')
    tmp = df_y[['Province', eff_col]].dropna().copy()
    tmp = tmp.groupby('Province', as_index=False)[eff_col].mean()
    try:
        tmp['_qbin'] = pd.qcut(tmp[eff_col], q=n_quantiles, labels=False, duplicates='drop')
    except Exception:
        tmp['_qbin'] = pd.qcut(tmp[eff_col].rank(method='average'), q=n_quantiles, labels=False, duplicates='drop')
    bins = sorted([b for b in tmp['_qbin'].dropna().unique().tolist()])
    Q = len(bins)
    if Q <= 1:
        rng = np.random.default_rng(seed)
        provs = tmp['Province'].tolist()
        k = min(n_total, len(provs))
        return rng.choice(provs, size=k, replace=False).tolist()
    base = n_total // Q
    rem = n_total % Q
    alloc = {b: base for b in bins}
    if rem > 0:
        if extra_to_extremes and Q >= 2:
            order = [bins[0], bins[-1]] + [b for b in bins[1:-1]]
        else:
            order = bins
        for b in order:
            if rem == 0:
                break
            alloc[b] += 1
            rem -= 1
    rng = np.random.default_rng(seed)
    selected = []
    for b in bins:
        pool = tmp.loc[tmp['_qbin'] == b, 'Province'].tolist()
        k = min(alloc[b], len(pool))
        if k <= 0:
            continue
        if len(pool) <= k:
            selected.extend(pool)
        else:
            selected.extend(rng.choice(pool, size=k, replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    target_n = min(n_total, len(tmp))
    if len(selected) < target_n:
        remaining = [p for p in tmp['Province'].tolist() if p not in selected]
        k = target_n - len(selected)
        if k > 0 and len(remaining) > 0:
            selected.extend(rng.choice(remaining, size=min(k, len(remaining)), replace=False).tolist())
    return selected[:target_n]

def dea_recomputation_check(df_raw, feats, year=2023, provinces=('Anhui', 'Chongqing', 'Shandong', 'Beijing'), scenario_out=None, out_xlsx='tables/DEA_recomputation_check_2023.xlsx', id_col='ID'):
    """
    Recompute DEA for a sampled set of provinces under scenarios S1, S2, and S3,
    and compare surrogate delta-yhat with recomputed DEA delta-y values.
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    df0 = _attach_province_name(df_raw, id_col=id_col, new_col='Province')
    year_col = 'Year' if 'Year' in df0.columns else 'year' if 'year' in df0.columns else None
    if year_col is None:
        raise ValueError('Year column not found (Year/year).')
    df_y = df0[df0[year_col] == year].copy()
    if df_y.empty:
        raise ValueError(f'No data found for year={year}.')
    input_cols = ['TPAM', 'EIA', 'CS', 'AFA', 'PU', 'ADY', 'PFU', 'NRP']
    good_cols = ['GAO']
    bad_cols = ['CEA']
    need_cols = [id_col, 'Province', year_col, 'efficiency'] + input_cols + good_cols + bad_cols
    for c in need_cols:
        if c not in df_y.columns:
            raise ValueError(f'DEA recomputation is missing required column: {c}')
    df_sel = df_y[df_y['Province'].isin(list(provinces))].copy().reset_index(drop=True)
    if df_sel.empty:
        raise ValueError(f'provinces={provinces} not found for year={year}.')
    scen_keys = ['S1_input10', 'S2_cea10', 'S3_gao5']
    rows = []
    errors = []
    for p in provinces:
        try:
            df_base = df_y.copy().reset_index(drop=True)
            hit = df_base.index[df_base['Province'] == p]
            if len(hit) == 0:
                raise ValueError(f'Province={p} not found in year={year}')
            idx = int(hit[0])
            score_base = super_sbm_undesirable_vrs_score_one(df_year=df_base, dmu_row=idx, input_cols=input_cols, good_cols=good_cols, bad_cols=bad_cols)
            score_s = {}
            delta_s = {}
            for sk in scen_keys:
                try:
                    df_cf = df_base.copy()
                    df_cf.loc[idx, :] = _apply_scenario(df_cf.loc[[idx], :], sk).iloc[0]
                    sc = super_sbm_undesirable_vrs_score_one(df_year=df_cf, dmu_row=idx, input_cols=input_cols, good_cols=good_cols, bad_cols=bad_cols)
                    score_s[sk] = sc
                    delta_s[sk] = sc - score_base
                except Exception as e_sk:
                    score_s[sk] = np.nan
                    delta_s[sk] = np.nan
                    errors.append({'Province': p, 'Scenario': sk, 'Stage': 'scenario_solve', 'Error': str(e_sk)})
            sur = {'S1_input10': np.nan, 'S2_cea10': np.nan, 'S3_gao5': np.nan}
            if scenario_out is not None and isinstance(scenario_out, pd.DataFrame) and ('Province' in scenario_out.columns):
                r = scenario_out[scenario_out['Province'] == p]
                if not r.empty:
                    sur['S1_input10'] = float(r['delta_S1_input10'].iloc[0])
                    sur['S2_cea10'] = float(r['delta_S2_cea10'].iloc[0])
                    sur['S3_gao5'] = float(r['delta_S3_gao5'].iloc[0])
            obs = float(df_base.loc[idx, 'efficiency'])
            rows.append({'Province': p, 'SuperSBM_observed': obs, 'DEA_base': score_base, 'DEA_minus_observed': score_base - obs, 'DEA_S1': score_s['S1_input10'], 'DEA_delta_S1': delta_s['S1_input10'], 'SUR_delta_S1': sur['S1_input10'], 'DEA_S2': score_s['S2_cea10'], 'DEA_delta_S2': delta_s['S2_cea10'], 'SUR_delta_S2': sur['S2_cea10'], 'DEA_S3': score_s['S3_gao5'], 'DEA_delta_S3': delta_s['S3_gao5'], 'SUR_delta_S3': sur['S3_gao5']})
        except Exception as e_p:
            errors.append({'Province': p, 'Scenario': 'BASE', 'Stage': 'baseline_solve', 'Error': str(e_p)})
            rows.append({'Province': p, 'SuperSBM_observed': np.nan, 'DEA_base': np.nan, 'DEA_minus_observed': np.nan, 'DEA_S1': np.nan, 'DEA_delta_S1': np.nan, 'SUR_delta_S1': np.nan, 'DEA_S2': np.nan, 'DEA_delta_S2': np.nan, 'SUR_delta_S2': np.nan, 'DEA_S3': np.nan, 'DEA_delta_S3': np.nan, 'SUR_delta_S3': np.nan})
    check_df = pd.DataFrame(rows)

    def _sign_agree(a, b, tol=1e-12):
        a = float(a)
        b = float(b)
        if abs(a) <= tol and abs(b) <= tol:
            return 1.0
        return 1.0 if np.sign(a) == np.sign(b) else 0.0
    summary = []
    for tag, dcol, scol in [('S1', 'DEA_delta_S1', 'SUR_delta_S1'), ('S2', 'DEA_delta_S2', 'SUR_delta_S2'), ('S3', 'DEA_delta_S3', 'SUR_delta_S3')]:
        tmp = check_df[[dcol, scol]].dropna()
        if tmp.empty:
            summary.append({'Scenario': tag, 'N': 0, 'sign_agreement': np.nan, 'MAE': np.nan})
        else:
            sa = np.mean([_sign_agree(a, b) for a, b in zip(tmp[dcol], tmp[scol])])
            mae = np.mean(np.abs(tmp[dcol].values - tmp[scol].values))
            summary.append({'Scenario': tag, 'N': int(len(tmp)), 'sign_agreement': float(sa), 'MAE': float(mae)})
    base_ok = check_df[['DEA_minus_observed']].dropna()
    base_mae = float(np.mean(np.abs(base_ok['DEA_minus_observed'].values))) if len(base_ok) else np.nan
    summary.append({'Scenario': 'BASE', 'N': int(len(base_ok)), 'sign_agreement': np.nan, 'MAE': base_mae})
    summary_df = pd.DataFrame(summary)
    err_df = pd.DataFrame(errors) if len(errors) else pd.DataFrame(columns=['Province', 'Scenario', 'Stage', 'Error'])
    with pd.ExcelWriter(out_xlsx) as w:
        check_df.to_excel(w, sheet_name='check_by_province', index=False)
        summary_df.to_excel(w, sheet_name='summary', index=False)
        if len(err_df):
            err_df.to_excel(w, sheet_name='errors', index=False)
    print(f'[DEA recompute check] Saved: {out_xlsx} (errors={len(err_df)})')
    return (check_df, summary_df)

def main():
    RUN_ABLATION = True
    RUN_DEA_RECOMPUTE_CHECK = True
    ABLATION_OUT = 'tables/Ablation and leakage-sensitivity checks.xlsx'
    df = load_data('./data.xlsx', 'Sheet1')
    X, y, feats = preprocess_numeric(df)
    N, F = X.shape
    print(f'Samples: {N}, Features: {F}')
    print(f'Target summary: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}')
    tr_idx, va_idx, te_idx, used_bins = stratified_train_val_test_split(X, y, test_size=0.2, val_size=0.2, max_bins=10, random_state=42)
    if used_bins:
        print(f'Outer stratified split used {used_bins} quantile bins')
    X_tr, y_tr = (X[tr_idx], y[tr_idx])
    X_va, y_va = (X[va_idx], y[va_idx])
    X_te, y_te = (X[te_idx], y[te_idx])
    models_flat_dl = {'BPNN': lambda F: build_bpnn(F), 'TSLR-MLP': lambda F: build_tslr_mlp_base(F), 'VanillaMLP': lambda F: build_mlp(F)}
    monotone = '(-1,-1,-1,-1,-1,-1,-1,-1,1,-1)'
    models_ml = {'XGBoost': lambda F: make_xgb_regressor(monotone_constraints=monotone, random_state=42, n_jobs=-1), 'Random Forest': lambda F: RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1), 'SVM': lambda F: SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'), 'Ridge': lambda F: Ridge(alpha=1.0), 'Lasso': lambda F: Lasso(alpha=0.001, max_iter=10000)}
    print('\n===== Repeated stratified K-fold cross-validation on the training domain =====')
    cv_summary = cross_validate(models_flat_dl, models_ml, X_tr, y_tr, repeats=2, folds=5, max_bins=10)
    df_cv = pd.DataFrame({'Model': list(cv_summary.keys()), 'Val_R2_mean': [cv_summary[m]['R2'][0] for m in cv_summary], 'Val_R2_std': [cv_summary[m]['R2'][1] for m in cv_summary], 'Val_MSE_mean': [cv_summary[m]['MSE'][0] for m in cv_summary], 'Val_MSE_std': [cv_summary[m]['MSE'][1] for m in cv_summary], 'Val_MAE_mean': [cv_summary[m]['MAE'][0] for m in cv_summary], 'Val_MAE_std': [cv_summary[m]['MAE'][1] for m in cv_summary]}).sort_values(by='Val_R2_mean', ascending=False)
    print('\n==== Cross-validation summary on the training domain (sorted by Val_R2_mean) ====')
    print(df_cv.to_string(index=False))
    os.makedirs('tables', exist_ok=True)
    df_cv.to_excel('tables/cv_summary.xlsx', index=False)
    print('Saved: tables/cv_summary.xlsx')
    best_cv_name = df_cv.iloc[0]['Model']
    print(f'\nRepresentative model selected from repeated K-fold CV: {best_cv_name}')
    print('\n===== Outer validation for early stopping + final test evaluation =====')
    scaler_outer = RobustScaler().fit(X_tr)
    X_tr_s = scaler_outer.transform(X_tr)
    X_va_s = scaler_outer.transform(X_va)
    X_te_s = scaler_outer.transform(X_te)
    results_test = {}
    history_dict = {}
    tslr_model_keep = None
    for name, builder in models_flat_dl.items():
        tf.keras.backend.clear_session()
        model = builder(F)
        if name == 'TSLR-MLP':
            hist, model = train_tslr_mlp(model, X_tr_s, y_tr, X_va_s, y_va, epochs=60, batch=32, name=name)
        else:
            hist, model = train_dl(model, X_tr_s, y_tr, X_va_s, y_va, epochs=120, batch=32, name=name)
        y_pred_te = model.predict(X_te_s, verbose=0).ravel()
        y_pred_va = model.predict(X_va_s, verbose=0).ravel()
        results_test[name] = {'y_pred': y_pred_te, 'test_metrics': metrics_dict(y_te, y_pred_te), 'y_pred_val': y_pred_va}
        history_dict[name] = hist
        if name == 'TSLR-MLP':
            tslr_model_keep = model
        else:
            del model
            gc.collect()
    xgb_model = models_ml['XGBoost'](F)
    xgb_model, y_pred_va_xgb = fit_xgb_compat(xgb_model, X_tr_s, y_tr, X_va_s, y_va)
    y_pred_te_xgb = xgb_model.predict(X_te_s)
    results_test['XGBoost'] = {'y_pred': y_pred_te_xgb, 'test_metrics': metrics_dict(y_te, y_pred_te_xgb), 'y_pred_val': y_pred_va_xgb}
    rf_model = models_ml['Random Forest'](F)
    rf_model.fit(X_tr_s, y_tr)
    y_pred = rf_model.predict(X_te_s)
    results_test['Random Forest'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': rf_model.predict(X_va_s)}
    svm_model = models_ml['SVM'](F)
    svm_model.fit(X_tr_s, y_tr)
    y_pred = svm_model.predict(X_te_s)
    results_test['SVM'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': svm_model.predict(X_va_s)}
    ridge_model = models_ml['Ridge'](F)
    ridge_model.fit(X_tr_s, y_tr)
    y_pred = ridge_model.predict(X_te_s)
    results_test['Ridge'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': ridge_model.predict(X_va_s)}
    lasso_model = models_ml['Lasso'](F)
    lasso_model.fit(X_tr_s, y_tr)
    y_pred = lasso_model.predict(X_te_s)
    results_test['Lasso'] = {'y_pred': y_pred, 'test_metrics': metrics_dict(y_te, y_pred), 'y_pred_val': lasso_model.predict(X_va_s)}
    if 'XGBoost' in results_test and 'TSLR-MLP' in results_test:
        w, val_mse = best_weight_by_val(y_va, results_test['XGBoost']['y_pred_val'], results_test['TSLR-MLP']['y_pred_val'])
        y_pred_ens = w * results_test['XGBoost']['y_pred'] + (1.0 - w) * results_test['TSLR-MLP']['y_pred']
        results_test[f'Ensemble(XGB{w:.2f}+TSLR-MLP{1 - w:.2f})'] = {'y_pred': y_pred_ens, 'test_metrics': metrics_dict(y_te, y_pred_ens), 'info': {'w_xgb': w, 'val_mse': val_mse}}
        print(f'\n[VWLB] Outer-val auto-selected weight: w_xgb={w:.2f} (Val MSE={val_mse:.6f})')
        try:
            if tslr_model_keep is not None:
                scenario_out = scenario_analysis_demo(df_raw=df, feats=feats, scaler=scaler_outer, xgb_model=xgb_model, tslr_model=tslr_model_keep, w_xgb=w, year=2023, provinces=('Anhui', 'Chongqing', 'Shandong', 'Beijing'))
                if RUN_DEA_RECOMPUTE_CHECK:
                    try:
                        prov_q = select_provinces_quantile_stratified(df_raw=df, year=2023, id_col='ID', eff_col='efficiency', n_quantiles=5, n_total=12, seed=42, extra_to_extremes=True)
                        print(f'[DEA recompute check] Stratified sample size={len(prov_q)}: {prov_q}')
                        scenario_out_q = scenario_analysis_demo(df_raw=df, feats=feats, scaler=scaler_outer, xgb_model=xgb_model, tslr_model=tslr_model_keep, w_xgb=w, year=2023, provinces=tuple(prov_q), out_xlsx='tables/Scenario_analysis_recompute_subset_2023.xlsx', out_fig='fig/Figure_SX_scenario_recompute_subset_2023.tiff')
                        _check_df, _summary_df = dea_recomputation_check(df_raw=df, feats=feats, year=2023, provinces=tuple(prov_q), scenario_out=scenario_out_q, out_xlsx='tables/DEA_recomputation_check_2023_stratified12.xlsx')
                        print('[DEA recompute check] summary:')
                        print(_summary_df.to_string(index=False))
                    except Exception as _e:
                        print('[DEA recompute check] Failed:', str(_e))
            else:
                print('[Scenario] tslr_model_keep=None; skipping the scenario demo.')
        except Exception as e:
            print('[Scenario] Failed to generate the scenario demo:', str(e))
    print('\n===== OOF stacking =====')
    base_names, X_tr_meta, X_va_meta, X_te_meta = stacking_oof_predict(models_flat_dl, models_ml, X_tr, y_tr, X_va, X_te, folds=5, max_bins=10)
    meta_name, y_va_hat_stack, y_te_hat_stack, meta_info = choose_meta_and_predict(base_names, X_tr_meta, y_tr, X_va_meta, y_va, X_te_meta)
    results_test[f'Stacking[{meta_name}]'] = {'y_pred': y_te_hat_stack, 'test_metrics': metrics_dict(y_te, y_te_hat_stack), 'y_pred_val': y_va_hat_stack, 'info': meta_info}
    print(f"[Stacking] Selected meta-learner: {meta_name} | Val MSE={meta_info['val_mse']:.6f}")
    df_test = pd.DataFrame({'Model': list(results_test.keys()), 'MSE': [results_test[m]['test_metrics']['MSE'] for m in results_test], 'MAE': [results_test[m]['test_metrics']['MAE'] for m in results_test], 'R²': [results_test[m]['test_metrics']['R2'] for m in results_test]}).sort_values(by='R²', ascending=False).reset_index(drop=True)
    print('\n' + '=' * 76)
    print('Test-set model performance summary (sorted by R2 descending):')
    print('=' * 76)
    print(df_test.to_string(index=False))
    df_test.to_excel('tables/test_results.xlsx', index=False)
    print('Saved: tables/test_results.xlsx')
    print('\n==== Bootstrap 95% CI for test-set MSE differences (A-B) vs the representative CV model ====')
    base = str(best_cv_name)
    if base not in results_test:
        base = 'XGBoost' if 'XGBoost' in results_test else list(results_test.keys())[0]
    y_base = results_test[base]['y_pred']
    rows = []
    for name in results_test:
        if name == base:
            continue
        mean_diff, ci, p = bootstrap_mse_diff(y_te, results_test[name]['y_pred'], y_base, B=2000, seed=42)
        rows.append({'Model_vs_Base': f'{name} - {base}', 'MeanDiff(A-B)': mean_diff, 'CI95_low': ci[0], 'CI95_high': ci[1], 'p(two-sided)': p})
    df_boot = pd.DataFrame(rows).sort_values(by='MeanDiff(A-B)')
    print(df_boot.to_string(index=False))
    df_boot.to_excel('tables/bootstrap_vs_base.xlsx', index=False)
    print('Saved: tables/bootstrap_vs_base.xlsx')
    plot_comparison(results_test, y_te, history_dict, title_suffix=' (Test)')
    df_tbl3 = make_figure5_and_table3(y_true=y_te, results_test=results_test, out_fig='fig/Figure_7_Bootstrap_distribution_of_MSE_differences_(model_minus_monotone_XGBoost_baseline).tiff', out_xlsx='tables/Table_3.xlsx', B=2000, seed=42)
    print('\nTable 3 preview:')
    print(df_tbl3.to_string(index=False))
    print('\n===== Ablation and leakage-sensitivity checks =====')
    df_ab_m, df_ab_b = run_ablation_and_leakage_sensitivity_checks(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te, X_tr_s=X_tr_s, X_va_s=X_va_s, X_te_s=X_te_s, models_flat_dl=models_flat_dl, models_ml=models_ml, results_test=results_test, monotone_constraints_str=monotone, meta_name_proper=meta_name, B_boot=2000, seed=42, out_xlsx=ABLATION_OUT, RUN=RUN_ABLATION)
    if df_ab_m is not None:
        print('\n[Ablation] Metrics preview (top 10 rows):')
        print(df_ab_m.head(10).to_string(index=False))
    if df_ab_b is not None:
        print('\n[Ablation] Bootstrap_MSEdiff preview (top 10 rows):')
        print(df_ab_b.head(10).to_string(index=False))
    print('\n===== Panel-dependence robustness checks (rolling time-forward / province-block) =====')
    df_panel = run_panel_dependence_robustness_checks(df_raw=df, models_flat_dl=models_flat_dl, models_ml=models_ml, monotone_constraints_str=monotone, base_results_random_split=results_test, out_xlsx='tables/Panel_dependence_robustness_checks.xlsx', RUN=True, TF_VAL_WINDOW=1, TF_TEST_WINDOW=3, TF_MIN_TRAIN_YEARS=10, TF_MAX_SPLITS=None, PB_RANDOM_STATE=42)
    if df_panel is not None:
        print('\n[Panel-robustness] summary_mean_std preview (top 20 rows):')
        print(df_panel.head(20).to_string(index=False))
    try:
        import shap
        print('\n===== Generate and save local SHAP plots for XGBoost =====')
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_te_s)
        shap.summary_plot(shap_values, features=X_te_s, feature_names=feats, show=False)
        plt.tight_layout()
        plt.savefig(f'fig/Figure_8.tiff', dpi=300, bbox_inches='tight')
        plt.close()

        def mean_abs_importance(sv):
            if isinstance(sv, list):
                return np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            return np.abs(sv).mean(axis=0)
        importances = mean_abs_importance(shap_values)
        top_idx = np.argsort(importances)[::-1][:3]
        sv_for_dep = shap_values[0] if isinstance(shap_values, list) else shap_values
        for rank, i in enumerate(top_idx, start=1):
            plt.figure()
            shap.dependence_plot(i, sv_for_dep, X_te_s, feature_names=feats, show=False, interaction_index=None)
            plt.tight_layout()
            fname = f'fig/Figure_9_top{rank}_{feats[i]}.tiff'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print('\nSHAP plots were not generated/saved:', str(e))
if __name__ == '__main__':
    main()


# Auxiliary post-processing for panel robustness output

import re
import numpy as np
import pandas as pd

PATH = "tables/Panel_dependence_robustness_checks.xlsx"
sheets = pd.read_excel(PATH, sheet_name=None)
print("Sheets:", list(sheets.keys()))

def pick_best_sheet(sheets: dict) -> str:
    best_detail = None
    best_summary = None
    for name, df in sheets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = set(df.columns)
        if {"SplitType", "Model", "MSE", "MAE"}.issubset(cols) and (
            "R2" in cols or "R²" in cols or "R^2" in cols
        ):
            best_detail = name
            break
        if {"SplitType", "Model", "n_splits", "MSE_mean", "MAE_mean", "R2_mean"}.issubset(cols):
            best_summary = name
    return best_detail or best_summary

sheet_name = pick_best_sheet(sheets)
if sheet_name is None:
    raise ValueError(
        "No usable sheet found. Expected either detail columns (MSE/MAE/R2) "
        "or summary columns (MSE_mean/...)."
    )

df = sheets[sheet_name].copy()
print(f"[Use sheet] {sheet_name}")
print("[Columns]", list(df.columns))

rename_map = {}
if "R²" in df.columns:
    rename_map["R²"] = "R2"
if "R^2" in df.columns:
    rename_map["R^2"] = "R2"
if "R²_mean" in df.columns:
    rename_map["R²_mean"] = "R2_mean"
if "R^2_mean" in df.columns:
    rename_map["R^2_mean"] = "R2_mean"
df = df.rename(columns=rename_map)

df["Model_raw"] = df["Model"].astype(str).str.strip()

def canonical_model(name: str) -> str:
    name = str(name).strip()
    if name.startswith("Ensemble("):
        return "VWLB"
    if name.startswith("Stacking["):
        return "OOF-Stacking"
    if name == "VanillaMLP":
        return "MLP"
    return name

df["Model_canon"] = df["Model_raw"].apply(canonical_model)

def extract_w_xgb(name: str):
    match = re.search(r"Ensemble\(XGB([0-9]*\.?[0-9]+)\+TSLR-MLP", str(name))
    return float(match.group(1)) if match else np.nan

def extract_meta(name: str):
    match = re.search(r"^Stacking\[(.+?)\]$", str(name).strip())
    return match.group(1) if match else np.nan

df["VWLB_w_xgb"] = df["Model_raw"].apply(extract_w_xgb)
df["Stack_meta"] = df["Model_raw"].apply(extract_meta)

is_detail = {"MSE", "MAE"}.issubset(df.columns) and ("R2" in df.columns)
is_summary = {"n_splits", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "R2_mean"}.issubset(df.columns)

if not (is_detail or is_summary):
    raise ValueError(
        "The selected sheet does not look like either a detail table or a summary table."
    )

def agg_from_detail(df_detail: pd.DataFrame) -> pd.DataFrame:
    for col in ["MSE", "MAE", "R2"]:
        df_detail[col] = pd.to_numeric(df_detail[col], errors="coerce")
    df_detail = df_detail.dropna(subset=["MSE", "MAE", "R2"]).reset_index(drop=True)

    out = (
        df_detail.groupby(["SplitType", "Model_canon"], as_index=False)
        .agg(
            n_splits=("MSE", "size"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            R2_mean=("R2", "mean"),
            R2_std=("R2", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        )
    )
    return out.sort_values(["SplitType", "R2_mean"], ascending=[True, False]).reset_index(drop=True)

def pooled_mean_std(n, mean, std):
    n = np.asarray(n, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    n = np.where(np.isnan(n), 0.0, n)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 0.0, std)

    total_n = n.sum()
    if total_n <= 0:
        return 0, np.nan, np.nan

    mean_pool = (n * mean).sum() / total_n
    if total_n == 1:
        return int(total_n), float(mean_pool), 0.0

    ss_within = ((np.maximum(n - 1, 0.0)) * (std ** 2)).sum()
    ss_between = (n * (mean - mean_pool) ** 2).sum()
    var_pool = (ss_within + ss_between) / (total_n - 1)
    std_pool = np.sqrt(max(var_pool, 0.0))
    return int(total_n), float(mean_pool), float(std_pool)

def agg_from_summary(df_sum: pd.DataFrame) -> pd.DataFrame:
    for col in ["n_splits", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"]:
        if col in df_sum.columns:
            df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce")
    df_sum["n_splits"] = df_sum["n_splits"].fillna(0).astype(int)

    rows = []
    for (split, model), group in df_sum.groupby(["SplitType", "Model_canon"]):
        n = group["n_splits"].values

        n_mse, mse_mean, mse_std = pooled_mean_std(
            n, group["MSE_mean"].values, group.get("MSE_std", pd.Series([0] * len(group))).values
        )
        n_mae, mae_mean, mae_std = pooled_mean_std(
            n, group["MAE_mean"].values, group.get("MAE_std", pd.Series([0] * len(group))).values
        )
        n_r2, r2_mean, r2_std = pooled_mean_std(
            n, group["R2_mean"].values, group.get("R2_std", pd.Series([0] * len(group))).values
        )

        rows.append(
            {
                "SplitType": split,
                "Model": model,
                "n_splits": int(max(n_mse, n_mae, n_r2)),
                "MSE_mean": mse_mean,
                "MSE_std": mse_std,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "R2_mean": r2_mean,
                "R2_std": r2_std,
            }
        )

    out = pd.DataFrame(rows).sort_values(["SplitType", "R2_mean"], ascending=[True, False]).reset_index(drop=True)
    return out

if is_detail:
    print("[Mode] DETAIL: regroup from split-level records")
    df_agg = agg_from_detail(df)
else:
    print("[Mode] SUMMARY: pooled recombination for VWLB and OOF stacking")
    df_agg = agg_from_summary(df)

print("\n=== Re-aggregated results with canonical model names ===")
print(df_agg.to_string(index=False))

df_w = df[df["Model_raw"].str.startswith("Ensemble(")].copy()
if not df_w.empty:
    if is_detail:
        w_stats = (
            df_w.groupby("SplitType")["VWLB_w_xgb"]
            .agg(n="count", mean="mean", std="std", min="min", max="max")
            .reset_index()
        )
        w_stats["std"] = w_stats["std"].fillna(0.0)
    else:
        def w_agg(group):
            w = group["VWLB_w_xgb"].values.astype(float)
            n = group["n_splits"].values.astype(float)
            mask = (~np.isnan(w)) & (n > 0)
            w, n = w[mask], n[mask]
            if len(w) == 0:
                return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
            total_w = n.sum()
            mu = (w * n).sum() / total_w
            var = (n * (w - mu) ** 2).sum() / total_w if total_w > 0 else 0.0
            return pd.Series(
                {
                    "n": int(total_w),
                    "mean": float(mu),
                    "std": float(np.sqrt(max(var, 0.0))),
                    "min": float(w.min()),
                    "max": float(w.max()),
                }
            )

        w_stats = df_w.groupby("SplitType").apply(w_agg).reset_index()

    print("\n=== VWLB w_xgb distribution by SplitType ===")
    print(w_stats.to_string(index=False))
else:
    print("\n[Info] No Ensemble(...) rows found; VWLB weights were not summarized.")

df_s = df[df["Model_raw"].str.startswith("Stacking[")].copy()
if not df_s.empty:
    if is_detail:
        meta_freq = (
            df_s.groupby(["SplitType", "Stack_meta"])
            .size()
            .reset_index(name="count")
            .sort_values(["SplitType", "count"], ascending=[True, False])
        )
    else:
        meta_freq = (
            df_s.groupby(["SplitType", "Stack_meta"])["n_splits"]
            .sum()
            .reset_index(name="count")
            .sort_values(["SplitType", "count"], ascending=[True, False])
        )

    print("\n=== Stacking meta-learner frequency by SplitType ===")
    print(meta_freq.to_string(index=False))
else:
    print("\n[Info] No Stacking[...] rows found; meta-learner frequency was not summarized.")

OUT = "tables/Panel_dependence_robustness_checks_reaggregated.xlsx"
with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name=f"source_{sheet_name}", index=False)
    df_agg.to_excel(writer, sheet_name="agg_canonical_mean_std", index=False)
    if "w_stats" in locals() and isinstance(w_stats, pd.DataFrame):
        w_stats.to_excel(writer, sheet_name="vwlb_weight_stats", index=False)
    if "meta_freq" in locals() and isinstance(meta_freq, pd.DataFrame):
        meta_freq.to_excel(writer, sheet_name="stack_meta_frequency", index=False)

print(f"\n[Saved] -> {OUT}")
