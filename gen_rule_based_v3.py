#!/usr/bin/env python3
"""Generate rule_based_strategy.ipynb — optimization-based v3."""
import json
from pathlib import Path

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = []

# ── 0: Title ──────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# Rule-Based Regime Trading Strategy\n\n"
    "Stocks scored by a **regime-aware optimisation**: jointly learned factor "
    "coefficients β, industry weights w, and softmax sharpness parameters ε/τ "
    "per regime, maximising Sharpe over the training window via SLSQP.\n"
    "Walk-forward evaluation identical to the baseline model notebook."
))

# ── 1: Imports header ─────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Imports"))

# ── 2: Imports ────────────────────────────────────────────────────────────────
cells.append(code_cell(
"""import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams.update({'figure.dpi': 100, 'axes.spines.top': False, 'axes.spines.right': False})"""
))

# ── 3: Config header ──────────────────────────────────────────────────────────
cells.append(md_cell("## 2. Configuration"))

# ── 4: Config ─────────────────────────────────────────────────────────────────
cells.append(code_cell(
"""BASE_DIR     = Path('/Users/gabriel/Desktop/Sentiment Model')
SNAPSHOT_DIR = BASE_DIR / 'portfolio_snapshots'

# Target (used for training signal and IC evaluation)
RETURN_HORIZON = '21d'
TARGET_COL     = f'target_ret_{RETURN_HORIZON}'
HORIZON_DAYS   = int(RETURN_HORIZON.replace('d', ''))
ANN_FACTOR     = 252 / HORIZON_DAYS   # 12  — for IC-based metrics only

# OOS portfolio returns are 1d, so portfolio metrics use this factor
PORT_ANN_FACTOR = 252

# Date range
DATA_START = '2018-01-01'
DATA_END   = '2025-12-31'

# Walk-forward (expanding window — identical to baseline)
TRAIN_YEARS = 2
TEST_YEARS  = 1

# Universe filter
MCAP_FILTER_B = 5.0

# FNG direction
USE_DIRECTION   = True
SMOOTH_WINDOW   = 15
SLOPE_THRESHOLD = 0

# Regimes where we rebalance; inactive regimes -> hold last portfolio
ACTIVE_REGIMES = {
    'Extreme Fear', 'Fear-Falling', 'Fear-Rising',
    'Extreme Greed', 'Greed-Rising', 'Greed-Falling',
}

# Portfolio construction
QUANTILE = 0.10    # top / bottom X% of stocks per industry

# Factors used to score stocks (z-scored cross-sectionally per date)
FACTOR_COLS = ['beta_spy', 'vol_std_21', 'roc_21', 'rsi_21']

# Optimisation hyperparameters
RIDGE_ALPHA     = 1.0    # L2 penalty for β ridge regression (stage 1)
L2_LAMBDA       = 0.5    # L2 penalty on industry weights w — increased from 0.05 to prevent w overfitting
SOFT_EPS        = 0.05   # initial value for ε (learned in stage 2)
N_RESTARTS      = 5      # random SLSQP restarts (stage 2 only)
MIN_REGIME_DAYS = 30     # min training dates before fallback to heuristic

# Fixed hyperparameters — NOT learned by SLSQP.
# τ: SLSQP always pushes τ→0 (single-stock selection) to maximise in-sample Sharpe.
#    Fix at a value that keeps selection broad: 0.10 means top stock gets ~2x median weight.
FIXED_TAU = 0.10
# λ: SLSQP always pushes λ→1 (full daily rebalance) in-sample.
#    Fix based on return horizon: half-life = log(0.5)/log(1-λ).
#    FIXED_LAM=0.10 → ~6 day half-life. Appropriate for capturing 21d alpha without excess turnover.
FIXED_LAM = 0.10

# ── β sharing toggle ─────────────────────────────────────────────────────────
# True  → single β fit on ALL training data (all regimes combined).
#          Only w and ε are regime-specific. More observations → stabler OOS IC.
# False → β fit per-regime (original behaviour). Fewer observations per regime
#         (~100–200 dates) can yield unstable β and negative OOS IC.
SHARED_BETA = True

print(f'Target        : {TARGET_COL}  (ANN_FACTOR={ANN_FACTOR}, PORT_ANN_FACTOR={PORT_ANN_FACTOR})')
print(f'WF folds      : {TRAIN_YEARS}yr expanding train -> {TEST_YEARS}yr test')
print(f'MCAP filter   : >= ${MCAP_FILTER_B}B')
print(f'USE_DIRECTION : {USE_DIRECTION}  (window={SMOOTH_WINDOW}d, threshold={SLOPE_THRESHOLD})')
print(f'QUANTILE      : top/bottom {int(QUANTILE*100)}% per industry')
print(f'FACTOR_COLS   : {FACTOR_COLS}')
print(f'Active regimes: {sorted(ACTIVE_REGIMES)}')
print(f'RIDGE_ALPHA   : {RIDGE_ALPHA}')
print(f'L2_LAMBDA     : {L2_LAMBDA}')
print(f'FIXED_TAU     : {FIXED_TAU}  (softmax temperature, fixed)')
print(f'FIXED_LAM     : {FIXED_LAM}  (EWM speed, half-life ~{round(-np.log(2)/np.log(1-FIXED_LAM), 1)}d)')
print(f'SHARED_BETA   : {SHARED_BETA}  ({"global β across all regimes" if SHARED_BETA else "per-regime β"})')
print(f'Snapshots     : {SNAPSHOT_DIR}')"""
))

# ── 5: Regime header ──────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 3. Regime Map\n\n"
    "Regimes are defined by FNG quintile × rolling-slope direction. "
    "**Active regimes** trigger a portfolio rebalance; **inactive regimes** "
    "hold the most recent active portfolio and track its actual 1d returns.\n\n"
    "`is_held = True` on a given day means the regime was inactive or a fallback — "
    "the portfolio was not rebalanced, we simply carry forward the previous selection."
))

# ── 6: Regime code ────────────────────────────────────────────────────────────
cells.append(code_cell(
"""REGIME_BINS  = [0, 25, 45, 55, 75, 100]
REGIME_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
REGIME_COLORS = {
    'Extreme Fear':  '#d62728',
    'Fear':          '#ff7f0e',
    'Neutral':       '#aec7e8',
    'Greed':         '#2ca02c',
    'Extreme Greed': '#17becf',
}

all_possible = {
    'Extreme Fear', 'Fear-Rising', 'Fear-Falling', 'Fear-Stable',
    'Neutral', 'Greed-Rising', 'Greed-Falling', 'Greed-Stable', 'Extreme Greed',
}
print('Active regimes (portfolio rebalanced):')
for r in sorted(ACTIVE_REGIMES):
    print(f'  {r}')
print('\\nInactive regimes (hold last portfolio):')
for r in sorted(all_possible - ACTIVE_REGIMES):
    print(f'  {r}')"""
))

# ── 7: Data Loading header ────────────────────────────────────────────────────
cells.append(md_cell("## 4. Data Loading"))

# ── 8: Stock parquets ─────────────────────────────────────────────────────────
cells.append(code_cell(
"""STOCK_COLS = ['date', 'ticker',
              'beta_spy', 'vol_std_21', 'roc_21', 'rsi_21',
              'target_ret_1d', 'target_ret_3d', 'target_ret_5d',
              'target_ret_10d', 'target_ret_21d']

batches = sorted((BASE_DIR / 'stock_training_data_final').glob('batch_*.parquet'))
print(f'Loading {len(batches)} parquet batches...')
stock_df = pd.concat(
    [pd.read_parquet(p, columns=STOCK_COLS) for p in tqdm(batches)],
    ignore_index=True
)
if stock_df['date'].dt.tz is not None:
    stock_df['date'] = stock_df['date'].dt.tz_localize(None)
stock_df['date'] = pd.to_datetime(stock_df['date']).dt.normalize()
print(f'Stock data: {stock_df.shape}')"""
))

# ── 9: Market data ────────────────────────────────────────────────────────────
cells.append(code_cell(
"""mkt_path = BASE_DIR / 'market_training_data_final' / 'market_state_vector.parquet'
mkt_df   = pd.read_parquet(mkt_path, columns=['fear_greed']).reset_index()
mkt_df   = mkt_df.rename(columns={mkt_df.columns[0]: 'date'})
if mkt_df['date'].dt.tz is not None:
    mkt_df['date'] = mkt_df['date'].dt.tz_localize(None)
mkt_df['date'] = pd.to_datetime(mkt_df['date']).dt.normalize()
mkt_df = mkt_df.sort_values('date').reset_index(drop=True)

mkt_df['regime_base'] = pd.cut(
    mkt_df['fear_greed'], bins=REGIME_BINS,
    labels=REGIME_ORDER, include_lowest=True
).astype(str)

print(f'Market data: {mkt_df.shape}  |  {mkt_df["date"].min().date()} to {mkt_df["date"].max().date()}')"""
))

# ── 10: FNG slope + direction ─────────────────────────────────────────────────
cells.append(code_cell(
"""def rolling_fng_slope(fng_series, window):
    slopes = np.full(len(fng_series), np.nan)
    vals   = fng_series.values
    x      = np.arange(window, dtype=float)
    for i in range(window, len(vals)):
        y = vals[i - window: i]
        if np.isnan(y).any():
            continue
        slopes[i] = np.polyfit(x, y, 1)[0]
    return pd.Series(slopes, index=fng_series.index)

mkt_df['fng_slope'] = rolling_fng_slope(mkt_df['fear_greed'], SMOOTH_WINDOW)

def slope_to_direction(slope_series, threshold):
    d = pd.Series('Stable', index=slope_series.index)
    d[slope_series >= threshold] = 'Rising'
    d[slope_series < -threshold] = 'Falling'
    return d

mkt_df['direction'] = slope_to_direction(mkt_df['fng_slope'], SLOPE_THRESHOLD)

def build_regime_label(row):
    base = row['regime_base']
    if not USE_DIRECTION or base in ('Extreme Fear', 'Extreme Greed', 'Neutral'):
        return base
    return f"{base}-{row['direction']}"

mkt_df['regime'] = mkt_df.apply(build_regime_label, axis=1)

fng_regime_series = mkt_df.set_index('date')['regime_base']

print('Regime distribution:')
print(mkt_df['regime'].value_counts().sort_index().rename('days'))"""
))

# ── 11: Sector metadata + merge ───────────────────────────────────────────────
cells.append(code_cell(
"""meta = (
    pd.read_csv(BASE_DIR / 'us_stocks_500m.csv',
                usecols=['ticker', 'sector', 'market_cap'])
    .drop_duplicates('ticker')
)
meta['market_cap_b'] = meta['market_cap'] / 1e9

df = (
    stock_df
    .merge(mkt_df[['date', 'fear_greed', 'regime_base', 'regime',
                    'fng_slope', 'direction']], on='date', how='left')
    .merge(meta[['ticker', 'sector', 'market_cap_b']], on='ticker', how='left')
)

df = df[(df['date'] >= DATA_START) & (df['date'] <= DATA_END)].copy()

if MCAP_FILTER_B is not None:
    before = df['ticker'].nunique()
    df = df[df['market_cap_b'] >= MCAP_FILTER_B].copy()
    print(f'MCAP filter >= ${MCAP_FILTER_B}B: {before:,} -> {df["ticker"].nunique():,} tickers')

df.dropna(subset=['beta_spy', 'vol_std_21', 'roc_21', TARGET_COL, 'regime'], inplace=True)
df.reset_index(drop=True, inplace=True)

ALL_SECTORS = sorted(df['sector'].dropna().unique())

print(f'Merged   : {df.shape}')
print(f'Tickers  : {df["ticker"].nunique():,}  |  Dates: {df["date"].nunique():,}')
print(f'Date range: {df["date"].min().date()} to {df["date"].max().date()}')
print(f'Sectors ({len(ALL_SECTORS)}): {ALL_SECTORS}')"""
))

# ── 12: WF Folds header ───────────────────────────────────────────────────────
cells.append(md_cell("## 5. Walk-Forward Folds"))

# ── 13: generate_wf_folds ─────────────────────────────────────────────────────
cells.append(code_cell(
"""def generate_wf_folds(start_year, end_year, min_train_years, test_years, step_years=1):
    \"\"\"Expanding-window walk-forward. Training always anchored at start_year.\"\"\"
    folds = []
    test_start = pd.Timestamp(f'{start_year + min_train_years}-01-01')
    while True:
        train_start = pd.Timestamp(f'{start_year}-01-01')
        train_end   = test_start - pd.Timedelta(days=1)
        test_end    = test_start + relativedelta(years=test_years) - pd.Timedelta(days=1)
        if test_end.year > end_year:
            break
        label = f'Train {train_start.year}-{train_end.year} | Test {test_start.year}'
        folds.append((label, train_start, train_end, test_start, test_end))
        test_start += relativedelta(years=step_years)
    return folds

WF_FOLDS = generate_wf_folds(
    start_year=2018, end_year=2025,
    min_train_years=TRAIN_YEARS, test_years=TEST_YEARS, step_years=1
)

print('Walk-forward folds:')
for name, tr_s, tr_e, te_s, te_e in WF_FOLDS:
    print(f'  {name}  [{(te_s - tr_s).days // 365}yr train]')"""
))

# ── 14: Strategy Functions header ─────────────────────────────────────────────
cells.append(md_cell("## 6. Strategy Functions"))

# ── 15: fit_regime_params (two-stage: ridge β then SLSQP w/ε/τ/λ) ────────────
cells.append(code_cell(
"""def fit_regime_params(train_df, regime, beta_fixed=None):
    \"\"\"
    Two-stage optimisation per regime.

    Stage 1 — β via ridge regression on 21d returns (closed-form).
              Skipped when beta_fixed is provided (SHARED_BETA=True): the caller
              pre-computed β on all training data across all regimes.

    Stage 2 — w and ε via SLSQP on 1d sequential EWM portfolio Sharpe.
              β is frozen; raw scores precomputed once before the optimiser loop.

    Returns dict: beta, w, eps, tau, lam, sectors, ic_train, sharpe_train, fallback.
    \"\"\"
    sub = train_df[train_df['regime'] == regime].dropna(
        subset=FACTOR_COLS + [TARGET_COL, 'target_ret_1d', 'sector'])

    n_dates = sub['date'].nunique()
    sectors = sorted(sub['sector'].unique())
    K       = len(sectors)
    sec_idx = {s: i for i, s in enumerate(sectors)}
    F       = len(FACTOR_COLS)

    # ── Fallback: insufficient data ───────────────────────────────────────────
    if n_dates < MIN_REGIME_DAYS or K < 2:
        means = sub.groupby('sector')[TARGET_COL].mean()
        if len(means) == 0:
            return {'beta': np.zeros(F), 'w': np.array([]),
                    'eps': SOFT_EPS, 'tau': FIXED_TAU, 'lam': FIXED_LAM,
                    'sectors': sectors, 'ic_train': np.nan,
                    'sharpe_train': np.nan, 'fallback': True}
        pos  = means[means > 0].sum() or 1.0
        neg  = means[means < 0].abs().sum() or 1.0
        w_fb = means.apply(lambda m: m / pos if m > 0 else (m / neg if m < 0 else 0.0))
        return {'beta': np.zeros(F),
                'w':    w_fb.reindex(sectors).fillna(0.0).values,
                'eps':  SOFT_EPS, 'tau': FIXED_TAU, 'lam': FIXED_LAM,
                'sectors': sectors, 'ic_train': np.nan,
                'sharpe_train': np.nan, 'fallback': True}

    sub = sub.copy().reset_index(drop=True)
    for col in FACTOR_COLS:
        sub[col] = sub.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))

    # ── Stage 1: β via ridge regression (or use shared β) ────────────────────
    X_all = sub[FACTOR_COLS].values.astype(np.float64)
    R_21d = sub[TARGET_COL].values.astype(np.float64)
    if beta_fixed is not None:
        beta = beta_fixed          # shared β pre-computed on all training data
    else:
        A    = X_all.T @ X_all + RIDGE_ALPHA * np.eye(F)
        beta = np.linalg.solve(A, X_all.T @ R_21d)

    # IC diagnostic evaluated on this regime's data (works for both shared and per-regime β)
    scores_all = X_all @ beta
    sub['_score'] = scores_all
    daily_ic = sub.groupby('date').apply(
        lambda g: np.corrcoef(g['_score'], g[TARGET_COL])[0, 1]
                  if len(g) >= 5 else np.nan
    ).dropna()
    ic_train = float(daily_ic.mean())

    # ── Stage 2 prep: precompute raw scores, build integer ticker index ───────
    raw_all       = scores_all   # frozen beta scores, shape (n_obs,)
    all_tickers   = sorted(sub['ticker'].unique())
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    N_T           = len(all_tickers)

    # date_data: [{sec_idx: (tidx, raw, R_1d)}] — chronological
    date_data = []
    for _, dgrp in sub.groupby('date'):
        d = {}
        for s, sgrp in dgrp.groupby('sector'):
            tidx = np.array([ticker_to_idx[t] for t in sgrp['ticker'].values])
            raw  = raw_all[sgrp.index.values]
            R_1d = sgrp['target_ret_1d'].values.astype(np.float64)
            d[sec_idx[s]] = (tidx, raw, R_1d)
        date_data.append(d)

    # ── Stage 2: SLSQP for w and ε only ──────────────────────────────────────
    # τ and λ are fixed hyperparameters — SLSQP always exploits them to overfit:
    #   τ→0 collapses softmax to argmax (single-stock selection in-sample)
    #   λ→1 means full daily rebalance (no EWM stickiness in-sample)
    # theta = [w(K), log_eps(1)]
    one_minus_lam = 1.0 - FIXED_LAM

    def portfolio_returns(theta):
        w   = theta[:K]
        eps = np.exp(theta[K])

        ema_arr     = np.zeros(N_T)
        initialized = np.zeros(N_T, dtype=bool)
        r_series    = []

        for d in date_data:
            rp = 0.0
            for k, (tidx, raw, R_1d) in d.items():
                if len(R_1d) < 2:
                    continue
                prev          = np.where(initialized[tidx], ema_arr[tidx], raw)
                ema           = one_minus_lam * prev + FIXED_LAM * raw
                ema_arr[tidx] = ema
                initialized[tidx] = True

                dir_k  = np.tanh(w[k] / eps)
                logits = ema * dir_k / FIXED_TAU
                logits -= logits.max()
                alpha  = np.exp(logits) / np.exp(logits).sum()
                rp    += w[k] * (alpha @ R_1d)
            r_series.append(rp)
        return np.array(r_series)

    def loss(theta):
        r   = portfolio_returns(theta)
        std = r.std()
        if std < 1e-10:
            return 0.0
        sharpe = r.mean() / std * np.sqrt(252)
        l2     = L2_LAMBDA * np.sum(theta[:K] ** 2)
        return -sharpe + l2

    constraints = [{'type': 'eq', 'fun': lambda t: t[:K].sum()}]
    # w unconstrained; log_eps bounded so eps in [e^-4, e^-0.5] = [0.018, 0.607]
    bounds = [(None, None)] * K + [(-4.0, -0.5)]

    best_res, best_val = None, np.inf
    rng = np.random.default_rng(42)
    for _ in range(N_RESTARTS):
        w0       = rng.standard_normal(K) * 0.1
        w0      -= w0.mean()
        log_eps0 = np.log(SOFT_EPS) + rng.standard_normal() * 0.5
        theta0   = np.concatenate([w0, [log_eps0]])
        res      = minimize(loss, theta0, method='SLSQP',
                            constraints=constraints, bounds=bounds,
                            options={'maxiter': 500, 'ftol': 1e-7})
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    if best_res is None:
        return {'beta': beta, 'w': np.zeros(K),
                'eps': SOFT_EPS, 'tau': FIXED_TAU, 'lam': FIXED_LAM,
                'sectors': sectors, 'ic_train': ic_train,
                'sharpe_train': np.nan, 'fallback': True}

    w_opt   = best_res.x[:K]
    eps_opt = np.exp(best_res.x[K])

    pos_sum = w_opt[w_opt > 0].sum()
    if pos_sum > 1e-8:
        w_opt = w_opt / pos_sum

    train_sharpe = -best_val + L2_LAMBDA * np.sum(w_opt ** 2)
    return {'beta': beta, 'w': w_opt, 'eps': eps_opt,
            'tau': FIXED_TAU, 'lam': FIXED_LAM,
            'sectors': sectors, 'ic_train': ic_train,
            'sharpe_train': train_sharpe, 'fallback': False}"""
))

# ── 16: build_date_portfolio ──────────────────────────────────────────────────
cells.append(code_cell(
"""def build_date_portfolio(date_df, params, prev_scores=None):
    \"\"\"
    Apply learned (beta, w) to one date's cross-section.
    EWM score: ema_t = (1-FIXED_LAM)*ema_{t-1} + FIXED_LAM*raw_score_t.
    FIXED_LAM and FIXED_TAU are global hyperparameters, not per-regime learned values.
    Realised return is target_ret_1d (daily P&L).
    Returns: port_ret, long_ret, short_ret (all 1d), score_records, new_scores.
    \"\"\"
    beta    = params['beta']
    sectors = params['sectors']
    w_map   = dict(zip(sectors, params['w']))
    if prev_scores is None:
        prev_scores = {}

    date_df = date_df.copy()
    for col in FACTOR_COLS:
        if col in date_df.columns:
            mu, sig      = date_df[col].mean(), date_df[col].std() + 1e-8
            date_df[col] = (date_df[col] - mu) / sig

    raw_scores = date_df[FACTOR_COLS].fillna(0.0).values @ beta

    # EWM score using fixed rebalancing speed
    new_scores = {}
    ema_col    = np.empty(len(date_df))
    tickers    = date_df['ticker'].values
    for i, (ticker, raw) in enumerate(zip(tickers, raw_scores)):
        ema = (1.0 - FIXED_LAM) * prev_scores.get(ticker, raw) + FIXED_LAM * raw
        new_scores[ticker] = ema
        ema_col[i]         = ema
    date_df['_score'] = ema_col

    long_contrib, short_contrib = 0.0, 0.0
    long_w_sum,   short_w_sum   = 0.0, 0.0
    score_records = []

    for sector, grp in date_df.groupby('sector'):
        w_k = w_map.get(sector, 0.0)
        if abs(w_k) < 1e-6:
            continue
        grp    = grp.dropna(subset=['_score', TARGET_COL, 'target_ret_1d'])
        n      = len(grp)
        n_pick = max(1, int(np.ceil(n * QUANTILE)))
        if n < max(5, n_pick):
            continue
        ranked = grp.sort_values('_score')

        if w_k > 0:
            sel           = ranked.iloc[-n_pick:]
            long_contrib += w_k * sel['target_ret_1d'].mean()
            long_w_sum   += w_k
            side          = 'long'
        else:
            sel            = ranked.iloc[:n_pick]
            short_contrib += w_k * sel['target_ret_1d'].mean()
            short_w_sum   += abs(w_k)
            side           = 'short'

        for _, row in sel.iterrows():
            score_records.append({
                'ticker':          row['ticker'],
                'sector':          sector,
                'y_pred':          row['_score'],
                'y_true':          row[TARGET_COL],   # 21d — for IC evaluation
                'side':            side,
                'industry_weight': w_k,
            })

    port_ret  = long_contrib + short_contrib
    long_ret  = long_contrib  / long_w_sum  if long_w_sum  > 0 else np.nan
    short_ret = short_contrib / short_w_sum if short_w_sum > 0 else np.nan
    return port_ret, long_ret, short_ret, score_records, new_scores"""
))

# ── 17: Evaluation Functions header ───────────────────────────────────────────
cells.append(md_cell("## 7. Evaluation Functions\n\nIdentical to baseline_models.ipynb."))

# ── 18: Evaluation functions ──────────────────────────────────────────────────
cells.append(code_cell(
"""def compute_ic_metrics(y_true, y_pred, dates, tickers):
    \"\"\"IC, ICIR, hit rate, R2, RMSE.\"\"\"
    ic_global, _ = spearmanr(y_pred, y_true)

    results_df = pd.DataFrame({
        'date': dates, 'ticker': tickers,
        'y_true': y_true, 'y_pred': y_pred
    })

    daily_ic = (
        results_df.groupby('date')
        .apply(lambda g: spearmanr(g['y_pred'], g['y_true'])[0] if len(g) >= 5 else np.nan)
        .dropna()
    )

    mean_ic  = daily_ic.mean()
    icir     = mean_ic / daily_ic.std() if daily_ic.std() > 0 else np.nan
    hit_rate = np.mean(np.sign(y_pred) == np.sign(y_true))

    return {
        'IC (global)':   ic_global,
        'Mean Daily IC': mean_ic,
        'ICIR':          icir,
        'Hit Rate':      hit_rate,
        'R2':            r2_score(y_true, y_pred),
        'RMSE':          np.sqrt(mean_squared_error(y_true, y_pred)),
        '_daily_ic':     daily_ic,
        '_df':           results_df,
    }


def compute_ls_returns(preds_df, quantile, min_stocks=10):
    \"\"\"
    For each date rank stocks by y_pred.
    Return DataFrame of daily L/S returns (long top q%, short bottom q%).
    \"\"\"
    records = []
    for date, g in preds_df.groupby('date'):
        n = len(g)
        if n < min_stocks:
            continue
        n_port    = max(1, int(np.ceil(n * quantile)))
        ranked    = g.sort_values('y_pred')
        long_ret  = ranked.iloc[-n_port:]['y_true'].mean()
        short_ret = ranked.iloc[:n_port]['y_true'].mean()
        records.append({
            'date':      date,
            'ls_ret':    long_ret - short_ret,
            'long_ret':  long_ret,
            'short_ret': short_ret,
            'n':         n,
        })
    return pd.DataFrame(records).set_index('date').sort_index()


def compute_strategy_metrics(ls_series, ann_factor=PORT_ANN_FACTOR):
    \"\"\"
    Annualised return, vol, Sharpe, Sortino, Calmar, max drawdown, win rate.
    ann_factor=PORT_ANN_FACTOR (252) for 1d port returns;
    pass ANN_FACTOR (12) for 21d IC-based series.
    \"\"\"
    s = ls_series.dropna()
    if len(s) < 20:
        return {k: np.nan for k in
                ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino',
                 'Calmar', 'Max DD', 'Win Rate']}

    ann_ret = s.mean() * ann_factor
    ann_vol = s.std()  * np.sqrt(ann_factor)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = s[s < 0].std() * np.sqrt(ann_factor)
    sortino  = ann_ret / downside if downside > 0 else np.nan

    cum    = (1 + s).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        'Ann. Return': ann_ret,
        'Ann. Vol':    ann_vol,
        'Sharpe':      sharpe,
        'Sortino':     sortino,
        'Calmar':      calmar,
        'Max DD':      max_dd,
        'Win Rate':    (s > 0).mean(),
    }"""
))

# ── 19: WF Training header ────────────────────────────────────────────────────
cells.append(md_cell(
    "## 8. Walk-Forward Training\n\n"
    "**Two-stage optimisation per regime per fold:**\n\n"
    "**Stage 1 — β (factor coefficients)** via ridge regression on 21d returns. "
    "Closed-form, instant. Answers *what predicts returns?* "
    "Objective: maximise cross-sectional Pearson IC between β·X and future 21d return.\n\n"
    "**Stage 2 — w, ε, τ, λ** via SLSQP on the 1d sequential EWM portfolio. "
    "β is frozen; raw scores are precomputed once before optimisation. "
    "Answers *how do I trade the signal?* per regime. "
    "~14 free params vs 18 previously — vectorised EWM via numpy integer indexing (no dict lookups).\n\n"
    "**EWM score**: `ema_t = (1-λ)·ema_{t-1} + λ·raw_score_t` — λ ∈ (0,1) per regime. "
    "λ→1 = full daily rebalance; λ→0 = hold forever. "
    "EWM scores persist across regime transitions at test time (`held_scores` dict).\n\n"
    "Active-regime days rebalance; inactive/fallback days hold last portfolio (`is_held=True`) "
    "but EWM scores still update so the signal stays current.\n\n"
    "OOS snapshot saved to `portfolio_snapshots/oos/`: one row per day, "
    "sector columns contain lists of held tickers."
))

# ── 20: WF loop ───────────────────────────────────────────────────────────────
cells.append(code_cell(
"""SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
OOS_DIR = SNAPSHOT_DIR / 'oos'
OOS_DIR.mkdir(exist_ok=True)

all_params    = {}
wf_port_rows  = []
wf_pred_rows  = []
wf_results    = []
oos_snap_rows = []   # one row per OOS day (wide format)

for fold in WF_FOLDS:
    fold_name, tr_s, tr_e, te_s, te_e = fold
    print(f'\\n── {fold_name} ──')

    train_df = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test_df  = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    if len(train_df) < 1000 or len(test_df) < 100:
        print('  Skipping — insufficient data')
        continue

    # ── Compute global β if SHARED_BETA=True ─────────────────────────────────
    if SHARED_BETA:
        all_sub = train_df.dropna(subset=FACTOR_COLS + [TARGET_COL]).copy().reset_index(drop=True)
        for col in FACTOR_COLS:
            all_sub[col] = all_sub.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8))
        X_g   = all_sub[FACTOR_COLS].values.astype(np.float64)
        R_g   = all_sub[TARGET_COL].values.astype(np.float64)
        A_g   = X_g.T @ X_g + RIDGE_ALPHA * np.eye(len(FACTOR_COLS))
        beta_global = np.linalg.solve(A_g, X_g.T @ R_g)
        ic_g_daily  = all_sub.assign(_s=X_g @ beta_global).groupby('date').apply(
            lambda g: np.corrcoef(g['_s'], g[TARGET_COL])[0, 1] if len(g) >= 5 else np.nan
        ).dropna()
        print(f'  Global β (all regimes): n_obs={len(all_sub):,}  '
              f'IC={ic_g_daily.mean():.4f}  '
              f'coefs={np.round(beta_global, 3).tolist()}')
    else:
        beta_global = None

    fold_params = {}
    for regime in sorted(ACTIVE_REGIMES):
        p      = fit_regime_params(train_df, regime, beta_fixed=beta_global)
        fold_params[regime] = p
        fb_str = 'FALLBACK' if p['fallback'] else 'ok'
        sh_str = f'{p["sharpe_train"]:.3f}' if not np.isnan(p['sharpe_train']) else 'n/a'
        ic_str = f'{p["ic_train"]:.4f}'     if not np.isnan(p['ic_train'])     else 'n/a'
        print(f'  {regime:25s}  IC={ic_str:>7}  sharpe={sh_str:>7}  {fb_str}  '
              f'eps={p["eps"]:.4f}  sectors={len(p["sectors"])}')
    all_params[fold_name] = fold_params

    held_params    = None
    held_scores    = {}   # EWM scores persist across all days in fold (active + held)
    fold_pred_rows = []
    fold_port_rows = []

    for date, date_df in test_df.groupby('date'):
        regime      = date_df['regime'].iloc[0]
        regime_base = regime.split('-')[0] if '-' in regime else regime

        if (regime in ACTIVE_REGIMES and regime in fold_params
                and not fold_params[regime]['fallback']):
            params      = fold_params[regime]
            held_params = params
            is_held     = False
        elif held_params is not None:
            params  = held_params
            is_held = True
        else:
            continue

        port_ret, long_ret, short_ret, score_recs, held_scores = build_date_portfolio(
            date_df, params, prev_scores=held_scores)

        # ── OOS snapshot row (wide format) ────────────────────────────────────
        sec_tickers = {}
        for rec in score_recs:
            sec_tickers.setdefault(rec['sector'], []).append(rec['ticker'])
        w_map_day = dict(zip(params['sectors'], params['w']))

        snap = {
            'date':     date,
            'fold':     fold_name,
            'regime':   regime,
            'is_held':  is_held,
            'port_ret': port_ret,
            'lam':      FIXED_LAM,
            **{f'coef_{col}': b for col, b in zip(FACTOR_COLS, params['beta'])},
            **{f'w_{s}': w_map_day.get(s, 0.0) for s in ALL_SECTORS},
            **{s: sec_tickers.get(s, [])        for s in ALL_SECTORS},
        }
        oos_snap_rows.append(snap)

        # ── Portfolio return tracking ──────────────────────────────────────────
        row_dict = {
            'date': date, 'port_ret': port_ret, 'long_ret': long_ret,
            'short_ret': short_ret, 'regime': regime, 'regime_base': regime_base,
            'fold': fold_name, 'is_held': is_held,
        }
        wf_port_rows.append(row_dict)
        fold_port_rows.append(row_dict)

        if not is_held:
            for rec in score_recs:
                row = {'date': date, 'fold': fold_name, **rec}
                wf_pred_rows.append(row)
                fold_pred_rows.append(row)

    all_dates = test_df['date'].nunique()

    if fold_port_rows:
        fp_port  = pd.Series([r['port_ret'] for r in fold_port_rows])
        m_oos    = compute_strategy_metrics(fp_port)
        n_active = sum(1 for r in fold_port_rows if not r['is_held'])
        ann_str  = f'{m_oos["Ann. Return"]:.4f}' if not np.isnan(m_oos["Ann. Return"]) else 'n/a'
        sh_str   = f'{m_oos["Sharpe"]:.3f}'       if not np.isnan(m_oos["Sharpe"])      else 'n/a'
        print(f'  OOS portfolio  active={n_active}/{all_dates}  Ann.Ret={ann_str}  Sharpe={sh_str}')

    if fold_pred_rows:
        fp      = pd.DataFrame(fold_pred_rows)
        metrics = compute_ic_metrics(
            fp['y_true'].values, fp['y_pred'].values,
            fp['date'].values,   fp['ticker'].values,
        )
        print(f'  OOS IC         active dates={fp["date"].nunique()}  '
              f'IC={metrics["Mean Daily IC"]:.4f}  ICIR={metrics["ICIR"]:.3f}')
        wf_results.append({
            'Fold': fold_name,
            **{k: v for k, v in metrics.items() if not k.startswith('_')},
        })

# ── Save OOS snapshot ─────────────────────────────────────────────────────────
if oos_snap_rows:
    oos_snap_df = pd.DataFrame(oos_snap_rows)
    oos_snap_df['date'] = pd.to_datetime(oos_snap_df['date'])
    snap_path = OOS_DIR / 'oos_daily_holdings.parquet'
    oos_snap_df.to_parquet(snap_path, index=False)
    print(f'\\nSaved OOS snapshot: {len(oos_snap_df)} days -> {snap_path}')

# ── Build analysis structures ─────────────────────────────────────────────────
wf_port_df = (
    pd.DataFrame(wf_port_rows)
    .assign(date=lambda x: pd.to_datetime(x['date']))
    .set_index('date')
    .sort_index()
)
wf_pred_df  = pd.DataFrame(wf_pred_rows)
wf_combined = {'RuleBase': wf_pred_df}

print(f'\\nTotal portfolio days : {len(wf_port_df)}'
      f'  (active: {(~wf_port_df["is_held"]).sum()}, held: {wf_port_df["is_held"].sum()})')"""
))

# ── 21: Analysis header ───────────────────────────────────────────────────────
cells.append(md_cell("## 9. Analysis"))

# ── 22: IC Summary ────────────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9a. IC Summary Table ─────────────────────────────────────────────────────
wf_df = pd.DataFrame(wf_results)
display_cols = ['Fold', 'Mean Daily IC', 'ICIR', 'Hit Rate', 'R2', 'RMSE']

print('=== Walk-Forward IC Summary ===')
print(wf_df[display_cols].to_string(index=False))

print('\\n=== Walk-Forward Averages ===')
avg = wf_df[['Mean Daily IC', 'ICIR', 'Hit Rate', 'R2']].mean()
print(avg.round(4).to_string())"""
))

# ── 23: Optimised Portfolio Strategy Metrics ──────────────────────────────────
cells.append(code_cell(
"""# ── 9b. Optimised Portfolio — Strategy Metrics ───────────────────────────────
metric_cols = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Win Rate']

m       = compute_strategy_metrics(wf_port_df['port_ret'])
m_activ = compute_strategy_metrics(wf_port_df.loc[~wf_port_df['is_held'], 'port_ret'])
m_held  = compute_strategy_metrics(wf_port_df.loc[ wf_port_df['is_held'], 'port_ret'])

rows = [
    {'Portfolio': 'All days (L/S)',   **m},
    {'Portfolio': 'Active days only', **m_activ},
    {'Portfolio': 'Held days only',   **m_held},
]
tbl = pd.DataFrame(rows).set_index('Portfolio')
print('=== Optimised Portfolio — Strategy Metrics (Walk-Forward, ann_factor=252) ===')
print(tbl[metric_cols].round(3).to_string())"""
))

# ── 24: Long-only metrics ─────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9b-ii. Long-Only Strategy Metrics ────────────────────────────────────────
m_long    = compute_strategy_metrics(wf_port_df['long_ret'].dropna())
rows_long = [{'Portfolio': f'Long-only top {int(QUANTILE*100)}%', **m_long}]
tbl_long  = pd.DataFrame(rows_long).set_index('Portfolio')
print('=== Long-Only Strategy Metrics (Walk-Forward, ann_factor=252) ===')
print(tbl_long[metric_cols].round(3).to_string())"""
))

# ── 25: Monthly table ─────────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9b-iii. Monthly L / S / L-S Return Table ─────────────────────────────────
print('\\n=== Monthly Performance — WF Combined | Optimised Portfolio ===')

monthly  = wf_port_df[['port_ret', 'long_ret', 'short_ret']].resample('ME').last()
roll_std = wf_port_df['port_ret'].rolling(21).std()

rows_m = []
for dt, row in monthly.iterrows():
    std_val  = roll_std.reindex([dt]).iloc[0]
    sharpe_m = ((row['port_ret'] * PORT_ANN_FACTOR) /
                (std_val * (PORT_ANN_FACTOR ** 0.5) + 1e-8)
                if not np.isnan(std_val) else np.nan)
    rows_m.append({
        'Month':          dt.strftime('%Y-%m'),
        'L Return':       round(row['long_ret'],  4),
        'S Return (P&L)': round(row['short_ret'], 4),
        'L/S Return':     round(row['port_ret'],  4),
        'Sharpe (ann)':   round(sharpe_m, 3) if not np.isnan(sharpe_m) else np.nan,
    })

print(pd.DataFrame(rows_m).set_index('Month').to_string())"""
))

# ── 26: Regime period table ───────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9b-iv. Regime-Period L / S / L-S Return Table ────────────────────────────
print('\\n=== Regime Period Performance — WF Combined ===')

port_s           = wf_port_df[['port_ret', 'long_ret', 'short_ret', 'regime_base']].copy()
port_s['block']  = (port_s['regime_base'] != port_s['regime_base'].shift()).cumsum()

rows_rp = []
for _, grp in port_s.groupby('block'):
    regime_name = grp['regime_base'].iloc[0]
    if regime_name in ('nan', 'None', ''):
        continue
    n      = len(grp)
    l_ret  = grp['long_ret'].mean()
    s_pnl  = grp['short_ret'].mean()
    ls_ret = grp['port_ret'].mean()
    std    = grp['port_ret'].std()
    sharpe = (ls_ret / std) * np.sqrt(PORT_ANN_FACTOR) if std > 0 and n > 1 else np.nan
    rows_rp.append({
        'Regime':         regime_name,
        'Start':          grp.index.min().strftime('%Y-%m-%d'),
        'End':            grp.index.max().strftime('%Y-%m-%d'),
        'Days':           n,
        'L Return':       round(l_ret,  4),
        'S Return (P&L)': round(s_pnl,  4),
        'L/S Return':     round(ls_ret, 4),
        'Sharpe':         round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
    })

print(pd.DataFrame(rows_rp).to_string(index=False))"""
))

# ── 27: Equity curves ─────────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9c. Equity Curves — Optimised Portfolio ──────────────────────────────────
def add_regime_background(ax, dates, regime_series):
    reg  = regime_series.reindex(pd.DatetimeIndex(dates)).ffill()
    x    = np.arange(len(dates))
    prev, start = None, 0
    for i, r in enumerate(reg):
        if r != prev:
            if prev is not None and str(prev) in REGIME_COLORS:
                ax.axvspan(start, i, alpha=0.18,
                           color=REGIME_COLORS[str(prev)], lw=0, zorder=0)
            prev, start = r, i
    if prev is not None and str(prev) in REGIME_COLORS:
        ax.axvspan(start, len(dates), alpha=0.18,
                   color=REGIME_COLORS[str(prev)], lw=0, zorder=0)


fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                         gridspec_kw={'height_ratios': [3, 1]})
eq_ax, dd_ax = axes

dates   = wf_port_df.index
cum     = (1 + wf_port_df['port_ret']).cumprod()
long_c  = (1 + wf_port_df['long_ret'].fillna(0)).cumprod()
short_c = (1 + wf_port_df['short_ret'].fillna(0)).cumprod()

x = np.arange(len(cum))
add_regime_background(eq_ax, dates, fng_regime_series)

held_mask = wf_port_df['is_held'].values
for i in range(len(held_mask)):
    if held_mask[i]:
        eq_ax.axvspan(i, i + 1, alpha=0.08, color='gray', lw=0, zorder=1)

eq_ax.plot(x, cum.values,     color='black',     lw=1.4, label='L/S (opt)',  zorder=5)
eq_ax.plot(x, long_c.values,  color='steelblue', lw=0.9, label='Long',       zorder=4, alpha=0.8)
eq_ax.plot(x, short_c.values, color='tomato',    lw=0.9, label='Short P&L',  zorder=4, alpha=0.8)
eq_ax.axhline(1, color='gray', lw=0.7, ls='--')
eq_ax.set_title(f'Optimised Portfolio — Walk-Forward ({TARGET_COL} signal, 1d realised returns)',
                fontsize=10)
eq_ax.set_ylabel('Cumulative Return')

patches    = [mpatches.Patch(color=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
held_patch = mpatches.Patch(color='gray', alpha=0.3, label='Held (inactive regime)')
eq_ax.legend(handles=patches + list(eq_ax.get_lines()[:3]) + [held_patch],
             fontsize=6, loc='upper left', ncol=3)

dd = (cum - cum.cummax()) / cum.cummax()
dd_ax.fill_between(x, dd.values, 0, color='tomato', alpha=0.5)
dd_ax.set_ylabel('Drawdown')
dd_ax.set_xlabel('Calendar Day (test window)')

plt.tight_layout()
plt.show()"""
))

# ── 28: Regime analysis ───────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9d. Regime Analysis — Mean Return per Regime ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

regime_mean = wf_port_df.groupby('regime_base')['port_ret'].mean().reindex(REGIME_ORDER)
colors      = [REGIME_COLORS[r] for r in REGIME_ORDER]
axes[0].bar(REGIME_ORDER, regime_mean.values, color=colors, alpha=0.85)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_title('Mean Daily 1d Return by Regime (Optimised Portfolio)', fontsize=9)
axes[0].set_ylabel('Mean Portfolio Return (1d)')
axes[0].tick_params(axis='x', rotation=25)

plot_data = [
    wf_port_df.loc[wf_port_df['regime_base'] == r, 'port_ret'].dropna().values
    for r in REGIME_ORDER
]
parts = axes[1].violinplot(plot_data, positions=range(len(REGIME_ORDER)),
                           showmedians=True, showextrema=False)
for pc, r in zip(parts['bodies'], REGIME_ORDER):
    pc.set_facecolor(REGIME_COLORS[r])
    pc.set_alpha(0.6)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xticks(range(len(REGIME_ORDER)))
axes[1].set_xticklabels(REGIME_ORDER, rotation=25, ha='right', fontsize=8)
axes[1].set_title('Return Distribution by Regime', fontsize=9)
axes[1].set_ylabel('Portfolio Return (1d)')

plt.suptitle(f'Regime Analysis — Walk-Forward ({TARGET_COL})', fontsize=11)
plt.tight_layout()
plt.show()"""
))

# ── 29: Regime-conditional Sharpe ─────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9e. Regime-Conditional Sharpe ────────────────────────────────────────────
rows_rc = []
for regime_b in REGIME_ORDER:
    sub = wf_port_df.loc[wf_port_df['regime_base'] == regime_b, 'port_ret']
    if len(sub) < 20:
        continue
    m = compute_strategy_metrics(sub)
    rows_rc.append({
        'Regime':      regime_b,
        'N days':      len(sub),
        'Sharpe':      m['Sharpe'],
        'Ann. Return': m['Ann. Return'],
        'Win Rate':    m['Win Rate'],
    })

tbl_rc = pd.DataFrame(rows_rc).set_index('Regime').reindex(REGIME_ORDER).dropna(how='all')
print('=== Regime-Conditional Sharpe (Optimised Portfolio, ann_factor=252) ===')
print(tbl_rc.round(3).to_string())"""
))

# ── 30: Monthly heatmap ───────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9f. Monthly Return Heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

monthly_sum = wf_port_df['port_ret'].resample('ME').sum()
monthly_sum.index = monthly_sum.index.to_period('M')

pivot          = monthly_sum.to_frame('ret')
pivot['year']  = pivot.index.year
pivot['month'] = pivot.index.month
heat           = pivot.pivot(index='year', columns='month', values='ret')
heat.columns   = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec'][:len(heat.columns)]

sns.heatmap(heat, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', linewidths=0.4,
            cbar_kws={'shrink': 0.6})
ax.set_title('Monthly L/S Returns — Optimised Portfolio (1d returns, summed)', fontsize=9)
plt.suptitle('Monthly Return Heatmap — Walk-Forward', fontsize=11)
plt.tight_layout()
plt.show()"""
))

# ── 31: Rolling Sharpe ────────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9g. Rolling Sharpe (63-day window) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))

ls_s   = wf_port_df['port_ret']
window = 63
roll_sharpe = (ls_s.rolling(window).mean() /
               ls_s.rolling(window).std()) * np.sqrt(PORT_ANN_FACTOR)

x = np.arange(len(roll_sharpe))
ax.plot(x, roll_sharpe.values, lw=1.2, label='Optimised Portfolio', color='steelblue')
ax.axhline(0, color='black', lw=0.8)
ax.set_title(f'{window}-Day Rolling Sharpe — WF Combined (ann_factor=252)')
ax.set_xlabel('Calendar Day (test window)')
ax.set_ylabel('Rolling Sharpe')
ax.legend()
plt.tight_layout()
plt.show()"""
))

# ── 32: IC decay ──────────────────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9h. IC Decay Curve ────────────────────────────────────────────────────────
horizons     = ['target_ret_1d', 'target_ret_3d', 'target_ret_5d',
                'target_ret_10d', 'target_ret_21d']
horizon_days = [1, 3, 5, 10, 21]

test_dates   = set(pd.to_datetime(wf_pred_df['date'].unique()).tolist())
test_actuals = (
    df[df['date'].isin(test_dates)][['date', 'ticker'] + horizons]
    .dropna()
)

ic_decay = []
for model_name, pred_df in wf_combined.items():
    merged = pred_df.merge(test_actuals, on=['date', 'ticker'], how='inner')
    for h, h_col in zip(horizon_days, horizons):
        if h_col not in merged.columns:
            continue
        daily_ic = (
            merged.groupby('date')
            .apply(lambda g: spearmanr(g['y_pred'], g[h_col])[0]
                   if len(g) >= 5 else np.nan)
            .dropna()
        )
        ic_decay.append({'Model': model_name,
                         'Horizon (days)': h,
                         'Mean IC': daily_ic.mean()})

ic_decay_df = pd.DataFrame(ic_decay)

fig, ax = plt.subplots(figsize=(8, 4))
for model_name in wf_combined:
    sub = ic_decay_df[ic_decay_df['Model'] == model_name].sort_values('Horizon (days)')
    ax.plot(sub['Horizon (days)'], sub['Mean IC'], marker='o', lw=1.5, label=model_name)
ax.axhline(0, color='gray', lw=0.7, ls='--')
ax.set_title('IC Decay Curve — WF Combined')
ax.set_xlabel('Forecast Horizon (days)')
ax.set_ylabel('Mean Daily IC')
ax.legend()
plt.tight_layout()
plt.show()

print(ic_decay_df.pivot(index='Horizon (days)', columns='Model',
                        values='Mean IC').round(4).to_string())"""
))

# ── 33: Learned params summary ────────────────────────────────────────────────
cells.append(code_cell(
"""# ── 9i. Learned Parameters Summary ──────────────────────────────────────────
# τ and λ are fixed hyperparameters (FIXED_TAU, FIXED_LAM) — not shown per regime.
# Only ε is learned per regime; β is learned via ridge regression (stage 1).
factor_hdr = '  '.join(f'{c:>12}' for c in FACTOR_COLS)
print(f'  τ={FIXED_TAU}  λ={FIXED_LAM}  (fixed — not learned per regime)')
print(f'{"Fold":<35} {"Regime":<25} {"IC":>7}  {"Sharpe":>7}  {"FB":>5}  {"eps":>7}  {factor_hdr}')
print('-' * 170)

for fold_name, fold_params in all_params.items():
    for regime in sorted(fold_params.keys()):
        p      = fold_params[regime]
        ic_s   = f'{p["ic_train"]:.4f}'     if not np.isnan(p['ic_train'])     else 'n/a'
        sh_str = f'{p["sharpe_train"]:.3f}' if not np.isnan(p['sharpe_train']) else 'n/a'
        fb_str = 'yes' if p['fallback'] else 'no'
        eps_s  = f'{p["eps"]:.4f}'
        beta_s = '  '.join(f'{b:>12.4f}' for b in p['beta'])
        print(f'{fold_name:<35} {regime:<25} {ic_s:>7}  {sh_str:>7}  {fb_str:>5}  {eps_s:>7}  {beta_s}')

print('\\n=== Industry Weights — Last Fold ===')
last_fold = list(all_params.keys())[-1]
for regime in sorted(all_params[last_fold].keys()):
    p = all_params[last_fold][regime]
    print(f'\\n{regime}  (IC={p["ic_train"]:.4f}  eps={p["eps"]:.4f}):')
    if len(p['sectors']) == 0:
        print('  (no sectors)')
        continue
    w_df = pd.Series(p['w'], index=p['sectors']).sort_values(ascending=False)
    for sector, wt in w_df.items():
        bar  = chr(9608) * min(40, int(abs(wt) * 30))
        side = 'LONG ' if wt > 0 else 'SHORT'
        print(f'  {side} {sector:<35} {wt:+.4f} {bar}')"""
))

# ── 34: Final Scorecard header ────────────────────────────────────────────────
cells.append(md_cell("## 10. Final Scorecard"))

# ── 35: Final Scorecard ───────────────────────────────────────────────────────
cells.append(code_cell(
"""print(f'\\n{"="*70}')
print(f'RULE-BASED STRATEGY (OPTIMISED) — FINAL SCORECARD — Target: {TARGET_COL}')
print(f'{"="*70}')
print(f'  USE_DIRECTION   : {USE_DIRECTION}  (window={SMOOTH_WINDOW}d, threshold={SLOPE_THRESHOLD})')
print(f'  MCAP filter     : >= ${MCAP_FILTER_B}B')
print(f'  FACTOR_COLS     : {FACTOR_COLS}')
print(f'  QUANTILE        : top/bottom {int(QUANTILE*100)}%')
print(f'  L2_LAMBDA       : {L2_LAMBDA}')
print(f'  N_RESTARTS      : {N_RESTARTS}')
print(f'  Active regimes  : {sorted(ACTIVE_REGIMES)}')
print(f'  Portfolio return: 1d realised  |  Signal: {TARGET_COL}')
print(f'  FIXED_TAU       : {FIXED_TAU}  (softmax temperature, not learned)')
print(f'  FIXED_LAM       : {FIXED_LAM}  (EWM rebalancing speed, not learned)')
print()

if wf_results:
    print('--- Walk-Forward Average IC (active days, signal vs 21d return) ---')
    wf_df_s = pd.DataFrame(wf_results)
    print(wf_df_s[['Mean Daily IC', 'ICIR', 'Hit Rate', 'R2']].mean().round(4).to_string())

print()
print('--- Optimised Portfolio Metrics (all days, ann_factor=252) ---')
m_all       = compute_strategy_metrics(wf_port_df['port_ret'])
metric_cols = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Win Rate']
for k in metric_cols:
    print(f'  {k:<15}: {m_all[k]:.4f}')

print()
print('--- Regime-Conditional Sharpe ---')
for regime_b in REGIME_ORDER:
    sub = wf_port_df.loc[wf_port_df['regime_base'] == regime_b, 'port_ret']
    if len(sub) < 20:
        continue
    sharpe = compute_strategy_metrics(sub)['Sharpe']
    print(f'  {regime_b:<20}: {sharpe:.3f}')

if (OOS_DIR / 'oos_daily_holdings.parquet').exists():
    print(f'\\nOOS snapshot : {OOS_DIR}/oos_daily_holdings.parquet')"""
))

# ── Build notebook JSON ───────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells,
}

out = Path('/Users/gabriel/Desktop/Sentiment Model/rule_based_strategy.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Written {len(cells)} cells -> {out}')
