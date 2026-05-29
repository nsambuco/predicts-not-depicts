"""
slim_analyses.py
Exploratory analyses: pre-scan perceived satiety (SLIM) and fMRI measures.

Input:  ../data_demo/SLIM_analyses_demo.csv (SLIM scores, demographics, ROI betas)
Output: ../figures/SLIM_results.txt

Three analysis blocks:
  Q1. SLIM x behavioral juice acceptance (%Yes)
  Q2. SLIM x food-specific brain responses (Food+ minus Neutral, Food- minus Neutral, Food+ minus Food-)
  Q3. SLIM x wanting/liking Yes/No contrasts (foodplus_YesVsNo, juice_YesVsNo)

For Q2 and Q3: zero-order Pearson, partial Pearson (controlling BMI + age), FDR correction.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

# ---------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data_demo')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

INPUT = os.path.join(DATA_DIR, 'SLIM_analyses_demo.csv')
OUTPUT = os.path.join(FIG_DIR, 'SLIM_results.txt')
ROIS = ['Visual', 'mPFC', 'IFG', 'Amygdala', 'NAc', 'vmPFC', 'PUT', 'dAI', 'vAI', 'PI']
ALPHA = 0.05

# ---------- HELPERS ----------
def partial_corr(x, y, covariates):
    """Partial Pearson r between x and y, controlling for list of covariate arrays."""
    X_cov = np.column_stack([np.ones(len(x))] + list(covariates))
    bx = np.linalg.lstsq(X_cov, x, rcond=None)[0]
    by = np.linalg.lstsq(X_cov, y, rcond=None)[0]
    res_x = x - X_cov @ bx
    res_y = y - X_cov @ by
    r, p = stats.pearsonr(res_x, res_y)
    return r, p

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns array of corrected p-values."""
    n = len(pvals)
    pvals = np.array(pvals)
    ranked = np.argsort(pvals)
    corrected = np.zeros(n)
    for i, idx in enumerate(ranked):
        corrected[idx] = pvals[idx] * n / (i + 1)
    # Enforce monotonicity (from largest rank downward)
    order = np.argsort(pvals)[::-1]
    running_min = 1.0
    for idx in order:
        corrected[idx] = min(corrected[idx], running_min)
        running_min = min(running_min, corrected[idx])
    return np.clip(corrected, 0, 1)

def star(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    if p < .1:   return '†'
    return ''

# ---------- LOAD DATA ----------
df = pd.read_csv(INPUT)
lines = []
def W(s=''):
    lines.append(s)

W("=" * 90)
W("SLIM (Satiety) Exploratory Analyses")
W("=" * 90)
W(f"Input file: {os.path.basename(INPUT)}")
W(f"Total subjects with SLIM + fMRI betas: N = {len(df)}")
W(f"  SLIM score: mean = {df['slim_score'].mean():.1f}, SD = {df['slim_score'].std():.1f}, "
  f"range = [{df['slim_score'].min():.1f}, {df['slim_score'].max():.1f}]")
W(f"  BMI: mean = {df['bmi'].mean():.1f}, SD = {df['bmi'].std():.1f}")
W(f"  Age: mean = {df['age'].mean():.1f}, SD = {df['age'].std():.1f}")

# Covariate intercorrelations
W(f"\nCovariate intercorrelations:")
for a, b, lab in [('slim_score','bmi','SLIM x BMI'),
                   ('slim_score','age','SLIM x Age'),
                   ('bmi','age','BMI x Age')]:
    mask = df[[a,b]].notna().all(axis=1)
    r, p = stats.pearsonr(df.loc[mask, a], df.loc[mask, b])
    W(f"  {lab:<15s}  r = {r:+.3f}  p = {p:.4f}")

# =====================================================================
# Q1: SLIM x %Yes
# =====================================================================
W(f"\n{'=' * 90}")
W("Q1: SLIM x Behavioral Juice Acceptance (%Yes)")
W("=" * 90)
mask = df[['slim_score','pct_yes']].notna().all(axis=1)
sub = df[mask]
n = len(sub)
W(f"  N = {n}")
W(f"  %Yes: mean = {sub['pct_yes'].mean():.1f}, SD = {sub['pct_yes'].std():.1f}")
r, p = stats.pearsonr(sub['slim_score'], sub['pct_yes'])
rp, pp = partial_corr(sub['slim_score'].values, sub['pct_yes'].values,
                       [sub['bmi'].values, sub['age'].values])
W(f"  Zero-order:      r = {r:+.3f}  p = {p:.4f}  {star(p)}")
W(f"  Partial (BMI,Age): r = {rp:+.3f}  p = {pp:.4f}  {star(pp)}")

# =====================================================================
# Q2: SLIM x food-specific brain responses
# =====================================================================
W(f"\n{'=' * 90}")
W("Q2: SLIM x Food-Specific Brain Responses (change from Neutral)")
W("=" * 90)

contrast_suffixes = [
    ('Food+ minus Neutral', 'foodplus_minus_neu'),
    ('Food- minus Neutral', 'foodminus_minus_neu'),
    ('Food+ minus Food-',   'foodplus_minus_foodminus'),
]

# Collect all p-values for FDR across the Q2 family (3 contrasts x 10 ROIs = 30 tests)
q2_results = []

for label, suffix in contrast_suffixes:
    cols = [f'{r}_{suffix}' for r in ROIS]
    mask = df[['slim_score','bmi','age'] + cols].notna().all(axis=1)
    sub = df[mask]
    n = len(sub)
    W(f"\n--- {label} --- (N = {n})")
    W(f"  {'ROI':<12s}  {'Zero-order':>18s}  {'Partial (BMI,Age)':>22s}")

    for roi, col in zip(ROIS, cols):
        r0, p0 = stats.pearsonr(sub['slim_score'].values, sub[col].values)
        rp, pp = partial_corr(sub['slim_score'].values, sub[col].values,
                               [sub['bmi'].values, sub['age'].values])
        q2_results.append({'contrast': label, 'roi': roi, 'n': n,
                           'r_zero': r0, 'p_zero': p0,
                           'r_partial': rp, 'p_partial': pp})
        W(f"  {roi:<12s}  r={r0:+.3f} p={p0:.4f}{star(p0):>4s}   r={rp:+.3f} p={pp:.4f}{star(pp):>4s}")

# FDR correction across all Q2 partial p-values
q2_df = pd.DataFrame(q2_results)
q2_df['p_partial_fdr'] = fdr_bh(q2_df['p_partial'].values)

W(f"\nFDR correction across Q2 family ({len(q2_df)} tests, Benjamini-Hochberg):")
W(f"  {'Contrast':<25s}  {'ROI':<12s}  {'r_partial':>10s}  {'p_uncorr':>10s}  {'p_FDR':>10s}")
for _, row in q2_df.sort_values('p_partial').iterrows():
    if row['p_partial'] < 0.10:
        W(f"  {row['contrast']:<25s}  {row['roi']:<12s}  {row['r_partial']:+.3f}      "
          f"{row['p_partial']:.4f}      {row['p_partial_fdr']:.4f}  {star(row['p_partial_fdr'])}")

# =====================================================================
# Q3: SLIM x wanting/liking Yes/No contrasts
# =====================================================================
W(f"\n{'=' * 90}")
W("Q3: SLIM x Wanting/Liking Yes/No Contrasts")
W("=" * 90)

yesno_suffixes = [
    ('Food+ Yes > No (cue, anticipatory)', 'foodplusYesNo'),
    ('Juice Yes > No (outcome, consummatory)', 'juiceYesNo'),
]

q3_results = []

for label, suffix in yesno_suffixes:
    cols = [f'{r}_{suffix}' for r in ROIS]
    mask = df[['slim_score','bmi','age'] + cols].notna().all(axis=1)
    sub = df[mask]
    n = len(sub)
    W(f"\n--- {label} --- (N = {n})")
    W(f"  {'ROI':<12s}  {'Zero-order':>18s}  {'Partial (BMI,Age)':>22s}")

    for roi, col in zip(ROIS, cols):
        r0, p0 = stats.pearsonr(sub['slim_score'].values, sub[col].values)
        rp, pp = partial_corr(sub['slim_score'].values, sub[col].values,
                               [sub['bmi'].values, sub['age'].values])
        q3_results.append({'contrast': label, 'roi': roi, 'n': n,
                           'r_zero': r0, 'p_zero': p0,
                           'r_partial': rp, 'p_partial': pp})
        W(f"  {roi:<12s}  r={r0:+.3f} p={p0:.4f}{star(p0):>4s}   r={rp:+.3f} p={pp:.4f}{star(pp):>4s}")

# FDR correction across all Q3 partial p-values
q3_df = pd.DataFrame(q3_results)
q3_df['p_partial_fdr'] = fdr_bh(q3_df['p_partial'].values)

W(f"\nFDR correction across Q3 family ({len(q3_df)} tests, Benjamini-Hochberg):")
W(f"  {'Contrast':<45s}  {'ROI':<12s}  {'r_partial':>10s}  {'p_uncorr':>10s}  {'p_FDR':>10s}")
for _, row in q3_df.sort_values('p_partial').iterrows():
    if row['p_partial'] < 0.10:
        W(f"  {row['contrast']:<45s}  {row['roi']:<12s}  {row['r_partial']:+.3f}      "
          f"{row['p_partial']:.4f}      {row['p_partial_fdr']:.4f}  {star(row['p_partial_fdr'])}")

# =====================================================================
# COMBINED SUMMARY
# =====================================================================
W(f"\n{'=' * 90}")
W("SUMMARY: All uncorrected p < .10 (partial correlations controlling BMI and Age)")
W("=" * 90)

all_results = pd.concat([q2_df, q3_df], ignore_index=True)
# FDR across the FULL set (Q2 + Q3 combined = 50 tests)
all_results['p_partial_fdr_global'] = fdr_bh(all_results['p_partial'].values)

W(f"  Total tests: {len(all_results)} (Q2: {len(q2_df)}, Q3: {len(q3_df)})")
W(f"\n  {'Contrast':<45s}  {'ROI':<12s}  N  {'r_partial':>10s}  {'p_uncorr':>9s}  "
  f"{'p_FDR_family':>12s}  {'p_FDR_global':>12s}")

for _, row in all_results.sort_values('p_partial').iterrows():
    if row['p_partial'] < 0.10:
        # Get family-level FDR
        if row['contrast'] in [c[0] for c in contrast_suffixes]:
            fdr_fam = q2_df.loc[(q2_df['contrast']==row['contrast']) &
                                (q2_df['roi']==row['roi']), 'p_partial_fdr'].values[0]
        else:
            fdr_fam = q3_df.loc[(q3_df['contrast']==row['contrast']) &
                                (q3_df['roi']==row['roi']), 'p_partial_fdr'].values[0]

        W(f"  {row['contrast']:<45s}  {row['roi']:<12s}  {int(row['n'])}  "
          f"{row['r_partial']:+.3f}      {row['p_partial']:.4f}    "
          f"{fdr_fam:.4f} {star(fdr_fam):>4s}    "
          f"{row['p_partial_fdr_global']:.4f} {star(row['p_partial_fdr_global']):>4s}")

W(f"\nNote: p_FDR_family = FDR within Q2 (30 tests) or Q3 (20 tests) separately.")
W(f"      p_FDR_global = FDR across all 50 tests combined.")

# Write output
with open(OUTPUT, 'w') as f:
    f.write('\n'.join(lines))
print(f"Results saved to {OUTPUT}")
print(f"{len(lines)} lines written.")
