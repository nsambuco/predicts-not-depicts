#!/usr/bin/env python3
"""
MRIeat - Statistical Analysis & Figure Pipeline
=================================================
Runs all the ROI analyses and makes the figures for the MRIeat paper
(Sambuco et al.).

What it produces:
  FunctionalROIs.png/.pdf   - Functional localiser ROI responses
  APrioriROIs.png/.pdf      - A priori ROI responses
  WantingLiking.png/.pdf    - Wanting vs liking (Yes > No) contrasts
  RSA.png/.pdf              - Representational similarity analysis
  Behavior.png/.pdf         - Behavioral results (alliesthesia)
  MRIeat_stats_report.txt   - Full statistical report
  pairwise_comparisons.tsv  - All pairwise ROI contrasts (for tables)

Inputs (read from ../data_demo/):
  ROIs_betas_demo.csv   - ROI-level neural betas
  behavior_demo.csv     - trial-level behavioral data

Outputs all go to ../figures/.

Run it with:
  python mrieat_analysis_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from itertools import combinations
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe

# White outline for significance-star text - keeps them readable on any color background
STAR_OUTLINE = [pe.withStroke(linewidth=2.5, foreground='white')]

# =========================================================
# CONFIG
# =========================================================
N_PERMS = 10000
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data_demo')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NEURAL_CSV = os.path.join(DATA_DIR, 'ROIs_betas_demo.csv')
BEHAV_CSV = os.path.join(DATA_DIR, 'behavior_demo.csv')
STATS_FILE = os.path.join(FIG_DIR, 'MRIeat_stats_report.txt')

# Colors (exact RGB)
COL_ERO = np.array([0, 0, 255]) / 255
COL_NEU = np.array([0, 255, 0]) / 255
COL_MUT = np.array([255, 0, 0]) / 255
COL_FP  = np.array([0, 0, 0]) / 255
COL_FM  = np.array([153, 153, 153]) / 255
COL_JUICE = np.array([0, 0, 255]) / 255
COL_RSA_SIG = '#2c3e50'

# ROI sets
FUNC_ROIS   = ['Visual', 'mPFC', 'IFG', 'Amygdala']
FUNC_LABELS = ['Visual', 'mPFC', 'IFG', 'Amygdala']
AP_ROIS     = ['NAc', 'PUT', 'vmPFC', 'dAI', 'vAI', 'PI']
AP_LABELS   = ['NAc', 'Putamen', 'vmPFC', 'dAI', 'vAI', 'PI']
ALL_ROIS    = FUNC_ROIS + AP_ROIS
ALL_LABELS  = ['Visual', 'mPFC', 'IFG', 'Amygdala', 'NAc', 'Putamen', 'vmPFC', 'dAI', 'vAI', 'PI']

CONDITIONS  = ['neu', 'ero', 'mut', 'foodminus', 'foodplus']
COND_LABELS = ['Neutral', 'Erotica', 'Mutilations', 'Food-', 'Food+']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# =========================================================
# LOAD DATA
# =========================================================
print("Loading data...")
ndf = pd.read_csv(NEURAL_CSV)
ndf.columns = [c.strip() for c in ndf.columns]
ndf['Contrast'] = ndf['Contrast'].str.strip()
ndf['Subject'] = ndf['Subject'].str.strip()

bdf = pd.read_csv(BEHAV_CSV)
bdf.columns = [c.strip() for c in bdf.columns]
bdf['rename2'] = bdf['rename2'].str.strip()
bdf['Subject'] = bdf['Subject'].astype(str).str.strip()

subs5 = sorted(set.intersection(*[set(ndf[ndf['Contrast'] == c]['Subject']) for c in CONDITIONS]))
N5 = len(subs5)
print(f"  N={N5} subjects")


# =========================================================
# HELPERS
# =========================================================
def get_data(roi, cond):
    sd = ndf[ndf['Contrast'] == cond].set_index('Subject')
    return np.array([sd.loc[s, roi] for s in subs5])


def get_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '†'
    return None


def add_bracket(ax, x1, x2, y, stars, lw=0.9, fs=14):
    h = 0.004
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h],
            lw=lw, color='black', zorder=5, clip_on=False)
    # Stars sit slightly above the bracket so they don't touch the line.
    # Anchor them close to their own bracket so they read as belonging to the
    # line beneath them, not the next stacked bracket above. va='baseline' with a
    # small positive offset plus a large step between stacked brackets makes the
    # gap below the star much smaller than the gap above.
    ax.text((x1 + x2) / 2, y + 0.006, stars, ha='center', va='baseline',
            fontsize=fs, fontweight='bold', color='black', zorder=6,
            path_effects=STAR_OUTLINE)


def draw_raincloud(ax, x, vals, color, w=0.18, alpha_fill=0.55,
                   dot_size=12, zorder=3, rng_seed=42):
    """Half-violin (right) + aligned dots (left) + mean+SEM marker at center.

    Publication-style raincloud element centered at x with total horizontal
    extent roughly +/- w (violin extends right of x, dots sit left of x).
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return

    # --- Half-violin (right side) via KDE ---
    try:
        kde = stats.gaussian_kde(vals, bw_method=0.4)
        y_min, y_max = vals.min(), vals.max()
        rng_span = y_max - y_min if y_max > y_min else max(abs(y_max), 0.1)
        pad = rng_span * 0.15
        y_grid = np.linspace(y_min - pad, y_max + pad, 120)
        dens = kde(y_grid)
        dens = dens / dens.max() * w  # half-width in x-axis units
        ax.fill_betweenx(y_grid, x, x + dens, color=color, alpha=alpha_fill,
                         edgecolor='black', linewidth=0.5, zorder=zorder)
    except Exception:
        pass

    # --- Aligned dots (left side) ---
    rng = np.random.default_rng(rng_seed)
    jitter = rng.uniform(-w * 0.85, -w * 0.15, size=len(vals))
    ax.scatter(x + jitter, vals, color=color, edgecolor='white', linewidth=0.3,
               s=dot_size, alpha=0.75, zorder=zorder + 1)

    # --- Mean +/- SEM marker on the center axis ---
    m = float(np.mean(vals))
    se = float(stats.sem(vals))
    ax.plot([x - w * 0.12, x + w * 0.12], [m, m],
            color='black', lw=1.6, zorder=zorder + 2, solid_capstyle='butt')
    ax.plot([x, x], [m - se, m + se], color='black', lw=1.1,
            zorder=zorder + 2, solid_capstyle='butt')


# =========================================================
# STATS REPORT
# =========================================================
SF = open(STATS_FILE, 'w')


def report(t):
    print(t)
    SF.write(t + '\n')


report("=" * 80)
report("MRIeat - Complete Statistical Report")
report(f"N={N5} subjects")
report("=" * 80)

# --- Section 1: t-tests vs zero ---
report("\n\nSECTION 1: ONE-SAMPLE T-TESTS VS ZERO")
report("=" * 80)
for roi in ALL_ROIS:
    report(f"\n--- {roi} ---")
    report(f"  {'Contrast':<25} {'N':>3} {'Mean':>10} {'SD':>10} {'t':>8} {'p':>10} {'d':>8}")
    report(f"  {'-' * 70}")
    for c in CONDITIONS + ['foodplus_YesVsNo', 'juice_YesVsNo']:
        vals = ndf[ndf['Contrast'] == c][roi].values
        n = len(vals); m = vals.mean(); sd = vals.std(ddof=1)
        t, p = stats.ttest_1samp(vals, 0)
        d = m / sd if sd > 0 else 0
        sig = get_stars(p) or ''
        report(f"  {c:<25} {n:>3} {m:>10.4f} {sd:>10.4f} {t:>8.3f} {p:>10.4f} {d:>8.3f} {sig}")

# --- Section 2: Paired t-tests ---
report("\n\nSECTION 2: PAIRED T-TESTS")
report("=" * 80)
pair_list = [
    ('ero', 'neu', 'Ero vs Neutral'), ('mut', 'neu', 'Mut vs Neutral'),
    ('foodplus', 'neu', 'Food+ vs Neutral'), ('foodminus', 'neu', 'Food- vs Neutral'),
    ('foodplus', 'foodminus', 'Food+ vs Food-'),
    ('ero', 'foodplus', 'Ero vs Food+'), ('mut', 'foodplus', 'Mut vs Food+'),
    ('ero', 'foodminus', 'Ero vs Food-'), ('mut', 'foodminus', 'Mut vs Food-'),
    ('ero', 'mut', 'Ero vs Mut'),
]
for roi in ALL_ROIS:
    report(f"\n--- {roi} ---")
    report(f"  {'Comparison':<30} {'diff':>10} {'t':>8} {'p':>10} {'d':>8}")
    report(f"  {'-' * 70}")
    for c1, c2, label in pair_list:
        v1, v2 = get_data(roi, c1), get_data(roi, c2)
        diff = v1 - v2
        t, p = stats.ttest_rel(v1, v2)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
        sig = get_stars(p) or ''
        report(f"  {label:<30} {diff.mean():>10.4f} {t:>8.3f} {p:>10.4f} {d:>8.3f} {sig}")

# --- Section 3: Brain-behavior ---
report("\n\nSECTION 3: BRAIN-BEHAVIOR CORRELATIONS")
report("=" * 80)
food_plus_beh = bdf[bdf['rename2'].isin(['food+Y', 'food+N'])]
all_runs = sorted(food_plus_beh['RunList'].unique())
behav = {}
for subj in food_plus_beh['Subject'].unique():
    sub = food_plus_beh[food_plus_beh['Subject'] == subj]
    n_yes = (sub['rename2'] == 'food+Y').sum()
    pct = n_yes / len(sub) * 100
    rpcts = []
    for r in all_runs:
        rs = sub[sub['RunList'] == r]
        ny = (rs['rename2'] == 'food+Y').sum()
        nt = len(rs)
        rpcts.append(ny / nt * 100 if nt > 0 else np.nan)
    sl = stats.linregress(range(len(rpcts)), rpcts).slope if len(rpcts) >= 3 else np.nan
    behav[subj] = {'Pct_Yes': pct, 'Alliesthesia_slope': sl}

for bvar, blab in [('Pct_Yes', '% Yes'), ('Alliesthesia_slope', 'Alliesthesia Slope')]:
    report(f"\n--- {blab} ---")
    results = []
    for contrast in CONDITIONS + ['foodplus_YesVsNo', 'juice_YesVsNo']:
        nsub = ndf[ndf['Contrast'] == contrast]
        for roi in ALL_ROIS:
            bv, nv = [], []
            for _, row in nsub.iterrows():
                nid = row['Subject'].split('_')[0]
                if nid in behav and not np.isnan(behav[nid].get(bvar, np.nan)):
                    nv.append(row[roi])
                    bv.append(behav[nid][bvar])
            if len(bv) >= 5:
                rho, p = stats.spearmanr(nv, bv)
                results.append((roi, contrast, len(bv), rho, p))
    results.sort(key=lambda x: x[4])
    for roi, contrast, n, rho, p in results[:20]:
        sig = get_stars(p) or ''
        report(f"  {roi} x {contrast:<28} N={n:>2} rho={rho:>7.3f} p={p:.4f} {sig}")

# --- Section 4: Model-based RSA ---
report(f"\n\nSECTION 4: MODEL-BASED RSA (N={N5}, {N_PERMS} permutations)")
report("=" * 80)

nc = len(CONDITIONS)
ti, tj = np.triu_indices(nc, k=1)

profiles = np.zeros((N5, nc, len(ALL_ROIS)))
for ci, c in enumerate(CONDITIONS):
    sd = ndf[ndf['Contrast'] == c].set_index('Subject')
    for si, s in enumerate(subs5):
        profiles[si, ci] = [sd.loc[s, r] for r in ALL_ROIS]

m1 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]], dtype=float)
m2 = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=float)
m3 = np.array([[0, .5, .5, 0, 0], [.5, 0, 1, .5, .5], [.5, 1, 0, .5, .5], [0, .5, .5, 0, 0], [0, .5, .5, 0, 0]], dtype=float)
m4 = np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=float)

MODEL_MATS = [m1, m2, m3, m4]
MODEL_NAMES = ['Food Reward', 'Arousal', 'Valence', 'Visual Category']
nm = len(MODEL_MATS)
mvecs = np.array([m[ti, tj] for m in MODEL_MATS])


def rdm_vec(prof):
    p = prof - prof.mean(1, keepdims=True)
    n = np.sqrt((p ** 2).sum(1, keepdims=True))
    n[n == 0] = 1
    p = p / n
    return 1 - (p @ p.T)[ti, tj]


def sprho(nvec, mvecs):
    nr = rankdata(nvec)
    return np.array([np.corrcoef(nr, rankdata(mv))[0, 1] for mv in mvecs])


report("Computing neural RDMs...")
srdm = np.array([rdm_vec(profiles[si]) for si in range(N5)])
srho = np.array([sprho(srdm[si], mvecs) for si in range(N5)])

report(f"Running {N_PERMS} permutations...")
rng = np.random.default_rng(SEED)
pmean = np.zeros((N_PERMS, nm))
for pi in range(N_PERMS):
    pr = np.zeros((N5, nm))
    for si in range(N5):
        pr[si] = sprho(rdm_vec(profiles[si, rng.permutation(nc)]), mvecs)
    pmean[pi] = pr.mean(0)
    if (pi + 1) % 2000 == 0:
        report(f"  {pi + 1}/{N_PERMS}")
report("Done.")

mrdm = srdm.mean(0)
nu = np.mean([stats.spearmanr(mrdm, srdm[si])[0] for si in range(N5)])
nl = np.mean([stats.spearmanr(np.mean(np.delete(srdm, si, 0), 0), srdm[si])[0] for si in range(N5)])

means_r = srho.mean(0)
perm_p = np.array([(np.sum(pmean[:, mi] >= means_r[mi]) + 1) / (N_PERMS + 1) for mi in range(nm)])

report(f"\n  {'Model':<20} {'rho':>8} {'SD':>8} {'t':>8} {'p(par)':>9} {'p(perm)':>9}")
report(f"  {'-' * 65}")
for mi in range(nm):
    sd = srho[:, mi].std(ddof=1)
    t, pp = stats.ttest_1samp(srho[:, mi], 0)
    sig = get_stars(perm_p[mi]) or ''
    report(f"  {MODEL_NAMES[mi]:<20}{means_r[mi]:>8.4f}{sd:>8.4f}{t:>8.3f}{pp:>9.4f}{perm_p[mi]:>9.4f} {sig}")
report(f"\n  Noise ceiling: [{nl:.4f}, {nu:.4f}]")

report(f"\n  Pairwise comparisons:")
for i, j in combinations(range(nm), 2):
    d = srho[:, i] - srho[:, j]
    t, p = stats.ttest_rel(srho[:, i], srho[:, j])
    sig = get_stars(p) or ''
    report(f"  {MODEL_NAMES[i]} vs {MODEL_NAMES[j]:<20} diff={d.mean():.4f} t={t:.3f} p={p:.4f} {sig}")

# --- Section 5: Behavioral ---
report("\n\nSECTION 5: BEHAVIORAL")
report("=" * 80)
subjects_beh = sorted(food_plus_beh['Subject'].unique())
pcts_all = [behav[s]['Pct_Yes'] for s in subjects_beh]
slopes_all = [behav[s]['Alliesthesia_slope'] for s in subjects_beh]
report(f"  %Yes: M={np.mean(pcts_all):.1f}, SD={np.std(pcts_all, ddof=1):.1f}")
t_sl, p_sl = stats.ttest_1samp(slopes_all, 0)
report(f"  Slope: M={np.mean(slopes_all):.2f}%/run, t={t_sl:.3f}, p={p_sl:.4f}")
report(f"  Negative slopes: {sum(1 for s in slopes_all if s < 0)}/{len(slopes_all)}")

run_data = {}
for subj in subjects_beh:
    sub = food_plus_beh[food_plus_beh['Subject'] == subj]
    rpcts = []
    for r in all_runs:
        rs = sub[sub['RunList'] == r]
        ny = (rs['rename2'] == 'food+Y').sum()
        nt = len(rs)
        rpcts.append(ny / nt * 100 if nt > 0 else np.nan)
    run_data[subj] = rpcts

SF.close()
print(f"Stats saved to {STATS_FILE}")


# =============================================================================
# =============================================================================
# FIGURES
# =============================================================================
# =============================================================================

def make_twopanel_fig(rois, roi_labels, figsize, outname):
    """Figs 1B and 2B: two-panel grouped bars with post-hoc brackets."""
    nr = len(rois)
    bw = 0.18
    offsets = np.array([-bw - 0.02, 0, bw + 0.02])
    x_pos = np.arange(nr)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

    for ax, pconds, pcols, pleg in [
        (axL, ['ero', 'mut', 'neu'], [COL_ERO, COL_MUT, COL_NEU],
         [Patch(fc=COL_ERO, ec='black', lw=0.5, label='Erotica'),
          Patch(fc=COL_MUT, ec='black', lw=0.5, label='Mutilations'),
          Patch(fc=COL_NEU, ec='black', lw=0.5, label='Neutral')]),
        (axR, ['foodplus', 'foodminus', 'neu'], [COL_FP, COL_FM, COL_NEU],
         [Patch(fc=COL_FP, ec='black', lw=0.5, label='Food+'),
          Patch(fc=COL_FM, ec='black', lw=0.5, label='Food−'),
          Patch(fc=COL_NEU, ec='black', lw=0.5, label='Neutral')]),
    ]:
        for ri, roi in enumerate(rois):
            for ci, (cond, col, off) in enumerate(zip(pconds, pcols, offsets)):
                vals = get_data(roi, cond)
                ax.bar(x_pos[ri] + off, vals.mean(), width=bw, color=col,
                       edgecolor='black', linewidth=0.5,
                       yerr=stats.sem(vals), capsize=2,
                       error_kw={'linewidth': 0.8}, zorder=3)

            v0 = get_data(roi, pconds[0])
            v1 = get_data(roi, pconds[1])
            v2 = get_data(roi, pconds[2])
            ms = [get_data(roi, c).mean() for c in pconds]
            ses = [stats.sem(get_data(roi, c)) for c in pconds]
            yt = max(m + s for m, s in zip(ms, ses)) + 0.012

            t, p = stats.ttest_rel(v0, v2)
            s = get_stars(p)
            if s:
                add_bracket(ax, x_pos[ri] + offsets[0], x_pos[ri] + offsets[2], yt + 0.025, s)
            t, p = stats.ttest_rel(v1, v2)
            s = get_stars(p)
            if s:
                add_bracket(ax, x_pos[ri] + offsets[1], x_pos[ri] + offsets[2], yt + 0.0, s)
            if pconds[0] == 'foodplus':
                t, p = stats.ttest_rel(v0, v1)
                s = get_stars(p)
                if s:
                    add_bracket(ax, x_pos[ri] + offsets[0], x_pos[ri] + offsets[1], yt + 0.05, s)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(roi_labels, fontsize=14)
        ax.axhline(0, color='black', linewidth=0.6, zorder=2)
        ax.set_ylabel('Mean Beta (a.u.)', fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(handles=pleg, fontsize=11, framealpha=0.9, loc='upper right')

    ym = min(axL.get_ylim()[0], axR.get_ylim()[0])
    yM = max(axL.get_ylim()[1], axR.get_ylim()[1])
    axL.set_ylim(ym, yM)
    axR.set_ylim(ym, yM)

    plt.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {outname}")


# --- Functional ROIs (streamlined single panel) ---
print("Making FunctionalROIs...")

def make_streamlined_fig(rois, roi_labels, figsize, outname):
    """Single-panel grouped bars: Pls Neu Unp | gap | Food+ Food-  per ROI."""
    nr = len(rois)
    bw = 0.12
    # 3 emotion bars, small gap, then 2 food bars
    # offsets relative to each ROI center
    gap = 0.06  # extra space between emotion group and food group
    off_emo = np.array([-bw - 0.02, 0, bw + 0.02])                # Pls, Neu, Unp
    off_food = np.array([2 * (bw + 0.02) + gap, 3 * (bw + 0.02) + gap])  # Food+, Food-
    # shift everything so the full group is centered
    all_off = np.concatenate([off_emo, off_food])
    center_shift = (all_off.min() + all_off.max()) / 2
    off_emo  -= center_shift
    off_food -= center_shift
    all_off  -= center_shift

    group_width = all_off.max() - all_off.min() + bw + 0.25
    x_pos = np.arange(nr) * group_width

    conds  = ['ero',     'neu',     'mut',     'foodplus', 'foodminus']
    colors = [COL_ERO,   COL_NEU,   COL_MUT,   COL_FP,     COL_FM]
    labels = ['Pleasant', 'Neutral', 'Unpleasant', 'Food+', 'Food−']
    offsets_all = np.concatenate([off_emo, off_food])

    fig, ax = plt.subplots(figsize=figsize)

    max_bracket_y = -np.inf
    data_min = np.inf

    for ri, roi in enumerate(rois):
        for ci, (cond, col, off) in enumerate(zip(conds, colors, offsets_all)):
            vals = get_data(roi, cond)
            data_min = min(data_min, float(np.nanmin(vals)))
            # Raincloud: half-violin + aligned dots + mean/SEM marker
            draw_raincloud(ax, x_pos[ri] + off, vals, col,
                           w=bw * 0.55, dot_size=10, zorder=3)

        # --- significance brackets ---
        v_pls = get_data(roi, 'ero')
        v_neu = get_data(roi, 'neu')
        v_unp = get_data(roi, 'mut')
        v_fp  = get_data(roi, 'foodplus')
        v_fm  = get_data(roi, 'foodminus')

        # Brackets must clear the violin tops (KDE extends ~15% past data max - see draw_raincloud)
        all_vals = [v_pls, v_neu, v_unp, v_fp, v_fm]
        def _violin_top(v):
            rng_span = v.max() - v.min() if v.max() > v.min() else max(abs(v.max()), 0.1)
            return v.max() + rng_span * 0.15
        yt = max(_violin_top(v) for v in all_vals) + 0.06  # clear gap above violin tops
        step = 0.15  # vertical step between stacked brackets - large gap above each star so stars read as belonging to the line beneath

        # Emotion cluster: Unp vs Neu, Pls vs Neu, Pls vs Unp
        lvl = 0
        t, p = stats.ttest_rel(v_unp, v_neu)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_emo[1], x_pos[ri] + off_emo[2], yt + step * lvl, s)
            lvl += 1
        t, p = stats.ttest_rel(v_pls, v_neu)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_emo[0], x_pos[ri] + off_emo[1], yt + step * lvl, s)
            lvl += 1
        t, p = stats.ttest_rel(v_pls, v_unp)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_emo[0], x_pos[ri] + off_emo[2], yt + step * lvl, s)
            lvl += 1

        # Food cluster: Food+ vs Food-, Food+ vs Neu, Food- vs Neu
        t, p = stats.ttest_rel(v_fp, v_fm)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_food[0], x_pos[ri] + off_food[1], yt + step * lvl, s)
            lvl += 1
        t, p = stats.ttest_rel(v_fm, v_neu)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_emo[1], x_pos[ri] + off_food[1], yt + step * lvl, s)
            lvl += 1
        t, p = stats.ttest_rel(v_fp, v_neu)
        s = get_stars(p)
        if s:
            add_bracket(ax, x_pos[ri] + off_emo[1], x_pos[ri] + off_food[0], yt + step * lvl, s)
            lvl += 1

        # Track highest bracket y so we can give the plot enough headroom
        if lvl > 0:
            max_bracket_y = max(max_bracket_y, yt + step * (lvl - 1))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(roi_labels, fontsize=14)
    ax.axhline(0, color='black', linewidth=0.6, zorder=2)
    ax.set_ylabel('Beta Value', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Expand y-limits so top brackets + significance text have room (not clipped),
    # and the bottom has breathing space for violin tails.
    cur_ylim = ax.get_ylim()
    top = max_bracket_y + 0.15 if np.isfinite(max_bracket_y) else cur_ylim[1]
    bottom = min(cur_ylim[0], data_min - 0.08) if np.isfinite(data_min) else cur_ylim[0]
    ax.set_ylim(bottom, top)

    # Legend
    legend_patches = [
        Patch(fc=COL_ERO, ec='black', lw=0.5, label='Pleasant'),
        Patch(fc=COL_NEU, ec='black', lw=0.5, label='Neutral'),
        Patch(fc=COL_MUT, ec='black', lw=0.5, label='Unpleasant'),
        Patch(fc=COL_FP,  ec='black', lw=0.5, label='Food+'),
        Patch(fc=COL_FM,  ec='black', lw=0.5, label='Food−'),
    ]
    ax.legend(handles=legend_patches, fontsize=11, framealpha=0.9, loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {outname}")

make_streamlined_fig(FUNC_ROIS, FUNC_LABELS, (12, 5), 'FunctionalROIs')

# --- A priori ROIs ---
print("Making APrioriROIs...")
make_streamlined_fig(AP_ROIS, AP_LABELS, (16, 5), 'APrioriROIs')

# --- Wanting vs Liking ---
print("Making WantingLiking...")
fig3, ax3 = plt.subplots(figsize=(15, 5))
x3 = np.arange(len(ALL_ROIS)) * 0.9
bw3 = 0.25
sub_fp = ndf[ndf['Contrast'] == 'foodplus_YesVsNo']
sub_jy = ndf[ndf['Contrast'] == 'juice_YesVsNo']

for ri, roi in enumerate(ALL_ROIS):
    for sub_data, color, offset, star_color in [
        (sub_fp, 'black', -bw3 / 2, 'black'),
        (sub_jy, COL_JUICE, bw3 / 2, COL_JUICE),
    ]:
        vals = sub_data[roi].values
        m, se = vals.mean(), stats.sem(vals)
        t, p = stats.ttest_1samp(vals, 0)
        # Raincloud: half-violin + aligned dots + mean/SEM marker
        draw_raincloud(ax3, x3[ri] + offset, vals, color,
                       w=bw3 * 0.5, dot_size=10, zorder=3)
        s = get_stars(p)
        if s:
            vmax = float(np.nanmax(vals))
            vmin = float(np.nanmin(vals))
            rng_span = vmax - vmin if vmax > vmin else max(abs(vmax), 0.1)
            # Place star well above/below the violin tail (KDE extends ~15% past data)
            yp = (vmax + rng_span * 0.15 + 0.06) if m >= 0 else (vmin - rng_span * 0.15 - 0.09)
            ax3.text(x3[ri] + offset, yp, s, ha='center', va='bottom' if m >= 0 else 'top',
                     fontsize=14, fontweight='bold', color='black', zorder=6,
                     path_effects=STAR_OUTLINE)

ax3.axvline(x3[3] + 0.45, color='grey', lw=0.6, ls='--', alpha=0.4)
ax3.axhline(0, color='black', lw=0.6, zorder=2)
ax3.set_xticks(x3)
ax3.set_xticklabels(ALL_LABELS, fontsize=14)
ax3.set_ylabel('Beta Value (Yes − No)', fontsize=14)
ax3.tick_params(axis='y', labelsize=12)
ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(handles=[
    Patch(fc='black', ec='black', lw=0.5, label='Cue: Food+ Yes > No'),
    Patch(fc=COL_JUICE, ec='black', lw=0.5, label='Outcome: Juice Yes > No')],
    fontsize=11, loc='upper right', framealpha=0.9)
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'WantingLiking.png'), dpi=300, bbox_inches='tight', facecolor='white')
fig3.savefig(os.path.join(FIG_DIR, 'WantingLiking.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("  Saved WantingLiking")

# =============================================================================
# RSA (3 panels - Model RDMs | Neural RDM | Bar plot)
# =============================================================================
print("Making RSA...")

fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(17, 5),
                                          gridspec_kw={'width_ratios': [1.7, 0.9, 0.9]})

# --- Panel A: Theoretical Model RDMs (2x2 grid) ---
ax4a.axis('off')
cond_short = ['N', 'E', 'M', 'F−', 'F+']
model_titles = ['Food Reward', 'Arousal', 'Valence', 'Visual Category']
model_cmap = ListedColormap(['white', '#999999', '#333333'])
# 2x2 layout - inset dims chosen so each inset is SQUARE in figure units
# Panel A is ~8.26 in wide x 5 in tall -> width_frac = height_frac * (5/8.26) ~ height_frac * 0.605
# height_frac = 0.40 -> width_frac ~ 0.242 (gives square insets -> axis box matches matrix exactly)
inset_positions = [
    [0.12, 0.55, 0.242, 0.40],
    [0.60, 0.55, 0.242, 0.40],
    [0.12, 0.05, 0.242, 0.40],
    [0.60, 0.05, 0.242, 0.40],
]
for mi, (mat, title, pos) in enumerate(zip(MODEL_MATS, model_titles, inset_positions)):
    ins = ax4a.inset_axes(pos)
    ins.set_aspect('equal')
    masked = np.copy(mat).astype(float)
    # Plot lower triangle only using imshow for clean square cells
    masked[np.triu_indices(5)] = np.nan
    ins.imshow(np.full((5, 5), np.nan), aspect='equal', extent=[-0.5, 4.5, 4.5, -0.5])
    # Draw lower-triangle cells manually as colored rectangles with borders
    for i in range(5):
        for j in range(i):
            v = mat[i, j]
            if v <= 0.01:
                fc = 'white'
            elif v <= 0.6:
                fc = '#999999'
            else:
                fc = '#333333'
            ins.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                          facecolor=fc, edgecolor='black', linewidth=0.3, zorder=1))
    ins.set_xticks(range(5))
    ins.set_yticks(range(5))
    ins.set_xticklabels(cond_short, fontsize=12)
    ins.set_yticklabels(cond_short, fontsize=12)
    # Label under the matrix (below the x-tick labels) instead of as a title above
    ins.set_xlabel(title, fontsize=13, fontweight='bold', labelpad=6)
    ins.tick_params(length=0, pad=1)
    ins.set_xlim(-0.5, 4.5)
    ins.set_ylim(4.5, -0.5)
    for spine in ins.spines.values():
        spine.set_visible(False)

# --- Panel B: Observed Neural RDM (group mean, lower triangle only) ---
mean_rdm_vec = srdm.mean(axis=0)
mean_rdm_mat = np.zeros((nc, nc))
mean_rdm_mat[ti, tj] = mean_rdm_vec
mean_rdm_mat[tj, ti] = mean_rdm_vec

masked_rdm = np.copy(mean_rdm_mat)
masked_rdm[np.triu_indices(nc)] = np.nan

cond_labels_rdm = ['Neutral', 'Erotica', 'Mut.', 'Food−', 'Food+']
# Use masked array so upper triangle + diagonal are truly transparent
masked_rdm_ma = np.ma.array(masked_rdm, mask=np.isnan(masked_rdm))
im4b = ax4b.pcolormesh(np.arange(nc+1), np.arange(nc+1), masked_rdm_ma,
                        cmap='Greys', vmin=0.2, vmax=0.85, edgecolors='white', linewidths=0.5)
ax4b.set_xticks(np.arange(nc) + 0.5)
ax4b.set_yticks(np.arange(nc) + 0.5)
ax4b.set_xticklabels(cond_labels_rdm, fontsize=13, rotation=35, ha='right')
ax4b.set_yticklabels(cond_labels_rdm, fontsize=13)
ax4b.tick_params(length=0)
for i in range(nc):
    for j in range(i):  # lower triangle only
        v = mean_rdm_mat[i, j]
        tc = 'white' if v > 0.55 else 'black'
        ax4b.text(j + 0.5, i + 0.5, f'{v:.2f}', ha='center', va='center',
                  fontsize=14, fontweight='bold', color=tc, zorder=3)
ax4b.set_xlim(0, nc)
ax4b.set_ylim(nc, 0)
ax4b.set_aspect('equal')
for spine in ax4b.spines.values():
    spine.set_visible(False)
# Cell values annotated directly; no colorbar needed

# --- Panel C: Model Fit Bar Plot ---
# Compress x positions so bars sit closer together; also make bars thinner
x4 = np.arange(nm) * 0.65
sems_r = srho.std(0, ddof=1) / np.sqrt(N5)
bcols = [COL_RSA_SIG if perm_p[mi] < 0.05 else '#DDDDDD' for mi in range(nm)]

ax4c.bar(x4, means_r, yerr=sems_r, capsize=3, color=bcols,
         edgecolor='none', width=0.28, zorder=3,
         error_kw={'linewidth': 1.0, 'color': 'black'})

# Noise ceiling band
ax4c.fill_between([x4[0] - 0.35, x4[-1] + 0.35], nl, nu, color='#B0C4DE', alpha=0.1, zorder=1)
ax4c.axhline(nl, color='grey', lw=0.5, ls=':', alpha=0.3)
ax4c.axhline(nu, color='grey', lw=0.5, ls=':', alpha=0.3)

# Stars above significant bars
for mi in range(nm):
    s = get_stars(perm_p[mi])
    if s:
        ax4c.text(x4[mi], means_r[mi] + sems_r[mi] + 0.025, s, ha='center', va='bottom',
                  fontsize=16, fontweight='bold', color='black', zorder=6,
                  path_effects=STAR_OUTLINE)

ax4c.axhline(0, color='black', lw=0.6, zorder=2)
ax4c.set_xticks(x4)
ax4c.set_xticklabels(MODEL_NAMES, fontsize=12, rotation=25, ha='right')
ax4c.set_xlim(x4[0] - 0.35, x4[-1] + 0.35)
# Fewer, larger y-axis ticks
ax4c.set_yticks([-0.3, 0, 0.3])
ax4c.tick_params(axis='y', labelsize=13)
ax4c.set_ylabel('Spearman ρ\n(model − neural RDM)', fontsize=13)
ax4c.spines['top'].set_visible(False)
ax4c.spines['right'].set_visible(False)

# Noise ceiling legend
nc_patch = Patch(facecolor='#B0C4DE', alpha=0.3, edgecolor='grey', label='Noise ceiling')
ax4c.legend(handles=[nc_patch], loc='upper right', fontsize=10, framealpha=0.8)

plt.tight_layout(w_pad=2)

# --- Panel letters (A, B, C) ---
for ax, letter in [(ax4a, 'A'), (ax4b, 'B'), (ax4c, 'C')]:
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', ha='right', va='bottom')

fig4.savefig(os.path.join(FIG_DIR, 'RSA.png'), dpi=300, bbox_inches='tight', facecolor='white')
fig4.savefig(os.path.join(FIG_DIR, 'RSA.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig4)
print("  Saved RSA")

# =============================================================================
# Behavior (2 panels - Spaghetti | Histogram)
# =============================================================================
print("Making Behavior...")
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Panel A: Spaghetti plot ---
x_runs = np.arange(1, len(all_runs) + 1)
run_mat = np.array([run_data[s] for s in subjects_beh])
gm = np.nanmean(run_mat, axis=0)
gsem = np.nanstd(run_mat, axis=0, ddof=1) / np.sqrt(len(subjects_beh))

overall_pcts = np.array([np.nanmean(run_data[s]) for s in subjects_beh])
cmap = plt.colormaps['viridis']
norm_pcts = overall_pcts / 100.0

for si, subj in enumerate(subjects_beh):
    ax5a.plot(x_runs, run_data[subj], color=cmap(norm_pcts[si]),
              linewidth=0.4, alpha=0.2, zorder=2)

ax5a.plot(x_runs, gm, color='red', linewidth=3, zorder=4,
          marker='D', markersize=7, markeredgecolor='darkred', markeredgewidth=0.5)
ax5a.fill_between(x_runs, gm - gsem, gm + gsem, color='red', alpha=0.1, zorder=3)
ax5a.axhline(50, color='grey', lw=0.6, ls='--', alpha=0.4)

ax5a.set_xlabel('Run', fontsize=11)
ax5a.set_ylabel('Juice Acceptance Rate (%)', fontsize=11)
ax5a.set_xticks(x_runs)
ax5a.set_ylim(-5, 105)
ax5a.spines['top'].set_visible(False)
ax5a.spines['right'].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax5a, shrink=0.7, pad=0.02)
cbar.set_label('Overall acceptance %', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# --- Panel B: Histogram of slopes (clean - stats in report only) ---
ax5b.hist(slopes_all, bins=8, color='#6baed6', edgecolor='white', linewidth=0.8, alpha=0.8, zorder=3)
ax5b.axvline(np.mean(slopes_all), color='red', lw=2, ls='--',
             label=f'Mean = {np.mean(slopes_all):.2f} %/run')
ax5b.axvline(0, color='black', lw=0.6)

ax5b.set_xlabel('Acceptance Slope (%/run)', fontsize=11)
ax5b.set_ylabel('Count', fontsize=11)
ax5b.legend(fontsize=8, loc='upper left')
ax5b.spines['top'].set_visible(False)
ax5b.spines['right'].set_visible(False)

plt.tight_layout(w_pad=3)
fig5.savefig(os.path.join(FIG_DIR, 'Behavior.png'), dpi=300, bbox_inches='tight', facecolor='white')
fig5.savefig(os.path.join(FIG_DIR, 'Behavior.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig5)
print("  Saved Behavior")

# =============================================================================
# TSV: All pairwise comparisons for every ROI (for table creation)
# =============================================================================
print("Saving pairwise comparisons TSV...")
tsv_path = os.path.join(FIG_DIR, 'pairwise_comparisons.tsv')
tsv_rows = []
comp_list = [
    ('ero',      'neu',       'Pleasant vs Neutral'),
    ('mut',      'neu',       'Unpleasant vs Neutral'),
    ('ero',      'mut',       'Pleasant vs Unpleasant'),
    ('foodplus', 'neu',       'Food+ vs Neutral'),
    ('foodminus','neu',       'Food− vs Neutral'),
    ('foodplus', 'foodminus', 'Food+ vs Food−'),
    ('ero',      'foodplus',  'Pleasant vs Food+'),
    ('ero',      'foodminus', 'Pleasant vs Food−'),
    ('mut',      'foodplus',  'Unpleasant vs Food+'),
    ('mut',      'foodminus', 'Unpleasant vs Food−'),
]
for roi, roi_label in zip(ALL_ROIS, ALL_LABELS):
    for c1, c2, comp_label in comp_list:
        v1, v2 = get_data(roi, c1), get_data(roi, c2)
        diff = v1 - v2
        t_val, p_val = stats.ttest_rel(v1, v2)
        d_val = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
        sig = get_stars(p_val) or 'ns'
        tsv_rows.append({
            'ROI': roi_label,
            'Comparison': comp_label,
            'Mean_Diff': f'{diff.mean():.4f}',
            'SD_Diff': f'{diff.std(ddof=1):.4f}',
            't': f'{t_val:.3f}',
            'df': N5 - 1,
            'p': f'{p_val:.4f}',
            'Cohen_d': f'{d_val:.3f}',
            'Sig': sig,
        })

tsv_df = pd.DataFrame(tsv_rows)
tsv_df.to_csv(tsv_path, sep='\t', index=False)
print(f"  Saved {tsv_path}")

print(f"\nAll done! Figures and stats in: {FIG_DIR}")
