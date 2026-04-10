"""
power_law_fit.py
----------------
Fit power-law P(s) ~ s^(-tau) to avalanche size distributions
using Maximum Likelihood Estimation (MLE) following Clauset et al. (2009).

Produces:
  - figures/freq_dist_<topology>.png  (log-log plot per topology)
  - figures/exponent_summary.png      (tau vs mean degree)
  - data/fit_results.csv
"""

import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    'lattice':     '#185FA5',
    'random':      '#0F6E56',
    'small_world': '#BA7517',
    'scale_free':  '#A32D2D',
}

# ── MLE power-law estimator ───────────────────────────────────────────────────

def mle_powerlaw_exponent(data: np.ndarray, s_min: int = 1) -> float:
    """
    Maximum Likelihood Estimator for discrete power-law exponent tau.
    Clauset, Shalizi & Newman (2009), eq. 3.1
    """
    x = data[data >= s_min].astype(float)
    n = len(x)
    if n < 10:
        return np.nan
    tau = 1.0 + n / np.sum(np.log(x / (s_min - 0.5)))
    return tau


def ks_statistic(data: np.ndarray, tau: float, s_min: int = 1) -> float:
    """Kolmogorov-Smirnov statistic between empirical CDF and fitted power-law CDF."""
    x = np.sort(data[data >= s_min])
    n = len(x)
    if n == 0:
        return np.nan
    empirical_cdf = np.arange(1, n + 1) / n
    # theoretical CDF: P(X <= x) = 1 - (x/s_min)^(1-tau)  [continuous approx]
    theoretical_cdf = 1.0 - (x / s_min) ** (1.0 - tau)
    theoretical_cdf = np.clip(theoretical_cdf, 0, 1)
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def ccdf(data: np.ndarray):
    """Complementary CDF: P(S >= s) for each unique s."""
    data = np.sort(data)
    n = len(data)
    unique_s = np.unique(data)
    p = np.array([(data >= s).sum() / n for s in unique_s])
    return unique_s, p


# ── Load all results ──────────────────────────────────────────────────────────

def load_results() -> list:
    files = glob.glob(os.path.join(DATA_DIR, '*.pkl'))
    results = []
    for f in files:
        with open(f, 'rb') as fh:
            results.append(pickle.load(fh))
    results.sort(key=lambda r: (r['topology'], r['mean_degree']))
    return results


# ── Plot: log-log frequency distribution per topology ────────────────────────

def plot_freq_distributions(results: list):
    topologies = list(dict.fromkeys(r['topology'] for r in results))

    for topo in topologies:
        topo_results = [r for r in results if r['topology'] == topo]
        fig, ax = plt.subplots(figsize=(5, 4))

        fit_lines = []
        for res in sorted(topo_results, key=lambda r: r['mean_degree']):
            avl = np.array(res['avalanches'])
            if len(avl) < 20:
                continue
            avl = avl[avl > 0]
            s_vals, p_vals = ccdf(avl)

            k_label = f"$\\langle k \\rangle = {res['mean_degree']:.1f}$"
            col = plt.cm.Blues(0.3 + 0.35 * topo_results.index(res))
            ax.scatter(s_vals, p_vals, s=6, alpha=0.6, color=col, label=k_label)

            # Power-law fit overlay
            tau = mle_powerlaw_exponent(avl, s_min=2)
            if not np.isnan(tau):
                s_fit = np.logspace(np.log10(2), np.log10(s_vals.max()), 80)
                norm  = p_vals[0] * (s_vals[0] ** (tau - 1))
                p_fit = norm * s_fit ** (-(tau - 1))
                ax.plot(s_fit, p_fit, '--', lw=1.2,
                        color=COLORS.get(topo, 'gray'),
                        label=f'fit $\\tau={tau:.2f}$' if topo_results.index(res) == len(topo_results)-1 else None)
                fit_lines.append(tau)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Avalanche size $s$', fontsize=11)
        ax.set_ylabel('$P(S \geq s)$', fontsize=11)
        ax.set_title(f'{topo.replace("_"," ").title()} network', fontsize=11)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
        plt.tight_layout()

        fname = os.path.join(FIGURES_DIR, f'freq_dist_{topo}.pdf')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"  Saved {fname}")
        plt.close(fig)


# ── Plot: tau vs mean degree (connectivity) ───────────────────────────────────

def plot_exponent_vs_connectivity(results: list):
    topologies = list(dict.fromkeys(r['topology'] for r in results))

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for topo in topologies:
        topo_results = sorted([r for r in results if r['topology'] == topo],
                              key=lambda r: r['mean_degree'])
        k_vals, tau_vals, ks_vals = [], [], []

        for res in topo_results:
            avl = np.array(res['avalanches'])
            avl = avl[avl > 0]
            tau = mle_powerlaw_exponent(avl, s_min=2)
            ks  = ks_statistic(avl, tau, s_min=2) if not np.isnan(tau) else np.nan
            if not np.isnan(tau):
                k_vals.append(res['mean_degree'])
                tau_vals.append(tau)
                ks_vals.append(ks)

        if k_vals:
            ax.plot(k_vals, tau_vals, 'o-', lw=1.5, ms=6,
                    color=COLORS.get(topo, 'gray'),
                    label=topo.replace('_', ' '))

    ax.set_xlabel('Mean degree $\\langle k \\rangle$ (connectivity)', fontsize=11)
    ax.set_ylabel('Power-law exponent $\\tau$', fontsize=11)
    ax.set_title('Avalanche exponent vs. network connectivity', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, 'exponent_vs_connectivity.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ── Save CSV summary ──────────────────────────────────────────────────────────

def save_csv(results: list):
    import csv
    fname = os.path.join(DATA_DIR, 'fit_results.csv')
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['topology', 'n_nodes', 'n_edges', 'mean_degree',
                         'n_avalanches', 'mean_size', 'max_size', 'tau', 'ks'])
        for res in results:
            avl = np.array(res['avalanches'])
            avl = avl[avl > 0]
            tau = mle_powerlaw_exponent(avl, s_min=2)
            ks  = ks_statistic(avl, tau, s_min=2) if not np.isnan(tau) else np.nan
            writer.writerow([
                res['topology'],
                res['n_nodes'],
                res['n_edges'],
                f"{res['mean_degree']:.2f}",
                len(avl),
                f"{avl.mean():.2f}" if len(avl) else 'N/A',
                int(avl.max()) if len(avl) else 'N/A',
                f"{tau:.3f}" if not np.isnan(tau) else 'N/A',
                f"{ks:.3f}" if not np.isnan(ks) else 'N/A',
            ])
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading results...")
    results = load_results()
    print(f"  Found {len(results)} result files.\n")

    print("Plotting frequency distributions...")
    plot_freq_distributions(results)

    print("\nPlotting exponent vs connectivity...")
    plot_exponent_vs_connectivity(results)

    print("\nSaving CSV summary...")
    save_csv(results)

    print("\nDone.")
