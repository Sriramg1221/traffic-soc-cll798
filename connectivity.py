"""
connectivity.py
---------------
Analyses the role of network connectivity in traffic jam cascade dynamics.

Produces:
  - figures/density_phase_diagram.pdf   : flow vs density (fundamental diagram)
  - figures/jam_timeseries.pdf          : avalanche size time-series
  - figures/network_comparison.pdf      : side-by-side network visualisations
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from traffic_model import TrafficNetwork, build_network, DEFAULT_PARAMS

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

TOPO_COLORS = {
    'lattice':     '#185FA5',
    'random':      '#0F6E56',
    'small_world': '#BA7517',
    'scale_free':  '#A32D2D',
}


# ── Fundamental diagram: flow vs density ─────────────────────────────────────

def plot_fundamental_diagram():
    """
    Classic NaSch fundamental diagram on a single road segment.
    Identifies critical density where jams emerge.
    """
    from traffic_model import RoadSegment
    rng = np.random.default_rng(42)

    densities   = np.linspace(0.05, 0.95, 30)
    mean_flows  = []

    for rho in densities:
        seg = RoadSegment(length=200, v_max=5, p_brake=0.3, density=rho, rng=rng)
        # burn-in
        for _ in range(300):
            seg.step()
        # measure
        flows = []
        for _ in range(500):
            seg.step()
            pos = seg.vehicle_positions
            if len(pos):
                flows.append(seg.cells[pos].mean() * rho)
        mean_flows.append(np.mean(flows) if flows else 0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(densities, mean_flows, 'o-', ms=4, color='#185FA5', lw=1.5)
    ax.axvline(densities[np.argmax(mean_flows)], ls='--', color='#A32D2D',
               lw=1, label=f"$\\rho^* \\approx {densities[np.argmax(mean_flows)]:.2f}$")
    ax.set_xlabel('Vehicle density $\\rho$', fontsize=11)
    ax.set_ylabel('Mean flow $q$', fontsize=11)
    ax.set_title('Fundamental diagram (NaSch model)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', lw=0.4, alpha=0.6)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, 'fundamental_diagram.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ── Avalanche time-series ─────────────────────────────────────────────────────

def plot_timeseries():
    """Plot avalanche size vs time for two topologies side-by-side."""
    params = {**DEFAULT_PARAMS, 'n_steps': 1500, 'burn_in': 300,
              'n_nodes': 25, 'road_len': 60}

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for ax, topo in zip(axes, ['lattice', 'scale_free']):
        G   = build_network(topo, n_nodes=params.get('n_nodes', 25), k=4)
        sim = TrafficNetwork(G, params)

        sizes = []
        for _ in range(params['n_steps']):
            sim.step()
            if sim.avalanche_log:
                sizes.append(sim.avalanche_log[-1])
            else:
                sizes.append(0)

        t = np.arange(len(sizes))
        ax.fill_between(t, sizes, alpha=0.4, color=TOPO_COLORS[topo])
        ax.plot(t, sizes, lw=0.5, color=TOPO_COLORS[topo])
        ax.set_ylabel('Avalanche size', fontsize=9)
        ax.set_title(topo.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, ls=':', lw=0.4, alpha=0.5)

    axes[-1].set_xlabel('Time step', fontsize=11)
    plt.suptitle('Avalanche time-series: intermittency across scales', fontsize=11, y=1.01)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, 'jam_timeseries.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ── Network topology visualisation ───────────────────────────────────────────

def plot_network_comparison():
    """Draw the four network topologies for the paper figure."""
    topologies = ['lattice', 'random', 'small_world', 'scale_free']
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, topo in zip(axes, topologies):
        G = build_network(topo, n_nodes=20, k=4, seed=42)
        if topo == 'lattice':
            pos = nx.spring_layout(G, seed=42)
        elif topo == 'scale_free':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        degrees = dict(G.degree())
        node_sizes = [20 + 8 * degrees[n] for n in G.nodes()]

        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_size=node_sizes,
            node_color=TOPO_COLORS[topo],
            edge_color='#888780',
            width=0.6, alpha=0.85,
            with_labels=False,
        )
        k_mean = 2 * G.number_of_edges() / G.number_of_nodes()
        ax.set_title(f"{topo.replace('_', ' ')}\n$\\langle k\\rangle={k_mean:.1f}$",
                     fontsize=9)
        ax.axis('off')

    plt.suptitle('Road network topologies used in simulation', fontsize=11)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, 'network_comparison.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Plotting fundamental diagram...")
    plot_fundamental_diagram()

    print("Plotting avalanche time-series...")
    plot_timeseries()

    print("Plotting network comparison...")
    plot_network_comparison()

    print("\nDone.")
