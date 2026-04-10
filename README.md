# Traffic Jam Cascade on a Road Network
### CLL798: Complexity Science in Chemical Industry — Individual Project

**Author:** Sriram G | 2023CH70173 | IIT Delhi  
**Submitted:** April 10, 2026

---

## Overview

This project applies **self-organised criticality (SOC)** theory to traffic 
jam cascades on road networks. Using a Nagel–Schreckenberg (NaSch) cellular 
automaton extended to arbitrary graph topologies, we show that:

- Traffic jam cascades follow a **power-law size distribution** P(s) ∝ s^(−τ)
- The power-law exponent τ **decreases with network connectivity** ⟨k⟩
- The system self-organises to a critical state **without parameter tuning**
- **Scale-free networks** produce heavier-tailed distributions than lattices

---

## Complexity Science Mapping

| Concept | Traffic System |
|---|---|
| **Noise** | Stochastic braking (probability p_brake per step) |
| **Avalanche** | Contiguous cluster of stopped vehicles (v=0) |
| **Connectivity** | Mean network degree ⟨k⟩ (intersections per node) |

---

## Repository Structure

```
traffic-soc-cll798/
├── README.md
├── requirements.txt
├── simulation/
│   ├── traffic_model.py       # NaSch model + network simulation core
│   └── run_simulation.py      # Driver: topology × connectivity sweep
├── analysis/
│   ├── power_law_fit.py       # MLE fitting, log-log plots, CSV export
│   └── connectivity.py        # Fundamental diagram, time-series, network viz
├── data/
│   ├── *.pkl                  # Pickled simulation results
│   └── fit_results.csv        # Summary: τ, KS, n_avalanches per config
├── figures/
│   └── *.pdf                  # All paper figures (auto-generated)
└── report/
    ├── main.tex               # LaTeX manuscript
    └── main.pdf               # Compiled PDF
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run simulations (fast test mode)

```bash
python simulation/run_simulation.py --fast
```

This runs a reduced parameter sweep (~2 minutes) and saves results to `data/`.

### 3. Run full simulations

```bash
python simulation/run_simulation.py
```

Full sweep across 4 topologies × 3 connectivity levels (~5–10 minutes).

### 4. Generate all figures and fit results

```bash
python analysis/connectivity.py
python analysis/power_law_fit.py
```

Figures saved to `figures/`. Fit summary saved to `data/fit_results.csv`.

### 5. Compile the report

```bash
cd report
pdflatex main.tex
pdflatex main.tex   # run twice for TOC
```

---

## Model Parameters

| Parameter | Symbol | Default |
|---|---|---|
| Maximum speed | v_max | 5 cells/step |
| Braking probability (noise) | p_brake | 0.3 |
| Vehicle density | ρ | 0.25 |
| Road segment length | L_e | 80 cells |
| Network nodes | n | 40 |
| Mean degree sweep | ⟨k⟩ | 3, 5, 8 |
| Simulation steps | T | 8000 |
| Burn-in steps | T_burn | 1000 |

---

## Network Topologies

| Topology | Description | Notes |
|---|---|---|
| `lattice` | Regular 2-D grid | ⟨k⟩ ≈ 4, uniform |
| `random` | Erdős–Rényi G(n,m) | Poisson degree dist. |
| `small_world` | Watts–Strogatz β=0.1 | Short path lengths |
| `scale_free` | Barabási–Albert | P(k) ∝ k^(−γ) hub structure |

---

## Requirements

```
numpy>=1.24
networkx>=3.0
matplotlib>=3.7
scipy>=1.10
```

---

## Acknowledgements

- Simulation based on: Nagel & Schreckenberg (1992), *J. Phys. I France*
- Power-law fitting: Clauset, Shalizi & Newman (2009), *SIAM Review*
- Network models: NetworkX library (Hagberg et al., 2008)
- Project assistance: Claude by Anthropic (prompts logged in report appendix)

---

## License

MIT License. See `LICENSE` for details.
