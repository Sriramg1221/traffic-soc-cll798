"""
run_simulation.py
-----------------
Driver script: runs the traffic simulation across four topologies
and three connectivity levels, saving all results to data/.

Usage:
    python run_simulation.py           # full run (~5 min)
    python run_simulation.py --fast    # quick test run
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_model import run_experiment, DEFAULT_PARAMS

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Experiment configurations ─────────────────────────────────────────────────

TOPOLOGIES = ['lattice', 'random', 'small_world', 'scale_free']

# Connectivity sweep: mean degree k
K_VALUES = [3, 5, 8]

FULL_PARAMS = dict(
    n_nodes  = 40,
    n_steps  = 8000,
    burn_in  = 1000,
    density  = 0.25,
    p_brake  = 0.3,
    v_max    = 5,
    road_len = 80,
    seed     = 42,
)

FAST_PARAMS = dict(
    n_nodes  = 20,
    n_steps  = 2000,
    burn_in  = 200,
    density  = 0.25,
    p_brake  = 0.3,
    v_max    = 5,
    road_len = 50,
    seed     = 42,
)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Quick test run with reduced parameters')
    args = parser.parse_args()

    params = FAST_PARAMS if args.fast else FULL_PARAMS
    label  = 'FAST' if args.fast else 'FULL'
    print(f"=== Traffic SOC Simulation  [{label} mode] ===\n")

    total = len(TOPOLOGIES) * len(K_VALUES)
    done  = 0

    for topo in TOPOLOGIES:
        for k in K_VALUES:
            done += 1
            print(f"\n[{done}/{total}] topology={topo}  k={k}")
            run_experiment(
                topology  = topo,
                n_nodes   = params['n_nodes'],
                k         = k,
                params    = params,
                save_dir  = DATA_DIR,
            )

    print(f"\n=== All done. Results in {os.path.abspath(DATA_DIR)} ===")


if __name__ == '__main__':
    main()
