"""
Traffic Jam Cascade on a Road Network
Core simulation: Nagel-Schreckenberg (NaSch) cellular automaton
extended to arbitrary graph topologies.

Deliverable 4b mapping:
  Noise       -> stochastic braking probability p_brake
  Avalanche   -> spatially connected cluster of vehicles at v=0
  Connectivity-> number of road segments per node (mean degree <k>)
"""

import numpy as np
import networkx as nx
import pickle
import os


# ── Model parameters (defaults) ───────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    v_max      = 5,       # maximum speed (cells/step)
    p_brake    = 0.3,     # random braking probability (noise)
    density    = 0.25,    # vehicle density (vehicles / road cell)
    road_len   = 100,     # cells per road segment
    n_steps    = 5000,    # total simulation steps
    burn_in    = 500,     # steps discarded before recording
    seed       = 42,
)


# ── Road network builder ───────────────────────────────────────────────────────

def build_network(topology: str, n_nodes: int = 20, k: int = 4, seed: int = 42) -> nx.Graph:
    """
    Return an undirected graph representing the road network.

    topology options
    ----------------
    'lattice'   : 2-D grid (regular, k~4)
    'random'    : Erdos-Renyi G(n, m) with mean degree k
    'small_world': Watts-Strogatz with rewiring prob 0.1
    'scale_free': Barabasi-Albert preferential attachment
    """
    rng = np.random.default_rng(seed)
    if topology == 'lattice':
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif topology == 'random':
        m = int(n_nodes * k / 2)
        G = nx.gnm_random_graph(n_nodes, m, seed=seed)
    elif topology == 'small_world':
        G = nx.watts_strogatz_graph(n_nodes, k, 0.1, seed=seed)
    elif topology == 'scale_free':
        m_ba = max(1, k // 2)
        G = nx.barabasi_albert_graph(n_nodes, m_ba, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Remove self-loops and isolated nodes
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# ── Road segment (1-D NaSch rule) ─────────────────────────────────────────────

class RoadSegment:
    """
    Single directed road segment of `length` cells.
    Vehicles are stored as an array of speeds (-1 = empty cell).
    """

    def __init__(self, length: int, v_max: int, p_brake: float, density: float, rng):
        self.length  = length
        self.v_max   = v_max
        self.p_brake = p_brake
        self.rng     = rng

        # Initialise vehicles randomly
        n_vehicles = max(1, int(density * length))
        positions  = rng.choice(length, size=n_vehicles, replace=False)
        self.cells = np.full(length, -1, dtype=int)   # -1 = empty
        initial_v  = rng.integers(0, v_max + 1, size=n_vehicles)
        self.cells[positions] = initial_v

    @property
    def vehicle_positions(self):
        return np.where(self.cells >= 0)[0]

    @property
    def n_stopped(self):
        return int(np.sum(self.cells == 0))

    def step(self):
        """One NaSch update step (periodic boundary within segment)."""
        positions = self.vehicle_positions
        n = len(positions)
        if n == 0:
            return

        new_cells = np.full(self.length, -1, dtype=int)

        for idx, pos in enumerate(positions):
            v = self.cells[pos]

            # 1. Acceleration
            v = min(v + 1, self.v_max)

            # 2. Braking (gap to next vehicle)
            next_pos = positions[(idx + 1) % n]
            if next_pos > pos:
                gap = next_pos - pos - 1
            else:
                gap = (self.length - pos - 1) + next_pos  # wrap-around
            v = min(v, gap)

            # 3. Random braking (noise)
            if v > 0 and self.rng.random() < self.p_brake:
                v = max(v - 1, 0)

            # 4. Move
            new_pos = (pos + v) % self.length
            new_cells[new_pos] = v

        self.cells = new_cells


# ── Network simulation ─────────────────────────────────────────────────────────

class TrafficNetwork:
    """
    Road network simulation.

    Each edge in the graph is a bi-directional RoadSegment.
    An 'avalanche' is defined as a spatially contiguous cluster of
    stopped vehicles (v=0) across multiple segments.
    """

    def __init__(self, G: nx.Graph, params: dict):
        self.G      = G
        self.params = {**DEFAULT_PARAMS, **params}
        self.rng    = np.random.default_rng(self.params['seed'])

        # One RoadSegment per edge (undirected)
        self.segments = {}
        for u, v in G.edges():
            seg = RoadSegment(
                length   = self.params['road_len'],
                v_max    = self.params['v_max'],
                p_brake  = self.params['p_brake'],
                density  = self.params['density'],
                rng      = self.rng,
            )
            self.segments[(u, v)] = seg
            self.segments[(v, u)] = seg   # same object, both directions

        self.step_count    = 0
        self.avalanche_log = []   # list of avalanche sizes recorded after burn-in

    # ── One global time step ──────────────────────────────────────────────────

    def step(self):
        for seg in set(self.segments.values()):
            seg.step()
        self.step_count += 1

        if self.step_count > self.params['burn_in']:
            size = self._measure_avalanche()
            if size > 0:
                self.avalanche_log.append(size)

    # ── Avalanche measurement ─────────────────────────────────────────────────

    def _measure_avalanche(self) -> int:
        """
        Avalanche size = total number of stopped vehicles in the largest
        contiguous stopped-vehicle cluster across the network graph.

        Algorithm: build a subgraph of edges that contain at least one
        stopped vehicle; connected components of that subgraph give clusters.
        Return the size of the largest cluster.
        """
        # Edges with ≥1 stopped vehicle
        active_edges = []
        stopped_per_edge = {}
        for (u, v), seg in self.segments.items():
            if u < v:   # avoid double-counting
                n_stop = seg.n_stopped
                stopped_per_edge[(u, v)] = n_stop
                if n_stop > 0:
                    active_edges.append((u, v))

        if not active_edges:
            return 0

        H = nx.Graph()
        H.add_edges_from(active_edges)
        components = list(nx.connected_components(H))

        # Largest cluster: sum stopped vehicles across its edges
        max_size = 0
        for comp in components:
            subedges = [(u, v) for (u, v) in active_edges
                        if u in comp and v in comp]
            size = sum(stopped_per_edge.get((u, v), 0) for (u, v) in subedges)
            max_size = max(max_size, size)

        return max_size

    # ── Run full simulation ───────────────────────────────────────────────────

    def run(self, verbose: bool = True):
        total = self.params['n_steps']
        for t in range(total):
            self.step()
            if verbose and t % 500 == 0:
                print(f"  step {t:5d}/{total}  |  avalanches recorded: {len(self.avalanche_log)}")
        if verbose:
            print(f"  Done. Total avalanche events: {len(self.avalanche_log)}")

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def global_flow(self) -> float:
        """Mean vehicle speed across all segments (flow proxy)."""
        speeds = []
        for seg in set(self.segments.values()):
            pos = seg.vehicle_positions
            if len(pos) > 0:
                speeds.extend(seg.cells[pos].tolist())
        return float(np.mean(speeds)) if speeds else 0.0


# ── Convenience runner ────────────────────────────────────────────────────────

def run_experiment(topology: str, n_nodes: int = 30, k: int = 4,
                   params: dict = None, save_dir: str = None) -> dict:
    """
    Build network, run simulation, return results dict.
    Optionally pickle results to save_dir.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    G = build_network(topology, n_nodes=n_nodes, k=k, seed=p['seed'])

    print(f"[{topology}] nodes={G.number_of_nodes()}  "
          f"edges={G.number_of_edges()}  "
          f"mean_degree={2*G.number_of_edges()/G.number_of_nodes():.2f}")

    sim = TrafficNetwork(G, p)
    sim.run(verbose=True)

    result = dict(
        topology      = topology,
        n_nodes       = G.number_of_nodes(),
        n_edges       = G.number_of_edges(),
        mean_degree   = 2 * G.number_of_edges() / G.number_of_nodes(),
        avalanches    = sim.avalanche_log,
        params        = p,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{topology}_k{k}.pkl")
        with open(fname, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Saved -> {fname}")

    return result


if __name__ == '__main__':
    # Quick smoke test
    result = run_experiment('lattice', n_nodes=25, k=4,
                            params={'n_steps': 1000, 'burn_in': 100})
    print(f"Avalanche sizes (first 10): {result['avalanches'][:10]}")
