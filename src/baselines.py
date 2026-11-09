"""Sampling and HLL baselines used by the experiments."""
import numpy as np
try:
    from .sketches import HyperLogLog
except ImportError:
    from sketches import HyperLogLog


class UniformSamplerBudget:
    """Reservoir-style uniform sampler that fills the entire memory budget."""

    def __init__(self, space_budget=1024 * 1024, bytes_per_edge=24, seed=42):
        self.space_budget = space_budget
        self.bytes_per_edge = bytes_per_edge
        self.sample_size = max(1024, space_budget // bytes_per_edge)
        self.edges = []
        self.total_edges = 0
        self.t_min = 0
        self.t_max = 1
        self.seed = seed

    def build(self, df, label_map=None):
        self.total_edges = len(df)
        times = df['time'].values
        self.t_min, self.t_max = int(times.min()), int(times.max())
        if self.t_max == self.t_min:
            self.t_max = self.t_min + 1
        n = min(self.sample_size, self.total_edges)
        idx = np.random.RandomState(self.seed).choice(self.total_edges, n, replace=False)
        if label_map is not None:
            label_arr = df['src'].map(label_map).fillna(-1).astype(int).values
        else:
            label_arr = np.full(self.total_edges, -1, dtype=int)
        src = df['src'].values
        dst = df['dst'].values
        self.edges = np.column_stack([
            src[idx].astype(np.int64),
            dst[idx].astype(np.int64),
            times[idx].astype(np.int64),
            label_arr[idx].astype(np.int64),
        ])

    def query_count(self, t_l, t_r):
        ts = self.edges[:, 2]
        m = (ts >= t_l) & (ts <= t_r)
        return int(float(m.sum()) * self.total_edges / max(len(self.edges), 1))

    def query_count_pred(self, t_l, t_r, label_val):
        ts = self.edges[:, 2]
        lb = self.edges[:, 3]
        m = (ts >= t_l) & (ts <= t_r) & (lb == label_val)
        return int(float(m.sum()) * self.total_edges / max(len(self.edges), 1))

    def cms_width(self):
        return 'N/A'

    def _use_label_correction(self):
        return True

    def memory_usage(self):
        return len(self.edges) * self.bytes_per_edge


class StratifiedHLLOnly:
    """K_HLL-segment HLL with no CMS/label storage; lower bound for CD error
    under the same budget."""

    def __init__(self, space_budget=1024 * 1024, n_hll_segments=16, hll_b=12):
        self.space_budget = space_budget
        self.n_hll_segments = n_hll_segments
        self.hll_b = hll_b
        self.segments = [HyperLogLog(hll_b) for _ in range(n_hll_segments)]
        self.boundaries = None
        self.t_min = 0
        self.t_max = 1

    def build(self, df, label_map=None):
        times = df['time'].values
        self.t_min, self.t_max = int(times.min()), int(times.max())
        if self.t_max == self.t_min:
            self.t_max = self.t_min + 1
        self.boundaries = np.linspace(
            self.t_min, self.t_max + 1, self.n_hll_segments + 1).astype(int).tolist()
        src = df['src'].values
        dst = df['dst'].values
        for i in range(len(df)):
            t = int(times[i])
            sidx = min(int((t - self.t_min) / (self.t_max - self.t_min) * self.n_hll_segments),
                       self.n_hll_segments - 1)
            self.segments[sidx].add(str(int(src[i])) + '_' + str(int(dst[i])))

    def query_count_distinct(self, t_l, t_r):
        merged = HyperLogLog(self.hll_b)
        for sidx in range(self.n_hll_segments):
            sl, sr = self.boundaries[sidx], self.boundaries[sidx + 1]
            if sr <= t_l or sl >= t_r:
                continue
            merged.merge(self.segments[sidx])
        return merged.estimate()

    def memory_usage(self):
        return sum(h.__sizeof__() for h in self.segments)
