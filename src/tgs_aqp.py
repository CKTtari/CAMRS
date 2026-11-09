"""Coupled temporal graph sketch baseline (TGS-AQP)."""
import numpy as np

try:
    from .camrs import SpaceSaving, COUNTER_BYTES, GLOBAL_METADATA_BYTES, predicate_slot_bytes
    from .sketches import CountMinSketch, HyperLogLog
except ImportError:
    from camrs import SpaceSaving, COUNTER_BYTES, GLOBAL_METADATA_BYTES, predicate_slot_bytes
    from sketches import CountMinSketch, HyperLogLog


def tgs_logical_allocation(space_budget, n_buckets, cms_depth, hll_b,
                           micro_hll_b, top_k_labels, min_cms_width=2):
    """Return (CMS width, charged bytes) for a coupled compact layout."""
    per_bucket_fixed = COUNTER_BYTES + (1 << hll_b)
    if top_k_labels > 0:
        per_bucket_fixed += top_k_labels * predicate_slot_bytes(micro_hll_b)
    cms_cell_bytes = n_buckets * cms_depth * 8
    width = (space_budget - GLOBAL_METADATA_BYTES
             - n_buckets * per_bucket_fixed) // cms_cell_bytes
    width = max(min_cms_width, int(width))
    logical_bytes = (GLOBAL_METADATA_BYTES + n_buckets * per_bucket_fixed
                     + cms_cell_bytes * width)
    return width, int(logical_bytes)


class TGSAQP:
    """Per-bucket sketch with budget coupling, for direct comparison."""

    def __init__(self, space_budget=1024*1024, n_buckets=52,
                 cms_depth=4, hll_b=12, seed=42, top_k_labels=5,
                 use_uniform=True, micro_hll_b=6, min_label_count=3,
                 enforce_budget=True):
        self.space_budget = space_budget
        self.n_buckets = n_buckets
        self.seed = seed
        self.cms_depth = cms_depth
        self.hll_b = hll_b
        self.top_k_labels = top_k_labels
        self.use_uniform = use_uniform
        self.micro_hll_b = micro_hll_b
        self.min_label_count = min_label_count
        self.enforce_budget = enforce_budget

        cms_width, logical_bytes = tgs_logical_allocation(
            space_budget, n_buckets, cms_depth, hll_b, micro_hll_b,
            top_k_labels)
        if logical_bytes > space_budget and enforce_budget:
            raise ValueError(
                f'B={n_buckets} requires {logical_bytes} logical bytes; '
                f'budget is {space_budget}')
        self.cms_width = cms_width
        self.logical_bytes = logical_bytes

        self.bucket_sketches = []
        for i in range(n_buckets):
            self.bucket_sketches.append({
                'cms': CountMinSketch(cms_width, cms_depth, seed + i),
                'hll': HyperLogLog(hll_b),
                'count': 0,
                'space_saving': SpaceSaving(top_k_labels),
                'top_labels': set(),
                'pred_counts': {},
                'label_hll': {},
                'label_exact': {},
            })
        self.bucket_boundaries = None
        self.edge_count = 0
        self.t_min = 0
        self.t_max = 0
        self._topk_hits = 0
        self._topk_queries = 0
        self._cd_pred_hits = 0
        self._cd_pred_queries = 0

    def build(self, df, label_map=None):
        self.edge_count = len(df)
        times = df['time'].values
        self.t_min, self.t_max = int(times.min()), int(times.max())
        if self.t_max == self.t_min:
            self.t_max = self.t_min + 1
        if self.use_uniform:
            self.bucket_boundaries = np.linspace(
                self.t_min, self.t_max + 1, self.n_buckets + 1).astype(int).tolist()
        else:
            self._fit_adaptive_boundaries(df)
        src = df['src'].values
        dst = df['dst'].values
        for i in range(len(df)):
            s, d, t = int(src[i]), int(dst[i]), int(times[i])
            bidx = self._get_bucket_idx(t)
            sk = self.bucket_sketches[bidx]
            key = str(s) + '_' + str(d)
            sk['hll'].add(key)
            sk['count'] += 1
            if label_map and s in label_map:
                lbl = label_map[s]
                sk['cms'].add(lbl, 1)
                sk['space_saving'].add(lbl)
        for sk in self.bucket_sketches:
            sk['top_labels'] = set(sk['space_saving'].counters)
            sk['pred_counts'] = {lbl: 0 for lbl in sk['top_labels']}
            sk['label_hll'] = {}
            sk['label_exact'] = {}
            if self.micro_hll_b > 0:
                if self.min_label_count > 0:
                    sk['label_exact_sets'] = {
                        lbl: set() for lbl in sk['top_labels']
                    }
                else:
                    sk['label_hll'] = {
                        lbl: HyperLogLog(self.micro_hll_b)
                        for lbl in sk['top_labels']
                    }

        if label_map:
            for i in range(len(df)):
                s, d, t = int(src[i]), int(dst[i]), int(times[i])
                lbl = label_map.get(s)
                if lbl is None:
                    continue
                sk = self.bucket_sketches[self._get_bucket_idx(t)]
                if lbl in sk['top_labels']:
                    sk['pred_counts'][lbl] += 1
                    if self.micro_hll_b <= 0:
                        continue
                    key = str(s) + '_' + str(d)
                    if lbl in sk.get('label_exact_sets', {}):
                        state = sk['label_exact_sets'][lbl]
                        state.add(key)
                        if len(state) > self.min_label_count:
                            hll = HyperLogLog(self.micro_hll_b)
                            for item in state:
                                hll.add(item)
                            sk['label_hll'][lbl] = hll
                            del sk['label_exact_sets'][lbl]
                    elif lbl in sk['label_hll']:
                        sk['label_hll'][lbl].add(key)

            if self.micro_hll_b > 0 and self.min_label_count > 0:
                for sk in self.bucket_sketches:
                    for lbl, state in sk.get('label_exact_sets', {}).items():
                        sk['label_exact'][lbl] = frozenset(state)
                    sk.pop('label_exact_sets', None)

    def _fit_adaptive_boundaries(self, df):
        span = self.t_max - self.t_min
        n_samples = min(1000, self.n_buckets * 10)
        slice_size = span / n_samples
        activity = np.zeros(n_samples)
        for t in df['time'].values:
            idx = min(int((t - self.t_min) / slice_size), n_samples - 1)
            activity[idx] += 1
        activity_smooth = np.convolve(activity, np.ones(5)/5, mode='same')
        activity_smooth = np.maximum(activity_smooth, 1)
        alloc = np.sqrt(activity_smooth)
        alloc = alloc / alloc.sum() * self.n_buckets
        cum = np.cumsum(alloc)
        boundaries = [self.t_min]
        for i in range(1, self.n_buckets):
            target = i / self.n_buckets * cum[-1]
            idx = np.searchsorted(cum, target)
            t = self.t_min + int((idx + 0.5) / n_samples * span)
            boundaries.append(min(t, self.t_max))
        boundaries.append(self.t_max + 1)
        self.bucket_boundaries = boundaries

    def _get_bucket_idx(self, t):
        for i in range(len(self.bucket_boundaries) - 1):
            if self.bucket_boundaries[i] <= t < self.bucket_boundaries[i + 1]:
                return i
        return len(self.bucket_boundaries) - 2

    def query_count(self, t_l, t_r):
        total = 0.0
        for bidx, sk in enumerate(self.bucket_sketches):
            b_l = int(self.bucket_boundaries[bidx])
            b_r = int(self.bucket_boundaries[bidx + 1])
            if b_r <= t_l or b_l >= t_r:
                continue
            overlap = min(b_r, t_r) - max(b_l, t_l)
            ratio = overlap / max(b_r - b_l, 1)
            total += sk['count'] * ratio
        return int(total)

    def query_count_pred(self, t_l, t_r, label_val):
        total = 0.0
        self._topk_queries += 1
        hit = False
        for bidx, sk in enumerate(self.bucket_sketches):
            b_l = int(self.bucket_boundaries[bidx])
            b_r = int(self.bucket_boundaries[bidx + 1])
            if b_r <= t_l or b_l >= t_r:
                continue
            overlap = min(b_r, t_r) - max(b_l, t_l)
            ratio = overlap / max(b_r - b_l, 1)
            if label_val in sk['top_labels']:
                total += sk['pred_counts'].get(label_val, 0) * ratio
                hit = True
            else:
                total += sk['cms'].estimate(label_val) * ratio
        if hit:
            self._topk_hits += 1
        return int(total)

    def query_count_distinct(self, t_l, t_r):
        merged = HyperLogLog(self.hll_b)
        for bidx, sk in enumerate(self.bucket_sketches):
            b_l = int(self.bucket_boundaries[bidx])
            b_r = int(self.bucket_boundaries[bidx + 1])
            if b_r <= t_l or b_l >= t_r:
                continue
            merged.merge(sk['hll'])
        return merged.estimate()

    def query_count_distinct_pred(self, t_l, t_r, label_val):
        """Predicate COUNT_DISTINCT over the coupled per-bucket state."""
        merged = HyperLogLog(self.micro_hll_b)
        exact_items = set()
        has_hll = False
        hit = False
        self._cd_pred_queries += 1
        for bidx, sk in enumerate(self.bucket_sketches):
            b_l = int(self.bucket_boundaries[bidx])
            b_r = int(self.bucket_boundaries[bidx + 1])
            if b_r <= t_l or b_l >= t_r:
                continue
            if label_val in sk['label_hll']:
                merged.merge(sk['label_hll'][label_val])
                has_hll = True
                hit = True
            elif label_val in sk['label_exact']:
                exact_items.update(sk['label_exact'][label_val])
                hit = True
        if hit:
            self._cd_pred_hits += 1
        if has_hll:
            for item in exact_items:
                merged.add(item)
            return merged.estimate()
        return len(exact_items)

    def topk_hit_rate(self):
        if self._topk_queries == 0:
            return 0.0
        return self._topk_hits / self._topk_queries

    def cd_pred_hit_rate(self):
        if self._cd_pred_queries == 0:
            return 0.0
        return self._cd_pred_hits / self._cd_pred_queries

    def reset_hit_rate(self):
        self._topk_hits = 0
        self._topk_queries = 0
        self._cd_pred_hits = 0
        self._cd_pred_queries = 0

    def memory_usage(self):
        return self.logical_bytes
