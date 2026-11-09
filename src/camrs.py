"""CAMRS implementation."""
import bisect
import numpy as np
try:
    from .sketches import CountMinSketch, HyperLogLog
except ImportError:
    from sketches import CountMinSketch, HyperLogLog


COUNTER_BYTES = 8
PREDICATE_SLOT_METADATA_BYTES = 16
GLOBAL_METADATA_BYTES = 32


def predicate_slot_bytes(micro_hll_b):
    """Compact bytes for one retained label and its reusable query state."""
    payload = (1 << micro_hll_b) if micro_hll_b > 0 else 0
    return PREDICATE_SLOT_METADATA_BYTES + payload


def camrs_logical_allocation(space_budget, n_buckets, n_hll_segments,
                             cms_depth, hll_b, micro_hll_b,
                             top_k_labels):
    """Return (CMS width, charged bytes) for the compact logical layout."""
    hll_bytes = n_hll_segments * (1 << hll_b)
    per_bucket_fixed = COUNTER_BYTES
    if top_k_labels > 0:
        per_bucket_fixed += top_k_labels * predicate_slot_bytes(micro_hll_b)
    fixed_bytes = GLOBAL_METADATA_BYTES + hll_bytes + n_buckets * per_bucket_fixed
    cms_cell_bytes = n_buckets * cms_depth * 8
    width = max(1, (space_budget - fixed_bytes) // cms_cell_bytes)
    logical_bytes = fixed_bytes + cms_cell_bytes * width
    if width < 1 or logical_bytes > space_budget:
        raise ValueError(
            f'CAMRS allocation requires at least {fixed_bytes + cms_cell_bytes} '
            f'logical bytes; budget is {space_budget}')
    return int(width), int(logical_bytes)




class CAMRS:
    """Budget-decoupled multi-resolution sketch with instrumentation."""

    def __init__(self, space_budget=1024*1024, n_buckets=500,
                 n_hll_segments=16, cms_depth=4, hll_b=12,
                 micro_hll_b=6, seed=42, top_k_labels=5,
                 min_label_count=3):
        self.space_budget = space_budget
        self.n_buckets = n_buckets
        self.n_hll_segments = n_hll_segments
        self.seed = seed
        self.cms_depth = cms_depth
        self.hll_b = hll_b
        self.micro_hll_b = micro_hll_b
        self.top_k_labels = top_k_labels
        self.min_label_count = min_label_count

        self.cms_width, self.logical_bytes = camrs_logical_allocation(
            space_budget, n_buckets, n_hll_segments, cms_depth, hll_b,
            micro_hll_b, top_k_labels)

        self.bucket_cms = []
        for i in range(n_buckets):
            self.bucket_cms.append({
                'cms': CountMinSketch(self.cms_width, cms_depth, seed + i),
                'count': 0,
                'space_saving': SpaceSaving(top_k_labels),
                'top_labels': set(),
                'pred_counts': {},
                'label_hll': {},
                'label_exact': {},
            })

        self.segment_hll = [HyperLogLog(hll_b) for _ in range(n_hll_segments)]

        self.bucket_boundaries = None
        self.segment_boundaries = None
        self.t_min = 0
        self.t_max = 0
        self.edge_count = 0


        self._cv_per_bucket = None
        self._topk_hits = 0
        self._topk_queries = 0
        self._cd_pred_hits = 0
        self._cd_pred_queries = 0
        self._query_latencies = []

    def build(self, df, label_map=None):
        self.edge_count = len(df)
        times = df['time'].values
        self.t_min, self.t_max = int(times.min()), int(times.max())
        if self.t_max == self.t_min:
            self.t_max = self.t_min + 1

        self.bucket_boundaries = np.linspace(
            self.t_min, self.t_max + 1, self.n_buckets + 1).astype(int).tolist()
        self.segment_boundaries = np.linspace(
            self.t_min, self.t_max + 1, self.n_hll_segments + 1).astype(int).tolist()


        src = df['src'].values
        dst = df['dst'].values
        for i in range(len(df)):
            s, d, t = int(src[i]), int(dst[i]), int(times[i])
            key = str(s) + '_' + str(d)
            bidx = bisect.bisect_right(self.bucket_boundaries, t) - 1
            sk = self.bucket_cms[bidx]
            sk['count'] += 1
            if self.n_hll_segments > 0:
                sidx = bisect.bisect_right(self.segment_boundaries, t) - 1
                self.segment_hll[sidx].add(key)
            if label_map and s in label_map:
                lbl = label_map[s]
                sk['cms'].add(lbl, 1)
                sk['space_saving'].add(lbl)


        for sk in self.bucket_cms:
            sk['top_labels'] = set(sk['space_saving'].counters)
            sk['pred_counts'] = {lbl: 0 for lbl in sk['top_labels']}


        if self.micro_hll_b > 0:
            for sk in self.bucket_cms:
                sk['label_hll'] = {}
                sk['label_exact'] = {}
                if self.min_label_count > 0:
                    sk['label_exact_sets'] = {}
                    for lbl in sk['top_labels']:
                        sk['label_exact_sets'][lbl] = set()
                else:
                    for lbl in sk['top_labels']:
                        sk['label_hll'][lbl] = HyperLogLog(self.micro_hll_b)


        if label_map:
            for i in range(len(df)):
                s, d, t = int(src[i]), int(dst[i]), int(times[i])
                lbl = label_map.get(s)
                if lbl is None:
                    continue
                bidx = bisect.bisect_right(self.bucket_boundaries, t) - 1
                sk = self.bucket_cms[bidx]
                if lbl not in sk['top_labels']:
                    continue
                sk['pred_counts'][lbl] += 1
                if self.micro_hll_b <= 0:
                    continue
                key = str(s) + '_' + str(d)
                if lbl in sk.get('label_exact_sets', {}):
                    st = sk['label_exact_sets'][lbl]
                    st.add(key)
                    if len(st) > self.min_label_count:
                        hll = HyperLogLog(self.micro_hll_b)
                        for k in st:
                            hll.add(k)
                        sk['label_hll'][lbl] = hll
                        del sk['label_exact_sets'][lbl]
                elif lbl in sk['label_hll']:
                    sk['label_hll'][lbl].add(key)


            if self.min_label_count > 0:
                for sk in self.bucket_cms:
                    for lbl, st in sk.get('label_exact_sets', {}).items():
                        sk['label_exact'][lbl] = frozenset(st)
                    sk.pop('label_exact_sets', None)


        self._cv_per_bucket = []
        bin_edges = np.array(self.bucket_boundaries)
        for j in range(self.n_buckets):
            mask = (times >= bin_edges[j]) & (times < bin_edges[j + 1])
            n_sub = max(1, np.sum(mask))
            if n_sub <= 1:
                self._cv_per_bucket.append(0.0)
                continue

            sub_l = bin_edges[j]
            sub_r = bin_edges[j + 1]
            sub_width = max(1, (sub_r - sub_l) // 8)

            hist_bins = [sub_l + k * sub_width for k in range(9)]
            counts, _ = np.histogram(times, bins=hist_bins)
            counts = counts.astype(float)
            mean_c = counts.mean()
            if mean_c > 0:
                cv = counts.std(ddof=1) / mean_c
            else:
                cv = 0.0
            self._cv_per_bucket.append(cv)

    def query_count(self, t_l, t_r):
        total = 0.0
        for bidx, sk in enumerate(self.bucket_cms):
            b_l = self.bucket_boundaries[bidx]
            b_r = self.bucket_boundaries[bidx + 1]
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
        for bidx, sk in enumerate(self.bucket_cms):
            b_l = self.bucket_boundaries[bidx]
            b_r = self.bucket_boundaries[bidx + 1]
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
        for sidx in range(self.n_hll_segments):
            s_l = self.segment_boundaries[sidx]
            s_r = self.segment_boundaries[sidx + 1]
            if s_r <= t_l or s_l >= t_r:
                continue
            merged.merge(self.segment_hll[sidx])
        return merged.estimate()

    def query_count_distinct_pred(self, t_l, t_r, label_val):
        """Predicate COUNT_DISTINCT using hybrid exact + micro-HLL."""
        merged = HyperLogLog(self.micro_hll_b)
        self._cd_pred_queries += 1
        hit = False
        exact_items = set()
        has_hll = False
        for bidx, sk in enumerate(self.bucket_cms):
            b_l = self.bucket_boundaries[bidx]
            b_r = self.bucket_boundaries[bidx + 1]
            if b_r <= t_l or b_l >= t_r:
                continue
            if label_val in sk['label_hll']:
                merged.merge(sk['label_hll'][label_val])
                hit = True
                has_hll = True
            elif label_val in sk.get('label_exact', {}):
                exact_items.update(sk['label_exact'][label_val])
                hit = True
        if hit:
            self._cd_pred_hits += 1
        if has_hll:
            for item in exact_items:
                merged.add(item)
            return merged.estimate()
        return len(exact_items)

    def predicted_error_bound(self, t_l, t_r):
        """CV-aware empirical risk score used by adaptive routing."""
        f_c = (t_r - t_l) / max(1, self.t_max - self.t_min)
        if f_c <= 0:
            return float('inf')
        return (2.0 * self.max_cv()) / (f_c * self.n_buckets)

    def mean_cv(self):
        return float(np.mean(self._cv_per_bucket)) if self._cv_per_bucket else 0.0

    def max_cv(self):
        return float(np.max(self._cv_per_bucket)) if self._cv_per_bucket else 0.0

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




class SpaceSaving:
    """Streaming top-K frequency estimator (Metwally et al., 2005).

    Maintains at most *k* counters.  On arrival of a new item:
    - if the item is already tracked, increment its counter;
    - else if fewer than *k* counters exist, allocate one;
    - else evict the item with the smallest counter and give the new item
      that counter value plus one (over-estimate guarantee).
    """

    def __init__(self, k):
        self.k = k
        self.counters = {}

    def add(self, label):
        """Process one arrival.

        Returns (is_tracked, evicted_label) where *evicted_label* is the
        label displaced to make room (None if no eviction occurred).
        """
        if label in self.counters:
            self.counters[label] += 1
            return True, None
        if len(self.counters) < self.k:
            self.counters[label] = 1
            return True, None

        min_label = min(self.counters, key=self.counters.get)
        min_count = self.counters[min_label]
        del self.counters[min_label]
        self.counters[label] = min_count + 1
        return True, min_label

    def __contains__(self, label):
        return label in self.counters
