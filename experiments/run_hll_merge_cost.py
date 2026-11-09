"""Measure the CPU-memory cost of coarse HLL segment merging.

The query protocol matches run_count_distinct.py: seed 42, 200
COUNT_DISTINCT windows, and fractions {5%, 10%, 20%, 50%}. K_HLL={16,52}
compares the paper default with the existing CD-heavy setting.
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS


SEED = 42
N_QUERIES = 200
FRACTIONS = [0.05, 0.10, 0.20, 0.50]
K_VALUES = [16, 52]
TIMING_TRIALS = 5


def generate_queries(t_min, t_max):
    rng = random.Random(SEED)
    span = max(1, t_max - t_min)
    queries = []
    for frac in FRACTIONS:
        for _ in range(N_QUERIES // len(FRACTIONS)):
            width = max(1, int(frac * span))
            left = rng.randint(t_min, max(t_min, t_max - width))
            queries.append((left, left + width, frac))
    return queries


def touched_segments(cam, left, right):
    return sum(
        1 for seg_left, seg_right in zip(cam.segment_boundaries[:-1], cam.segment_boundaries[1:])
        if seg_right > left and seg_left < right
    )


def time_trial(cam, queries, trial):
    per_query_us = []
    checksum = 0
    timing_start = time.perf_counter()
    for left, right, _ in queries:
        start_ns = time.perf_counter_ns()
        checksum += cam.query_count_distinct(left, right)
        per_query_us.append((time.perf_counter_ns() - start_ns) / 1000.0)
    timing_s = time.perf_counter() - timing_start
    return {
        'trial': trial,
        'query_median_us': float(np.median(per_query_us)),
        'query_p95_us': float(np.percentile(per_query_us, 95)),
        'query_p99_us': float(np.percentile(per_query_us, 99)),
        'query_qps': len(queries) / timing_s,
        'checksum': checksum,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*')
    parser.add_argument('--k-order', default='16,52')
    parser.add_argument('--output', default='hll_merge_cost.csv')
    parser.add_argument('--interleaved', action='store_true')
    args = parser.parse_args()
    k_values = [int(value) for value in args.k_order.split(',')]
    if sorted(k_values) != sorted(K_VALUES):
        raise ValueError(f'--k-order must contain {K_VALUES}')

    np.random.seed(SEED)
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = set(args.datasets)
    if selected:
        datasets = [(name, loader) for name, loader in datasets if name in selected]

    rows = []
    for name, loader in datasets:
        print(f'[hll] loading {name}', flush=True)
        frame = ds.preprocess_temporal(loader())
        t_min = int(frame['time'].min())
        t_max = int(frame['time'].max())
        queries = generate_queries(t_min, t_max)
        cams = {}
        build_times = {}
        for k_hll in k_values:
            cam = CAMRS(n_buckets=500, n_hll_segments=k_hll)
            build_start = time.perf_counter()
            cam.build(frame)
            build_times[k_hll] = time.perf_counter() - build_start
            cams[k_hll] = cam

            for left, right, _ in queries[:20]:
                cam.query_count_distinct(left, right)

        trial_rows_by_k = {k_hll: [] for k_hll in k_values}
        for trial in range(1, TIMING_TRIALS + 1):
            trial_order = k_values
            if args.interleaved and trial % 2 == 0:
                trial_order = list(reversed(k_values))
            for k_hll in trial_order:
                trial_rows_by_k[k_hll].append(time_trial(cams[k_hll], queries, trial))

        for k_hll in k_values:
            cam = cams[k_hll]
            trial_rows = trial_rows_by_k[k_hll]
            touched = [touched_segments(cam, left, right) for left, right, _ in queries]
            hll_bytes = k_hll * (1 << cam.hll_b)
            for trial_row in trial_rows:
                rows.append({
                    'dataset': name,
                    'n_edges': len(frame),
                    'K_HLL': k_hll,
                    'cms_width': cam.cms_width,
                    'hll_bytes': hll_bytes,
                    'total_reported_bytes': cam.memory_usage(),
                    'build_s': build_times[k_hll],
                    'n_queries': len(queries),
                    'segments_mean': float(np.mean(touched)),
                    'segments_p95': float(np.percentile(touched, 95)),
                    **trial_row,
                })
            median_latency = float(np.median([row['query_median_us'] for row in trial_rows]))
            median_p95 = float(np.median([row['query_p95_us'] for row in trial_rows]))
            print(
                f"  K={k_hll}: hll={hll_bytes / 1024:.0f}KB, "
                f"segments={np.mean(touched):.2f}, median={median_latency:.1f}us, "
                f"median-p95={median_p95:.1f}us",
                flush=True,
            )

    output = os.path.join(os.path.dirname(__file__), '..', 'results', args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
