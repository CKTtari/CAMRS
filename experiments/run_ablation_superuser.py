"""Run the RQ1 four-configuration ablation on sx-superuser."""

import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS
from tgs_aqp import TGSAQP


SEED = 42
COUNT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
WORKLOAD_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'results',
    'superuser_workload.csv'))


def degree_terciles(frame):
    nodes = sorted(set(frame['src'].values) | set(frame['dst'].values))
    src_counts = frame['src'].value_counts().to_dict()
    dst_counts = frame['dst'].value_counts().to_dict()
    degree = {node: src_counts.get(node, 0) + dst_counts.get(node, 0) for node in nodes}
    ordered = sorted(nodes, key=lambda node: degree[node], reverse=True)
    first, second = len(nodes) // 3, 2 * len(nodes) // 3
    return {
        node: 0 if rank < first else 1 if rank < second else 2
        for rank, node in enumerate(ordered)
    }


def count_queries(frame, rng):
    t_min, t_max = int(frame['time'].min()), int(frame['time'].max())
    span = max(1, t_max - t_min)
    queries = []
    for fraction in COUNT_FRACTIONS:
        width = max(1, int(fraction * span))
        for _ in range(100):
            left = rng.randint(t_min, max(t_min, t_max - width))
            queries.append((left, left + width))
    return queries


def predicate_queries(frame, labels, rng):
    t_min, t_max = int(frame['time'].min()), int(frame['time'].max())
    span = max(1, t_max - t_min)
    label_values = sorted(set(labels.values()))
    queries = []
    per_fraction, remainder = divmod(200, len(COUNT_FRACTIONS))
    for index, fraction in enumerate(COUNT_FRACTIONS):
        width = max(1, int(fraction * span))
        for _ in range(per_fraction + (index < remainder)):
            left = rng.randint(t_min, max(t_min, t_max - width))
            queries.append((left, left + width, rng.choice(label_values)))
    return queries


def load_formal_workloads():
    count_workload = []
    predicate_workload = []
    with open(WORKLOAD_FILE, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            left, right = int(row['t_l']), int(row['t_r'])
            if row['query_type'] == 'COUNT':
                count_workload.append((left, right))
            elif row['query_type'] == 'PREDICATE_COUNT':
                predicate_workload.append((left, right, int(row['label'])))
    if len(count_workload) != 600 or len(predicate_workload) != 200:
        raise RuntimeError(f'Unexpected workload size in {WORKLOAD_FILE}')
    return count_workload, predicate_workload


def exact_count(times, left, right):
    return int(((times >= left) & (times <= right)).sum())


def exact_predicate_count(frame, label_lookup, left, right, label):
    times = frame['time'].values
    source_labels = label_lookup[frame['src'].values]
    return int(((times >= left) & (times <= right) & (source_labels == label)).sum())


def mean_relative_error(estimates, exacts):
    errors = [abs(estimate - exact) / exact for estimate, exact in zip(estimates, exacts) if exact > 0]
    return float(np.mean(errors))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    frame = ds.preprocess_temporal(ds.load_sx_superuser())
    labels = degree_terciles(frame)
    label_lookup = np.full(max(labels) + 1, -1, dtype=np.int8)
    for node, label in labels.items():
        label_lookup[node] = label

    count_workload, predicate_workload = load_formal_workloads()
    times = frame['time'].values
    count_exact = [exact_count(times, left, right) for left, right in count_workload]
    predicate_exact = [
        exact_predicate_count(frame, label_lookup, left, right, label)
        for left, right, label in predicate_workload
    ]

    configurations = [
        ('TGS-B52', lambda: TGSAQP(n_buckets=52, use_uniform=True), 'within 1 MB'),
        ('TGS-B300', lambda: TGSAQP(
            n_buckets=300, use_uniform=True, micro_hll_b=0,
            enforce_budget=False,
        ), 'exceeds 1 MB'),
        ('MR-B52', lambda: CAMRS(n_buckets=52, n_hll_segments=16, micro_hll_b=6), 'within 1 MB'),
        ('MR-B500', lambda: CAMRS(n_buckets=500, n_hll_segments=16, micro_hll_b=6), 'within 1 MB'),
    ]
    rows = []
    for variant, constructor, budget_status in configurations:
        print(f'Building {variant}...')
        sketch = constructor()
        sketch.build(frame, label_map=labels)
        count_estimates = [sketch.query_count(left, right) for left, right in count_workload]
        predicate_estimates = [
            sketch.query_count_pred(left, right, label)
            for left, right, label in predicate_workload
        ]
        row = {
            'dataset': 'sx_superuser',
            'variant': variant,
            'count_error': mean_relative_error(count_estimates, count_exact),
            'predicate_error': mean_relative_error(predicate_estimates, predicate_exact),
            'count_queries': len(count_workload),
            'predicate_queries': len(predicate_workload),
            'budget_status': budget_status,
        }
        rows.append(row)
        print(row)

    output = os.path.join(os.path.dirname(__file__), '..', 'results', 'ablation_sx_superuser.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
