"""Sweep complete-R2 coupled allocations plus the COUNT-only resolution limit.

The protocol matches the paper's formal COUNT/predicate experiments:
six datasets, seed 42, 1 MiB logical sketch payload, D=4, b=12,
600 COUNT queries and 200 predicate COUNT queries over the six published
window fractions.  The evaluator is an exact vectorized specialization of
TGS/CAMRS for COUNT and predicate COUNT.  Published B=52 and B=500
endpoint values are checked against the reported reference values.

Full-R2 allocations reserve a per-bucket full HLL, compact top-K slots, and
an exact bucket total.  The additional count-only row is the corresponding
resolution limit without predicate slots and is
marked explicitly in the output.
"""

import csv
import os
import random
import sys

import mmh3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds


MEMORY_BYTES = 1024 * 1024
CMS_DEPTH = 4
HLL_B = 12
HLL_BYTES = 1 << HLL_B
TOP_K = 5
MICRO_HLL_B = 6
PREDICATE_CD_BYTES = TOP_K * (1 << MICRO_HLL_B)
PREDICATE_SLOT_METADATA_BYTES = TOP_K * 16
BUCKET_TOTAL_BYTES = 8
GLOBAL_METADATA_BYTES = 32
SEED = 42
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
PREDICATE_B_VALUES = (52, 96, 128, 160, 192, 224, 229)

DATASET_SEQUENCE = (
    ('email_eu', ds.load_email_eu),
    ('college_msg', ds.load_college_msg),
    ('wiki_talk', ds.load_wiki_talk),
    ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
    ('sx_superuser', ds.load_sx_superuser),
    ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
)

FORMAL_COUNT = {
    'email_eu': (0.4758, 0.0268),
    'college_msg': (0.1284, 0.0161),
    'wiki_talk': (0.0303, 0.0057),
    'reddit_hyperlinks': (0.0583, 0.0071),
    'sx_superuser': (0.1632, 0.0055),
    'sx_stackoverflow_a2q': (0.0136, 0.0022),
}

FORMAL_PREDICATE = {
    'email_eu': (0.37446681287692685, 0.047598586893689177),
    'college_msg': (0.14919555634779946, 0.031640826642953986),
    'wiki_talk': (0.05421410713294236, 0.010758092947884903),
    'reddit_hyperlinks': (0.03342250623609591, 0.007840663580591453),
    'sx_superuser': (0.0338419083815381, 0.007972040914942517),
    'sx_stackoverflow_a2q': (0.017623916180609503, 0.007326551589818074),
}


def tgs_allocation(n_buckets, complete_r2=True):
    """Return CMS width and charged logical bytes for one coupled allocation."""
    predicate_bytes = (
        PREDICATE_CD_BYTES + PREDICATE_SLOT_METADATA_BYTES
        if complete_r2 else 0)
    fixed_bytes = GLOBAL_METADATA_BYTES + n_buckets * (
        HLL_BYTES + BUCKET_TOTAL_BYTES + predicate_bytes)
    width = max(1, (MEMORY_BYTES - fixed_bytes) // (
        n_buckets * CMS_DEPTH * 8))
    if width < 2:
        width = 2
    logical_bytes = fixed_bytes + n_buckets * CMS_DEPTH * width * 8
    return width, logical_bytes


def feasible_count_allocations():
    rows = []
    for n_buckets in range(52, 513):
        width, logical_bytes = tgs_allocation(n_buckets, complete_r2=True)
        if logical_bytes <= MEMORY_BYTES:
            rows.append((n_buckets, width, logical_bytes))
    return rows


def generate_count_queries(t_min, t_max, rng, n=600):
    span = max(1, t_max - t_min)
    queries = []
    for fraction in FRACTIONS:
        for _ in range(n // len(FRACTIONS)):
            window = max(1, int(fraction * span))
            left = rng.randint(t_min, max(t_min, t_max - window))
            queries.append((left, left + window))
    return queries


def generate_predicate_queries(t_min, t_max, label_values, rng, n=200):
    span = max(1, t_max - t_min)
    queries = []
    per_fraction, remainder = divmod(n, len(FRACTIONS))
    for index, fraction in enumerate(FRACTIONS):
        for _ in range(per_fraction + (index < remainder)):
            window = max(1, int(fraction * span))
            left = rng.randint(t_min, max(t_min, t_max - window))
            label = rng.choice(label_values)
            queries.append((left, left + window, label))
    return queries


def uniform_boundaries(t_min, t_max, n_buckets):
    return np.linspace(t_min, t_max + 1, n_buckets + 1).astype(np.int64)


def bucket_counts_from_sorted_times(sorted_times, boundaries):
    positions = np.searchsorted(sorted_times, boundaries, side='left')
    return np.diff(positions).astype(np.int64)


def interpolated_count(boundaries, bucket_values, left, right):
    starts = boundaries[:-1]
    ends = boundaries[1:]
    overlap = np.minimum(ends, right) - np.maximum(starts, left)
    mask = overlap > 0
    if not np.any(mask):
        return 0
    widths = np.maximum(ends[mask] - starts[mask], 1)
    estimate = np.sum(bucket_values[mask] * overlap[mask] / widths)
    return int(estimate)


def exact_counts(sorted_times, queries):
    return [
        int(np.searchsorted(sorted_times, right, side='right')
            - np.searchsorted(sorted_times, left, side='left'))
        for left, right in queries
    ]


def mean_relative_error(estimates, exact_values):
    errors = [abs(estimate - exact) / exact
              for estimate, exact in zip(estimates, exact_values) if exact > 0]
    return float(np.mean(errors)) if errors else 0.0


def degree_tercile_labels(frame):
    nodes = sorted(set(frame['src'].values) | set(frame['dst'].values))
    src_counts = frame['src'].value_counts().to_dict()
    dst_counts = frame['dst'].value_counts().to_dict()
    degree = {node: src_counts.get(node, 0) + dst_counts.get(node, 0) for node in nodes}
    sorted_nodes = sorted(nodes, key=lambda node: degree[node], reverse=True)
    first, second = len(nodes) // 3, 2 * len(nodes) // 3
    labels = {}
    for node in sorted_nodes[:first]:
        labels[node] = 0
    for node in sorted_nodes[first:second]:
        labels[node] = 1
    for node in sorted_nodes[second:]:
        labels[node] = 2
    return labels


def load_dataset(name, loader):
    if name == 'reddit_hyperlinks':
        raw_frame, label_map = ds.load_reddit_hyperlinks_with_labels(n_labels=10)
        frame = ds.preprocess_temporal(raw_frame)
    else:
        frame = ds.preprocess_temporal(loader())
        label_map = degree_tercile_labels(frame)
    return frame, label_map


def edge_labels(frame, label_map):
    source = frame['src'].values
    return np.fromiter((label_map.get(int(node), -1) for node in source),
                       dtype=np.int16, count=len(source))


def space_saving_labels(bucket_index, labels, n_buckets, label_values):
    if len(label_values) <= TOP_K:
        present = np.zeros((n_buckets, len(label_values)), dtype=bool)
        valid = labels >= 0
        np.logical_or.at(present, (bucket_index[valid], labels[valid]), True)
        return [set(np.flatnonzero(present[index]).tolist()) for index in range(n_buckets)]

    counters = [dict() for _ in range(n_buckets)]
    for bucket, label in zip(bucket_index.tolist(), labels.tolist()):
        if label < 0:
            continue
        state = counters[bucket]
        if label in state:
            state[label] += 1
        elif len(state) < TOP_K:
            state[label] = 1
        else:
            evicted = min(state, key=state.get)
            replacement_count = state[evicted] + 1
            del state[evicted]
            state[label] = replacement_count
    return [set(state) for state in counters]


def cms_estimates(exact_matrix, width):
    n_buckets, n_labels = exact_matrix.shape
    estimates = np.zeros_like(exact_matrix)
    for bucket in range(n_buckets):
        seed = SEED + bucket
        row_estimates = []
        for depth in range(CMS_DEPTH):
            indices = [mmh3.hash(str(label).encode(), depth + seed) % width
                       for label in range(n_labels)]
            table = np.zeros(width, dtype=np.int64)
            np.add.at(table, indices, exact_matrix[bucket])
            row_estimates.append(table[np.asarray(indices, dtype=np.int64)])
        estimates[bucket] = np.min(np.stack(row_estimates, axis=0), axis=0)
    return estimates


def predicate_bucket_values(times, labels, t_min, t_max, n_buckets, width, label_values):
    boundaries = uniform_boundaries(t_min, t_max, n_buckets)
    bucket_index = np.searchsorted(boundaries, times, side='right') - 1
    bucket_index = np.clip(bucket_index, 0, n_buckets - 1)
    exact_matrix = np.zeros((n_buckets, len(label_values)), dtype=np.int64)
    valid = labels >= 0
    np.add.at(exact_matrix, (bucket_index[valid], labels[valid]), 1)

    retained = space_saving_labels(bucket_index, labels, n_buckets, label_values)
    fallback = cms_estimates(exact_matrix, width)
    values = fallback.copy()
    for bucket, retained_labels in enumerate(retained):
        for label in retained_labels:
            values[bucket, label] = exact_matrix[bucket, label]
    return boundaries, values


def predicate_exact_values(times_by_label, queries):
    exact = []
    for left, right, label in queries:
        label_times = times_by_label[label]
        exact.append(int(np.searchsorted(label_times, right, side='right')
                         - np.searchsorted(label_times, left, side='left')))
    return exact


def evaluate_dataset(name, frame, label_map, count_rng, predicate_rng, allocations):
    print(f'[{name}] edges={len(frame):,}', flush=True)
    times = frame['time'].to_numpy(dtype=np.int64, copy=False)
    sorted_times = np.sort(times)
    t_min, t_max = int(sorted_times[0]), int(sorted_times[-1])

    count_queries = generate_count_queries(t_min, t_max, count_rng)
    count_exact = exact_counts(sorted_times, count_queries)
    count_rows = []
    for n_buckets, width, logical_bytes in allocations:
        boundaries = uniform_boundaries(t_min, t_max, n_buckets)
        values = bucket_counts_from_sorted_times(sorted_times, boundaries)
        estimates = [interpolated_count(boundaries, values, left, right)
                     for left, right in count_queries]
        error = mean_relative_error(estimates, count_exact)
        count_rows.append({
            'dataset': name,
            'query_type': 'COUNT',
            'buckets': n_buckets,
            'cms_width': width,
            'logical_bytes': logical_bytes,
            'mean_error': error,
            'r2_complete': 1,
        })

    count_only_buckets = 251
    count_only_width, count_only_bytes = tgs_allocation(
        count_only_buckets, complete_r2=False)
    count_only_boundaries = uniform_boundaries(t_min, t_max, count_only_buckets)
    count_only_values = bucket_counts_from_sorted_times(
        sorted_times, count_only_boundaries)
    count_only_estimates = [
        interpolated_count(count_only_boundaries, count_only_values, left, right)
        for left, right in count_queries
    ]
    count_only_row = {
        'dataset': name,
        'query_type': 'COUNT',
        'buckets': count_only_buckets,
        'cms_width': count_only_width,
        'logical_bytes': count_only_bytes,
        'mean_error': mean_relative_error(count_only_estimates, count_exact),
        'r2_complete': 0,
    }

    cam_boundaries = uniform_boundaries(t_min, t_max, 500)
    cam_values = bucket_counts_from_sorted_times(sorted_times, cam_boundaries)
    cam_estimates = [interpolated_count(cam_boundaries, cam_values, left, right)
                     for left, right in count_queries]
    cam_count_error = mean_relative_error(cam_estimates, count_exact)

    count_b52 = next(row['mean_error'] for row in count_rows if row['buckets'] == 52)
    expected_tgs, expected_cam = FORMAL_COUNT[name]
    if abs(count_b52 - expected_tgs) > 1.5e-4 or abs(cam_count_error - expected_cam) > 1.5e-4:
        print(
            f'  WARNING: COUNT endpoint differs from the reference: computed '
            f'B52={count_b52:.8f}, CAMRS={cam_count_error:.8f}; '
            f'formal={expected_tgs:.8f},{expected_cam:.8f}',
            flush=True)

    label_values = sorted(set(label_map.values()))
    if label_values != list(range(len(label_values))):
        raise RuntimeError(f'{name}: labels must be contiguous integers, got {label_values}')
    labels = edge_labels(frame, label_map)
    predicate_queries = generate_predicate_queries(
        t_min, t_max, label_values, predicate_rng)
    times_by_label = {label: np.sort(times[labels == label]) for label in label_values}
    predicate_exact = predicate_exact_values(times_by_label, predicate_queries)

    predicate_rows = []
    for n_buckets in PREDICATE_B_VALUES:
        width, logical_bytes = tgs_allocation(n_buckets, complete_r2=True)
        if logical_bytes > MEMORY_BYTES:
            continue
        boundaries, values = predicate_bucket_values(
            times, labels, t_min, t_max, n_buckets, width, label_values)
        estimates = [interpolated_count(boundaries, values[:, label], left, right)
                     for left, right, label in predicate_queries]
        error = mean_relative_error(estimates, predicate_exact)
        predicate_rows.append({
            'dataset': name,
            'query_type': 'PREDICATE_COUNT',
            'buckets': n_buckets,
            'cms_width': width,
            'logical_bytes': logical_bytes,
            'mean_error': error,
            'r2_complete': 1,
        })

    cam_pred_boundaries, cam_pred_values = predicate_bucket_values(
        times, labels, t_min, t_max, 500, 48, label_values)
    cam_pred_estimates = [
        interpolated_count(cam_pred_boundaries, cam_pred_values[:, label], left, right)
        for left, right, label in predicate_queries
    ]
    cam_pred_error = mean_relative_error(cam_pred_estimates, predicate_exact)
    pred_b52 = next(row['mean_error'] for row in predicate_rows if row['buckets'] == 52)
    expected_tgs_pred, expected_cam_pred = FORMAL_PREDICATE[name]
    if abs(pred_b52 - expected_tgs_pred) > 1e-5 or abs(cam_pred_error - expected_cam_pred) > 1e-5:
        print(
            f'  WARNING: predicate endpoint differs from the reference: computed '
            f'B52={pred_b52:.12f}, CAMRS={cam_pred_error:.12f}; '
            f'formal={expected_tgs_pred:.12f},{expected_cam_pred:.12f}',
            flush=True)

    for row in count_rows:
        row['camrs_mean_error'] = cam_count_error
        row['tgs_over_camrs'] = row['mean_error'] / max(cam_count_error, 1e-15)
    count_only_row['camrs_mean_error'] = cam_count_error
    count_only_row['tgs_over_camrs'] = (
        count_only_row['mean_error'] / max(cam_count_error, 1e-15))
    for row in predicate_rows:
        row['camrs_mean_error'] = cam_pred_error
        row['tgs_over_camrs'] = row['mean_error'] / max(cam_pred_error, 1e-15)

    best_count = min(count_rows, key=lambda row: row['mean_error'])
    best_predicate = min(predicate_rows, key=lambda row: row['mean_error'])
    max_resolution = next(row for row in count_rows if row['buckets'] == allocations[-1][0])
    print(
        f"  COUNT: CAMRS={cam_count_error:.6f}; B52={count_b52:.6f}; "
        f"full-R2 B{max_resolution['buckets']}={max_resolution['mean_error']:.6f}; "
        f"COUNT-only B{count_only_buckets}={count_only_row['mean_error']:.6f}; "
        f"oracle B={best_count['buckets']} err={best_count['mean_error']:.6f}",
        flush=True)
    print(
        f"  PRED:  CAMRS={cam_pred_error:.6f}; B52={pred_b52:.6f}; "
        f"best grid B={best_predicate['buckets']} err={best_predicate['mean_error']:.6f}",
        flush=True)
    return count_rows + [count_only_row] + predicate_rows


def main():
    allocations = feasible_count_allocations()
    if not allocations or allocations[0][0] != 52:
        raise RuntimeError('Expected feasible coupled allocation range to start at B=52')
    print(
        f'Feasible complete-R2 TGS allocations: B={allocations[0][0]}..{allocations[-1][0]} '
        f'({len(allocations)} integer settings); max-resolution w={allocations[-1][1]}',
        flush=True)

    selected = set(sys.argv[1:])
    selected_indices = [index for index, (name, _) in enumerate(DATASET_SEQUENCE)
                        if name in selected]
    last_selected_index = max(selected_indices) if selected_indices else None
    count_rng = random.Random(SEED)
    predicate_rng = random.Random(SEED)
    rows = []
    for index, (name, loader) in enumerate(DATASET_SEQUENCE):
        if last_selected_index is not None and index > last_selected_index:
            break
        if index == 2:


            bitcoin = ds.preprocess_temporal(ds.load_bitcoin_otc())
            generate_count_queries(
                int(bitcoin['time'].min()), int(bitcoin['time'].max()), count_rng)
            del bitcoin

            print('[protocol] advanced the COUNT random stream', flush=True)
        frame, label_map = load_dataset(name, loader)
        if selected and name not in selected:


            t_min = int(frame['time'].min())
            t_max = int(frame['time'].max())
            generate_count_queries(t_min, t_max, count_rng)
            generate_predicate_queries(
                t_min, t_max, sorted(set(label_map.values())), predicate_rng)
            del frame
            continue
        rows.extend(evaluate_dataset(
            name, frame, label_map, count_rng, predicate_rng, allocations))
        del frame

    output = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tgs_allocation_sweep.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fields = [
        'dataset', 'query_type', 'buckets', 'cms_width', 'logical_bytes',
        'mean_error', 'camrs_mean_error', 'tgs_over_camrs', 'r2_complete',
    ]
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
