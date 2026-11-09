"""Prepare window endpoints and router decisions for the C++ benchmark."""
import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds


SEED = 42
N_QUERIES = 600
EPSILON = 0.10


def generate_queries(t_min, t_max, fractions):
    span = max(1, t_max - t_min)
    queries = []
    for fraction in fractions:
        width = max(1, int(fraction * span))
        left = random.randint(t_min, max(t_min, t_max - width))
        queries.append((left, left + width, fraction))
    return queries


def main():
    random.seed(SEED)
    golden_path = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        'router_validation.csv',
    )
    golden_rows = {}
    with open(golden_path, encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            golden_rows.setdefault(row['dataset'], []).append(row)
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = set(sys.argv[1:])
    if selected:
        datasets = [(name, loader) for name, loader in datasets if name in selected]
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'router-cpp-queries')
    os.makedirs(output_dir, exist_ok=True)
    for name, loader in datasets:
        times_path = os.path.join(output_dir, f'{name}.times.i64')
        if name == 'sx_stackoverflow_a2q' and os.path.exists(times_path):
            times = np.memmap(times_path, dtype=np.int64, mode='r')
        else:
            frame = ds.preprocess_temporal(loader())
            times = frame['time'].to_numpy(dtype=np.int64, copy=True)
            times.tofile(times_path)
            del frame
        t_min, t_max = int(times.min()), int(times.max())
        accepted_rows = golden_rows[name]
        fractions = [float(row['window_frac']) for row in accepted_rows]
        queries = generate_queries(t_min, t_max, fractions)
        boundaries = np.linspace(t_min, t_max + 1, 501).astype(np.int64)
        bucket_ids = np.searchsorted(boundaries, times, side='right') - 1
        bucket_ids = np.clip(bucket_ids, 0, 499)
        bucket_counts = np.bincount(bucket_ids, minlength=500).astype(np.int64)
        np.concatenate([boundaries, bucket_counts]).tofile(
            os.path.join(output_dir, f'{name}.state.i64')
        )
        output = os.path.join(output_dir, f'{name}.csv')



        accepted_bounds = [float(row['predicted_bound']) for row in accepted_rows]
        if len(accepted_bounds) != len(queries):
            raise RuntimeError(f'{name}: accepted/query length mismatch')
        span = max(1, t_max - t_min)
        sorted_times = np.sort(np.asarray(times))
        with open(output, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'left', 'right', 'window_fraction', 'predicted_bound', 'fallback',
                'expected_sketch', 'expected_exact', 'expected_router',
            ])
            fallback_count = 0
            for (left, right, fraction), bound in zip(queries, accepted_bounds):
                fallback = bound > EPSILON
                fallback_count += fallback
                total = 0.0
                for index, count in enumerate(bucket_counts):
                    bucket_left, bucket_right = boundaries[index], boundaries[index + 1]
                    if bucket_right <= left or bucket_left >= right:
                        continue
                    overlap = min(bucket_right, right) - max(bucket_left, left)
                    total += int(count) * overlap / max(1, bucket_right - bucket_left)
                sketch = int(total)
                exact = int(np.searchsorted(sorted_times, right, side='right') - np.searchsorted(sorted_times, left, side='left'))
                writer.writerow([left, right, fraction, bound, int(fallback), sketch, exact, exact if fallback else sketch])
        print(f'{name}: n={len(times)}, accepted_fallback={fallback_count / len(queries):.4f}')


if __name__ == '__main__':
    main()
