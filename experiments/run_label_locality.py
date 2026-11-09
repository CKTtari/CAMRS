"""Measure label locality under the paper's dataset-specific label protocol.

Unlabeled graphs use degree terciles. Reddit uses the nine most frequent
source subreddits plus an ``other`` category, exactly matching the paper's
native-label definition (L=10).
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds


N_BUCKETS = 500
TOP_K = 5


def degree_tercile_labels(frame):
    nodes = sorted(set(frame['src'].values) | set(frame['dst'].values))
    src_counts = frame['src'].value_counts().to_dict()
    dst_counts = frame['dst'].value_counts().to_dict()
    degree = {node: src_counts.get(node, 0) + dst_counts.get(node, 0) for node in nodes}
    sorted_nodes = sorted(nodes, key=lambda node: degree[node], reverse=True)
    t1, t2 = len(nodes) // 3, 2 * len(nodes) // 3
    label_map = {}
    for node in sorted_nodes[:t1]:
        label_map[node] = 0
    for node in sorted_nodes[t1:t2]:
        label_map[node] = 1
    for node in sorted_nodes[t2:]:
        label_map[node] = 2
    return np.array([label_map[int(node)] for node in frame['src'].values], dtype=np.int8)


def analyze(dataset, protocol, times, labels):
    boundaries = np.linspace(int(times.min()), int(times.max()) + 1, N_BUCKETS + 1)
    bucket_ids = np.searchsorted(boundaries, times, side='right') - 1
    bucket_ids = np.clip(bucket_ids, 0, N_BUCKETS - 1)
    global_counts = pd.Series(labels).value_counts()
    global_top = set(global_counts.head(TOP_K).index)
    local_sets = []
    local_coverage = []
    global_coverage = []
    active_labels = []
    nonempty_buckets = 0
    bucket_label_counts = []
    for bucket in range(N_BUCKETS):
        bucket_labels = labels[bucket_ids == bucket]
        counts = pd.Series(bucket_labels).value_counts()
        local = set(counts.head(TOP_K).index)
        local_sets.append(local)
        active_labels.append(len(counts))
        bucket_label_counts.append(counts)
        denom = max(1, len(bucket_labels))
        if len(bucket_labels) > 0:
            nonempty_buckets += 1
            local_coverage.append(float(counts.head(TOP_K).sum()) / denom)
            global_coverage.append(float(counts[counts.index.isin(global_top)].sum()) / denom)

    adjacent_jaccard = []
    for left, right in zip(local_sets[:-1], local_sets[1:]):
        union = left | right
        adjacent_jaccard.append(len(left & right) / len(union) if union else 1.0)
    tracked_labels = list(global_counts.head(min(50, len(global_counts))).index)
    label_cvs = []
    for label in tracked_labels:
        counts = np.array([bucket.get(label, 0) for bucket in bucket_label_counts], dtype=float)
        if counts.mean() > 0:
            label_cvs.append(float(counts.std(ddof=1) / counts.mean()))
    return {
        'dataset': dataset,
        'protocol': protocol,
        'n_edges': len(labels),
        'n_buckets': N_BUCKETS,
        'top_k': TOP_K,
        'n_unique_labels': int(global_counts.size),
        'nonempty_bucket_fraction': nonempty_buckets / N_BUCKETS,
        'active_labels_mean': float(np.mean(active_labels)),
        'local_topk_coverage_mean': float(np.mean(local_coverage)),
        'global_topk_coverage_mean': float(np.mean(global_coverage)),
        'local_minus_global_coverage': float(np.mean(local_coverage) - np.mean(global_coverage)),
        'adjacent_topk_jaccard_median': float(np.median(adjacent_jaccard)),
        'label_temporal_cv_median': float(np.median(label_cvs)),
        'label_temporal_cv_p90': float(np.percentile(label_cvs, 90)),
    }


def main():
    rows = []
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', None),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = set(sys.argv[1:])
    if selected:
        datasets = [(name, loader) for name, loader in datasets if name in selected]
    for name, loader in datasets:
        if name == 'reddit_hyperlinks':
            raw, label_map = ds.load_reddit_hyperlinks_with_labels(n_labels=10)
            frame = ds.preprocess_temporal(raw)
            labels = np.array(
                [label_map[int(node)] for node in frame['src'].to_numpy()],
                dtype=np.int8,
            )
            protocol = 'native_top9_plus_other'
        else:
            frame = ds.preprocess_temporal(loader())
            labels = degree_tercile_labels(frame)
            protocol = 'degree_terciles'
        rows.append(analyze(
            name,
            protocol,
            frame['time'].to_numpy(),
            labels,
        ))
        del frame, labels

    output = os.path.join(os.path.dirname(__file__), '..', 'results', 'label_locality.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
