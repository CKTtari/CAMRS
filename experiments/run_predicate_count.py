"""Predicate COUNT error on the six paper datasets.
Methods: Uniform-40K (with label filtering), TGS-AQP (B=52), CAMRS (B=500)."""
import os, sys, csv, random
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS
from tgs_aqp import TGSAQP
from baselines import UniformSamplerBudget

random.seed(42)
np.random.seed(42)
WORKLOAD_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'results',
    'superuser_workload.csv'))


def generate_pred_queries(df, label_map, n=200):
    t_min, t_max = df['time'].min(), df['time'].max()
    span = max(1, t_max - t_min)
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    queries = []
    all_labels = sorted(set(label_map.values()))
    per_fraction, remainder = divmod(n, len(fractions))
    for index, frac in enumerate(fractions):
        for _ in range(per_fraction + (index < remainder)):
            win = max(1, int(frac * span))
            tl = random.randint(int(t_min), max(int(t_min), int(t_max) - win))
            lbl = random.choice(all_labels)
            queries.append((tl, tl + win, lbl))
    return queries


def load_sx_predicate_workload():
    queries = []
    with open(WORKLOAD_FILE, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['query_type'] == 'PREDICATE_COUNT':
                queries.append((
                    int(row['t_l']), int(row['t_r']), int(row['label'])))
    if len(queries) != 200:
        raise RuntimeError(f'Unexpected workload size in {WORKLOAD_FILE}')
    return queries


def build_label_lookup(label_map, max_node):
    lookup = np.full(max_node + 1, -1, dtype=np.int8)
    for n, lbl in label_map.items():
        lookup[n] = lbl
    return lookup


def degree_tercile_labels(df):
    nodes = sorted(set(df['src'].values) | set(df['dst'].values))
    src_counts = df['src'].value_counts().to_dict()
    dst_counts = df['dst'].value_counts().to_dict()
    degree = {n: src_counts.get(n, 0) + dst_counts.get(n, 0) for n in nodes}
    sorted_nodes = sorted(nodes, key=lambda x: degree[x], reverse=True)
    t1, t2 = len(nodes) // 3, 2 * len(nodes) // 3
    label_map = {}
    for n in sorted_nodes[:t1]:
        label_map[n] = 0
    for n in sorted_nodes[t1:t2]:
        label_map[n] = 1
    for n in sorted_nodes[t2:]:
        label_map[n] = 2
    return label_map


def exact_count_pred(df, label_lookup, t_l, t_r, label_val):
    mask = (df['time'].values >= t_l) & (df['time'].values <= t_r)
    src = df['src'].values
    labels = label_lookup[src]
    mask = mask & (labels == label_val)
    return int(mask.sum())


def rel_err(est, exact):
    if exact == 0:
        return None
    return abs(est - exact) / exact


def run_dataset(name, df, label_map, label_lookup, verify_formal_workload=False):
    print(f"  [PREDICATE COUNT] {name}")
    generated_queries = generate_pred_queries(df, label_map, n=200)
    if name == 'sx_superuser':
        queries = load_sx_predicate_workload()
        if verify_formal_workload and generated_queries != queries:
            raise RuntimeError('sx-superuser predicate workload drifted from formal archive')
    else:
        queries = generated_queries


    print(f"    Uniform-40K...")
    unif = UniformSamplerBudget(space_budget=960*1024)
    unif.build(df, label_map=label_map)
    unif_errs = []
    for tl, tr, lbl in queries:
        exact = exact_count_pred(df, label_lookup, tl, tr, lbl)
        if exact == 0:
            continue
        e = rel_err(unif.query_count_pred(tl, tr, lbl), exact)
        if e is not None:
            unif_errs.append(e)


    print(f"    TGS-AQP...")
    tgs = TGSAQP(n_buckets=52, use_uniform=True)
    tgs.build(df, label_map=label_map)
    tgs_errs = []
    for tl, tr, lbl in queries:
        exact = exact_count_pred(df, label_lookup, tl, tr, lbl)
        if exact == 0:
            continue
        e = rel_err(tgs.query_count_pred(tl, tr, lbl), exact)
        if e is not None:
            tgs_errs.append(e)


    print(f"    CAMRS...")
    cam = CAMRS(n_buckets=500, n_hll_segments=16)
    cam.build(df, label_map=label_map)
    cam_errs = []
    for tl, tr, lbl in queries:
        exact = exact_count_pred(df, label_lookup, tl, tr, lbl)
        if exact == 0:
            continue
        e = rel_err(cam.query_count_pred(tl, tr, lbl), exact)
        if e is not None:
            cam_errs.append(e)

    unif_mean = float(np.mean(unif_errs)) if unif_errs else 0
    tgs_mean = float(np.mean(tgs_errs)) if tgs_errs else 0
    cam_mean = float(np.mean(cam_errs)) if cam_errs else 0
    print(f"    Uniform={unif_mean:.4f}  TGS={tgs_mean:.4f}  CAMRS={cam_mean:.4f}")
    return {
        'dataset': name,
        'Uniform-40K': unif_mean,
        'TGS-AQP': tgs_mean,
        'CAMRS': cam_mean,
    }


def main():
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = sys.argv[1:] if len(sys.argv) > 1 else []
    last_selected = max((i for i, (n, _) in enumerate(datasets) if n in selected),
                        default=None)
    rows = []
    for index, (name, loader) in enumerate(datasets):
        if last_selected is not None and index > last_selected:
            break
        if name == 'reddit_hyperlinks':
            raw_df, label_map = ds.load_reddit_hyperlinks_with_labels(n_labels=10)
            df = ds.preprocess_temporal(raw_df)
        else:
            df = ds.preprocess_temporal(loader())
            label_map = degree_tercile_labels(df)
        if selected and name not in selected:
            generate_pred_queries(df, label_map, n=200)
            del df
            continue
        nodes = sorted(set(df['src'].values) | set(df['dst'].values))
        max_node = max(nodes)
        label_lookup = build_label_lookup(label_map, max_node)
        r = run_dataset(
            name, df, label_map, label_lookup,
            verify_formal_workload=not selected)
        rows.append(r)
        del df

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'predicate_count.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'Uniform-40K', 'TGS-AQP', 'CAMRS'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
