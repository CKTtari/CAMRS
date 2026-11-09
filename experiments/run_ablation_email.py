"""Experiment for the RQ1 ablation on Email-Eu-Core.
Variants: TGS-B52, TGS-B300, MR-B52, MR-B500(CAMRS).
The COUNT and predicate query streams match their formal main experiments."""
import argparse
import os, sys, csv, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS
from tgs_aqp import TGSAQP


def generate_queries(t_min, t_max, rng, n=600):
    span = max(1, t_max - t_min)
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    queries = []
    per_fraction, remainder = divmod(n, len(fractions))
    for index, frac in enumerate(fractions):
        for _ in range(per_fraction + (index < remainder)):
            win = int(frac * span)
            if win < 1:
                win = 1
            tl = rng.randint(t_min, max(t_min, t_max - win))
            queries.append((tl, tl + win))
    return queries


def generate_pred_queries(df, label_map, rng, n=200):
    t_min, t_max = df['time'].min(), df['time'].max()
    span = max(1, t_max - t_min)
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    queries = []
    all_labels = sorted(set(label_map.values()))
    per_fraction, remainder = divmod(n, len(fractions))
    for index, frac in enumerate(fractions):
        for _ in range(per_fraction + (index < remainder)):
            win = max(1, int(frac * span))
            tl = rng.randint(int(t_min), max(int(t_min), int(t_max) - win))
            lbl = rng.choice(all_labels)
            queries.append((tl, tl + win, lbl))
    return queries


def exact_count(df, t_l, t_r):
    return int(((df['time'] >= t_l) & (df['time'] <= t_r)).sum())


def exact_count_pred(df, label_map, t_l, t_r, label_val):
    mask = (df['time'] >= t_l) & (df['time'] <= t_r)
    src = df['src'].values
    mask = mask & np.array([label_map.get(s, -1) == label_val for s in src])
    return int(mask.sum())


def rel_err(est, exact):
    if exact == 0:
        return None
    return abs(est - exact) / exact


def run_variant(name, df, label_map, cnt_queries, pred_queries, B, use_camrs):
    if use_camrs:
        sk = CAMRS(n_buckets=B, n_hll_segments=16, micro_hll_b=6)
    else:
        if B > 229:


            sk = TGSAQP(
                n_buckets=B, use_uniform=True, micro_hll_b=0,
                enforce_budget=False,
            )
        else:
            sk = TGSAQP(n_buckets=B, use_uniform=True)
    sk.build(df, label_map=label_map)

    cnt_errs = []
    for tl, tr in cnt_queries:
        e = exact_count(df, tl, tr)
        if e == 0:
            continue
        err = rel_err(sk.query_count(tl, tr), e)
        if err is not None:
            cnt_errs.append(err)

    pred_errs = []
    for tl, tr, lbl in pred_queries:
        e = exact_count_pred(df, label_map, tl, tr, lbl)
        if e == 0:
            continue
        err = rel_err(sk.query_count_pred(tl, tr, lbl), e)
        if err is not None:
            pred_errs.append(err)

    fits = (sk.memory_usage() <= 1024 * 1024)
    return {
        'variant': name,
        'B': B,
        'cnt_mean': float(np.mean(cnt_errs)) if cnt_errs else 0,
        'pred_mean': float(np.mean(pred_errs)) if pred_errs else 0,
        'cms_width': sk.cms_width,
        'fits_1mb': fits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--intermediate-only', action='store_true',
        help='Run the TGS-B300 and MR-B52 configurations.',
    )
    args = parser.parse_args()
    np.random.seed(42)
    df = ds.preprocess_temporal(ds.load_email_eu())

    src_counts = df['src'].value_counts().to_dict()
    dst_counts = df['dst'].value_counts().to_dict()
    nodes = sorted(set(src_counts.keys()) | set(dst_counts.keys()))
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

    t_min, t_max = int(df['time'].min()), int(df['time'].max())
    cnt_queries = generate_queries(t_min, t_max, random.Random(42), n=600)
    pred_queries = generate_pred_queries(df, label_map, random.Random(42), n=200)

    rows = []
    variants = [
        ('TGS-B52', 52, False),
        ('TGS-B300', 300, False),
        ('MR-B52', 52, True),
        ('MR-B500(CAMRS)', 500, True),
    ]
    if args.intermediate_only:
        variants = [variant for variant in variants if variant[0] in {'TGS-B300', 'MR-B52'}]
    for name, B, use_camrs in variants:
        print(f"  [ABLATION] {name} B={B}")
        r = run_variant(name, df, label_map, cnt_queries, pred_queries, B, use_camrs)
        rows.append(r)
        print(f"    CNT={r['cnt_mean']:.4f} PRED={r['pred_mean']:.4f} w={r['cms_width']} fits={r['fits_1mb']}")

    filename = 'ablation_intermediate_email_eu.csv' if args.intermediate_only else 'ablation_email_eu.csv'
    out = os.path.join(os.path.dirname(__file__), '..', 'results', filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant', 'B', 'cnt_mean', 'pred_mean', 'cms_width', 'fits_1mb'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
