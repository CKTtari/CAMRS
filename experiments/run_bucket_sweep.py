"""Bucket-count sensitivity across datasets.
B=300,400,500,600,750 with K_HLL=16, b=12, 1MB budget."""
import os, sys, csv, random, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS


def generate_queries(t_min, t_max, n=300):
    span = max(1, t_max - t_min)
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    queries = []
    for frac in fractions:
        for _ in range(n // len(fractions)):
            win = int(frac * span)
            if win < 1:
                win = 1
            tl = random.randint(t_min, max(t_min, t_max - win))
            queries.append((tl, tl + win))
    return queries


def exact_count(df, t_l, t_r):
    return int(((df['time'] >= t_l) & (df['time'] <= t_r)).sum())


def rel_err(est, exact):
    if exact == 0:
        return None
    return abs(est - exact) / exact


def main():
    random.seed(42)
    np.random.seed(42)
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),
        ('dblp', ds.load_dblp),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = sys.argv[1:] if len(sys.argv) > 1 else []
    last_selected = max((i for i, (n, _) in enumerate(datasets) if n in selected),
                        default=None)
    B_values = [300, 400, 500, 600, 750]
    rows = []

    for index, (name, loader) in enumerate(datasets):
        if last_selected is not None and index > last_selected:
            break
        df = ds.preprocess_temporal(loader())
        t_min, t_max = int(df['time'].min()), int(df['time'].max())
        queries = generate_queries(t_min, t_max, n=300)
        if selected and name not in selected:
            del df
            continue
        exacts = [exact_count(df, tl, tr) for tl, tr in queries]

        for B in B_values:
            print(f"  [BUCKET SWEEP] {name} B={B}")
            sk = CAMRS(n_buckets=B, n_hll_segments=16)
            sk.build(df)
            errs = []
            for (tl, tr), e in zip(queries, exacts):
                if e == 0:
                    continue
                err = rel_err(sk.query_count(tl, tr), e)
                if err is not None:
                    errs.append(err)
            mean_err = float(np.mean(errs)) if errs else 0
            print(f"    CNT mean error={mean_err:.4f}, w={sk.cms_width}")
            rows.append({
                'dataset': name,
                'B': B,
                'cnt_mean': mean_err,
                'cms_width': sk.cms_width,
            })
        del df

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'bucket_sweep.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'B', 'cnt_mean', 'cms_width'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
