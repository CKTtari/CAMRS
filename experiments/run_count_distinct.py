"""COUNT_DISTINCT on temporal datasets, single-layer K_HLL=16 and 52.
f_c >= 5% only, B=500, micro_hll_b=6."""
import os, sys, csv, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS
from baselines import StratifiedHLLOnly


def generate_queries(t_min, t_max, n=200):
    span = max(1, t_max - t_min)
    fractions = [0.05, 0.10, 0.20, 0.50]
    queries = []
    for frac in fractions:
        for _ in range(n // len(fractions)):
            win = max(1, int(frac * span))
            tl = random.randint(int(t_min), max(int(t_min), int(t_max) - win))
            queries.append((tl, tl + win))
    return queries


def exact_cd(df, t_l, t_r):
    mask = (df['time'] >= t_l) & (df['time'] <= t_r)
    src = df['src'].values[mask]
    dst = df['dst'].values[mask]
    pairs = set(zip(src, dst))
    return len(pairs)


def rel_err(est, exact):
    if exact == 0 or est is None:
        return None
    return abs(est - exact) / exact


def run_dataset(name, loader, k):
    print(f"  [{name}] K_HLL={k}")
    df = ds.preprocess_temporal(loader())
    t_min, t_max = int(df['time'].min()), int(df['time'].max())
    queries = generate_queries(t_min, t_max, n=200)
    exacts = [exact_cd(df, tl, tr) for tl, tr in queries]




    sk = StratifiedHLLOnly(n_hll_segments=k, hll_b=12)
    sk.build(df)
    allocation = CAMRS(n_buckets=500, n_hll_segments=k, micro_hll_b=6)
    errs = []
    for (tl, tr), e in zip(queries, exacts):
        if e == 0:
            continue
        err = rel_err(sk.query_count_distinct(tl, tr), e)
        if err is not None:
            errs.append(err)
    mean_err = float(np.mean(errs)) if errs else 0
    print(f"    CD mean error={mean_err:.4f}, w={allocation.cms_width}, mem={allocation.memory_usage()}")
    return {
        'dataset': name,
        'K_HLL': k,
        'cd_mean_err': mean_err,
        'cms_width': allocation.cms_width,
        'mem_bytes': allocation.memory_usage(),
    }


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
    rows = []
    for index, (name, loader) in enumerate(datasets):
        if last_selected is not None and index > last_selected:
            break
        if selected and name not in selected:
            df = ds.preprocess_temporal(loader())
            t_min, t_max = int(df['time'].min()), int(df['time'].max())
            generate_queries(t_min, t_max, n=200)
            generate_queries(t_min, t_max, n=200)
            del df
            continue
        for k in [16, 52]:
            rows.append(run_dataset(name, loader, k))
        del loader

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'count_distinct.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'K_HLL', 'cd_mean_err', 'cms_width', 'mem_bytes'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
