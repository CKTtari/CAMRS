"""COUNT error on six paper datasets plus one RNG-compatibility dataset.
Methods: Uniform-40K, TGS-AQP, CMS-Only, CAMRS.
All under 1MB budget."""
import os, sys, csv, random, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS
from tgs_aqp import TGSAQP
from baselines import UniformSamplerBudget


WORKLOAD_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'results',
    'superuser_workload.csv'))


def generate_queries(t_min, t_max, n=600):
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


def load_sx_count_workload():
    queries = []
    with open(WORKLOAD_FILE, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['query_type'] == 'COUNT':
                queries.append((int(row['t_l']), int(row['t_r'])))
    if len(queries) != 600:
        raise RuntimeError(f'Unexpected workload size in {WORKLOAD_FILE}')
    return queries


def exact_count(df, t_l, t_r):
    return int(((df['time'] >= t_l) & (df['time'] <= t_r)).sum())


def rel_err(est, exact):
    if exact == 0:
        return None
    return abs(est - exact) / exact


def run_dataset(name, df, verify_formal_workload=False):
    print(f"  [COUNT] {name}")
    t_min, t_max = int(df['time'].min()), int(df['time'].max())
    generated_queries = generate_queries(t_min, t_max, n=600)
    if name == 'sx_superuser':
        queries = load_sx_count_workload()
        if verify_formal_workload and generated_queries != queries:
            raise RuntimeError('sx-superuser COUNT workload drifted from formal archive')
    else:
        queries = generated_queries
    exacts = [exact_count(df, tl, tr) for tl, tr in queries]

    results = []

    print(f"    Uniform-40K...")
    sk = UniformSamplerBudget(space_budget=960*1024)
    sk.build(df)
    errs = []
    for (tl, tr), e in zip(queries, exacts):
        if e == 0:
            continue
        err = rel_err(sk.query_count(tl, tr), e)
        if err is not None:
            errs.append(err)
    results.append({'method': 'Uniform-40K', 'cnt_mean': float(np.mean(errs)) if errs else 0})


    print(f"    TGS-AQP...")
    sk = TGSAQP(n_buckets=52, use_uniform=True)
    sk.build(df)
    errs = []
    for (tl, tr), e in zip(queries, exacts):
        if e == 0:
            continue
        err = rel_err(sk.query_count(tl, tr), e)
        if err is not None:
            errs.append(err)
    results.append({'method': 'TGS-AQP', 'cnt_mean': float(np.mean(errs)) if errs else 0})


    print(f"    CMS-Only...")
    sk = CAMRS(
        n_buckets=500, n_hll_segments=0, micro_hll_b=0,
        top_k_labels=0)
    sk.build(df)
    errs = []
    for (tl, tr), e in zip(queries, exacts):
        if e == 0:
            continue
        err = rel_err(sk.query_count(tl, tr), e)
        if err is not None:
            errs.append(err)
    results.append({'method': 'CMS-Only', 'cnt_mean': float(np.mean(errs)) if errs else 0})


    print(f"    CAMRS...")
    sk = CAMRS(n_buckets=500, n_hll_segments=16)
    sk.build(df)
    errs = []
    for (tl, tr), e in zip(queries, exacts):
        if e == 0:
            continue
        err = rel_err(sk.query_count(tl, tr), e)
        if err is not None:
            errs.append(err)
    results.append({'method': 'CAMRS', 'cnt_mean': float(np.mean(errs)) if errs else 0})

    for r in results:
        print(f"      {r['method']:20s}: {r['cnt_mean']:.4f}")

    return results


def main():
    random.seed(42)
    np.random.seed(42)
    datasets = [
        ('email_eu', ds.load_email_eu),
        ('college_msg', ds.load_college_msg),



        ('bitcoin_otc', ds.load_bitcoin_otc),
        ('wiki_talk', ds.load_wiki_talk),
        ('reddit_hyperlinks', ds.load_reddit_hyperlinks),
        ('sx_superuser', ds.load_sx_superuser),
        ('sx_stackoverflow_a2q', ds.load_sx_stackoverflow_a2q),
    ]
    selected = sys.argv[1:] if len(sys.argv) > 1 else []
    last_selected = max((i for i, (n, _) in enumerate(datasets) if n in selected),
                        default=None)
    all_rows = []
    for index, (name, loader) in enumerate(datasets):
        if last_selected is not None and index > last_selected:
            break
        df = ds.preprocess_temporal(loader())
        if selected and name not in selected:
            generate_queries(int(df['time'].min()), int(df['time'].max()), n=600)
            del df
            continue
        results = run_dataset(name, df, verify_formal_workload=not selected)
        for r in results:
            all_rows.append({'dataset': name, **r})
        del df

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'count.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'method', 'cnt_mean'])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
