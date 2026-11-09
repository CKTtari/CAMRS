"""Budget sensitivity sweep on Email-Eu-Core.
Sweeps feasible CAMRS budgets from 512KB to 4MB at fixed B=500, K_HLL=16.
Output: results/budget.csv
"""
import os, sys, csv, random, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS


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


def exact_count(df, t_l, t_r):
    return int(((df['time'] >= t_l) & (df['time'] <= t_r)).sum())


def rel_err(est, exact):
    if exact == 0:
        return None
    return abs(est - exact) / exact


def main():
    random.seed(42)
    np.random.seed(42)

    df = ds.preprocess_temporal(ds.load_email_eu())
    t_min, t_max = int(df['time'].min()), int(df['time'].max())
    queries = generate_queries(t_min, t_max, n=600)
    exacts = [exact_count(df, tl, tr) for tl, tr in queries]

    rows = []
    for budget_kb in [512, 1024, 2048, 4096]:
        budget_bytes = budget_kb * 1024
        print(f"  [BUDGET] {budget_kb}KB")
        sk = CAMRS(space_budget=budget_bytes, n_buckets=500, n_hll_segments=16)
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
            'budget_kb': budget_kb,
            'cnt_mean': mean_err,
            'cms_width': sk.cms_width,
        })

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'budget.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['budget_kb', 'cnt_mean', 'cms_width'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
