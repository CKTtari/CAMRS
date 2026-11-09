"""Router validation experiment: per-query error CDF for Always-Sketch vs Adaptive Router.

Collects per-query data:
  - exact count, sketch estimate, predicted error bound, actual sketch error
  - router decision and mixed error at epsilon = 0.10 and 0.50

Output: results/router_validation.csv
"""
import os, sys, csv, random, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds
from camrs import CAMRS


def generate_queries(t_min, t_max, n=600):
    span = max(1, t_max - t_min)
    queries = []
    for _ in range(n):
        frac = random.uniform(0.01, 0.50)
        win = int(frac * span)
        if win < 1:
            win = 1
        tl = random.randint(t_min, max(t_min, t_max - win))
        queries.append((tl, tl + win, frac))
    return queries


def run_dataset(name, df):
    print(f"  [ROUTER] {name}")
    t_min, t_max = int(df['time'].min()), int(df['time'].max())
    cam = CAMRS(n_buckets=500, n_hll_segments=16)
    cam.build(df)

    queries = generate_queries(t_min, t_max, n=600)
    exact = []
    for tl, tr, _ in queries:
        m = (df['time'] >= tl) & (df['time'] <= tr)
        exact.append(int(m.sum()))

    epsilons = [0.10, 0.50]
    rows = []
    for i, (tl, tr, frac) in enumerate(queries):
        est = cam.query_count(tl, tr)
        bound = cam.predicted_error_bound(tl, tr)
        actual = exact[i]
        rel_err = abs(est - actual) / max(1, actual)


        router_errs = {}
        for eps in epsilons:
            if bound <= eps:
                router_errs[f"router_err_{eps:.2f}"] = rel_err
            else:
                router_errs[f"router_err_{eps:.2f}"] = 0.0

        rows.append({
            'dataset': name,
            'query_id': i,
            'window_frac': frac,
            'predicted_bound': bound,
            'exact_count': actual,
            'sketch_estimate': est,
            'sketch_error': rel_err,
            **router_errs,
        })
    return rows


def compute_stats(errors):
    arr = np.array(errors)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(np.max(arr)),
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

    all_rows = []
    summary = []
    for index, (name, loader) in enumerate(datasets):
        if last_selected is not None and index > last_selected:
            break
        df = ds.preprocess_temporal(loader())
        if selected and name not in selected:
            generate_queries(int(df['time'].min()), int(df['time'].max()), n=600)
            del df
            continue
        rows = run_dataset(name, df)
        all_rows.extend(rows)


        sketch_errs = [r['sketch_error'] for r in rows]
        r010_errs = [r['router_err_0.10'] for r in rows]
        r050_errs = [r['router_err_0.50'] for r in rows]
        sketch_stats = compute_stats(sketch_errs)
        r010_stats = compute_stats(r010_errs)
        r050_stats = compute_stats(r050_errs)

        n_sketch = sum(1 for r in rows if r['predicted_bound'] <= 0.10)
        n_sketch_050 = sum(1 for r in rows if r['predicted_bound'] <= 0.50)

        print(f"    {name}: sketch_mean={sketch_stats['mean']:.4f}, "
              f"router010_mean={r010_stats['mean']:.4f}, router050_mean={r050_stats['mean']:.4f}, "
              f"sketch_usage@010={n_sketch/len(rows):.2%}, sketch_usage@050={n_sketch_050/len(rows):.2%}")

        summary.append({
            'dataset': name,
            'sketch_mean': sketch_stats['mean'],
            'sketch_p95': sketch_stats['p95'],
            'sketch_p99': sketch_stats['p99'],
            'sketch_max': sketch_stats['max'],
            'router010_mean': r010_stats['mean'],
            'router010_p95': r010_stats['p95'],
            'router010_p99': r010_stats['p99'],
            'router010_max': r010_stats['max'],
            'router050_mean': r050_stats['mean'],
            'router050_p95': r050_stats['p95'],
            'router050_p99': r050_stats['p99'],
            'router050_max': r050_stats['max'],
            'usage_010': n_sketch / len(rows),
            'usage_050': n_sketch_050 / len(rows),
        })
        del df


    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'router_validation.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"Wrote per-query detail: {out}")


    sum_out = os.path.join(os.path.dirname(__file__), '..', 'results', 'router_summary.csv')
    with open(sum_out, 'w', newline='') as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    print(f"Wrote summary: {sum_out}")


if __name__ == '__main__':
    main()
