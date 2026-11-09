"""Generate the shared sx-superuser COUNT and predicate workload."""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import dataset as ds


SEED = 42
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)


def count_queries(t_min, t_max, rng, n=600):
    span = max(1, t_max - t_min)
    queries = []
    for fraction in FRACTIONS:
        width = max(1, int(fraction * span))
        for _ in range(n // len(FRACTIONS)):
            left = rng.randint(t_min, max(t_min, t_max - width))
            queries.append((left, left + width))
    return queries


def predicate_queries(t_min, t_max, label_values, rng, n=200):
    span = max(1, t_max - t_min)
    queries = []
    per_fraction, remainder = divmod(n, len(FRACTIONS))
    for index, fraction in enumerate(FRACTIONS):
        width = max(1, int(fraction * span))
        for _ in range(per_fraction + (index < remainder)):
            left = rng.randint(t_min, max(t_min, t_max - width))
            queries.append((left, left + width, rng.choice(label_values)))
    return queries


def time_range(loader):
    frame = ds.preprocess_temporal(loader())
    result = int(frame['time'].min()), int(frame['time'].max())
    del frame
    return result


def main():
    count_rng = random.Random(SEED)
    for loader in (
        ds.load_email_eu,
        ds.load_college_msg,
        ds.load_bitcoin_otc,
        ds.load_wiki_talk,
        ds.load_reddit_hyperlinks,
    ):
        count_queries(*time_range(loader), count_rng)
    sx_range = time_range(ds.load_sx_superuser)
    formal_count = count_queries(*sx_range, count_rng)

    predicate_rng = random.Random(SEED)
    predicate_prefix = (
        (ds.load_email_eu, range(3)),
        (ds.load_college_msg, range(3)),
        (ds.load_wiki_talk, range(3)),
        (ds.load_reddit_hyperlinks, range(10)),
    )
    for loader, labels in predicate_prefix:
        predicate_queries(*time_range(loader), list(labels), predicate_rng)
    formal_predicate = predicate_queries(*sx_range, list(range(3)), predicate_rng)

    rows = []
    for query_id, (left, right) in enumerate(formal_count):
        rows.append({
            'query_type': 'COUNT', 'query_id': query_id,
            't_l': left, 't_r': right, 'label': '',
        })
    for query_id, (left, right, label) in enumerate(formal_predicate):
        rows.append({
            'query_type': 'PREDICATE_COUNT', 'query_id': query_id,
            't_l': left, 't_r': right, 'label': label,
        })

    output = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        'superuser_workload.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['query_type', 'query_id', 't_l', 't_r', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output} ({len(formal_count)} COUNT, '
          f'{len(formal_predicate)} predicate queries)')


if __name__ == '__main__':
    main()
