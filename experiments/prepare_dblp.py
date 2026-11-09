"""Process DBLP coauthorship data into a labeled temporal edge stream.

Reads: data/raw/dblp.json.gz
Writes: data/processed/dblp_coauthorship.csv

Steps:
1. Parse paper entries, extract venue category from key.
2. Generate coauthorship edges with category labels.
3. For each (author1, author2, year), keep the most common category.
4. Filter years to 2005--2024.
5. Assign deterministic month via string hash.
6. Map author names to integer IDs.
7. Output: src,dst,time,label  where time=(year-2005)*12+month, label=category int.
"""
import os, sys, json, gzip, csv, hashlib
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


VENUE_MAP = {

    'sigmod': 'DB', 'vldb': 'DB', 'icde': 'DB', 'pods': 'DB',
    'edbt': 'DB', 'cidr': 'DB', 'tkde': 'DB', 'vldbj': 'DB',
    'tods': 'DB', 'cikm': 'DB', 'icdt': 'DB', 'ssdbm': 'DB',
    'wise': 'DB', 'dasfaa': 'DB', 'pvldb': 'DB',

    'icml': 'AI', 'nips': 'AI', 'neurips': 'AI', 'aaai': 'AI',
    'ijcai': 'AI', 'cvpr': 'AI', 'iccv': 'AI', 'eccv': 'AI',
    'acl': 'AI', 'emnlp': 'AI', 'naacl': 'AI', 'coling': 'AI',
    'kdd': 'AI', 'www': 'AI', 'icdm': 'AI', 'sdm': 'AI',
    'ecml': 'AI', 'pkdd': 'AI', 'recsys': 'AI', 'wsdm': 'AI',
    'uai': 'AI', 'aistats': 'AI', 'ml': 'AI',

    'sosp': 'SYS', 'osdi': 'SYS', 'atc': 'SYS', 'eurosys': 'SYS',
    'fast': 'SYS', 'nsdi': 'SYS', 'sigcomm': 'SYS', 'mobisys': 'SYS',
    'mobicom': 'SYS', 'conext': 'SYS', 'hotos': 'SYS', 'hotnets': 'SYS',
    'europar': 'SYS', 'ics': 'SYS',

    'stoc': 'TH', 'focs': 'TH', 'soda': 'TH', 'icalp': 'TH',
    'approx': 'TH', 'random': 'TH', 'jacm': 'TH', 'sicomp': 'TH',
    'tcs': 'TH', 'algorithmica': 'TH', 'ipl': 'TH', 'iandc': 'TH',

    'popl': 'PL', 'pldi': 'PL', 'oopsla': 'PL', 'icfp': 'PL',
    'ecoop': 'PL', 'cc': 'PL', 'cav': 'PL', 'tacas': 'PL', 'lics': 'PL',

    'icse': 'SE', 'fse': 'SE', 'ase': 'SE', 'issta': 'SE',
    'tse': 'SE', 'tosem': 'SE', 'sfm': 'SE', 'esec': 'SE',

    'sp': 'SEC', 'oakland': 'SEC', 'ccs': 'SEC', 'usenixsecurity': 'SEC',
    'ndss': 'SEC', 'crypto': 'SEC', 'eurocrypt': 'SEC', 'asiacrypt': 'SEC',
    'tissec': 'SEC', 'tdsc': 'SEC', 'pets': 'SEC',

    'isca': 'ARCH', 'micro': 'ARCH', 'hpca': 'ARCH', 'asplos': 'ARCH',
    'dac': 'ARCH', 'iccad': 'ARCH', 'tcad': 'ARCH',

    'cgoo': 'PL', 'lctes': 'PL', 'codes': 'PL',
}

CATEGORY_ORDER = ['DB', 'AI', 'SYS', 'TH', 'PL', 'SE', 'SEC', 'ARCH']
CATEGORY_TO_INT = {c: i for i, c in enumerate(CATEGORY_ORDER)}


COARSE_MAP = {
    'DB':   'DataAI',
    'AI':   'DataAI',
    'SYS':  'SysSec',
    'SEC':  'SysSec',
    'ARCH': 'SysSec',
    'TH':   'TheorySE',
    'PL':   'TheorySE',
    'SE':   'TheorySE',
}

COARSE_ORDER = ['DataAI', 'SysSec', 'TheorySE']
COARSE_TO_INT = {c: i for i, c in enumerate(COARSE_ORDER)}


def extract_venue(key):
    """Extract venue from DBLP key: conf/SIGMOD/... -> 'sigmod' (lower-case)."""
    parts = key.split('/')
    if len(parts) >= 2 and parts[0] in ('conf', 'journals', 'series'):
        return parts[1].lower()
    return None


def deterministic_month(a1, a2, year):
    """Deterministic month 1..12 from author pair and year."""
    s = f"{a1}|{a2}|{year}"
    h = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
    return (h % 12) + 1


def main():
    in_path = os.path.join(DATA_DIR, 'dblp.json.gz')
    out_dir = PROC_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dblp_coauthorship.csv')


    edge_cats = {}
    total_papers = 0
    kept_papers = 0
    skipped_no_venue = 0
    skipped_year = 0
    skipped_authors = 0

    print("Pass 1: parsing dblp.json.gz...")
    with gzip.open(in_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line in ('[', ']'):
                continue
            if line.endswith(','):
                line = line[:-1]
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (len(obj) == 3):
                continue
            key, authors, year = obj
            if not (isinstance(authors, list) and len(authors) >= 2
                    and isinstance(year, int) and 2005 <= year <= 2024):
                if isinstance(year, int) and not (2005 <= year <= 2024):
                    skipped_year += 1
                elif not (isinstance(authors, list) and len(authors) >= 2):
                    skipped_authors += 1
                continue

            total_papers += 1
            venue = extract_venue(key)
            cat = VENUE_MAP.get(venue)
            if cat is None:
                skipped_no_venue += 1
                continue

            kept_papers += 1

            n = len(authors)
            for i in range(n):
                for j in range(i + 1, n):
                    a1, a2 = authors[i], authors[j]
                    if a1 == a2:
                        continue
                    if a1 > a2:
                        a1, a2 = a2, a1
                    k = (a1, a2, year)
                    if k not in edge_cats:
                        edge_cats[k] = Counter()
                    edge_cats[k][cat] += 1

    print(f"  Total papers checked: {total_papers}")
    print(f"  Kept papers (known venue): {kept_papers}")
    print(f"  Skipped (year out of range): {skipped_year}")
    print(f"  Skipped (<2 authors): {skipped_authors}")
    print(f"  Skipped (unknown venue): {skipped_no_venue}")
    print(f"  Unique (a1,a2,year) pairs: {len(edge_cats)}")


    print("Pass 2: resolving categories and writing edges...")
    author2id = {}
    next_id = 0

    rows = []
    for (a1, a2, year), cat_counter in edge_cats.items():

        cat = cat_counter.most_common(1)[0][0]
        month = deterministic_month(a1, a2, year)
        time_stamp = (year - 2005) * 12 + month

        for a in (a1, a2):
            if a not in author2id:
                author2id[a] = next_id
                next_id += 1
        coarse_cat = COARSE_MAP[cat]
        rows.append((author2id[a1], author2id[a2], time_stamp, COARSE_TO_INT[coarse_cat]))


    rows.sort(key=lambda x: x[2])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['src', 'dst', 'time', 'label'])
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} edges to {out_path}")
    print(f"Authors (nodes): {len(author2id)}")
    print(f"Time range: {rows[0][2]} -- {rows[-1][2]} (month units)")
    cat_counts = Counter(r[3] for r in rows)
    for c, i in COARSE_TO_INT.items():
        print(f"  {c}: {cat_counts[i]} edges")


    map_path = os.path.join(out_dir, 'dblp_author_map.csv')
    with open(map_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'name'])
        for name, idx in sorted(author2id.items(), key=lambda x: x[1]):
            w.writerow([idx, name])
    print(f"Wrote author map to {map_path}")


if __name__ == '__main__':
    main()
