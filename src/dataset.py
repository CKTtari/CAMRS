import os
import pandas as pd


DATA_DIR = os.environ.get('CAMRS_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
PROC_DIR = os.environ.get('CAMRS_PROCESSED_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))


def load_email_eu():
    path = os.path.join(DATA_DIR, 'email-Eu-core-temporal.txt')
    df = pd.read_csv(path, sep=' ', header=None, names=['src', 'dst', 'time'])
    return df


def load_college_msg():
    path = os.path.join(DATA_DIR, 'CollegeMsg.txt')
    df = pd.read_csv(path, sep=' ', header=None, names=['src', 'dst', 'time'])
    return df


def load_bitcoin_otc():
    path = os.path.join(DATA_DIR, 'soc-sign-bitcoinotc.csv')
    if not os.path.exists(path):
        import gzip
        gz_path = path + '.gz'
        with gzip.open(gz_path, 'rt') as f:
            content = f.read()
        with open(path, 'w') as f:
            f.write(content)
    df = pd.read_csv(path, header=None, names=['src', 'dst', 'rating', 'time'])
    return df[['src', 'dst', 'time']]


def load_wiki_talk(nrows=None):
    """Wikipedia talk network; load the complete local archive by default."""
    path = os.path.join(DATA_DIR, 'wiki-talk-temporal.txt')
    gz_path = path + '.gz'
    source = gz_path if os.path.exists(gz_path) else path
    df = pd.read_csv(
        source,
        sep=' ',
        header=None,
        names=['src', 'dst', 'time'],
        nrows=nrows,
        compression='gzip' if source.endswith('.gz') else None,
    )
    return df


def load_sx_superuser(nrows=None):
    """StackExchange Superuser temporal graph (1.44M edges). Format: SRC DST UNIXTS."""
    path = os.path.join(DATA_DIR, 'sx-superuser.txt')
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        import gzip
        with gzip.open(path + '.gz', 'rt') as f_in:
            content = f_in.read(200 * 1024 * 1024)
        with open(path, 'w') as f_out:
            f_out.write(content)
    df = pd.read_csv(path, sep=' ', header=None, names=['src', 'dst', 'time'], nrows=nrows)
    return df


def load_sx_stackoverflow_a2q(nrows=None):
    """StackExchange StackOverflow answer-to-question temporal graph (17.8M edges). Format: SRC DST UNIXTS."""
    path = os.path.join(DATA_DIR, 'sx-stackoverflow-a2q.txt')
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        import gzip
        with gzip.open(path + '.gz', 'rt') as f_in:
            content = f_in.read(500 * 1024 * 1024)
        with open(path, 'w') as f_out:
            f_out.write(content)
    df = pd.read_csv(path, sep=' ', header=None, names=['src', 'dst', 'time'], nrows=nrows)
    return df


def load_dblp():
    """DBLP coauthorship temporal graph (390K edges, 2005-2024).
    Labels are coarse venue categories: 0=DataAI, 1=SysSec, 2=TheorySE.
    Timestamps are month-level (1=Jan 2005, 132=Dec 2024).
    """
    path = os.path.join(PROC_DIR, 'dblp_coauthorship.csv')
    df = pd.read_csv(path)
    return df


def load_reddit_hyperlinks(nrows=500000):
    """Reddit hyperlinks network with subreddit names as labels."""
    path = os.path.join(DATA_DIR, 'soc-redditHyperlinks-title.tsv')
    df = pd.read_csv(path, sep='\t', nrows=nrows)
    if 'LINK_SENTIMENT' in df.columns:
        df = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']]
    else:
        df = df.iloc[:, :3]
    df.columns = ['src', 'dst', 'time']

    all_names = pd.concat([df['src'], df['dst']]).unique()
    name_to_id = {name: i for i, name in enumerate(all_names)}
    df['src'] = df['src'].map(name_to_id)
    df['dst'] = df['dst'].map(name_to_id)

    df['time'] = pd.to_datetime(df['time']).astype('datetime64[s]').astype('int64')
    return df


def load_reddit_hyperlinks_with_labels(nrows=500000, n_labels=10):
    """Load Reddit with the paper's native top-(L-1)+other source labels.

    Nodes retain the same integer encoding as ``load_reddit_hyperlinks``.
    The most frequent source subreddits receive labels 0..L-2 and every
    remaining node receives label L-1 (``other``).
    """
    if n_labels < 2:
        raise ValueError('n_labels must be at least 2')
    path = os.path.join(DATA_DIR, 'soc-redditHyperlinks-title.tsv')
    raw = pd.read_csv(
        path,
        sep='\t',
        nrows=nrows,
        usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'],
    )
    all_names = pd.concat([raw['SOURCE_SUBREDDIT'], raw['TARGET_SUBREDDIT']]).unique()
    name_to_id = {name: i for i, name in enumerate(all_names)}
    top_sources = list(raw['SOURCE_SUBREDDIT'].value_counts().head(n_labels - 1).index)
    source_to_label = {name: label for label, name in enumerate(top_sources)}
    other_label = n_labels - 1
    label_map = {
        node_id: source_to_label.get(name, other_label)
        for name, node_id in name_to_id.items()
    }
    frame = pd.DataFrame({
        'src': raw['SOURCE_SUBREDDIT'].map(name_to_id),
        'dst': raw['TARGET_SUBREDDIT'].map(name_to_id),
        'time': pd.to_datetime(raw['TIMESTAMP']).astype('datetime64[s]').astype('int64'),
    })
    return frame, label_map


def preprocess_temporal(df, normalize_time=True):
    df = df.copy()
    df['src'] = df['src'].astype(int)
    df['dst'] = df['dst'].astype(int)
    if 'time' in df.columns:
        df['time'] = df['time'].astype(int)
        if normalize_time:
            tmin = df['time'].min()
            df['time'] = df['time'] - tmin
    edges_unique = df.drop_duplicates(subset=['src', 'dst', 'time'])
    return edges_unique
