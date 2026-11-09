# CAMRS: Bounded-Memory Aggregate Query Processing on Streaming Temporal Graphs

Tianyi Gao, Hongchao Qin, Haotian Jiang, Guang Zeng, and Rong-Hua Li. *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026)*. [DOI: 10.1145/3799682.3840889](https://doi.org/10.1145/3799682.3840889).

CAMRS supports bounded-memory aggregate queries over streaming temporal graphs. This repository contains the method, experiment drivers, and data processing.

## Repository structure

- `src/`: CAMRS, the TGS-AQP baseline, shared sketch primitives, and data loaders.
- `experiments/`: data preparation, workload generation, and experiment scripts.
- `data/`: input files prepared locally.
- `results/`: generated experiment outputs.

The main implementations are `src/camrs.py` (CAMRS), `src/tgs_aqp.py` (TGS-AQP), and `src/sketches.py` (shared CMS and HLL primitives). The COUNT and predicate experiment scripts compare CAMRS with TGS-AQP.

## Environment

Use Python 3.11 and install the dependencies from `requirements.txt`. For example, create a virtual environment from the repository root:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```sh
source .venv/bin/activate
```

Then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

The scalability benchmark uses PowerShell and a C++17 compiler available as `g++` on PATH. Run commands from the repository root. Experiment outputs are written to `results/`.

## Data preparation

Download and extract the following files into `data/raw/`:

| Dataset | Source | File |
| --- | --- | --- |
| Email-Eu | [SNAP](https://snap.stanford.edu/data/email-Eu-core-temporal.html) | `email-Eu-core-temporal.txt` |
| CollegeMsg | [SNAP](https://snap.stanford.edu/data/CollegeMsg.html) | `CollegeMsg.txt` |
| Wiki-Talk temporal | [SNAP temporal motifs](https://snap.stanford.edu/temporal-motifs/data.html) | `wiki-talk-temporal.txt` or `.txt.gz` |
| Reddit hyperlinks | [SNAP](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) | `soc-redditHyperlinks-title.tsv` |
| Super User | [SNAP temporal motifs](https://snap.stanford.edu/temporal-motifs/data.html) | `sx-superuser.txt` |
| Stack Overflow answer-to-question | [SNAP temporal motifs](https://snap.stanford.edu/temporal-motifs/data.html) | `sx-stackoverflow-a2q.txt` |
| Bitcoin OTC | [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html) | `soc-sign-bitcoinotc.csv` or `.csv.gz` |

For workload preparation, download `dblp.json.gz` from [MIT BigDND DBLP Graphs](https://projects.csail.mit.edu/dnd/DBLP/) into `data/raw/`, then run:

```powershell
python experiments/prepare_dblp.py
```

The loader normalizes time origins and removes duplicate `(src, dst, time)` records. Reddit uses the nine most frequent source subreddits and an other category; the other paper datasets use degree terciles. Wiki-Talk uses the complete temporal input.

## Experiments

Run the following commands from the repository root. The preparation step generates the shared Super User query workload in `results/`.

```sh
python experiments/prepare_superuser_workload.py
```

RQ1: ablation and TGS-AQP allocation:

```sh
python experiments/run_ablation_email.py
python experiments/run_ablation_superuser.py
python experiments/run_tgs_allocation_sweep.py
```

RQ2: COUNT and predicate COUNT, plus label locality:

```sh
python experiments/run_count.py email_eu college_msg wiki_talk reddit_hyperlinks sx_superuser sx_stackoverflow_a2q
python experiments/run_predicate_count.py
python experiments/run_label_locality.py
```

RQ3: COUNT_DISTINCT and HLL merge cost:

```sh
python experiments/run_count_distinct.py email_eu college_msg wiki_talk reddit_hyperlinks sx_superuser sx_stackoverflow_a2q
python experiments/run_hll_merge_cost.py --interleaved
```

RQ4: bucket count and memory budget:

```sh
python experiments/run_bucket_sweep.py
python experiments/run_budget_sweep.py
```

RQ5: Router validation:

```sh
python experiments/run_router_validation.py email_eu college_msg wiki_talk reddit_hyperlinks sx_superuser sx_stackoverflow_a2q
```

RQ6: Router scalability. Run RQ5 first, then prepare the benchmark and execute it with PowerShell:

```powershell
python experiments/prepare_router_scalability.py
powershell -ExecutionPolicy Bypass -File experiments/run_router_scalability.ps1
```

Each experiment writes a CSV file to `results/` with a name corresponding to the experiment. Router error uses `abs(estimate - exact) / max(1, exact)`.

## Citation

```bibtex
@inproceedings{gao2026camrs,
  title = {Bounded-Memory Aggregate Query Processing on Streaming Temporal Graphs},
  author = {Gao, Tianyi and Qin, Hongchao and Jiang, Haotian and Zeng, Guang and Li, Rong-Hua},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year = {2026},
  doi = {10.1145/3799682.3840889}
}
```

## License

MIT License; see [LICENSE](LICENSE).
