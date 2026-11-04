#!/usr/bin/env python
"""Draw the articles for the paper-replication experiment.

This experiment reproduces the generation method of Silva et al. as published:
the LLM receives the **full true news article** and is asked to modify it into a
fake news story, with their prompt reproduced verbatim (``PAPER_ARTICLE_PROMPT``
in ``expanded_fake_news_corpus.prompts``). It runs beside — not instead of —
the headline-seeded pipeline whose output lives under ``corpus/headlines`` and
``corpus/fake_news``.

Unlike ``sample_rounds.py``, which samples from the pooled corpora, here the
draw is **stratified**: a fixed number of articles per source corpus, so both
genres are equally represented in a small sample. A single ``random.Random``
seeded once draws Fake.br first and FakeTrueBR second, so the selection is
reproducible from ``--seed`` alone.

Outputs:

* ``<work-dir>/FakeBr_true.csv`` and ``<work-dir>/FakeTrueBr_true.csv`` — the
  drawn articles, one file per source, with ``uid`` and ``source`` columns.
  They restate upstream text, so they are working input under ``true-corpus/``.
* ``<manifest>`` — seed, sizes and drawn identifiers; corpus provenance.

Usage::

    uv run python scripts/sample_paper_replication.py
    uv run python scripts/sample_paper_replication.py --per-source 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from sample_rounds import SOURCES, write_round

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_source(clean_dir: Path, source: str) -> list[dict]:
    """Read one cleaned corpus, tagging source, genre and ``uid``."""
    file_name, genre = SOURCES[source]
    path = clean_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run first: uv run python scripts/prepare_corpora.py"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {**row, "source": source, "genre": genre, "uid": f"{source}:{row['id']}"}
            for row in csv.DictReader(handle)
        ]


def draw_stratified(
    clean_dir: Path, *, per_source: int, seed: int
) -> tuple[list[dict], dict[str, int]]:
    """Draw ``per_source`` articles from each corpus with one seeded RNG.

    Sources are drawn in the order of :data:`SOURCES`, so the result depends
    only on the seed and the corpus files.

    Returns:
        Tuple ``(records, pool_sizes)``.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    pool_sizes: dict[str, int] = {}
    for source in SOURCES:
        rows = load_source(clean_dir, source)
        pool_sizes[source] = len(rows)
        if per_source > len(rows):
            raise ValueError(
                f"{source}: asked for {per_source} articles, pool holds {len(rows)}."
            )
        records.extend(rng.sample(rows, per_source))
    return records, pool_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--clean-dir", type=Path, default=Path("true-corpus/clean"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("true-corpus/clean/paper_replication"),
        help="Where the per-source CSVs go (working input, rebuildable).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/paper_replication/manifest.json"),
        help="Where the provenance manifest goes. This one is corpus data.",
    )
    parser.add_argument(
        "--per-source", type=int, default=10, help="Articles drawn from each corpus."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling seed (reproducibility)."
    )
    args = parser.parse_args(argv)

    records, pool_sizes = draw_stratified(
        args.clean_dir, per_source=args.per_source, seed=args.seed
    )
    counts = write_round(records, args.work_dir)

    manifest = {
        "corpus": "FakeGen.BR",
        "stage": "selection",
        "experiment": "paper_replication",
        "seed": args.seed,
        "per_source": args.per_source,
        "pool": {"total": sum(pool_sizes.values()), "by_source": pool_sizes},
        "sources": {s: {"genre": g, "file": f} for s, (f, g) in SOURCES.items()},
        "path": str(args.work_dir),
        "by_source": counts,
        "uids": [r["uid"] for r in records],
    }

    print(f"stratified draw: {args.per_source} per source (seed {args.seed})")
    for source, n in pool_sizes.items():
        print(f"   {source:<12} {counts[source]:>3} of {n:>5}")
    print(f"-> {args.work_dir}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest with the drawn ids -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
