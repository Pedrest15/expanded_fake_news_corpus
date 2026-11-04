#!/usr/bin/env python
"""Draw disjoint rounds of true news articles for generation.

Sampling is performed over the **pooled** set of both source corpora, with no
per-source quota: every article has the same probability, so each round's
composition mirrors the relative size of the corpora (3,599 from Fake.br against
1,403 from FakeTrueBR, i.e. roughly 72%/28%).

Rounds are **disjoint by construction**: the pool is shuffled once with the given
seed and then sliced. An article appearing in two rounds is impossible, with no
need for iterative exclusion.

The manifest — seed, sizes and drawn identifiers — is corpus provenance and is
written under ``corpus/``. The per-round CSVs restate the upstream article text
and are rebuildable from the manifest, so they are working input and are written
under ``true-corpus/`` instead.

Each round is written as two CSV files, one per source. The split is not
cosmetic: Fake.br articles are headlined with ``--genre news`` and FakeTrueBR
ones with ``--genre factcheck``, and mixing both in a single file would make it
impossible to pass the correct genre.

The manifest records the seed, the sizes and the drawn identifiers, so the
selection is auditable and reproducible. Because id 4 exists in both corpora,
every article also carries a ``uid`` shaped as ``source:id``.

Usage::

    uv run python scripts/sample_rounds.py
    uv run python scripts/sample_rounds.py --rounds 3 --size 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

#: Source corpora: file name and the genre their articles belong to.
SOURCES = {
    "fakebr": ("FakeBr_true.csv", "news"),
    "faketruebr": ("FakeTrueBr_true.csv", "factcheck"),
}


def load_pool(clean_dir: Path) -> list[dict]:
    """Read both cleaned corpora into a single pool, tagging source and genre."""
    pool: list[dict] = []
    for source, (file_name, genre) in SOURCES.items():
        path = clean_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run first: uv run python scripts/prepare_corpora.py"
            )
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                pool.append(
                    {
                        **row,
                        "source": source,
                        "genre": genre,
                        "uid": f"{source}:{row['id']}",
                    }
                )
    return pool


def draw_rounds(
    pool: list[dict], *, rounds: int, size: int, seed: int
) -> list[list[dict]]:
    """Shuffle once and slice, guaranteeing non-overlapping rounds."""
    needed = rounds * size
    if needed > len(pool):
        raise ValueError(
            f"{rounds} rounds of {size} require {needed} articles, "
            f"but the pool holds {len(pool)}."
        )
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i * size : (i + 1) * size] for i in range(rounds)]


def write_round(records: list[dict], destination: Path) -> dict[str, int]:
    """Write one CSV per source and return the per-source counts."""
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for source, (file_name, _) in SOURCES.items():
        items = [r for r in records if r["source"] == source]
        counts[source] = len(items)
        if not items:
            continue
        fields = [c for c in items[0] if c != "source"] + ["source"]
        with (destination / file_name).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--clean-dir", type=Path, default=Path("true-corpus/clean"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("true-corpus/clean/rounds"),
        help="Where the per-round CSVs go. They restate upstream text and are "
        "rebuildable from the manifest, so they are working input, not corpus data.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/rounds/manifest.json"),
        help="Where the provenance manifest goes. This one is corpus data.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling seed (reproducibility)."
    )
    args = parser.parse_args(argv)

    pool = load_pool(args.clean_dir)
    rounds = draw_rounds(pool, rounds=args.rounds, size=args.size, seed=args.seed)

    by_source = {s: sum(1 for r in pool if r["source"] == s) for s in SOURCES}
    manifest = {
        "corpus": "FakeGen.BR",
        "stage": "selection",
        "seed": args.seed,
        "rounds": args.rounds,
        "size": args.size,
        "pool": {"total": len(pool), "by_source": by_source},
        "sources": {s: {"genre": g, "file": f} for s, (f, g) in SOURCES.items()},
        "draws": [],
    }

    print(f"pooled set: {len(pool)} articles (seed {args.seed})")
    for source, n in by_source.items():
        print(f"   {source:<12} {n:>5}  ({100 * n / len(pool):.1f}%)")
    print()

    for index, records in enumerate(rounds, start=1):
        destination = args.work_dir / f"round{index}"
        counts = write_round(records, destination)
        manifest["draws"].append(
            {
                "round": index,
                "path": str(destination),
                "by_source": counts,
                "uids": [r["uid"] for r in records],
            }
        )
        detail = "  ".join(f"{s}={n}" for s, n in counts.items())
        print(f"round {index}: {len(records)} articles  ({detail})  -> {destination}")

    sets = [set(d["uids"]) for d in manifest["draws"]]
    print("\ndisjointness check:")
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            print(f"   round {i + 1} n round {j + 1} = {len(sets[i] & sets[j])}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nmanifest with the drawn ids -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
