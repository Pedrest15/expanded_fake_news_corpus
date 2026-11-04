#!/usr/bin/env python
"""Reparte um round entre os modelos geradores, em blocos contíguos.

Cada round tem 100 notícias sorteadas; cada modelo fica com um bloco de 20. A
repartição segue a **ordem do sorteio** registrada no manifesto, não a ordem dos
CSVs — que estão separados por corpus e dariam blocos enviesados para uma única
origem.

Os blocos são atribuídos por posição: o primeiro modelo fica com as posições
0–19, o segundo com 20–39, e assim por diante. Reexecutar com modelos novos
**preserva** as atribuições existentes e aloca os blocos livres seguintes, para
que acrescentar um modelo nunca redistribua o que já foi gerado.

Cada modelo recebe um CSV por corpus, porque o Fake.br é titulado com
``--genre news`` e o FakeTrueBR com ``--genre factcheck``.

Uso::

    uv run python scripts/assign_blocks.py --round 1 \\
        --model openai/gpt-4.1-mini-2025-04-14
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from expanded_fake_news_corpus.config import model_slug, split_model

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SOURCE_FILES = {"fakebr": "FakeBr_true.csv", "faketruebr": "FakeTrueBr_true.csv"}


def load_round_uids(manifest_path: Path, round_number: int) -> list[str]:
    """Lê os uids de um round na ordem em que foram sorteados.

    Raises:
        SystemExit: Se o manifesto não existir ou não tiver o round pedido.
    """
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} not found. Run first: "
            "uv run python scripts/sample_rounds.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for draw in manifest["draws"]:
        if draw["round"] == round_number:
            return list(draw["uids"])
    raise SystemExit(f"Round {round_number} not in {manifest_path}.")


def load_round_records(round_dir: Path) -> dict[str, dict]:
    """Indexa por uid os registros dos CSVs de um round."""
    records: dict[str, dict] = {}
    for file_name in SOURCE_FILES.values():
        path = round_dir / file_name
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records[row["uid"]] = row
    if not records:
        raise SystemExit(f"No round CSV under {round_dir}.")
    return records


def assign_blocks(
    uids: list[str], models: list[str], previous: dict[str, int], block_size: int
) -> dict[str, int]:
    """Mapeia modelo -> índice do bloco, preservando o que já estava atribuído.

    Args:
        uids: Uids do round, na ordem do sorteio.
        models: Modelos a atribuir, na ordem em que devem receber blocos livres.
        previous: Atribuições de uma execução anterior.
        block_size: Notícias por bloco.

    Returns:
        Modelo -> índice do bloco.

    Raises:
        SystemExit: Se não houver blocos suficientes para os modelos pedidos.
    """
    total_blocks = len(uids) // block_size
    # Preserva tudo que já estava atribuído, inclusive modelos fora desta
    # execução: acrescentar um modelo não pode redistribuir blocos.
    assignments = dict(previous)
    taken = set(assignments.values())

    for model in models:
        if model in assignments:
            continue
        free = next((b for b in range(total_blocks) if b not in taken), None)
        if free is None:
            raise SystemExit(
                f"No free block: {len(uids)} articles / {block_size} = "
                f"{total_blocks} blocks, all taken."
            )
        assignments[model] = free
        taken.add(free)
    return assignments


def write_model_inputs(
    uids: list[str], records: dict[str, dict], destination: Path
) -> dict[str, int]:
    """Grava um CSV por corpus com os registros do bloco de um modelo."""
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for source, file_name in SOURCE_FILES.items():
        rows = [records[u] for u in uids if records[u]["source"] == source]
        counts[source] = len(rows)
        if not rows:
            continue
        with (destination / file_name).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        metavar="PROVEDOR/MODELO",
        help="Modelo a receber um bloco; repita para atribuir vários.",
    )
    parser.add_argument(
        "--rounds-manifest", type=Path, default=Path("corpus/rounds/manifest.json")
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("true-corpus/clean/rounds")
    )
    parser.add_argument("--out-manifest", type=Path, default=Path("corpus/assignments"))
    args = parser.parse_args(argv)

    uids = load_round_uids(args.rounds_manifest, args.round)
    round_dir = args.work_dir / f"round{args.round}"
    records = load_round_records(round_dir)

    manifest_path = args.out_manifest / f"round{args.round}.json"
    previous: dict[str, int] = {}
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {a["model"]: a["block"] for a in stored["assignments"]}

    blocks = assign_blocks(uids, args.models, previous, args.block_size)

    assignments = []
    print(f"round {args.round}: {len(uids)} articles, blocks of {args.block_size}")
    for model, block in sorted(blocks.items(), key=lambda kv: kv[1]):
        start = block * args.block_size
        block_uids = uids[start : start + args.block_size]
        provider, name = split_model(model)
        destination = round_dir / model_slug(provider) / model_slug(name)
        counts = write_model_inputs(block_uids, records, destination)
        assignments.append(
            {
                "model": model,
                "block": block,
                "range": [start, start + args.block_size - 1],
                "by_source": counts,
                "input_dir": str(destination),
                "uids": block_uids,
            }
        )
        detail = "  ".join(f"{s}={n}" for s, n in counts.items())
        print(
            f"   block {block} [{start}:{start + args.block_size}] -> {model}"
            f"  ({detail})"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "round": args.round,
                "block_size": args.block_size,
                "total_blocks": len(uids) // args.block_size,
                "assignments": assignments,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nassignment manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
