#!/usr/bin/env python
"""Extrai as notícias VERDADEIRAS dos corpora de origem, em texto original.

**Fake.br** — lidos de ``full_texts/true/*.txt``, que trazem o texto como foi
coletado dos sites: com caixa, acentuação e pontuação. Não use o CSV
``preprocessed/`` do repositório: ele está em minúsculas, sem acentos e sem
*stopwords*, o que inviabiliza a titulação. Os metadados (autor, link,
categoria, data) vêm de ``full_texts/true-meta-information/<id>-meta.txt``.

**FakeTrueBR** — o corpus é distribuído apenas como ``FakeTrueBr_corpus.csv``,
uma linha por par (fake, true), e o texto já vem normalizado em minúsculas e
com parte da pontuação removida. Não há versão original publicada; recuperá-la
exigiria recoletar os artigos a partir de ``link_t``. A limpeza aqui mantém as
colunas ``true`` e ``link_t`` e deduplica as notícias verdadeiras, que se
repetem por aparecerem pareadas a mais de uma fake.

Ambas as saídas usam o mesmo esquema base::

    id, text, link, duplicate_rows[, author, category, date]

``id`` é o identificador de origem (nome do arquivo no Fake.br, linha do CSV no
FakeTrueBR) e ``duplicate_rows`` lista os registros com texto idêntico que foram
descartados — a rastreabilidade até o corpus original é preservada.

Uso::

    uv run python scripts/prepare_corpora.py
    uv run python scripts/prepare_corpora.py --out-dir true-corpus/clean
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

BASE_FIELDS = ["id", "text", "link", "duplicate_rows"]
FAKEBR_FIELDS = [*BASE_FIELDS, "author", "category", "date"]

#: Ordem das linhas em ``*-meta.txt`` (ver README do Fake.br). Só os quatro
#: primeiros campos interessam aqui; o resto são contagens linguísticas.
META_FIELDS = ["author", "link", "category", "date"]

_WHITESPACE_RE = re.compile(r"\s+")


def _key(text: str) -> str:
    """Chave de deduplicação: texto sem variação de espaçamento nem de caixa."""
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _dedupe(records: list[dict]) -> tuple[list[dict], int]:
    """Mantém a primeira ocorrência de cada texto e anota as repetições."""
    unique: OrderedDict[str, dict] = OrderedDict()
    for record in records:
        key = _key(record["text"])
        if key in unique:
            unique[key]["duplicate_rows"].append(record["id"])
        else:
            unique[key] = {**record, "duplicate_rows": []}
    return list(unique.values()), len(records) - len(unique)


def _sort_key(path: Path) -> tuple[int, int | str]:
    """Ordena 1.txt, 2.txt, ..., 10.txt numericamente."""
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def _read_meta(path: Path) -> dict[str, str]:
    """Lê autor, link, categoria e data do arquivo de metadados."""
    if not path.exists():
        return dict.fromkeys(META_FIELDS, "")
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    values = [line.strip() for line in lines[: len(META_FIELDS)]]
    values += [""] * (len(META_FIELDS) - len(values))
    return {
        name: ("" if value.lower() == "none" else value)
        for name, value in zip(META_FIELDS, values, strict=True)
    }


def prepare_fakebr(full_texts: Path) -> tuple[list[dict], dict[str, int]]:
    """Monta o conjunto verdadeiro do Fake.br a partir dos textos originais."""
    texts_dir = full_texts / "true"
    meta_dir = full_texts / "true-meta-information"
    if not texts_dir.is_dir():
        raise FileNotFoundError(texts_dir)

    files = sorted(texts_dir.glob("*.txt"), key=_sort_key)
    records = []
    for file in files:
        # utf-8-sig: parte dos arquivos do corpus vem com BOM.
        text = file.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not text:
            continue
        meta = _read_meta(meta_dir / f"{file.stem}-meta.txt")
        records.append(
            {
                "id": file.stem,
                "text": text,
                "link": meta["link"],
                "author": meta["author"],
                "category": meta["category"],
                "date": meta["date"],
            }
        )

    unique, removed = _dedupe(records)
    stats = {
        "source": len(files),
        "true_articles": len(records),
        "duplicates_removed": removed,
        "final": len(unique),
    }
    return unique, stats


def prepare_faketruebr(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Extrai as notícias verdadeiras (e seus links) do FakeTrueBR."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records = [
        {
            "id": str(number),
            "text": (row.get("true") or "").strip(),
            "link": (row.get("link_t") or "").strip(),
        }
        for number, row in enumerate(rows, start=1)
    ]
    records = [record for record in records if record["text"]]

    unique, removed = _dedupe(records)
    stats = {
        "source": len(rows),
        "true_articles": len(records),
        "duplicates_removed": removed,
        "final": len(unique),
    }
    return unique, stats


def write_csv(records: list[dict], path: Path, fieldnames: list[str]) -> None:
    """Grava os registros limpos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {**record, "duplicate_rows": ";".join(record["duplicate_rows"])}
            )


def _report(label: str, source: Path, destination: Path, stats: dict) -> None:
    print(f"{label} ({source} → {destination})")
    print(f"  registros na origem  : {stats['source']}")
    print(f"  notícias verdadeiras : {stats['true_articles']}")
    print(f"  duplicatas removidas : {stats['duplicates_removed']}")
    print(f"  TOTAL final          : {stats['final']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--fakebr-full-texts",
        type=Path,
        default=Path("true-corpus/raw/Fake.br-full_texts"),
        help="Pasta full_texts do Fake.br (com true/ e true-meta-information/).",
    )
    parser.add_argument(
        "--faketruebr-csv",
        type=Path,
        default=Path("true-corpus/FakeTrueBr_corpus.csv"),
        help="CSV do FakeTrueBR.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("true-corpus/clean"))
    args = parser.parse_args(argv)

    if not args.fakebr_full_texts.is_dir():
        print(
            f"erro: {args.fakebr_full_texts} não encontrado. Baixe o corpus com:\n"
            "  git clone --depth 1 https://github.com/roneysco/Fake.br-Corpus.git\n"
            "e aponte --fakebr-full-texts para a pasta full_texts.",
            file=sys.stderr,
        )
        return 1
    if not args.faketruebr_csv.exists():
        print(f"erro: {args.faketruebr_csv} não encontrado", file=sys.stderr)
        return 1

    records, stats = prepare_fakebr(args.fakebr_full_texts)
    destination = args.out_dir / "FakeBr_true.csv"
    write_csv(records, destination, FAKEBR_FIELDS)
    _report("Fake.br", args.fakebr_full_texts / "true", destination, stats)

    records, stats = prepare_faketruebr(args.faketruebr_csv)
    destination = args.out_dir / "FakeTrueBr_true.csv"
    write_csv(records, destination, BASE_FIELDS)
    _report("FakeTrueBR", args.faketruebr_csv, destination, stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
