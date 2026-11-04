#!/usr/bin/env python
"""Monta os dados do visualizador do corpus para o GitHub Pages.

Lê as fake news sintéticas da replicação do artigo
(``corpus/paper_replication/<provider>/<model>/*.jsonl``), acrescenta a
procedência de cada lote (``*.meta.json``) e os metadados da notícia de origem
(link, categoria, data), e grava tudo em ``docs/data.json``.

O texto verdadeiro de origem **não** entra: ele pertence aos corpora Fake.br e
FakeTrueBR, e republicá-lo num site é problema de licenciamento. A página traz o
link para a matéria original, não a matéria. O que se publica é o que o modelo
escreveu: o texto sintético e a explicação das mudanças.

Uso::

    uv run python scripts/build_site.py
    uv run python scripts/build_site.py --model openai/gpt-4.1-mini-2025-04-14
    uv run python scripts/build_site.py --inline preview.html
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
from datetime import datetime
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

#: Mediana de palavras das fake news humanas, medida nos corpora de origem.
#: Serve de referência para o comprimento das sintéticas.
HUMAN_MEDIAN = {"fakebr": 157, "faketruebr": 109}

SOURCE_LABEL = {"fakebr": "Fake.br", "faketruebr": "FakeTrueBR"}

#: Aviso que o parser grava quando a resposta veio sem as tags do artigo — é
#: como uma recusa do modelo aparece no JSONL. Recusa não é fake news e não
#: entra no visualizador; fica registrada em ``corpus/paper_replication/NOTES.md``.
REFUSAL_WARNING = "response without format tags"


def read_jsonl(path: Path) -> list[dict]:
    """Lê um JSONL, ignorando linhas em branco."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_source_metadata(work_dir: Path) -> dict[str, dict]:
    """Indexa por uid os metadados das notícias de origem.

    Só link, categoria, data e o tamanho do texto — o texto fica de fora de
    propósito.
    """
    metadata: dict[str, dict] = {}
    for path in sorted(work_dir.rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                uid = row.get("uid")
                if uid and uid not in metadata:
                    metadata[uid] = {
                        "link": row.get("link", ""),
                        "category": row.get("category", ""),
                        "date": row.get("date", ""),
                        "source_words": len((row.get("text") or "").split()),
                    }
    return metadata


def collect_items(
    experiment_dir: Path,
    source_meta: dict[str, dict],
    *,
    models: set[str] | None = None,
    excluded: set[str] | None = None,
) -> list[dict]:
    """Um registro por notícia sintética, com a procedência do lote.

    Args:
        experiment_dir: Raiz ``corpus/paper_replication``
        source_meta: Metadados das notícias de origem, por uid
        models: Restringe a estes modelos (``provedor/modelo``); todos se omitido
        excluded: Pares ``modelo:uid`` ou ``uid`` a descartar

    Returns:
        Registros na ordem dos arquivos, sem as recusas
    """
    items: list[dict] = []
    for jsonl in sorted(experiment_dir.rglob("*.jsonl")):
        meta = _read_meta(jsonl)
        for record in read_jsonl(jsonl):
            model = record.get("model", "")
            if models and model not in models:
                continue
            if REFUSAL_WARNING in record.get("warnings", []):
                continue
            uid = record["source_id"]
            if excluded and (uid in excluded or f"{model}:{uid}" in excluded):
                continue
            text = record.get("synthetic_text", "")
            items.append(
                {
                    "uid": uid,
                    "source": uid.split(":", 1)[0],
                    "fake_headline": _fake_headline(text),
                    "fake_text": text,
                    "changes": record.get("changes", ""),
                    "fake_model": model,
                    "fake_warnings": record.get("warnings", []),
                    "sampling": meta.get("sampling", {}),
                    "max_tokens": meta.get("max_tokens"),
                    "prompt_sha256": meta.get("prompt_sha256", ""),
                    **source_meta.get(
                        uid,
                        {"link": "", "category": "", "date": "", "source_words": 0},
                    ),
                }
            )
    return items


def _read_meta(jsonl_path: Path) -> dict:
    """Procedência do lote, gravada ao lado do JSONL."""
    meta_path = jsonl_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _fake_headline(text: str) -> str:
    """Manchete da fake news: no formato do artigo, é a primeira linha do texto."""
    text = text.strip()
    return text.splitlines()[0].strip() if text else ""


def build_stats(items: list[dict]) -> dict:
    """Contagens e comprimentos, com a referência humana para comparação."""
    by_source: dict[str, dict] = {}
    for source in sorted({i["source"] for i in items}):
        lengths = [
            len(i["fake_text"].split())
            for i in items
            if i["source"] == source and i["fake_text"]
        ]
        by_source[source] = {
            "label": SOURCE_LABEL.get(source, source),
            "items": sum(1 for i in items if i["source"] == source),
            "median_words": round(stats.median(lengths)) if lengths else 0,
            "human_median_words": HUMAN_MEDIAN.get(source, 0),
        }

    return {
        "items": len(items),
        "warnings": sum(len(i["fake_warnings"]) for i in items),
        "models": sorted({i["fake_model"] for i in items if i["fake_model"]}),
        "by_source": by_source,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("corpus/paper_replication")
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("true-corpus/clean/paper_replication")
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Só este modelo gerador (repetível). Sem a opção, todos.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="excluded",
        default=[],
        metavar="[MODELO:]UID",
        help=(
            "Descarta este registro (repetível). Com o modelo na frente, só "
            "daquele gerador. Para geração defeituosa que não representa o "
            "modelo — repetição degenerada, texto no idioma errado."
        ),
    )
    parser.add_argument(
        "--inline",
        type=Path,
        help="Também grava uma cópia autocontida, com os dados embutidos.",
    )
    args = parser.parse_args(argv)

    source_meta = load_source_metadata(args.work_dir) if args.work_dir.is_dir() else {}
    items = collect_items(
        args.experiment_dir,
        source_meta,
        models=set(args.models) if args.models else None,
        excluded=set(args.excluded) if args.excluded else None,
    )
    payload = {
        "corpus": "FakeGen.BR",
        "experiment": "paper_replication",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stats": build_stats(items),
        "items": items,
    }

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.docs_dir / "data.json"
    data_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{len(items)} notícias sintéticas -> {data_path}")
    for info in payload["stats"]["by_source"].values():
        print(
            f"   {info['label']:<12} {info['items']:>3} itens  "
            f"mediana {info['median_words']} palavras "
            f"(humana {info['human_median_words']})"
        )

    if args.inline:
        template = (args.docs_dir / "index.html").read_text(encoding="utf-8")
        blob = json.dumps(payload, ensure_ascii=False)
        args.inline.write_text(
            template.replace(
                '<script id="corpus-data" type="application/json"></script>',
                f'<script id="corpus-data" type="application/json">{blob}</script>',
            ),
            encoding="utf-8",
        )
        print(f"cópia autocontida -> {args.inline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
