#!/usr/bin/env python
"""Normaliza a planilha de métricas do NILC-Metrix calculada fora do projeto.

O NILC-Metrix (https://github.com/sidleal/nilcmetrix) não roda aqui: depende de
uma imagem Docker, de um Postgres com o léxico e do parser PALAVRAS. O cálculo
é feito no servidor do NILC (repositório irmão ``wrapper_nilcmetrix``), que
devolve um CSV com uma linha por texto e a coluna ``id_texto`` no formato
``/<pasta do gerador>/<source>_<id>`` — a pasta relativa a
``corpus/paper_replication``, ou ``human`` para a fake news humana::

    /anthropic/claude-sonnet-4-5-20250929/fakebr_1006
    /rodada2/ollama/llama3.1_8b/faketruebr_509
    /human/fakebr_1006

Este script traduz esse identificador para as chaves que as análises usam
(``group``, ``model``, ``round``, ``uid``) e grava ``data/nilc_metrix/metrics.csv``,
que é o que :mod:`expanded_fake_news_corpus.analysis.nilc_metrix` lê.

NOTA: o wrapper cortou o identificador no primeiro ponto, como se fosse a
extensão do arquivo: ``/openai/gpt-4.1-mini-2025-04-14/fakebr_1006`` chegou como
``/openai/gpt-4`` e ``/rodada2/ollama/llama3.1_8b/...`` como
``/rodada2/ollama/llama3``. As linhas desses blocos vêm na mesma ordem dos
blocos íntegros (ordem lexicográfica de ``<source>_<id>``), então o conserto é
posicional — e só é aceito quando o prefixo casa com uma única pasta do corpus e
o bloco tem exatamente os mesmos uids dos blocos íntegros.

O lado humano **não** vem da planilha do servidor: as linhas ``/human/...`` são
descartadas e substituídas pelas de ``nilc-metrix/results/human.csv`` do
repositório ``noticias_falsas_humano_maquina_semantica``, que já mediu as 5.391
fake news humanas do trabalho anterior. As nossas 20 são um subconjunto delas,
e tirá-las de lá mantém um único conjunto de métricas humanas para as duas
bases. A coluna ``metrics_source`` registra de onde veio cada linha.

Cada linha é conferida contra o texto que as análises veem (manchete composta,
Fake.br limpo, markdown removido): a coluna ``corpus_words`` traz a contagem
local e ``text_matches`` diz se ela bate com a ``words`` do NILC dentro de
:data:`WORD_COUNT_TOLERANCE`. Quando não bate, o NILC mediu outro texto, e o
módulo de análise descarta o par.

Uso::

    uv run python scripts/import_nilc_metrix.py data/nilc_metrix/raw/_pedro_metrics.csv
    uv run python scripts/import_nilc_metrix.py <planilha.csv> --human-metrics <h.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from expanded_fake_news_corpus.analysis.documents import (
    EXPERIMENTS,
    PROJECT_ROOT,
    CorpusPaths,
    Document,
    Group,
    Source,
    load_paired_corpus,
    parse_uid,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "nilc_metrix" / "metrics.csv"

#: Métricas humanas do repositório de semântica (ver a docstring do módulo).
DEFAULT_HUMAN_METRICS = (
    PROJECT_ROOT.parent
    / "noticias_falsas_humano_maquina_semantica"
    / "nilc-metrix"
    / "results"
    / "human.csv"
)

#: ``subset`` da planilha humana de lá, e o deslocamento do nome do arquivo em
#: relação ao nosso uid. NOTA: os ``.txt`` do FakeTrueBR de lá são 0-based e o
#: nosso uid é 1-based — o mesmo ajuste de ``documents._PRIOR_INDEX_OFFSET``.
_PRIOR_HUMAN_SUBSETS: dict[Source, tuple[str, int]] = {
    Source.FAKEBR: ("fake_br_human", 0),
    Source.FAKETRUEBR: ("fake_true_human", -1),
}

#: Colunas de identificação na planilha humana de lá, que não são métricas.
_PRIOR_ID_COLUMNS = ("file", "subset")

#: Coluna de identificação na planilha devolvida pelo servidor.
ID_COLUMN = "id_texto"

#: Pasta que identifica o lado humano em ``id_texto``.
HUMAN_DIR = "human"

#: Desvio relativo aceito entre a contagem de palavras do NILC e a local. Os
#: tokenizadores diferem (hífen, número, sigla), o que dá até ~5% nos textos
#: íntegros; um texto diferente erra por dezenas de pontos percentuais.
WORD_COUNT_TOLERANCE = 0.10

#: Colunas de chave gravadas antes das métricas, nesta ordem.
KEY_COLUMNS = (
    ID_COLUMN,
    "experiment",
    "group",
    "model",
    "round",
    "uid",
    "source",
    "corpus_words",
    "text_matches",
    "metrics_source",
)

_WORD_RE = re.compile(r"\w+")


class ImportError_(Exception):
    """A planilha não pôde ser traduzida para as chaves do corpus."""


@dataclass(frozen=True)
class Generator:
    """Uma pasta de gerador do experimento, com o que a identifica no corpus."""

    directory: str  # relativa à raiz do experimento
    model: str | None  # None do lado humano
    round_name: str | None


def count_words(text: str) -> int:
    """Contagem de palavras usada só para conferir o texto medido."""
    return len(_WORD_RE.findall(text))


def discover_generators(experiment_dir: Path) -> dict[str, Generator]:
    """Mapeia cada pasta de gerador ao modelo e à rodada do ``.meta.json``.

    Args:
        experiment_dir: Raiz do experimento (``corpus/paper_replication``)

    Returns:
        Geradores por pasta relativa, mais o lado humano em :data:`HUMAN_DIR`
    """
    generators = {HUMAN_DIR: Generator(HUMAN_DIR, None, None)}
    for meta_path in sorted(experiment_dir.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        directory = meta_path.parent.relative_to(experiment_dir).as_posix()
        generators.setdefault(
            directory,
            Generator(directory, meta["model_string"], meta.get("run_name")),
        )
    return generators


def split_id(text_id: str) -> tuple[str, str | None]:
    """Separa ``id_texto`` em pasta do gerador e nome do documento.

    Returns:
        ``(pasta, nome)``; o nome é None quando o wrapper o truncou
    """
    parts = text_id.strip().strip("/").split("/")
    if parts[-1].startswith(("fakebr_", "faketruebr_")):
        return "/".join(parts[:-1]), parts[-1]
    return "/".join(parts), None


def resolve_directory(prefix: str, generators: dict[str, Generator]) -> Generator:
    """Acha a pasta de um bloco, inclusive quando o nome foi cortado no ponto.

    Raises:
        ImportError_: Se o prefixo não casar com exatamente uma pasta
    """
    if prefix in generators:
        return generators[prefix]
    matches = [
        generator
        for directory, generator in generators.items()
        if directory.startswith(f"{prefix}.")
    ]
    if len(matches) != 1:
        found = ", ".join(m.directory for m in matches) or "nenhuma"
        raise ImportError_(f"prefixo {prefix!r} casa com {found}")
    logger.warning(f"Identificador truncado: {prefix!r} -> {matches[0].directory!r}")
    return matches[0]


def name_to_uid(name: str) -> str:
    """``fakebr_1006`` -> ``fakebr:1006``."""
    source, _, local_id = name.partition("_")
    return f"{source}:{local_id}"


def assign_uids(
    blocks: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    """Dá a cada linha o seu uid, consertando os blocos truncados pela posição.

    Args:
        blocks: Linhas da planilha agrupadas pela pasta do gerador

    Returns:
        A lista de uids de cada bloco, na ordem das linhas

    Raises:
        ImportError_: Se os blocos íntegros discordarem entre si ou um bloco
            truncado tiver tamanho diferente deles
    """
    complete: dict[str, list[str]] = {}
    for directory, rows in blocks.items():
        names = [split_id(row[ID_COLUMN])[1] for row in rows]
        if all(names):
            complete[directory] = [name_to_uid(name) for name in names]
        elif any(names):
            raise ImportError_(f"bloco {directory!r} mistura ids íntegros e truncados")

    orders = {tuple(uids) for uids in complete.values()}
    if len(orders) != 1:
        raise ImportError_("os blocos íntegros não têm os mesmos uids na mesma ordem")
    reference = list(orders.pop())

    assigned: dict[str, list[str]] = {}
    for directory, rows in blocks.items():
        if directory in complete:
            assigned[directory] = complete[directory]
            continue
        if len(rows) != len(reference):
            raise ImportError_(
                f"bloco truncado {directory!r} tem {len(rows)} linhas, "
                f"os íntegros têm {len(reference)}"
            )
        assigned[directory] = reference
    return assigned


def corpus_texts(experiment: str) -> dict[tuple[str | None, str | None, str], str]:
    """Texto de cada documento como as análises o veem, por (modelo, rodada, uid)."""
    paths = replace(CorpusPaths(), synthetic_dir=EXPERIMENTS[experiment])
    documents: list[Document] = load_paired_corpus(paths)
    return {
        (document.model, document.round_name, document.uid): document.text
        for document in documents
    }


def _normalized_row(
    *,
    directory: str,
    experiment: str,
    model: str | None,
    round_name: str | None,
    uid: str,
    text: str,
    metrics: dict[str, str],
    metrics_source: str,
) -> dict[str, object]:
    """Monta uma linha normalizada, conferindo o texto medido contra o corpus."""
    source, local_id = parse_uid(uid)
    corpus_words = count_words(text)
    nilc_words = float(metrics["words"])
    matches = abs(nilc_words / corpus_words - 1) <= WORD_COUNT_TOLERANCE
    if not matches:
        logger.warning(
            f"Texto medido difere do analisado: {directory} {uid} "
            f"(NILC {nilc_words:.0f} palavras, corpus {corpus_words})"
        )
    group = Group.HUMAN if model is None else Group.MACHINE
    return {
        ID_COLUMN: f"/{directory}/{source.value}_{local_id}",
        "experiment": experiment,
        "group": group.value,
        "model": model or "",
        "round": round_name or "",
        "uid": uid,
        "source": source.value,
        "corpus_words": corpus_words,
        "text_matches": matches,
        "metrics_source": metrics_source,
        **metrics,
    }


def normalize(
    rows: Sequence[dict[str, str]],
    *,
    experiment: str,
    generators: dict[str, Generator],
    texts: dict[tuple[str | None, str | None, str], str],
    metrics_source: str = "",
) -> list[dict[str, object]]:
    """Traduz as linhas da planilha do servidor para as chaves do corpus.

    O bloco humano serve de referência para consertar os truncados, mas não
    vira linha: o lado humano vem do repositório de semântica
    (:func:`prior_human_rows`).

    Args:
        rows: Linhas da planilha do servidor
        experiment: Experimento a que os textos pertencem
        generators: Saída de :func:`discover_generators`
        texts: Saída de :func:`corpus_texts`
        metrics_source: Procedência gravada em cada linha

    Returns:
        Linhas de máquina com :data:`KEY_COLUMNS` seguidas das métricas

    Raises:
        ImportError_: Se algum documento não for encontrado no corpus
    """
    blocks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        blocks[split_id(row[ID_COLUMN])[0]].append(row)
    uids_by_block = assign_uids(blocks)

    normalized: list[dict[str, object]] = []
    for prefix, block in blocks.items():
        generator = resolve_directory(prefix, generators)
        if generator.model is None:
            logger.info(f"Lado humano do servidor descartado: {len(block)} linhas")
            continue
        for row, uid in zip(block, uids_by_block[prefix], strict=True):
            key = (generator.model, generator.round_name, uid)
            if key not in texts:
                raise ImportError_(f"documento fora do corpus: {key}")
            normalized.append(
                _normalized_row(
                    directory=generator.directory,
                    experiment=experiment,
                    model=generator.model,
                    round_name=generator.round_name,
                    uid=uid,
                    text=texts[key],
                    metrics={
                        column.strip(): value
                        for column, value in row.items()
                        if column != ID_COLUMN
                    },
                    metrics_source=metrics_source,
                )
            )
    return normalized


def prior_human_rows(
    prior_rows: Sequence[dict[str, str]],
    *,
    experiment: str,
    texts: dict[tuple[str | None, str | None, str], str],
    metrics_source: str = "",
) -> list[dict[str, object]]:
    """Linhas humanas tiradas da planilha do repositório de semântica.

    Args:
        prior_rows: Linhas de ``nilc-metrix/results/human.csv`` de lá
        experiment: Experimento a que os textos pertencem
        texts: Saída de :func:`corpus_texts`
        metrics_source: Procedência gravada em cada linha

    Returns:
        Uma linha normalizada por fake news humana do corpus pareado; um uid
        que não esteja lá fica de fora com aviso, e a análise descarta o par
    """
    by_file = {(row["subset"], row["file"]): row for row in prior_rows}
    human_uids = sorted(uid for model, _, uid in texts if model is None)

    normalized: list[dict[str, object]] = []
    for uid in human_uids:
        source, local_id = parse_uid(uid)
        subset, offset = _PRIOR_HUMAN_SUBSETS[source]
        row = by_file.get((subset, f"{int(local_id) + offset}.txt"))
        if row is None:
            logger.warning(f"Sem métricas humanas no repositório de semântica: {uid}")
            continue
        normalized.append(
            _normalized_row(
                directory=HUMAN_DIR,
                experiment=experiment,
                model=None,
                round_name=None,
                uid=uid,
                text=texts[(None, None, uid)],
                metrics={
                    column: value
                    for column, value in row.items()
                    if column not in _PRIOR_ID_COLUMNS
                },
                metrics_source=metrics_source,
            )
        )
    return normalized


def merge_rows(
    machine_rows: Sequence[dict[str, object]],
    human_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Junta máquina (servidor) e humano (repositório de semântica).

    Raises:
        ImportError_: Se as duas planilhas não tiverem as mesmas métricas
    """
    if machine_rows and human_rows and set(machine_rows[0]) != set(human_rows[0]):
        difference = sorted(set(machine_rows[0]) ^ set(human_rows[0]))
        raise ImportError_(f"as planilhas divergem nas colunas: {difference}")
    logger.info(
        f"{len(machine_rows)} linhas de máquina do servidor, {len(human_rows)} "
        "humanas do repositório de semântica"
    )
    return [*machine_rows, *human_rows]


def _relative(path: Path) -> str:
    """Caminho relativo à pasta dos repositórios, para a coluna de procedência."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.parent).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_raw(path: Path) -> list[dict[str, str]]:
    """Lê a planilha do servidor (o cabeçalho traz espaços soltos)."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Grava a planilha normalizada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Gravado {path} ({len(rows)} linhas)")


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da importação."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("raw", type=Path, help="CSV devolvido pelo servidor do NILC")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="paper_replication",
        help="Experimento a que os textos pertencem (default: paper_replication)",
    )
    parser.add_argument(
        "--human-metrics",
        type=Path,
        default=DEFAULT_HUMAN_METRICS,
        help=(
            "Planilha humana do repositório de semântica "
            f"(default: {DEFAULT_HUMAN_METRICS})"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.human_metrics.exists():
        logger.error(
            f"Planilha humana não encontrada: {args.human_metrics}. Clone o "
            "repositório de semântica ao lado deste ou passe --human-metrics"
        )
        return 1

    try:
        texts = corpus_texts(args.experiment)
        rows = merge_rows(
            normalize(
                read_raw(args.raw),
                experiment=args.experiment,
                generators=discover_generators(EXPERIMENTS[args.experiment]),
                texts=texts,
                metrics_source=_relative(args.raw),
            ),
            prior_human_rows(
                read_raw(args.human_metrics),
                experiment=args.experiment,
                texts=texts,
                metrics_source=_relative(args.human_metrics),
            ),
        )
    except ImportError_ as err:
        logger.error(f"Importação abortada: {err}")
        return 1

    write_rows(rows, args.output)
    mismatched = [row for row in rows if not row["text_matches"]]
    logger.info(
        f"{len(rows)} textos, {len(mismatched)} com texto diferente do analisado"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
