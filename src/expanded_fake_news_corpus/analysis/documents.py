"""Carregamento pareado dos documentos que alimentam as análises.

As notícias sintéticas ficam em ``corpus/fake_news/<round>/<provider>/<model>/*.jsonl``
(pipeline por manchete) ou em ``corpus/paper_replication/<provider>/<model>/*.jsonl``
(replicação do artigo, sem rounds) — ver :data:`EXPERIMENTS`.
A contraparte humana de cada uma é a *fake news* escrita por humano no corpus de
origem, resolvida pelo ``uid``:

- ``fakebr:1386``    -> ``true-corpus/raw/Fake.br-full_texts/fake/1386.txt``
- ``faketruebr:1683`` -> linha 1683 de ``true-corpus/FakeTrueBr_corpus.csv``

O pareamento por ``uid`` é o que sustenta a comparação: as duas versões de um
mesmo fato jornalístico entram sempre juntas, então diferença medida entre os
grupos é diferença de autoria, não de assunto.

NOTA: manchete e corpo são sempre unidos, dos dois lados. No corpus anterior o
lado humano trazia o título na primeira linha e o lado LLM não, o que enviesava
comprimento de sentença e densidade de PROPN. Aqui a composição é simétrica.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

#: O texto de uma notícia do FakeTrueBR estoura o limite default do módulo csv.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Repositório do trabalho anterior, de onde vêm as coortes de referência.
#: Sobrescrevível por ``FAKEGEN_PRIOR_CORPUS`` ou pelo argumento ``--prior-dir``.
PRIOR_CORPUS_ENV_VAR = "FAKEGEN_PRIOR_CORPUS"
DEFAULT_PRIOR_DIR = (
    PROJECT_ROOT.parent / "noticias_falsas_humano_maquina_caracterizacao" / "corpus"
)

#: Experimentos de geração, pela pasta onde cada um grava as notícias sintéticas.
#:
#: ``pipeline`` é o corpus principal (manchete -> fake news, por rounds);
#: ``paper_replication`` é a replicação do método de Silva et al. (notícia
#: inteira -> fake news). As análises rodam sobre um de cada vez, e os
#: resultados de cada experimento vão para uma pasta própria em
#: ``data/analysis`` — ver :func:`output_dir_from_args`.
EXPERIMENTS: dict[str, Path] = {
    "pipeline": PROJECT_ROOT / "corpus" / "fake_news",
    "paper_replication": PROJECT_ROOT / "corpus" / "paper_replication",
}
DEFAULT_EXPERIMENT = "pipeline"
DEFAULT_SYNTHETIC_DIR = EXPERIMENTS[DEFAULT_EXPERIMENT]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DEFAULT_FAKEBR_DIR = (
    PROJECT_ROOT / "true-corpus" / "raw" / "Fake.br-full_texts" / "fake"
)
DEFAULT_FAKETRUEBR_CSV = PROJECT_ROOT / "true-corpus" / "FakeTrueBr_corpus.csv"


class Group(StrEnum):
    """Autoria do documento."""

    HUMAN = "human"
    MACHINE = "machine"


class Source(StrEnum):
    """Corpus de origem da notícia."""

    FAKEBR = "fakebr"
    FAKETRUEBR = "faketruebr"


#: Coorte da *fake news* humana — a mesma nos dois repositórios.
HUMAN_COHORT = "human"

#: Coorte gerada pelo trabalho anterior (Maritalk/Maritaca).
SABIA3_COHORT = "sabia-3"

# NOTA: os .txt do FakeTrueBR no repositório anterior são indexados a partir de
# zero, enquanto nosso uid guarda a linha 1-based do CSV. Sem este ajuste o
# pareamento sairia deslocado em uma notícia — e silenciosamente, porque todo
# uid continuaria encontrando um arquivo, só que o errado.
_PRIOR_INDEX_OFFSET = {Source.FAKEBR: 0, Source.FAKETRUEBR: -1}

#: Subpastas do repositório anterior, por coorte e corpus de origem.
_PRIOR_SUBDIRS = {
    (HUMAN_COHORT, Source.FAKEBR): Path(
        "Fake.br-Corpus-master/full_texts/fake_br_clean"
    ),
    (HUMAN_COHORT, Source.FAKETRUEBR): Path("FakeTrue.Br-main/fake"),
    (SABIA3_COHORT, Source.FAKEBR): Path(
        "fake-news-llm-ptbr-main/fake-news-llm-ptbr-main/data/Fake.Br"
    ),
    (SABIA3_COHORT, Source.FAKETRUEBR): Path(
        "fake-news-llm-ptbr-main/fake-news-llm-ptbr-main/data/FakeTrueBR"
    ),
}


class CorpusError(Exception):
    """Falha ao montar o corpus de análise."""


class InvalidUidError(CorpusError):
    """``uid`` fora do formato ``<source>:<id>``."""


class MissingCounterpartError(CorpusError):
    """Não há *fake news* humana correspondente ao ``uid``."""


class MissingPriorCorpusError(CorpusError):
    """O corpus do trabalho anterior não está acessível."""


@dataclass(frozen=True)
class Document:
    """Uma notícia pronta para análise."""

    uid: str
    source: Source
    group: Group
    text: str
    model: str | None = None  # None do lado humano
    round_name: str | None = None
    cohort: str = ""  # vazio: derivado do grupo em __post_init__

    def __post_init__(self) -> None:
        """Deriva a coorte do modelo quando ela não foi informada."""
        if not self.cohort:
            derived = (
                HUMAN_COHORT if self.group is Group.HUMAN else (self.model or "machine")
            )
            object.__setattr__(self, "cohort", derived)

    @property
    def dataset(self) -> str:
        """Rótulo ``<source>_<group>`` usado para agrupar as tabelas de saída."""
        return f"{self.source.value}_{self.group.value}"


@dataclass(frozen=True)
class CorpusPaths:
    """Onde estão o corpus sintético e os corpora de origem."""

    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR
    fakebr_dir: Path = DEFAULT_FAKEBR_DIR
    faketruebr_csv: Path = DEFAULT_FAKETRUEBR_CSV


_NON_ALNUM_RE = re.compile(r"[^0-9a-zà-ÿ]+")


def parse_uid(uid: str) -> tuple[Source, str]:
    """Separa o ``uid`` em corpus de origem e identificador local.

    Args:
        uid: Identificador qualificado, no formato ``fakebr:1386``

    Returns:
        Par ``(source, local_id)``

    Raises:
        InvalidUidError: Se faltar o prefixo ou o corpus for desconhecido
    """
    prefix, _, local_id = uid.partition(":")
    if not local_id:
        raise InvalidUidError(f"missing source prefix: {uid!r}")
    try:
        return Source(prefix), local_id
    except ValueError as err:
        raise InvalidUidError(f"unknown source: {uid!r}") from err


def load_paired_corpus(
    paths: CorpusPaths | None = None,
    *,
    sources: Sequence[Source] | None = None,
    models: Sequence[str] | None = None,
) -> list[Document]:
    """Carrega as notícias sintéticas e suas contrapartes humanas.

    Só entram os ``uid`` com os dois lados presentes — um documento sem par
    desequilibraria a comparação e é descartado com aviso.

    Args:
        paths: Localização dos corpora; usa os caminhos do repositório se omitido
        sources: Restringe a estes corpora de origem (todos, se omitido)
        models: Restringe a estes modelos geradores (todos, se omitido)

    Returns:
        Documentos dos dois grupos, ordenados por ``uid``
    """
    paths = paths or CorpusPaths()
    machine = load_machine_documents(
        paths.synthetic_dir, sources=sources, models=models
    )
    if not machine:
        logger.warning(f"Nenhuma notícia sintética encontrada em {paths.synthetic_dir}")
        return []

    human = load_human_documents({doc.uid for doc in machine}, paths=paths)
    paired_uids = {doc.uid for doc in human}

    dropped = [doc.uid for doc in machine if doc.uid not in paired_uids]
    if dropped:
        logger.warning(
            f"Descartados {len(dropped)} documentos sem par humano: {dropped}"
        )

    documents = [doc for doc in machine if doc.uid in paired_uids] + human
    logger.info(
        f"Corpus pareado: {len(paired_uids)} notícias, "
        f"{len(documents)} documentos ({Group.HUMAN} + {Group.MACHINE})"
    )
    return sorted(documents, key=lambda doc: (doc.uid, doc.group.value))


def load_machine_documents(
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    *,
    sources: Sequence[Source] | None = None,
    models: Sequence[str] | None = None,
) -> list[Document]:
    """Lê as notícias sintéticas dos JSONL de saída da geração.

    Args:
        synthetic_dir: Raiz de ``corpus/fake_news``
        sources: Restringe a estes corpora de origem (todos, se omitido)
        models: Restringe a estes modelos geradores (todos, se omitido)

    Returns:
        Documentos do grupo ``machine``
    """
    if not synthetic_dir.is_dir():
        logger.warning(
            f"Diretório de notícias sintéticas não encontrado: {synthetic_dir}"
        )
        return []

    wanted_sources = set(sources) if sources else None
    wanted_models = set(models) if models else None

    documents: list[Document] = []
    for jsonl_path in sorted(synthetic_dir.rglob("*.jsonl")):
        round_name = _round_name(jsonl_path, synthetic_dir)
        for record in _read_jsonl(jsonl_path):
            document = _machine_document(record, round_name=round_name)
            if document is None:
                continue
            if wanted_sources and document.source not in wanted_sources:
                continue
            if wanted_models and document.model not in wanted_models:
                continue
            documents.append(document)

    logger.info(f"Lidas {len(documents)} notícias sintéticas de {synthetic_dir}")
    return documents


def load_human_documents(
    uids: Iterable[str],
    *,
    paths: CorpusPaths | None = None,
) -> list[Document]:
    """Resolve as *fake news* humanas correspondentes aos ``uid`` pedidos.

    Args:
        uids: Identificadores qualificados a resolver
        paths: Localização dos corpora de origem

    Returns:
        Documentos do grupo ``human``, sem os ``uid`` que não puderam ser resolvidos
    """
    paths = paths or CorpusPaths()
    faketruebr_rows: list[dict[str, str]] | None = None

    documents: list[Document] = []
    for uid in sorted(set(uids)):
        source, local_id = parse_uid(uid)
        if source is Source.FAKETRUEBR and faketruebr_rows is None:
            faketruebr_rows = _read_faketruebr_rows(paths.faketruebr_csv)

        try:
            if source is Source.FAKEBR:
                text = _read_fakebr_fake(paths.fakebr_dir, local_id)
            else:
                text = _read_faketruebr_fake(faketruebr_rows or [], local_id)
        except MissingCounterpartError as err:
            logger.warning(f"Sem par humano para {uid}: {err}")
            continue

        documents.append(Document(uid=uid, source=source, group=Group.HUMAN, text=text))

    logger.info(f"Resolvidas {len(documents)} fake news humanas")
    return documents


def find_prior_dir(explicit: Path | None = None) -> Path:
    """Resolve a raiz do corpus do trabalho anterior.

    Ordem: caminho explícito, variável :data:`PRIOR_CORPUS_ENV_VAR`, repositório
    irmão em :data:`DEFAULT_PRIOR_DIR`.

    Args:
        explicit: Caminho passado na linha de comando

    Returns:
        Raiz do corpus anterior

    Raises:
        MissingPriorCorpusError: Se o diretório não existir
    """
    candidate = explicit or Path(
        os.environ.get(PRIOR_CORPUS_ENV_VAR, DEFAULT_PRIOR_DIR)
    )
    if not candidate.is_dir():
        raise MissingPriorCorpusError(
            f"prior corpus not found: {candidate}. Clone the companion repository "
            f"or set {PRIOR_CORPUS_ENV_VAR}"
        )
    return candidate


def load_prior_cohort(
    cohort: str,
    *,
    prior_dir: Path | None = None,
    sources: Sequence[Source] | None = None,
    uids: Iterable[str] | None = None,
) -> list[Document]:
    """Carrega uma coorte do repositório do trabalho anterior.

    As coortes disponíveis são :data:`HUMAN_COHORT` (as *fake news* escritas por
    humanos) e :data:`SABIA3_COHORT` (as geradas por sabia-3). Os dois lados são
    pareados por ``uid``, então servem de linha de base para as gerações novas.

    Args:
        cohort: Nome da coorte
        prior_dir: Raiz do corpus anterior
        sources: Restringe a estes corpora de origem (todos, se omitido)
        uids: Restringe a estes identificadores (todos, se omitido)

    Returns:
        Documentos da coorte, ordenados por ``uid``

    Raises:
        MissingPriorCorpusError: Se a raiz ou uma subpasta não existir
    """
    root = find_prior_dir(prior_dir)
    wanted_uids = set(uids) if uids is not None else None
    group = Group.HUMAN if cohort == HUMAN_COHORT else Group.MACHINE

    documents: list[Document] = []
    for source in sources or list(Source):
        subdir = _PRIOR_SUBDIRS.get((cohort, source))
        if subdir is None:
            raise MissingPriorCorpusError(
                f"unknown cohort {cohort!r} for {source.value}"
            )

        directory = root / subdir
        if not directory.is_dir():
            raise MissingPriorCorpusError(f"directory not found: {directory}")

        offset = _PRIOR_INDEX_OFFSET[source]
        for path in directory.glob("*.txt"):
            if not path.stem.isdigit():
                continue
            uid = f"{source.value}:{int(path.stem) - offset}"
            if wanted_uids is not None and uid not in wanted_uids:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                logger.warning(f"Arquivo vazio na coorte {cohort}: {path}")
                continue
            documents.append(
                Document(
                    uid=uid,
                    source=source,
                    group=group,
                    text=text,
                    model=None if group is Group.HUMAN else cohort,
                    cohort=cohort,
                )
            )

    logger.info(f"Coorte {cohort}: {len(documents)} documentos de {root}")
    return sorted(documents, key=lambda doc: doc.uid)


def _machine_document(record: dict, *, round_name: str | None) -> Document | None:
    """Converte um registro do JSONL em documento, ou None se estiver incompleto."""
    uid = record.get("source_id", "")
    body = (record.get("synthetic_text") or "").strip()
    if not uid or not body:
        logger.warning(
            f"Registro sem source_id ou synthetic_text: {uid or '<sem uid>'}"
        )
        return None

    try:
        source, _ = parse_uid(uid)
    except InvalidUidError as err:
        logger.warning(f"Registro ignorado: {err}")
        return None

    return Document(
        uid=uid,
        source=source,
        group=Group.MACHINE,
        text=compose_news(record.get("source_headline", ""), body),
        model=record.get("model"),
        round_name=round_name,
    )


def compose_news(headline: str, body: str) -> str:
    """Junta manchete e corpo sem duplicar a manchete já presente no corpo.

    Args:
        headline: Manchete da notícia (pode vir vazia)
        body: Corpo da notícia

    Returns:
        Texto com a manchete na primeira linha
    """
    headline = headline.strip()
    body = body.strip()
    if not headline:
        return body
    if _alnum_only(body).startswith(_alnum_only(headline)):
        return body
    return f"{headline}\n\n{body}"


def _alnum_only(text: str) -> str:
    """Texto reduzido a letras e dígitos em caixa baixa.

    Compara manchete e início do corpo sem tropeçar em aspas, travessões ou
    caixa alta, que o modelo aplica de forma inconsistente.
    """
    return _NON_ALNUM_RE.sub("", text.casefold())


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Itera os registros de um JSONL, pulando linhas em branco."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                logger.warning(f"JSON inválido em {path}:{line_number}: {err}")


def _round_name(jsonl_path: Path, synthetic_dir: Path) -> str | None:
    """Extrai o round do caminho ``<round>/<provider>/<model>/arquivo.jsonl``.

    Sem o nível do round (``<provider>/<model>/arquivo.jsonl``, como na
    replicação do artigo) devolve None em vez de confundir o provedor com round.
    """
    relative = jsonl_path.relative_to(synthetic_dir)
    return relative.parts[0] if len(relative.parts) > 3 else None


def _read_fakebr_fake(fakebr_dir: Path, local_id: str) -> str:
    """Lê a fake news humana do Fake.br.

    O arquivo já traz a manchete na primeira sentença, então não há o que compor.
    """
    path = fakebr_dir / f"{local_id}.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as err:
        raise MissingCounterpartError(f"file not found: {path}") from err
    if not text:
        raise MissingCounterpartError(f"empty file: {path}")
    return text


def _read_faketruebr_fake(rows: Sequence[dict[str, str]], local_id: str) -> str:
    """Lê a fake news humana do FakeTrueBR pela linha do CSV (1-based)."""
    try:
        index = int(local_id) - 1
    except ValueError as err:
        raise MissingCounterpartError(f"non-numeric row: {local_id!r}") from err
    if not 0 <= index < len(rows):
        raise MissingCounterpartError(f"row out of range: {local_id}")

    row = rows[index]
    text = compose_news(row.get("title_fake", ""), row.get("fake", ""))
    if not text:
        raise MissingCounterpartError(f"empty row: {local_id}")
    return text


def _read_faketruebr_rows(csv_path: Path) -> list[dict[str, str]]:
    """Carrega o CSV do FakeTrueBR inteiro — resolver linha a linha seria pior."""
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as err:
        raise MissingCounterpartError(f"file not found: {csv_path}") from err
    logger.info(f"FakeTrueBR carregado: {len(rows)} linhas de {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Linha de comando compartilhada pelos módulos de análise
# ---------------------------------------------------------------------------


def add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    """Acrescenta ``--experiment``, ``--source`` e ``--model`` a um parser.

    Os módulos de análise recebem os mesmos recortes de corpus; defini-los num
    lugar só garante que ``--experiment`` signifique a mesma coisa em todos.
    """
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=DEFAULT_EXPERIMENT,
        help=(
            "Experimento de geração a analisar: 'pipeline' (corpus/fake_news) "
            f"ou 'paper_replication' (corpus/paper_replication). "
            f"Default: {DEFAULT_EXPERIMENT}"
        ),
    )
    parser.add_argument(
        "--source",
        choices=[source.value for source in Source],
        action="append",
        help="Restringe a um corpus de origem (repetível)",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Restringe a um modelo gerador (repetível)",
    )


def documents_from_args(
    args: argparse.Namespace, *, paths: CorpusPaths | None = None
) -> list[Document]:
    """Carrega o corpus pareado conforme :func:`add_corpus_arguments`.

    Args:
        args: Namespace com ``experiment``, ``source`` e ``model``
        paths: Corpora de origem; a pasta sintética é trocada pela do experimento

    Returns:
        Documentos dos dois grupos, ordenados por ``uid``
    """
    sources = [Source(value) for value in args.source] if args.source else None
    paths = replace(paths or CorpusPaths(), synthetic_dir=EXPERIMENTS[args.experiment])
    return load_paired_corpus(paths, sources=sources, models=args.model)


def output_dir_from_args(args: argparse.Namespace, default: Path) -> Path:
    """Resolve a pasta de saída de um módulo de análise.

    ``--output-dir`` explícito vence. Sem ele, o experimento principal grava
    na pasta padrão do módulo (``data/analysis/<módulo>``) e os demais em
    ``data/analysis/<experimento>/<módulo>``, para uma execução nunca
    sobrescrever as tabelas da outra.
    """
    if args.output_dir != default:
        return args.output_dir
    if args.experiment == DEFAULT_EXPERIMENT:
        return default
    return DEFAULT_ANALYSIS_DIR / args.experiment / default.name
