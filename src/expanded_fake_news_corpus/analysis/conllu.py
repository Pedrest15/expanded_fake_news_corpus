"""Camada de corpus parseado: onde ficam os CoNLL-U e como lê-los.

O parsing sintático é caro (Portparser v2 sobre BERTimbau) e é feito uma vez
só, por :mod:`expanded_fake_news_corpus.parsing.portparser`, que grava um
arquivo por documento em::

    data/parsed/<experimento>/<grupo>/<stem>.conllu

onde ``<grupo>`` é ``human`` ou ``machine`` e ``<stem>`` é o ``uid`` com o
``:`` trocado por ``_`` (``fakebr:7`` -> ``fakebr_7``). As análises que
dependem de POS ou de sintaxe leem daqui, e nunca chamam o parser.

Este módulo é só o formato: localizar, ler e percorrer sentenças e tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    Group,
)

logger = logging.getLogger(__name__)

DEFAULT_PARSED_ROOT = PROJECT_ROOT / "data" / "parsed"

#: Colunas de uma linha de token no CoNLL-U.
CONLLU_FIELDS = (
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)


def parsed_dir(experiment: str, root: Path = DEFAULT_PARSED_ROOT) -> Path:
    """Pasta dos CoNLL-U de um experimento."""
    return root / experiment


def document_stem(uid: str) -> str:
    """Nome de arquivo de um documento: ``fakebr:7`` -> ``fakebr_7``."""
    return uid.replace(":", "_")


def conllu_path(
    document: Document,
    experiment: str,
    root: Path = DEFAULT_PARSED_ROOT,
    *,
    variant: str = "",
) -> Path:
    """Caminho do CoNLL-U de um documento do corpus pareado.

    Args:
        document: Documento do corpus pareado
        experiment: Nome do experimento
        root: Raiz dos CoNLL-U
        variant: Anotação derivada, gravada ao lado da básica com o nome
            ``<stem>.<variant>.conllu`` — ``"eud"`` para a versão com
            Enhanced Universal Dependencies (:mod:`parsing.eud`)
    """
    suffix = f".{variant}.conllu" if variant else ".conllu"
    return (
        parsed_dir(experiment, root)
        / document.group.value
        / f"{document_stem(document.uid)}{suffix}"
    )


@dataclass(frozen=True)
class Token:
    """Uma linha de token do CoNLL-U.

    ``id`` fica como string porque contrações vêm em intervalo (``8-9``) e
    palavras vazias em decimal (``8.1``); :attr:`is_word` separa os tokens
    sintáticos — os que têm HEAD e DEPREL — dos demais.
    """

    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str

    @property
    def is_word(self) -> bool:
        """Token sintático (não é intervalo de contração nem nó vazio)."""
        return self.id.isdigit()

    @property
    def index(self) -> int:
        """Posição inteira do token (só para :attr:`is_word`)."""
        return int(self.id)

    @property
    def head_index(self) -> int:
        """HEAD como inteiro; ``0`` é a raiz, ``-1`` quando ausente."""
        return int(self.head) if self.head.isdigit() else -1

    def enhanced_deps(self) -> list[tuple[int, str]]:
        """Arestas da coluna DEPS.

        ``"2:nsubj|6:nmod:de"`` vira ``[(2, "nsubj"), (6, "nmod:de")]``.
        Núcleos de nós vazios (``8.1``) não têm índice inteiro e são ignorados.
        """
        if not self.deps or self.deps == "_":
            return []
        edges: list[tuple[int, str]] = []
        for part in self.deps.split("|"):
            head, _, deprel = part.partition(":")
            if head.isdigit():
                edges.append((int(head), deprel or "_"))
        return edges


@dataclass
class Sentence:
    """Uma sentença anotada: comentários e tokens."""

    sent_id: str = ""
    text: str = ""
    comments: list[str] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)

    @property
    def words(self) -> list[Token]:
        """Só os tokens sintáticos, na ordem."""
        return [token for token in self.tokens if token.is_word]

    def to_conllu(self) -> str:
        """Serializa a sentença de volta ao formato, com linha em branco final."""
        lines = list(self.comments)
        for token in self.tokens:
            lines.append("\t".join(getattr(token, name) for name in CONLLU_FIELDS))
        return "\n".join(lines) + "\n\n"


def parse_conllu(text: str) -> list[Sentence]:
    """Lê sentenças de um CoNLL-U em memória.

    Args:
        text: Conteúdo do arquivo

    Returns:
        Sentenças na ordem do arquivo; linhas malformadas são ignoradas com aviso
    """
    sentences: list[Sentence] = []
    current = Sentence()

    def flush() -> None:
        nonlocal current
        if current.tokens or current.comments:
            sentences.append(current)
        current = Sentence()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            current.comments.append(line)
            key, _, value = line[1:].partition("=")
            key = key.strip()
            if key == "sent_id":
                current.sent_id = value.strip()
            elif key == "text":
                current.text = value.strip()
            continue
        fields = line.split("\t")
        if len(fields) != len(CONLLU_FIELDS):
            logger.warning(
                f"Linha CoNLL-U com {len(fields)} colunas ignorada: {line[:60]!r}"
            )
            continue
        current.tokens.append(Token(*fields))
    flush()
    return sentences


def read_conllu(path: Path) -> list[Sentence]:
    """Lê as sentenças de um arquivo CoNLL-U."""
    return parse_conllu(path.read_text(encoding="utf-8"))


def write_conllu(path: Path, sentences: Sequence[Sentence]) -> None:
    """Grava sentenças num arquivo CoNLL-U."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.to_conllu() for s in sentences), encoding="utf-8")


@dataclass(frozen=True)
class ParsedDocument:
    """Um documento do corpus pareado com sua anotação sintática."""

    document: Document
    sentences: list[Sentence]

    @property
    def uid(self) -> str:
        return self.document.uid

    @property
    def group(self) -> Group:
        return self.document.group

    @property
    def dataset(self) -> str:
        return self.document.dataset

    def words(self) -> Iterator[Token]:
        """Todos os tokens sintáticos do documento, sentença a sentença."""
        for sentence in self.sentences:
            yield from sentence.words


def load_parsed_documents(
    documents: Sequence[Document],
    experiment: str,
    root: Path = DEFAULT_PARSED_ROOT,
    *,
    variant: str = "",
) -> list[ParsedDocument]:
    """Anexa a cada documento o seu CoNLL-U.

    Documentos sem arquivo parseado são descartados com aviso — e o par deles
    também, porque a comparação é pareada por ``uid``.

    Args:
        documents: Corpus pareado (:func:`load_paired_corpus`)
        experiment: Nome do experimento (pasta em ``data/parsed``)
        root: Raiz dos CoNLL-U
        variant: Anotação derivada a ler (ver :func:`conllu_path`)

    Returns:
        Documentos parseados, na mesma ordem da entrada
    """
    parsed: dict[str, ParsedDocument] = {}
    missing: list[str] = []
    for document in documents:
        path = conllu_path(document, experiment, root, variant=variant)
        if not path.is_file():
            missing.append(f"{document.uid}/{document.group.value}")
            continue
        parsed[f"{document.uid}/{document.group.value}"] = ParsedDocument(
            document, read_conllu(path)
        )

    if missing:
        preview = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
        logger.warning(
            f"{len(missing)} documentos sem CoNLL-U em {parsed_dir(experiment, root)} "
            f"(rode parsing.portparser{' e parsing.eud' if variant == 'eud' else ''}): "
            f"{preview}"
        )

    complete = {
        uid
        for uid in {d.uid for d in documents}
        if f"{uid}/{Group.HUMAN.value}" in parsed
        and f"{uid}/{Group.MACHINE.value}" in parsed
    }
    dropped = {d.uid for d in documents} - complete
    if dropped:
        logger.warning(f"Descartados {len(dropped)} uids sem os dois lados parseados")

    result = [
        parsed[f"{d.uid}/{d.group.value}"] for d in documents if d.uid in complete
    ]
    logger.info(f"Corpus parseado: {len(result)} documentos de {len(complete)} pares")
    return result
