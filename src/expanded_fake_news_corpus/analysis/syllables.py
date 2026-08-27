"""Contagem de sílabas por palavra e por sentença.

Mede o quanto as palavras escolhidas por cada autoria são longas. No corpus
anterior essa foi uma das diferenças mais estáveis entre humano e máquina, e a
heurística de silabação aqui é a mesma usada lá, para os resultados seguirem
comparáveis.

Uso::

    python -m expanded_fake_news_corpus.analysis.syllables
    python -m expanded_fake_news_corpus.analysis.syllables --source fakebr
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    Group,
    Source,
    load_paired_corpus,
)
from expanded_fake_news_corpus.analysis.nltk_resources import (
    TOKENIZER_PACKAGES,
    ensure_nltk_resources,
)
from expanded_fake_news_corpus.analysis.significance import (
    compare_frame,
    compare_within,
    comparisons_to_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "syllables"

#: Métricas contínuas comparadas entre humano e máquina.
COMPARED_METRICS = ("syllables_per_sentence", "syllables_per_word")

#: Vogais do português. O "ç" é consoante e fica de fora de propósito.
VOWELS = "aeiouáàâãéèêíîóôõú"

#: Dígrafos que podem formar hiato quando precedidos de consoante.
POSSIBLE_HIATUS = ("ia", "ie", "io", "ua", "ue", "ui", "uo")

_NON_LETTER_RE = re.compile(r"[^a-záàâãéèêíîóôõúç]")


@dataclass(frozen=True)
class SyllableStats:
    """Estatísticas de silabação de um documento."""

    sentences: int
    words: int
    syllables: int
    syllables_per_sentence: float
    syllables_per_word: float


def count_syllables(word: str) -> int:
    """Conta as sílabas de uma palavra em português.

    Cada grupo de vogais contíguas vale uma sílaba, exceto quando o grupo é um
    dos dígrafos de :data:`POSSIBLE_HIATUS` precedido de consoante — aí o par se
    desfaz em duas sílabas.

    Args:
        word: Palavra a silabar

    Returns:
        Número de sílabas; 0 se não sobrar nenhuma letra
    """
    letters = _NON_LETTER_RE.sub("", word.casefold().strip())
    if not letters:
        return 0

    syllables = 0
    index = 0
    while index < len(letters):
        if letters[index] in VOWELS:
            syllables += 1
            while index + 1 < len(letters) and letters[index + 1] in VOWELS:
                if _is_hiatus(letters, index):
                    break
                index += 1
        index += 1

    return max(syllables, 1)


def analyze_text(text: str) -> SyllableStats | None:
    """Calcula as estatísticas de silabação de um texto.

    Args:
        text: Texto da notícia

    Returns:
        Estatísticas do documento, ou None se não houver sentença com palavras
    """
    sentences = sent_tokenize(text, language="portuguese")
    if not sentences:
        return None

    syllables_by_sentence: list[int] = []
    total_syllables = 0
    total_words = 0

    for sentence in sentences:
        # NOTA: sem language= aqui, como no pipeline anterior — trocar o
        # tokenizador mudaria a contagem e quebraria a comparabilidade.
        words = [
            token for token in word_tokenize(sentence.casefold()) if token.isalpha()
        ]

        sentence_syllables = sum(count_syllables(word) for word in words)
        total_syllables += sentence_syllables
        total_words += len(words)
        if sentence_syllables > 0:
            syllables_by_sentence.append(sentence_syllables)

    if not syllables_by_sentence:
        return None

    return SyllableStats(
        sentences=len(sentences),
        words=total_words,
        syllables=total_syllables,
        syllables_per_sentence=total_syllables / len(syllables_by_sentence),
        syllables_per_word=total_syllables / total_words if total_words else 0.0,
    )


def build_document_frame(documents: Sequence[Document]) -> pd.DataFrame:
    """Monta a tabela de silabação com uma linha por documento.

    Args:
        documents: Documentos pareados do corpus

    Returns:
        DataFrame com identificação do documento e as métricas de silabação
    """
    rows: list[dict[str, object]] = []
    for document in documents:
        stats = analyze_text(document.text)
        if stats is None:
            logger.warning(f"Documento sem sentenças analisáveis: {document.uid}")
            continue
        rows.append(
            {
                "uid": document.uid,
                "source": document.source.value,
                "group": document.group.value,
                "dataset": document.dataset,
                "model": document.model,
                **asdict(stats),
            }
        )

    logger.info(f"Silabação calculada para {len(rows)} documentos")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Agrega as métricas por uma coluna categórica.

    Args:
        frame: Tabela por documento
        by: Coluna de agrupamento (``dataset``, ``group``, ``model``, ...)

    Returns:
        Totais e médias por grupo. ``syllables_per_word`` é recalculada a partir
        dos totais, e não como média das médias por documento.
    """
    grouped = frame.groupby(by).agg(
        documents=("uid", "count"),
        sentences=("sentences", "sum"),
        words=("words", "sum"),
        syllables=("syllables", "sum"),
        syllables_per_sentence_mean=("syllables_per_sentence", "mean"),
        syllables_per_sentence_std=("syllables_per_sentence", "std"),
        syllables_per_word_mean=("syllables_per_word", "mean"),
        syllables_per_word_std=("syllables_per_word", "std"),
    )
    grouped["syllables_per_word_weighted"] = grouped["syllables"] / grouped["words"]
    return grouped.reset_index()


def run_analysis(documents: Sequence[Document], output_dir: Path) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos pareados do corpus
        output_dir: Pasta de saída dos CSVs

    Returns:
        Tabela por documento
    """
    frame = build_document_frame(documents)
    if frame.empty:
        logger.warning("Nenhum documento analisável — nada a gravar")
        return frame

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame, output_dir / "syllables_per_document.csv")
    _write_csv(summarize(frame, "dataset"), output_dir / "syllables_by_dataset.csv")
    _write_csv(summarize(frame, "group"), output_dir / "syllables_by_group.csv")

    comparisons = comparisons_to_frame(compare_frame(frame, COMPARED_METRICS))
    _write_csv(comparisons, output_dir / "syllables_significance.csv")
    _write_csv(
        compare_within(frame, COMPARED_METRICS, split_column="source"),
        output_dir / "syllables_significance_by_source.csv",
    )

    return frame


def _is_hiatus(letters: str, index: int) -> bool:
    """Indica se o par de vogais em ``index`` se desfaz em duas sílabas."""
    return (
        letters[index : index + 2] in POSSIBLE_HIATUS
        and index > 0
        and letters[index - 1] not in VOWELS
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Grava um DataFrame, avisando quando não há nada para gravar."""
    if frame.empty:
        logger.warning(f"Tabela vazia, arquivo não gerado: {path.name}")
        return
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Gravado {path} ({len(frame)} linhas)")


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada da análise de silabação."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    ensure_nltk_resources(*TOKENIZER_PACKAGES)

    sources = [Source(value) for value in args.source] if args.source else None
    documents = load_paired_corpus(sources=sources, models=args.model)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    frame = run_analysis(documents, args.output_dir)
    if frame.empty:
        return

    summary = summarize(frame, "group").set_index("group")
    for group in (Group.HUMAN, Group.MACHINE):
        if group.value not in summary.index:
            continue
        row = summary.loc[group.value]
        logger.info(
            f"{group.value}: {int(row['documents'])} documentos, "
            f"{row['syllables_per_word_weighted']:.4f} sílabas/palavra, "
            f"{row['syllables_per_sentence_mean']:.2f} sílabas/sentença"
        )


if __name__ == "__main__":
    main()
