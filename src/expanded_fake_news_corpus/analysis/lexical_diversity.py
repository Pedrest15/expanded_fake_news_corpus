"""Diversidade lexical: quanto vocabulário distinto cada autoria gasta.

A razão tipo/ocorrência bruta não serve para comparar textos de tamanhos
diferentes. Ela cai conforme o texto cresce — o vocabulário satura enquanto o
total de palavras continua subindo — então um texto mais longo parece menos
diverso mesmo quando não é. Como as notícias sintéticas são sistematicamente
mais longas que as humanas, a razão bruta mediria comprimento disfarçado de
diversidade.

O MATTR (*moving-average type-token ratio*, Covington & McFall 2010) resolve
isso medindo a razão numa janela deslizante de tamanho fixo e tirando a média:
toda janela tem o mesmo denominador, então o valor não depende do tamanho do
texto. A razão bruta continua na tabela, para o contraste ficar visível.

Uso::

    python -m expanded_fake_news_corpus.analysis.lexical_diversity
    python -m expanded_fake_news_corpus.analysis.lexical_diversity --window 40
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    Group,
    add_corpus_arguments,
    documents_from_args,
    output_dir_from_args,
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
from expanded_fake_news_corpus.analysis.zipf import tokenize

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "lexical_diversity"

#: Janela do MATTR. 50 é o valor da formulação original; textos mais curtos que
#: ela não têm MATTR definido e ficam de fora, o que a execução registra.
DEFAULT_WINDOW = 50

#: Métricas comparadas. A razão bruta entra para o contraste, não como resultado.
COMPARED_METRICS = ("mattr", "type_token_ratio")


@dataclass(frozen=True)
class DiversityStats:
    """Medidas de diversidade lexical de um documento."""

    tokens: int
    types: int
    type_token_ratio: float
    mattr: float | None


def moving_average_ttr(
    tokens: Sequence[str],
    *,
    window: int = DEFAULT_WINDOW,
) -> float | None:
    """Média das razões tipo/ocorrência de todas as janelas deslizantes.

    Args:
        tokens: Sequência de tokens, na ordem do texto
        window: Tamanho da janela

    Returns:
        MATTR entre 0 e 1, ou None se o texto for mais curto que a janela

    Raises:
        ValueError: Se a janela não for positiva
    """
    if window <= 0:
        raise ValueError(f"window must be positive: {window}")
    if len(tokens) < window:
        return None

    counts = Counter(tokens[:window])
    distinct_total = len(counts)
    windows = 1

    for index in range(window, len(tokens)):
        leaving = tokens[index - window]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            del counts[leaving]
        counts[tokens[index]] += 1
        distinct_total += len(counts)
        windows += 1

    return distinct_total / (windows * window)


def analyze_text(text: str, *, window: int = DEFAULT_WINDOW) -> DiversityStats | None:
    """Calcula as medidas de diversidade de um texto.

    Args:
        text: Texto da notícia
        window: Janela do MATTR

    Returns:
        Medidas do documento, ou None se não sobrar nenhum token
    """
    # NOTA: sem o recorte herdado do Zipf — descartar "sobre"/"após"/"contra"
    # abriria buracos na sequência que a janela deslizante percorre.
    tokens = tokenize(text, discard_extra_tokens=False)
    if not tokens:
        return None

    return DiversityStats(
        tokens=len(tokens),
        types=len(set(tokens)),
        type_token_ratio=len(set(tokens)) / len(tokens),
        mattr=moving_average_ttr(tokens, window=window),
    )


def build_document_frame(
    documents: Sequence[Document],
    *,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Monta a tabela de diversidade com uma linha por documento.

    Args:
        documents: Documentos pareados do corpus
        window: Janela do MATTR

    Returns:
        DataFrame com identificação do documento e as medidas de diversidade
    """
    rows: list[dict[str, object]] = []
    for document in documents:
        stats = analyze_text(document.text, window=window)
        if stats is None:
            logger.warning(f"Documento sem tokens: {document.uid}")
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

    frame = pd.DataFrame(rows)
    _warn_about_short_documents(frame, window)
    return frame


def summarize(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Agrega as medidas por uma coluna categórica.

    Args:
        frame: Tabela por documento
        by: Coluna de agrupamento (``group``, ``dataset``, ...)

    Returns:
        Médias e totais por grupo. ``type_token_ratio_pooled`` é a razão do
        vocabulário somado — é ela que o comprimento distorce.
    """
    grouped = frame.groupby(by).agg(
        documents=("uid", "count"),
        tokens=("tokens", "sum"),
        mattr_mean=("mattr", "mean"),
        mattr_std=("mattr", "std"),
        mattr_documents=("mattr", "count"),
        type_token_ratio_mean=("type_token_ratio", "mean"),
    )
    pooled = frame.groupby(by).apply(_pooled_ratio, include_groups=False)
    grouped["type_token_ratio_pooled"] = pooled
    return grouped.reset_index()


def run_analysis(
    documents: Sequence[Document],
    output_dir: Path,
    *,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos pareados do corpus
        output_dir: Pasta de saída dos CSVs
        window: Janela do MATTR

    Returns:
        Tabela por documento
    """
    frame = build_document_frame(documents, window=window)
    if frame.empty:
        logger.warning("Nenhum documento analisável — nada a gravar")
        return frame

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame, output_dir / "lexical_diversity_per_document.csv")
    _write_csv(summarize(frame, "group"), output_dir / "lexical_diversity_by_group.csv")
    _write_csv(
        summarize(frame, "dataset"), output_dir / "lexical_diversity_by_dataset.csv"
    )
    _write_csv(
        comparisons_to_frame(compare_frame(frame, COMPARED_METRICS)),
        output_dir / "lexical_diversity_significance.csv",
    )
    _write_csv(
        compare_within(frame, COMPARED_METRICS, split_column="source"),
        output_dir / "lexical_diversity_significance_by_source.csv",
    )
    return frame


def _pooled_ratio(group: pd.DataFrame) -> float:
    """Razão tipo/ocorrência do grupo inteiro somado, não a média das razões."""
    tokens = group["tokens"].sum()
    return float(group["types"].sum() / tokens) if tokens else 0.0


def _warn_about_short_documents(frame: pd.DataFrame, window: int) -> None:
    """Avisa quando a janela descarta documentos, e se o descarte é assimétrico."""
    if frame.empty:
        return
    dropped = frame[frame["mattr"].isna()]
    if dropped.empty:
        logger.info(f"MATTR definido para todos os {len(frame)} documentos")
        return

    by_group = dropped["group"].value_counts().to_dict()
    logger.warning(
        f"{len(dropped)} documentos mais curtos que a janela ({window}) ficaram "
        f"sem MATTR: {by_group}. Uma janela menor os recupera."
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Grava um DataFrame, avisando quando não há nada para gravar."""
    if frame.empty:
        logger.warning(f"Tabela vazia, arquivo não gerado: {path.name}")
        return
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Gravado {path} ({len(frame)} linhas)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando (``sys.argv`` se ``argv`` for None)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Janela do MATTR em tokens (default: {DEFAULT_WINDOW})",
    )
    add_corpus_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada da análise de diversidade lexical."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    ensure_nltk_resources(*TOKENIZER_PACKAGES)

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    frame = run_analysis(documents, output_dir, window=args.window)
    if frame.empty:
        return

    summary = summarize(frame, "group").set_index("group")
    for group in (Group.HUMAN, Group.MACHINE):
        if group.value not in summary.index:
            continue
        row = summary.loc[group.value]
        logger.info(
            f"{group.value}: MATTR {row['mattr_mean']:.3f} "
            f"(tipo/ocorrência bruta {row['type_token_ratio_pooled']:.3f}, "
            f"{int(row['tokens'])} tokens)"
        )


if __name__ == "__main__":
    main()
