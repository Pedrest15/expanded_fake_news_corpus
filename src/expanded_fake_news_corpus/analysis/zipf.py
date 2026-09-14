"""Distribuição de Zipf do vocabulário, por autoria e por corpus de origem.

Gera a tabela rank-frequência de cada recorte do corpus, as curvas log-log e
linear, e a comparação lado a lado das palavras mais frequentes. A análise roda
em duas variantes — com e sem *stopwords* — porque a lista de mais frequentes é
dominada por palavras funcionais, e é a variante sem elas que mostra o
vocabulário de conteúdo que cada autoria prefere.

Uso::

    python -m expanded_fake_news_corpus.analysis.zipf
    python -m expanded_fake_news_corpus.analysis.zipf --top-n 30
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: os gráficos vão direto para arquivo

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

from expanded_fake_news_corpus.analysis.documents import (  # noqa: E402
    PROJECT_ROOT,
    Document,
    add_corpus_arguments,
    documents_from_args,
    output_dir_from_args,
)
from expanded_fake_news_corpus.analysis.nltk_resources import (  # noqa: E402
    TOKENIZER_PACKAGES,
    ensure_nltk_resources,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "zipf"

#: Quantas palavras entram na tabela comparativa por default.
DEFAULT_TOP_N = 20

# NOTA: herdado do pipeline anterior, que descartava estes tokens junto com a
# pontuação. "r" vem de "R$" partido pelo tokenizador; as três preposições são um
# recorte manual. Mantido para os resultados seguirem comparáveis com o corpus
# anterior — mexer aqui muda a lista de mais frequentes.
EXTRA_DISCARDED_TOKENS = frozenset({"r", "sobre", "após", "contra"})

# NOTA: o pipeline anterior enumerava a pontuação a descartar e deixava passar as
# variações fora da lista (".." era a nona palavra mais frequente do lado humano).
# Exigir ao menos um caractere alfanumérico cobre todas elas.
_HAS_ALNUM_RE = re.compile(r"[0-9a-zà-ÿ]")


def tokenize(
    text: str,
    *,
    ignored_words: frozenset[str] = frozenset(),
    discard_extra_tokens: bool = True,
) -> list[str]:
    """Quebra o texto em tokens de palavra, em caixa baixa.

    Descarta pontuação, números e, por padrão, os tokens de
    :data:`EXTRA_DISCARDED_TOKENS`.

    Args:
        text: Texto da notícia
        ignored_words: Palavras adicionais a descartar (as *stopwords*, tipicamente)
        discard_extra_tokens: Aplica o recorte herdado do pipeline anterior. Ligado
            para o ranking de frequências seguir comparável; desligue para medidas
            que precisam da sequência íntegra de palavras

    Returns:
        Tokens na ordem em que aparecem
    """
    extra = EXTRA_DISCARDED_TOKENS if discard_extra_tokens else frozenset()
    tokens = word_tokenize(text.casefold(), language="portuguese")
    return [
        token
        for token in tokens
        if _HAS_ALNUM_RE.search(token)
        and token not in extra
        and token not in ignored_words
        and not token.isnumeric()
    ]


def build_frequency_table(
    documents: Sequence[Document],
    *,
    ignored_words: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Monta a tabela rank-frequência de um recorte do corpus.

    Args:
        documents: Documentos do recorte
        ignored_words: Palavras a descartar antes de contar

    Returns:
        DataFrame com ``rank``, ``frequency`` e ``word``, do mais para o menos frequente
    """
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize(document.text, ignored_words=ignored_words))

    ranked = counter.most_common()
    return pd.DataFrame(
        [
            {"rank": rank, "frequency": frequency, "word": word}
            for rank, (word, frequency) in enumerate(ranked, start=1)
        ]
    )


def group_documents(documents: Sequence[Document]) -> dict[str, list[Document]]:
    """Agrupa os documentos nos recortes analisados.

    São os quatro cruzamentos corpus × autoria (``fakebr_human``, ...) mais os
    dois agregados por autoria (``human``, ``machine``).

    Args:
        documents: Documentos pareados do corpus

    Returns:
        Recorte -> documentos, em ordem estável
    """
    groups: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        groups[document.dataset].append(document)
        groups[document.group.value].append(document)
    return dict(sorted(groups.items()))


def plot_zipf(
    frame: pd.DataFrame, label: str, output_path: Path, *, log_scale: bool
) -> None:
    """Grava a curva rank-frequência de um recorte.

    Args:
        frame: Tabela de :func:`build_frequency_table`
        label: Nome do recorte, usado no título
        output_path: Arquivo PNG de saída
        log_scale: Eixos log-log (a forma em que a lei de Zipf vira uma reta)
    """
    scale = "log-log" if log_scale else "linear"
    figure, axes = plt.subplots(figsize=(8, 6))

    if log_scale:
        axes.loglog(frame["rank"], frame["frequency"], marker=".", linestyle="none")
    else:
        axes.plot(
            frame["rank"], frame["frequency"], marker="o", markersize=3, linewidth=1
        )

    axes.set_title(f"Distribuição de Zipf ({scale}) — {label}")
    axes.set_xlabel("Rank da palavra")
    axes.set_ylabel("Frequência")
    axes.grid(True, linestyle="--", linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"Gravado {output_path}")


def compare_top_words(
    tables: Mapping[str, pd.DataFrame],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Põe as palavras mais frequentes de cada recorte lado a lado.

    Args:
        tables: Recorte -> tabela rank-frequência
        top_n: Quantas posições comparar

    Returns:
        DataFrame com ``rank`` e um par de colunas palavra/frequência por recorte
    """
    comparison = pd.DataFrame({"rank": range(1, top_n + 1)})
    for label, frame in tables.items():
        head = frame.head(top_n).reset_index(drop=True)
        comparison[f"{label}_word"] = head["word"]
        comparison[f"{label}_frequency"] = head["frequency"]
    return comparison


def run_analysis(
    documents: Sequence[Document],
    output_dir: Path,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> None:
    """Roda as duas variantes da análise e grava tabelas e gráficos.

    Args:
        documents: Documentos pareados do corpus
        output_dir: Pasta raiz de saída
        top_n: Quantas palavras entram na tabela comparativa
    """
    portuguese_stopwords = frozenset(stopwords.words("portuguese"))
    variants = {
        "with_stopwords": frozenset(),
        "without_stopwords": portuguese_stopwords,
    }

    groups = group_documents(documents)
    for variant, ignored_words in variants.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        tables: dict[str, pd.DataFrame] = {}
        for label, group_docs in groups.items():
            frame = build_frequency_table(group_docs, ignored_words=ignored_words)
            if frame.empty:
                logger.warning(f"Recorte sem tokens: {variant}/{label}")
                continue

            tables[label] = frame
            frame.to_csv(
                variant_dir / f"{label}_frequencies.csv", index=False, encoding="utf-8"
            )
            logger.info(
                f"{variant}/{label}: {len(group_docs)} documentos, "
                f"{frame['frequency'].sum()} tokens, {len(frame)} tipos"
            )
            plot_zipf(
                frame,
                f"{label} ({variant})",
                variant_dir / "plots" / f"{label}_zipf_loglog.png",
                log_scale=True,
            )
            plot_zipf(
                frame,
                f"{label} ({variant})",
                variant_dir / "plots" / f"{label}_zipf_linear.png",
                log_scale=False,
            )

        if tables:
            comparison = compare_top_words(tables, top_n=top_n)
            comparison.to_csv(
                variant_dir / "top_words_comparison.csv", index=False, encoding="utf-8"
            )
            logger.info(f"Gravado {variant_dir / 'top_words_comparison.csv'}")


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_corpus_arguments(parser)
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Palavras na tabela comparativa (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada da análise de Zipf."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    ensure_nltk_resources(*TOKENIZER_PACKAGES, "stopwords")

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    run_analysis(documents, output_dir, top_n=args.top_n)
    recuts = len(group_documents(documents))
    logger.info(f"Análise de Zipf concluída: {recuts} recortes em {output_dir}")


if __name__ == "__main__":
    main()
