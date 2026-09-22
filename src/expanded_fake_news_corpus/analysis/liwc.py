"""Perfil psicolinguístico LIWC das duas autorias.

Pontua cada documento nas categorias do LIWC — percentual de palavras que caem
em cada categoria — e compara humano contra máquina categoria a categoria.

A estatística por categoria espelha o módulo ``liwc/`` do repositório
``noticias_falsas_humano_maquina_semantica``, para que os resultados das duas
bases sejam lidos com o mesmo critério: Mann-Whitney bilateral, d de Cohen pela
média das variâncias e corte no p bruto (``significant_005``). O q-valor de
Benjamini-Hochberg continua na tabela como coluna informativa — com 74 testes
a 5%, umas quatro categorias "significativas" saem do acaso puro —, mas não é
o critério. Veja :func:`compare_categories`.

O dicionário LIWC é proprietário e não acompanha o repositório. Aponte para ele
com ``--dictionary``, com a variável ``FAKEGEN_LIWC_DICTIONARY`` ou deixando o
arquivo em ``resources/liwc/``.

Uso::

    python -m expanded_fake_news_corpus.analysis.liwc
    python -m expanded_fake_news_corpus.analysis.liwc --dictionary /caminho/pt.dic
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    Group,
    add_corpus_arguments,
    documents_from_args,
    output_dir_from_args,
)
from expanded_fake_news_corpus.analysis.liwc_dictionary import (
    LiwcDictionary,
    LiwcDictionaryError,
    load_liwc_dictionary,
)
from expanded_fake_news_corpus.analysis.significance import (
    add_fdr_correction,
    interpret_effect_size,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "liwc"

#: Onde procurar o ``.dic`` quando nada é passado (a pasta não vai para o git).
DEFAULT_DICTIONARY_DIR = PROJECT_ROOT / "resources" / "liwc"

#: Variável de ambiente que aponta o dicionário.
DICTIONARY_ENV_VAR = "FAKEGEN_LIWC_DICTIONARY"

#: Colunas de identificação do documento, que não são métricas.
_IDENTITY_COLUMNS = ("uid", "source", "group", "dataset", "model")

# NOTA: contagens brutas ficam no CSV por documento mas fora da comparação — a
# tabela é de percentuais, e o tamanho do documento já é medido em `syllables`.
_RAW_COUNT_COLUMNS = ("word_count", "matched_words")


def find_dictionary_path(explicit: Path | None = None) -> Path:
    """Resolve o caminho do dicionário LIWC.

    A ordem é: caminho explícito, variável de ambiente, primeiro ``.dic``
    encontrado em :data:`DEFAULT_DICTIONARY_DIR`.

    Args:
        explicit: Caminho passado na linha de comando

    Returns:
        Caminho do dicionário

    Raises:
        LiwcDictionaryError: Se nenhum dicionário for encontrado
    """
    if explicit is not None:
        return explicit

    from_env = os.environ.get(DICTIONARY_ENV_VAR)
    if from_env:
        return Path(from_env)

    candidates = sorted(DEFAULT_DICTIONARY_DIR.glob("*.dic"))
    if candidates:
        return candidates[0]

    raise LiwcDictionaryError(
        f"no LIWC dictionary found. Pass --dictionary, set {DICTIONARY_ENV_VAR}, "
        f"or place a .dic file under {DEFAULT_DICTIONARY_DIR}"
    )


def score_document(
    dictionary: LiwcDictionary,
    document: Document,
) -> dict[str, object] | None:
    """Pontua um documento nas categorias LIWC.

    Args:
        dictionary: Dicionário carregado
        document: Documento a pontuar

    Returns:
        Linha de tabela — identificação, contagens e o percentual de cada
        categoria — ou None se o documento não tiver nenhuma palavra
    """
    categorization = dictionary.categorize_text(document.text)
    if categorization.total_words == 0:
        return None

    row: dict[str, object] = {
        "uid": document.uid,
        "source": document.source.value,
        "group": document.group.value,
        "dataset": document.dataset,
        "model": document.model,
        "word_count": categorization.total_words,
        "matched_words": categorization.matched_words,
        "dictionary_coverage": categorization.coverage,
    }
    for name in dictionary.category_names:
        row[name] = (
            100 * categorization.counts.get(name, 0) / categorization.total_words
        )
    return row


def build_document_frame(
    documents: Sequence[Document],
    dictionary: LiwcDictionary,
) -> pd.DataFrame:
    """Monta a tabela LIWC com uma linha por documento.

    Args:
        documents: Documentos pareados do corpus
        dictionary: Dicionário carregado

    Returns:
        DataFrame com identificação do documento e o percentual por categoria
    """
    rows: list[dict[str, object]] = []
    for document in documents:
        row = score_document(dictionary, document)
        if row is None:
            logger.warning(f"Documento sem palavras: {document.uid}")
            continue
        rows.append(row)

    logger.info(f"LIWC calculado para {len(rows)} documentos")
    return pd.DataFrame(rows)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Colunas comparáveis: as categorias LIWC mais a cobertura do dicionário."""
    excluded = set(_IDENTITY_COLUMNS) | set(_RAW_COUNT_COLUMNS)
    return [column for column in frame.columns if column not in excluded]


# --------------------------------------------------------------------------
# Estatística espelhada de ``noticias_falsas_humano_maquina_semantica/liwc``
# --------------------------------------------------------------------------
#
# Este módulo não usa o `compare_frame` do `significance.py`: a comparação por
# categoria reproduz `liwc/analyzer.py` daquele repositório, para que os
# números das duas bases sejam lidos com o mesmo critério. As diferenças em
# relação aos outros sete módulos deste projeto são deliberadas:
#
# - **sem Shapiro-Wilk e sem teste t** — lá só o Mann-Whitney é computado;
# - **d de Cohen pela média das variâncias**, ``sqrt((σ_h² + σ_m²) / 2)`` com
#   ``σ`` populacional (``ddof=0``), e não o desvio agrupado ponderado por
#   ``n-1`` do `significance.py`. Com n igual nos dois grupos a diferença é o
#   fator ``sqrt(n / (n-1))`` — 2,6% com n = 20;
# - **sem correção de multiplicidade**: o corte é o p bruto.


def _mann_whitney(human: np.ndarray, machine: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U bilateral, com o mesmo fallback do módulo espelhado.

    Quando os dois grupos são constantes e iguais o teste não existe e o SciPy
    levanta ``ValueError``; lá isso vira ``(0, 1.0)``, ou seja, "sem diferença".
    """
    try:
        statistic, p_value = stats.mannwhitneyu(human, machine, alternative="two-sided")
    except ValueError:
        return 0.0, 1.0
    return float(statistic), float(p_value)


def _cohens_d(human: np.ndarray, machine: np.ndarray) -> float:
    """d de Cohen pela média das variâncias populacionais.

    É a fórmula de ``liwc/analyzer.py``: ``(m_h - m_m) / sqrt((σ_h² + σ_m²)/2)``
    com ``σ`` de ``ddof=0``. Devolve 0.0 quando não há variância nenhuma.
    """
    pooled_std = float(np.sqrt((human.std() ** 2 + machine.std() ** 2) / 2))
    if pooled_std == 0.0:
        return 0.0
    return float((human.mean() - machine.mean()) / pooled_std)


def compare_categories(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    group_column: str = "group",
) -> pd.DataFrame:
    """Compara cada categoria entre autorias, como no repositório de semântica.

    Args:
        frame: Tabela com uma linha por documento
        metrics: Colunas a comparar (categorias LIWC mais a cobertura)
        group_column: Coluna que separa ``human`` de ``machine``

    Returns:
        Uma linha por categoria, ordenada por ``|cohens_d|`` decrescente
    """
    human_rows = frame[frame[group_column] == Group.HUMAN.value]
    machine_rows = frame[frame[group_column] == Group.MACHINE.value]

    rows: list[dict[str, object]] = []
    for metric in metrics:
        human = human_rows[metric].to_numpy(dtype=float)
        machine = machine_rows[metric].to_numpy(dtype=float)
        human = human[~np.isnan(human)]
        machine = machine[~np.isnan(machine)]
        if len(human) < 2 or len(machine) < 2:
            logger.warning(f"Categoria sem observações suficientes: {metric}")
            continue

        difference = float(human.mean() - machine.mean())
        statistic, p_value = _mann_whitney(human, machine)
        cohens_d = _cohens_d(human, machine)

        rows.append(
            {
                "metric": metric,
                "human_n": len(human),
                "human_mean": float(human.mean()),
                "human_std": float(human.std()),
                "machine_n": len(machine),
                "machine_mean": float(machine.mean()),
                "machine_std": float(machine.std()),
                "difference": difference,
                "abs_difference": abs(difference),
                "u_statistic": statistic,
                "u_p_value": p_value,
                "cohens_d": cohens_d,
                "effect_size": interpret_effect_size(cohens_d),
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
                "characteristic_of": (
                    "human"
                    if difference > 0
                    else "machine"
                    if difference < 0
                    else "tie"
                ),
            }
        )

    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons
    return comparisons.reindex(
        comparisons["cohens_d"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def compare_categories_within(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split_column: str,
) -> pd.DataFrame:
    """Repete :func:`compare_categories` dentro de cada valor de uma coluna."""
    blocks: list[pd.DataFrame] = []
    for value in sorted(frame[split_column].unique()):
        block = compare_categories(frame[frame[split_column] == value], metrics)
        if block.empty:
            continue
        block.insert(0, split_column, value)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def run_analysis(
    documents: Sequence[Document],
    dictionary: LiwcDictionary,
    output_dir: Path,
) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos pareados do corpus
        dictionary: Dicionário carregado
        output_dir: Pasta de saída dos CSVs

    Returns:
        Tabela de significância por categoria, ordenada pelo tamanho do efeito
    """
    frame = build_document_frame(documents, dictionary)
    if frame.empty:
        logger.warning("Nenhum documento pontuável — nada a gravar")
        return frame

    metrics = metric_columns(frame)
    comparisons = compare_categories(frame, metrics)
    if not comparisons.empty:
        # O q-valor entra como coluna informativa, depois do critério: a página
        # em `docs/` publica "X com p < 0,05 · Y após correção FDR".
        comparisons = add_fdr_correction(comparisons)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame, output_dir / "liwc_per_document.csv")
    _write_csv(comparisons, output_dir / "liwc_significance.csv")
    _write_csv(
        compare_categories_within(frame, metrics, split_column="source"),
        output_dir / "liwc_significance_by_source.csv",
    )

    return comparisons


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
        "--dictionary",
        type=Path,
        help=f"Arquivo .dic do LIWC (default: {DICTIONARY_ENV_VAR} ou resources/liwc/)",
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
    """Ponto de entrada da análise LIWC."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    try:
        dictionary = load_liwc_dictionary(find_dictionary_path(args.dictionary))
    except LiwcDictionaryError as err:
        logger.error(f"Dicionário LIWC indisponível: {type(err).__name__}: {err}")
        return

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    comparisons = run_analysis(documents, dictionary, output_dir)
    if comparisons.empty:
        return

    significant = comparisons[comparisons["significant_005"]]
    logger.info(
        f"{len(significant)} de {len(comparisons)} categorias significativas "
        f"(p < 0.05); {int(comparisons['significant_fdr_005'].sum())} sobrevivem "
        f"à correção FDR"
    )
    for row in significant.head(8).itertuples():
        logger.info(
            f"  {row.metric}: {row.human_mean:.2f}% vs {row.machine_mean:.2f}% "
            f"(d={row.cohens_d:+.2f}, p={row.u_p_value:.1e}) "
            f"-> {row.characteristic_of}"
        )


if __name__ == "__main__":
    main()
