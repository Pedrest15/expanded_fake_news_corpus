"""Métricas do NILC-Metrix: complexidade, coesão e semântica das duas autorias.

O NILC-Metrix (https://github.com/sidleal/nilcmetrix) mede ~200 propriedades do
texto — legibilidade, frequência e diversidade lexical, coesão referencial e
por LSA, conectivos, ambiguidade, sintaxe, normas psicolinguísticas. Ele não
roda aqui: depende de uma imagem Docker, de um Postgres com o léxico e do
parser PALAVRAS. O cálculo é feito no servidor do NILC e volta como planilha,
que ``scripts/import_nilc_metrix.py`` traduz para ``data/nilc_metrix/metrics.csv``
com as chaves do corpus. Este módulo junta essas métricas ao corpus pareado e
compara humano contra máquina.

Três famílias de tabela, todas com o prefixo ``nilc_metrix_``:

- **as do projeto** — ``per_document``, ``significance`` e
  ``significance_by_source``, com a estatística de :mod:`significance` (t,
  Mann-Whitney, d de Cohen, q de Benjamini-Hochberg) sobre as 200 métricas,
  para a tabela conversar com as das outras análises;
- **``top_human`` / ``top_machine``** — espelham ``nilc-metrix/analyze_metrics.py``
  do repositório ``noticias_falsas_humano_maquina_semantica``: as métricas com
  q < 0,05, separadas pelo sinal do d. O d e o q de lá são os mesmos de
  :mod:`significance` (desvio agrupado com ``ddof=1``; BH sobre as 200), então
  as listas saem da tabela de significância, sem recálculo;
- **``ranked`` / ``strong_signals``** — espelham
  ``nilc-metrix/rank_discriminative_metrics.py`` do mesmo repositório: só as
  117 métricas de complexidade, coesão e semântica (:data:`METRICS_BY_CATEGORY`,
  morfologia e sintaxe ficam de fora por desenho), medianas e MAD, Cliff's δ
  como tamanho de efeito, AUC e BH sobre as 117. Sinal forte é
  ``|δ| >= 0.330`` e ``q < 0.05``. Veja :func:`rank_discriminative_metrics`.

NOTA: as métricas de parágrafo (:data:`LAYOUT_METRICS`) medem a diagramação da
distribuição, não a autoria. O Fake.br humano chega em um parágrafo só — a
limpeza cola as linhas, como no trabalho anterior —, e o FakeTrueBR humano em
dois (manchete e corpo), enquanto a máquina escreve em parágrafos. Elas ficam
nas tabelas, porque o repositório de semântica as inclui, mas com a coluna
``layout_dependent`` para o leitor poder descartá-las.

NOTA: o lado humano vem do repositório de semântica e o lado máquina do
servidor do NILC, e as duas instalações não medem igual algumas métricas
(:data:`INSTALLATION_DEPENDENT_METRICS`). Nelas a comparação mistura autoria
com instrumento; a coluna ``installation_dependent`` as marca.

Pares descartados: o documento sem métricas, ou cujas métricas descrevem outro
texto (``text_matches`` falso na importação — o NILC mediu só a manchete, ou
uma versão sem o apêndice que o modelo escreveu), sai junto com o seu par,
como no ``--exclude-uid``: a comparação é pareada.

Uso::

    uv run python scripts/import_nilc_metrix.py data/nilc_metrix/raw/<planilha>.csv
    python -m expanded_fake_news_corpus.analysis.nilc_metrix \\
        --experiment paper_replication --model openai/gpt-4.1-mini-2025-04-14
    python -m expanded_fake_news_corpus.analysis.nilc_metrix \\
        --experiment paper_replication --model ollama/qwen3:32b \\
        --output-dir data/analysis/paper_replication/qwen3_32b/nilc_metrix
"""

from __future__ import annotations

import argparse
import logging
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
from expanded_fake_news_corpus.analysis.significance import (
    add_fdr_correction,
    compare_frame,
    compare_within,
    comparisons_to_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "nilc_metrix"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "nilc_metrix" / "metrics.csv"

#: Colunas de chave da planilha importada, que não são métricas.
METRICS_KEY_COLUMNS = (
    "id_texto",
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

#: Colunas de identificação do documento na tabela por documento.
_IDENTITY_COLUMNS = ("uid", "source", "group", "dataset", "model")

#: Métricas por categoria, copiadas de ``rank_discriminative_metrics.py`` do
#: repositório de semântica — complexidade, coesão e semântica; morfologia e
#: sintaxe ficam de fora por desenho daquele script.
METRICS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "complexity_descriptive": (
        "paragraphs",
        "sentences",
        "words",
        "sentences_per_paragraph",
        "words_per_sentence",
        "syllables_per_content_word",
        "sentence_length_max",
        "sentence_length_min",
        "sentence_length_standard_deviation",
        "subtitles",
    ),
    "complexity_simplicity": (
        "simple_word_ratio",
        "easy_conjunctions_ratio",
        "hard_conjunctions_ratio",
        "short_sentence_ratio",
        "medium_short_sentence_ratio",
        "medium_long_sentence_ratio",
        "dialog_pronoun_ratio",
    ),
    "complexity_lexical_diversity": (
        "ttr",
        "content_density",
        "content_word_diversity",
        "content_word_max",
        "content_word_min",
        "content_word_standard_deviation",
        "function_word_diversity",
        "adjective_diversity_ratio",
        "adverbs_diversity_ratio",
        "noun_diversity",
        "verb_diversity",
        "pronoun_diversity",
        "indefinite_pronouns_diversity",
        "relative_pronouns_diversity_ratio",
        "preposition_diversity",
        "punctuation_diversity",
    ),
    "complexity_frequency": (
        "cw_freq",
        "cw_freq_bra",
        "cw_freq_brwac",
        "freq_bra",
        "freq_brwac",
        "min_cw_freq",
        "min_cw_freq_bra",
        "min_cw_freq_brwac",
        "min_freq_bra",
        "min_freq_brwac",
    ),
    "complexity_readability": (
        "flesch",
        "gunning_fox",
        "dalechall_adapted",
        "brunet",
        "honore",
    ),
    "complexity_psycholinguistic": tuple(
        f"{norm}_{suffix}"
        for norm in ("concretude", "familiaridade", "idade_aquisicao", "imageabilidade")
        for suffix in (
            "mean",
            "std",
            "1_25_ratio",
            "25_4_ratio",
            "4_55_ratio",
            "55_7_ratio",
        )
    ),
    "cohesion_referential": (
        "adj_arg_ovl",
        "arg_ovl",
        "adj_stem_ovl",
        "stem_ovl",
        "adj_cw_ovl",
        "adjacent_refs",
        "anaphoric_refs",
        "coreference_pronoun_ratio",
        "demonstrative_pronoun_ratio",
    ),
    "cohesion_semantic_lsa": (
        "cross_entropy",
        "lsa_adj_mean",
        "lsa_adj_std",
        "lsa_all_mean",
        "lsa_all_std",
        "lsa_paragraph_mean",
        "lsa_paragraph_std",
        "lsa_givenness_mean",
        "lsa_givenness_std",
        "lsa_span_mean",
        "lsa_span_std",
    ),
    "cohesion_connectives": (
        "conn_ratio",
        "add_pos_conn_ratio",
        "add_neg_conn_ratio",
        "cau_pos_conn_ratio",
        "cau_neg_conn_ratio",
        "log_pos_conn_ratio",
        "log_neg_conn_ratio",
        "tmp_pos_conn_ratio",
        "tmp_neg_conn_ratio",
        "logic_operators",
        "and_ratio",
        "or_ratio",
        "if_ratio",
        "negation_ratio",
    ),
    "semantics_lexical": (
        "abstract_nouns_ratio",
        "content_words_ambiguity",
        "adjectives_ambiguity",
        "adverbs_ambiguity",
        "nouns_ambiguity",
        "verbs_ambiguity",
        "hypernyms_verbs",
        "named_entity_ratio_sentence",
        "named_entity_ratio_text",
        "positive_words",
        "negative_words",
    ),
}

#: Categoria de cada métrica; as que não estão em :data:`METRICS_BY_CATEGORY`
#: (morfologia e sintaxe) ficam com :data:`UNCATEGORIZED`.
CATEGORY_BY_METRIC: dict[str, str] = {
    metric: category
    for category, metrics in METRICS_BY_CATEGORY.items()
    for metric in metrics
}
UNCATEGORIZED = "morphosyntax"

#: Métricas que dependem da quebra em parágrafos — ver a nota do módulo.
LAYOUT_METRICS = frozenset(
    {
        "paragraphs",
        "sentences_per_paragraph",
        "subtitles",
        "lsa_paragraph_mean",
        "lsa_paragraph_std",
    }
)

#: Métricas que as duas instalações do NILC medem diferente nos mesmos textos
#: humanos — servidor do NILC contra ``nilc-metrix/results/human.csv`` do
#: repositório de semântica, em 4 a 19 dos 19 textos comparados. As de
#: parágrafo também divergem, mas já estão em :data:`LAYOUT_METRICS`. Ver
#: ``corpus/paper_replication/NOTES.md``, entrada de 2026-10-08.
INSTALLATION_DEPENDENT_METRICS = frozenset(
    {
        "gunning_fox",
        "punctuation_ratio",
        "anaphoric_refs",
        "demonstrative_pronoun_ratio",
        "lsa_all_std",
        "named_entity_ratio_text",
        "named_entity_ratio_sentence",
    }
)

#: Cortes de Romano et al. (2006) para |δ| de Cliff, do maior para o menor.
CLIFFS_DELTA_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.474, "large"),
    (0.330, "medium"),
    (0.147, "small"),
)

#: Critério de sinal forte do repositório de semântica.
STRONG_SIGNAL_DELTA = 0.330
STRONG_SIGNAL_Q = 0.05

#: Quantas métricas vão para cada lista de ``top_human`` / ``top_machine``.
TOP_METRICS = 20


class MissingMetricsError(Exception):
    """A planilha importada do NILC-Metrix não existe."""


def load_metrics(path: Path = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    """Lê a planilha normalizada por ``scripts/import_nilc_metrix.py``.

    Args:
        path: Caminho do ``metrics.csv``

    Returns:
        Uma linha por texto medido, com as chaves e as métricas

    Raises:
        MissingMetricsError: Se o arquivo não existir
    """
    if not path.exists():
        raise MissingMetricsError(
            f"{path} not found. Import the server spreadsheet first: "
            "uv run python scripts/import_nilc_metrix.py <planilha.csv>"
        )
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in ("model", "round"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["text_matches"] = frame["text_matches"].astype(str) == "True"
    return frame


def metric_names(metrics: pd.DataFrame) -> list[str]:
    """As colunas de métrica da planilha, na ordem em que o NILC as devolve."""
    return [column for column in metrics.columns if column not in METRICS_KEY_COLUMNS]


def _lookup_key(document: Document) -> tuple[str, str, str, str]:
    """Chave de um documento na planilha: ``(group, model, round, uid)``.

    O lado humano não depende de gerador nem de rodada, então casa só pelo uid.
    """
    if document.group is Group.HUMAN:
        return (Group.HUMAN.value, "", "", document.uid)
    return (
        Group.MACHINE.value,
        document.model or "",
        document.round_name or "",
        document.uid,
    )


def build_document_frame(
    documents: Sequence[Document],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Junta as métricas do NILC aos documentos pareados.

    Um documento de máquina sem métricas válidas sai sozinho; o humano sai
    quando ele mesmo não tem métricas válidas ou quando nenhum documento de
    máquina do seu uid sobrou. O humano inválido leva junto as máquinas do uid.

    Args:
        documents: Documentos pareados do corpus
        metrics: Saída de :func:`load_metrics`

    Returns:
        Uma linha por documento, com identificação e as 200 métricas
    """
    names = metric_names(metrics)
    indexed = metrics.set_index(["group", "model", "round", "uid"])
    if not indexed.index.is_unique:
        duplicated = indexed.index[indexed.index.duplicated()].unique().tolist()
        raise ValueError(f"metrics.csv has duplicated keys: {duplicated}")

    valid: dict[int, pd.Series] = {}
    for position, document in enumerate(documents):
        key = _lookup_key(document)
        if key not in indexed.index:
            logger.warning(f"Sem métricas do NILC: {document.cohort} {document.uid}")
            continue
        row = indexed.loc[key]
        if not row["text_matches"]:
            logger.warning(
                f"Métricas de outro texto, descartadas: {document.cohort} "
                f"{document.uid} (NILC {row['words']:.0f} palavras, corpus "
                f"{row['corpus_words']})"
            )
            continue
        valid[position] = row

    human_ok = {
        documents[position].uid
        for position in valid
        if documents[position].group is Group.HUMAN
    }
    machine_ok = {
        documents[position].uid
        for position in valid
        if documents[position].group is Group.MACHINE
    }
    paired = human_ok & machine_ok

    rows: list[dict[str, object]] = []
    for position, row in valid.items():
        document = documents[position]
        if document.uid not in paired:
            continue
        rows.append(
            {
                "uid": document.uid,
                "source": document.source.value,
                "group": document.group.value,
                "dataset": document.dataset,
                "model": document.model,
                **{name: float(row[name]) for name in names},
            }
        )

    dropped = sorted({document.uid for document in documents} - paired)
    if dropped:
        logger.warning(
            f"Pares fora da análise do NILC ({len(dropped)}): {', '.join(dropped)}"
        )
    logger.info(f"Métricas do NILC juntadas a {len(rows)} documentos")
    return pd.DataFrame(rows)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Colunas comparáveis da tabela por documento."""
    return [column for column in frame.columns if column not in _IDENTITY_COLUMNS]


def annotate_metrics(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``category`` e as duas ressalvas logo depois de ``metric``."""
    if comparisons.empty:
        return comparisons
    annotated = comparisons.copy()
    position = annotated.columns.get_loc("metric") + 1
    annotated.insert(
        position,
        "category",
        annotated["metric"].map(CATEGORY_BY_METRIC).fillna(UNCATEGORIZED),
    )
    annotated.insert(
        position + 1, "layout_dependent", annotated["metric"].isin(LAYOUT_METRICS)
    )
    annotated.insert(
        position + 2,
        "installation_dependent",
        annotated["metric"].isin(INSTALLATION_DEPENDENT_METRICS),
    )
    return annotated


def summarize(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Média de algumas métricas de referência por recorte, para leitura rápida."""
    headline = [
        "words",
        "sentences",
        "words_per_sentence",
        "syllables_per_content_word",
        "flesch",
        "ttr",
        "content_density",
        "lsa_adj_mean",
    ]
    present = [metric for metric in headline if metric in frame.columns]
    grouped = frame.groupby(by)
    summary = grouped[present].mean().add_suffix("_mean")
    summary.insert(0, "documents", grouped.size())
    return summary.reset_index()


# --------------------------------------------------------------------------
# Estatística espelhada de ``noticias_falsas_humano_maquina_semantica/nilc-metrix``
# --------------------------------------------------------------------------


def top_metrics(
    comparisons: pd.DataFrame, *, top: int = TOP_METRICS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As listas de ``analyze_metrics.py``: q < 0,05, separadas pelo sinal do d.

    Args:
        comparisons: Tabela de significância, já com ``q_value``
        top: Tamanho de cada lista

    Returns:
        ``(maiores no humano, maiores na máquina)``, por ``|d|`` decrescente
    """
    significant = comparisons[comparisons["q_value"] < 0.05]
    ordered = significant.reindex(
        significant["cohens_d"].abs().sort_values(ascending=False).index
    )
    human = ordered[ordered["cohens_d"] > 0].head(top).reset_index(drop=True)
    machine = ordered[ordered["cohens_d"] < 0].head(top).reset_index(drop=True)
    return human, machine


def cliffs_delta_magnitude(delta: float) -> str:
    """Traduz |δ| para a escala de Romano et al. (2006)."""
    if np.isnan(delta):
        return "n/a"
    magnitude = abs(delta)
    for threshold, label in CLIFFS_DELTA_THRESHOLDS:
        if magnitude >= threshold:
            return label
    return "negligible"


def cliffs_delta_from_u(u_statistic: float, n_human: int, n_machine: int) -> float:
    """δ de Cliff a partir do U do Mann-Whitney com o humano como ``x``.

    O SciPy devolve ``U1 = #(x > y) + 0,5·#(x == y)``, então
    ``δ = 2·U1 / (n_x·n_y) − 1``: positivo quando o humano tende a ser maior.
    """
    if n_human == 0 or n_machine == 0:
        return float("nan")
    return 2.0 * u_statistic / (n_human * n_machine) - 1.0


def rank_discriminative_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str = "group",
) -> pd.DataFrame:
    """Ranqueia as métricas de :data:`METRICS_BY_CATEGORY` pelo δ de Cliff.

    Reproduz ``rank_discriminative_metrics.py``: medianas e MAD (escala 1),
    Mann-Whitney bilateral, δ de Cliff derivado do U, AUC = (δ + 1) / 2 e
    q de Benjamini-Hochberg sobre as métricas filtradas apenas.

    Args:
        frame: Tabela com uma linha por documento
        group_column: Coluna que separa ``human`` de ``machine``

    Returns:
        Uma linha por métrica, por ``|δ|`` decrescente
    """
    human_rows = frame[frame[group_column] == Group.HUMAN.value]
    machine_rows = frame[frame[group_column] == Group.MACHINE.value]

    rows: list[dict[str, object]] = []
    for category, metrics in METRICS_BY_CATEGORY.items():
        for metric in metrics:
            if metric not in frame.columns:
                logger.warning(f"Métrica ausente da planilha: {metric}")
                continue
            human = human_rows[metric].to_numpy(dtype=float)
            machine = machine_rows[metric].to_numpy(dtype=float)
            human = human[~np.isnan(human)]
            machine = machine[~np.isnan(machine)]
            if len(human) < 2 or len(machine) < 2:
                continue

            result = stats.mannwhitneyu(human, machine, alternative="two-sided")
            delta = cliffs_delta_from_u(
                float(result.statistic), len(human), len(machine)
            )
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "layout_dependent": metric in LAYOUT_METRICS,
                    "installation_dependent": metric in INSTALLATION_DEPENDENT_METRICS,
                    "n_human": len(human),
                    "n_machine": len(machine),
                    "median_human": float(np.median(human)),
                    "median_machine": float(np.median(machine)),
                    "mad_human": float(stats.median_abs_deviation(human)),
                    "mad_machine": float(stats.median_abs_deviation(machine)),
                    "cliffs_delta": delta,
                    "abs_delta": abs(delta),
                    "auc": (delta + 1.0) / 2.0,
                    "u_p_value": float(result.pvalue),
                    "magnitude": cliffs_delta_magnitude(delta),
                    "higher_in": (
                        "human" if delta > 0 else "machine" if delta < 0 else "tie"
                    ),
                }
            )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = add_fdr_correction(ranked)
    return ranked.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def strong_signals(ranked: pd.DataFrame) -> pd.DataFrame:
    """Sinais fortes: ``|δ| >= 0.330`` e ``q < 0.05``."""
    if ranked.empty:
        return ranked
    strong = ranked["abs_delta"].ge(STRONG_SIGNAL_DELTA) & ranked["q_value"].lt(
        STRONG_SIGNAL_Q
    )
    return ranked[strong].reset_index(drop=True)


def run_analysis(
    documents: Sequence[Document],
    metrics: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos pareados do corpus
        metrics: Saída de :func:`load_metrics`
        output_dir: Pasta de saída dos CSVs

    Returns:
        Ranking das métricas de complexidade, coesão e semântica
    """
    frame = build_document_frame(documents, metrics)
    if frame.empty:
        logger.warning("Nenhum documento com métricas do NILC — nada a gravar")
        return frame

    columns = metric_columns(frame)
    comparisons = comparisons_to_frame(compare_frame(frame, columns))
    comparisons = annotate_metrics(add_fdr_correction(comparisons))
    top_human, top_machine = top_metrics(comparisons)
    ranked = rank_discriminative_metrics(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame, output_dir / "nilc_metrix_per_document.csv")
    _write_csv(summarize(frame, "dataset"), output_dir / "nilc_metrix_by_dataset.csv")
    _write_csv(comparisons, output_dir / "nilc_metrix_significance.csv")
    _write_csv(
        annotate_metrics(compare_within(frame, columns, split_column="source")),
        output_dir / "nilc_metrix_significance_by_source.csv",
    )
    _write_csv(top_human, output_dir / "nilc_metrix_top_human.csv")
    _write_csv(top_machine, output_dir / "nilc_metrix_top_machine.csv")
    _write_csv(ranked, output_dir / "nilc_metrix_ranked.csv")
    _write_csv(strong_signals(ranked), output_dir / "nilc_metrix_strong_signals.csv")

    return ranked


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
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Planilha importada do NILC-Metrix (default: {DEFAULT_METRICS_PATH})",
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
    """Ponto de entrada da análise do NILC-Metrix."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    try:
        metrics = load_metrics(args.metrics)
    except MissingMetricsError as err:
        logger.error(f"Métricas do NILC indisponíveis: {err}")
        return

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    ranked = run_analysis(documents, metrics, output_dir)
    if ranked.empty:
        return

    strong = strong_signals(ranked)
    logger.info(
        f"{len(strong)} de {len(ranked)} métricas de complexidade, coesão e "
        f"semântica com sinal forte (|δ| >= {STRONG_SIGNAL_DELTA}, "
        f"q < {STRONG_SIGNAL_Q})"
    )
    for row in strong.head(10).itertuples():
        layout = " [parágrafo]" if row.layout_dependent else ""
        layout += " [instalação]" if row.installation_dependent else ""
        logger.info(
            f"  {row.metric}: {row.median_human:.3f} vs {row.median_machine:.3f} "
            f"(δ={row.cliffs_delta:+.2f}, q={row.q_value:.1e}) "
            f"-> {row.higher_in}{layout}"
        )


if __name__ == "__main__":
    main()
