"""Testes de significância compartilhados pelos módulos de caracterização.

Toda métrica contínua do projeto — sílabas por palavra, frequência de POS,
regras por sentença, categorias LIWC — é comparada da mesma forma: normalidade
por Shapiro-Wilk, teste t e Mann-Whitney U (que não pressupõe normalidade e é
o reportado), e Cohen's d para o tamanho do efeito. Centralizar isso mantém as
tabelas dos módulos comparáveis entre si.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from expanded_fake_news_corpus.analysis.documents import Group

logger = logging.getLogger(__name__)

#: Shapiro-Wilk perde sentido em amostras grandes — amostramos até este tamanho.
SHAPIRO_MAX_SAMPLE = 5_000

#: Semente da amostragem do Shapiro-Wilk, para o teste ser reproduzível.
SHAPIRO_SEED = 42

#: Shapiro-Wilk exige um mínimo de observações.
SHAPIRO_MIN_SAMPLE = 3

#: Cortes de Cohen (1988) para interpretar |d|.
EFFECT_SIZE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
)

DEFAULT_EFFECT_SIZE = "large"


@dataclass(frozen=True)
class GroupSummary:
    """Estatística descritiva de um dos grupos."""

    n: int
    mean: float
    std: float


@dataclass(frozen=True)
class ComparisonResult:
    """Resultado da comparação humano vs máquina para uma métrica.

    O sinal de ``cohens_d`` segue ``human - machine``: positivo significa valor
    maior no lado humano.
    """

    metric: str
    human: GroupSummary
    machine: GroupSummary
    shapiro_p_human: float | None
    shapiro_p_machine: float | None
    t_statistic: float
    t_p_value: float
    u_statistic: float
    u_p_value: float
    cohens_d: float
    effect_size: str

    def to_row(self) -> dict[str, object]:
        """Achata o resultado numa linha de tabela."""
        row: dict[str, object] = {"metric": self.metric}
        row.update({f"human_{k}": v for k, v in asdict(self.human).items()})
        row.update({f"machine_{k}": v for k, v in asdict(self.machine).items()})
        row.update(
            {
                "shapiro_p_human": self.shapiro_p_human,
                "shapiro_p_machine": self.shapiro_p_machine,
                "t_statistic": self.t_statistic,
                "t_p_value": self.t_p_value,
                "u_statistic": self.u_statistic,
                "u_p_value": self.u_p_value,
                "cohens_d": self.cohens_d,
                "effect_size": self.effect_size,
            }
        )
        return row


class InsufficientDataError(Exception):
    """Um dos grupos não tem observações suficientes para o teste."""


def compare_groups(
    metric: str,
    human_values: Sequence[float],
    machine_values: Sequence[float],
) -> ComparisonResult:
    """Compara uma métrica entre os grupos humano e máquina.

    Args:
        metric: Nome da métrica, repassado ao resultado
        human_values: Observações do lado humano
        machine_values: Observações do lado máquina

    Returns:
        Descritivas dos dois grupos, testes de hipótese e tamanho do efeito

    Raises:
        InsufficientDataError: Se algum grupo tiver menos de duas observações
    """
    human = np.asarray(human_values, dtype=float)
    machine = np.asarray(machine_values, dtype=float)
    human = human[~np.isnan(human)]
    machine = machine[~np.isnan(machine)]

    if len(human) < 2 or len(machine) < 2:
        raise InsufficientDataError(
            f"{metric}: human n={len(human)}, machine n={len(machine)}"
        )

    t_statistic, t_p_value = stats.ttest_ind(human, machine)
    u_statistic, u_p_value = stats.mannwhitneyu(human, machine, alternative="two-sided")
    cohens_d = _cohens_d(human, machine)

    return ComparisonResult(
        metric=metric,
        human=_summarize(human),
        machine=_summarize(machine),
        shapiro_p_human=_shapiro_p_value(human),
        shapiro_p_machine=_shapiro_p_value(machine),
        t_statistic=float(t_statistic),
        t_p_value=float(t_p_value),
        u_statistic=float(u_statistic),
        u_p_value=float(u_p_value),
        cohens_d=cohens_d,
        effect_size=interpret_effect_size(cohens_d),
    )


def compare_frame(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    group_column: str = "group",
) -> list[ComparisonResult]:
    """Roda :func:`compare_groups` para cada métrica de um DataFrame por documento.

    Args:
        frame: Tabela com uma linha por documento
        metrics: Colunas numéricas a comparar
        group_column: Coluna que separa ``human`` de ``machine``

    Returns:
        Um resultado por métrica; métricas sem dados suficientes são omitidas
    """
    human_rows = frame[frame[group_column] == Group.HUMAN.value]
    machine_rows = frame[frame[group_column] == Group.MACHINE.value]

    results: list[ComparisonResult] = []
    for metric in metrics:
        try:
            results.append(
                compare_groups(metric, human_rows[metric], machine_rows[metric])
            )
        except InsufficientDataError as err:
            logger.warning(f"Métrica ignorada por falta de dados: {err}")
    return results


def compare_within(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    split_column: str,
    group_column: str = "group",
) -> pd.DataFrame:
    """Repete a comparação humano vs máquina dentro de cada valor de uma coluna.

    Serve para separar efeito de autoria de efeito do recorte. O caso corrente é
    ``split_column="source"``: o FakeTrueBR humano é distribuído em texto
    degradado (minúsculo, sem acentos, sem pontuação plena), então o resultado
    agregado mistura autoria com tipografia.

    Args:
        frame: Tabela com uma linha por documento
        metrics: Colunas numéricas a comparar
        split_column: Coluna que define os recortes
        group_column: Coluna que separa ``human`` de ``machine``

    Returns:
        Resultados empilhados, com ``split_column`` na primeira posição
    """
    blocks: list[pd.DataFrame] = []
    for value in sorted(frame[split_column].unique()):
        comparisons = compare_frame(
            frame[frame[split_column] == value], metrics, group_column=group_column
        )
        if not comparisons:
            continue
        block = comparisons_to_frame(comparisons)
        block.insert(0, split_column, value)
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def comparisons_to_frame(results: Sequence[ComparisonResult]) -> pd.DataFrame:
    """Empilha os resultados numa tabela pronta para CSV."""
    return pd.DataFrame([result.to_row() for result in results])


def add_fdr_correction(
    frame: pd.DataFrame,
    *,
    p_value_column: str = "u_p_value",
) -> pd.DataFrame:
    """Acrescenta o q-valor de Benjamini-Hochberg à tabela de comparações.

    Quando dezenas de métricas são testadas de uma vez — as 73 categorias do
    LIWC, por exemplo — o p-valor bruto produz falsos positivos por construção:
    a 5% de significância, ~4 categorias "significativas" saem do acaso. O
    q-valor controla a taxa de descobertas falsas da família de testes.

    Métricas cujo teste não foi computável — os dois grupos constantes e iguais,
    caso em que o Mann-Whitney devolve ``nan`` — ficam de fora da correção e com
    ``q_value`` nulo: um teste que não rodou não é uma descoberta, e incluí-lo
    inflaria o denominador do procedimento.

    Args:
        frame: Tabela de :func:`comparisons_to_frame`
        p_value_column: Coluna com os p-valores brutos

    Returns:
        A tabela com as colunas ``q_value`` e ``significant_fdr_005``
    """
    if frame.empty:
        return frame

    corrected = frame.copy()
    testable = corrected[p_value_column].notna()
    if not testable.any():
        logger.warning("Nenhum p-valor computável: correção FDR não aplicada")
        corrected["q_value"] = float("nan")
        corrected["significant_fdr_005"] = False
        return corrected

    if (~testable).any():
        logger.warning(
            f"{int((~testable).sum())} métricas sem p-valor ficaram fora do FDR"
        )

    corrected["q_value"] = float("nan")
    corrected.loc[testable, "q_value"] = stats.false_discovery_control(
        corrected.loc[testable, p_value_column].to_numpy()
    )
    corrected["significant_fdr_005"] = corrected["q_value"] < 0.05
    return corrected


def interpret_effect_size(cohens_d: float) -> str:
    """Traduz |d| para a escala qualitativa de Cohen."""
    magnitude = abs(cohens_d)
    for threshold, label in EFFECT_SIZE_THRESHOLDS:
        if magnitude < threshold:
            return label
    return DEFAULT_EFFECT_SIZE


def _summarize(values: np.ndarray) -> GroupSummary:
    """Descritivas de um grupo, com desvio amostral (ddof=1)."""
    return GroupSummary(
        n=int(len(values)),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
    )


def _cohens_d(human: np.ndarray, machine: np.ndarray) -> float:
    """Cohen's d com desvio agrupado; 0.0 quando não há variância nos dois grupos."""
    n_human, n_machine = len(human), len(machine)
    pooled_variance = (
        (n_human - 1) * human.var(ddof=1) + (n_machine - 1) * machine.var(ddof=1)
    ) / (n_human + n_machine - 2)
    pooled_std = float(np.sqrt(pooled_variance))
    if pooled_std == 0.0:
        return 0.0
    return float((human.mean() - machine.mean()) / pooled_std)


def _shapiro_p_value(values: np.ndarray) -> float | None:
    """p-valor de Shapiro-Wilk sobre uma amostra determinística.

    Devolve None quando o teste não se aplica: amostra curta demais ou grupo
    constante (categoria LIWC que nunca casa, por exemplo), caso em que o scipy
    divide por zero e devolve ``nan``.
    """
    if len(values) < SHAPIRO_MIN_SAMPLE or values.var(ddof=1) == 0:
        return None
    sample = values
    if len(values) > SHAPIRO_MAX_SAMPLE:
        generator = np.random.default_rng(SHAPIRO_SEED)
        sample = generator.choice(values, size=SHAPIRO_MAX_SAMPLE, replace=False)
    return float(stats.shapiro(sample).pvalue)
