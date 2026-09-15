"""SAGE — termos distintivos de cada autoria.

Implementa o *Sparse Additive Generative Model* de Eisenstein et al. (2011):
a distribuição de palavras de cada classe é modelada como um desvio esparso
sobre uma distribuição de fundo comum,

    log P(w | classe) = η_classe + m − Z

onde ``m`` são as log-frequências globais e ``η`` o desvio da classe, empurrado
para zero por uma penalidade L1. O que sobra em ``η`` é o vocabulário que a
classe realmente usa a mais — não apenas as palavras frequentes, que a
distribuição de fundo já explica.

A distintividade de um termo é ``η_human − η_machine``: positiva, é do lado
humano; negativa, do lado da máquina.

NOTA: o texto é normalizado sem acentos antes de contar. Não é preciosismo — o
FakeTrueBR humano é distribuído já sem acentuação, e sem essa normalização o
SAGE acharia apenas que a máquina acentua e o humano não.

Uso::

    python -m expanded_fake_news_corpus.analysis.sage
    python -m expanded_fake_news_corpus.analysis.sage --source fakebr --min-df 3
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    Group,
    add_corpus_arguments,
    documents_from_args,
    output_dir_from_args,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "sage"

#: Ordem das classes no modelo: o sinal da distintividade depende dela.
CLASS_ORDER = (Group.HUMAN, Group.MACHINE)

#: Abaixo disso o desvio é ruído da otimização, não sinal.
DISTINCTIVENESS_THRESHOLD = 0.01

#: Evita log(0) na distribuição de fundo.
COUNT_SMOOTHING = 1e-10


@dataclass(frozen=True)
class SageConfig:
    """Hiperparâmetros da vetorização e do modelo.

    Os defaults são os da análise agregada do trabalho anterior. Em corpus
    pequeno, ``min_df=5`` derruba quase todo o vocabulário — vale baixar.
    """

    regularization: float = 0.1  # peso da penalidade L1: maior = mais esparso
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 5
    max_df: float = 0.95
    max_features: int = 3000
    max_iterations: int = 1000
    seed: int = 42


class SageError(Exception):
    """Falha ao ajustar o modelo SAGE."""


class SageModel:
    """Modelo SAGE de duas classes.

    Responsabilidades:
    - Estimar a distribuição de fundo e os desvios esparsos por classe
    - Ranquear os termos pela diferença entre os desvios das duas classes
    """

    def __init__(self, config: SageConfig | None = None) -> None:
        """Inicializa o modelo.

        Args:
            config: Hiperparâmetros; usa os defaults se omitido
        """
        self.config = config or SageConfig()
        self.background: np.ndarray | None = None
        self.components: np.ndarray | None = None

    def _objective(self, params: np.ndarray, class_counts: np.ndarray) -> float:
        """Log-verossimilhança negativa somada à penalidade L1."""
        components = params.reshape(class_counts.shape)
        log_likelihood = 0.0
        for index, counts in enumerate(class_counts):
            log_likelihood += float(counts @ self._log_probabilities(components[index]))

        penalty = self.config.regularization * float(np.abs(components).sum())
        return -(log_likelihood - penalty)

    def _gradient(self, params: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
        """Gradiente analítico: contagem observada menos a esperada sob o modelo."""
        components = params.reshape(class_counts.shape)
        gradient = np.zeros_like(components)
        for index, counts in enumerate(class_counts):
            probabilities = np.exp(self._log_probabilities(components[index]))
            expected = counts.sum() * probabilities
            gradient[index] = counts - expected
            gradient[index] -= self.config.regularization * np.sign(components[index])

        return -gradient.flatten()

    def _log_probabilities(self, component: np.ndarray) -> np.ndarray:
        """Distribuição normalizada de uma classe: η + m − log-sum-exp(η + m)."""
        unnormalized = component + self.background
        return unnormalized - logsumexp(unnormalized)

    def fit(self, counts: np.ndarray) -> SageModel:
        """Ajusta o modelo às contagens agregadas por classe.

        A verossimilhança depende só das contagens somadas de cada classe, então
        a matriz documento-termo é agregada antes de otimizar.

        Args:
            counts: Matriz ``(n_classes, vocabulário)`` de contagens

        Returns:
            O próprio modelo, ajustado

        Raises:
            SageError: Se o vocabulário estiver vazio
        """
        if counts.size == 0 or counts.shape[1] == 0:
            raise SageError("empty vocabulary: nothing to fit")

        total_counts = counts.sum(axis=0) + COUNT_SMOOTHING
        self.background = np.log(total_counts) - np.log(total_counts.sum())

        generator = np.random.default_rng(self.config.seed)
        initial = generator.normal(0, 0.01, counts.shape)

        logger.info(
            f"Ajustando SAGE: {counts.shape[0]} classes, "
            f"{counts.shape[1]} termos, L1={self.config.regularization}"
        )
        result = minimize(
            fun=self._objective,
            x0=initial.flatten(),
            args=(counts,),
            jac=self._gradient,
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iterations},
        )
        if not result.success:
            logger.warning(f"Otimização terminou sem convergir: {result.message}")

        self.components = result.x.reshape(counts.shape)
        return self

    def distinctive_terms(
        self,
        feature_names: Sequence[str],
        *,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """Ranqueia os termos pela diferença entre os desvios das duas classes.

        Args:
            feature_names: Vocabulário, na ordem das colunas
            top_k: Quantos termos extrair de cada lado

        Returns:
            DataFrame com ``term``, ``characteristic_of``, ``distinctiveness`` e os
            dois ``eta``, do mais para o menos distintivo

        Raises:
            SageError: Se o modelo ainda não foi ajustado
        """
        if self.components is None:
            raise SageError("model not fitted")

        difference = self.components[0] - self.components[1]
        order = np.argsort(difference)

        rows = [
            self._term_row(index, feature_names, difference, Group.HUMAN)
            for index in order[::-1][:top_k]
            if difference[index] > DISTINCTIVENESS_THRESHOLD
        ]
        rows.extend(
            self._term_row(index, feature_names, difference, Group.MACHINE)
            for index in order[:top_k]
            if difference[index] < -DISTINCTIVENESS_THRESHOLD
        )

        frame = pd.DataFrame(rows)
        if frame.empty:
            logger.warning("Nenhum termo passou do limiar de distintividade")
            return frame

        frame["abs_distinctiveness"] = frame["distinctiveness"].abs()
        return frame.sort_values("abs_distinctiveness", ascending=False).reset_index(
            drop=True
        )

    def _term_row(
        self,
        index: int,
        feature_names: Sequence[str],
        difference: np.ndarray,
        group: Group,
    ) -> dict[str, object]:
        """Monta a linha de um termo distintivo."""
        return {
            "term": feature_names[index],
            "characteristic_of": group.value,
            "distinctiveness": float(difference[index]),
            "eta_human": float(self.components[0, index]),
            "eta_machine": float(self.components[1, index]),
        }


def normalize_for_sage(text: str) -> str:
    """Reduz o texto a caixa baixa sem acentos.

    A decomposição NFD separa a letra do diacrítico; descartar as marcas
    (categoria Unicode ``Mn``) deixa só a letra base.

    Args:
        text: Texto original

    Returns:
        Texto normalizado
    """
    lowered = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in lowered if unicodedata.category(char) != "Mn")
    return stripped.replace("ç", "c")


def build_class_counts(
    documents: Sequence[Document],
    config: SageConfig,
) -> tuple[np.ndarray, list[str]]:
    """Vetoriza os documentos e agrega as contagens por autoria.

    Args:
        documents: Documentos pareados do corpus
        config: Hiperparâmetros da vetorização

    Returns:
        Par ``(contagens por classe, vocabulário)``, com as classes na ordem de
        :data:`CLASS_ORDER`

    Raises:
        SageError: Se algum dos grupos estiver vazio ou o vocabulário ficar vazio
    """
    texts = [normalize_for_sage(document.text) for document in documents]
    groups = np.array([document.group.value for document in documents])

    for group in CLASS_ORDER:
        if not np.any(groups == group.value):
            raise SageError(f"no documents for group {group.value!r}")

    vectorizer = CountVectorizer(
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
    )
    try:
        document_term = vectorizer.fit_transform(texts)
    except ValueError as err:
        raise SageError(f"vectorization failed: {err}") from err

    feature_names = list(vectorizer.get_feature_names_out())
    dense = document_term.toarray()

    class_counts = np.vstack(
        [dense[groups == group.value].sum(axis=0) for group in CLASS_ORDER]
    ).astype(float)

    logger.info(
        f"Vetorizados {len(documents)} documentos: {len(feature_names)} termos, "
        f"{int(class_counts.sum())} ocorrências"
    )
    return class_counts, feature_names


def run_analysis(
    documents: Sequence[Document],
    output_dir: Path,
    *,
    config: SageConfig | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Roda o SAGE sobre o corpus e grava os termos distintivos.

    Args:
        documents: Documentos pareados do corpus
        output_dir: Pasta de saída
        config: Hiperparâmetros; usa os defaults se omitido
        top_k: Quantos termos extrair de cada lado

    Returns:
        Tabela de termos distintivos
    """
    config = config or SageConfig()
    class_counts, feature_names = build_class_counts(documents, config)

    model = SageModel(config).fit(class_counts)
    terms = model.distinctive_terms(feature_names, top_k=top_k)
    if terms.empty:
        return terms

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sage_distinctive_terms.csv"
    terms.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Gravado {output_path} ({len(terms)} termos)")

    return terms


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando (``sys.argv`` se ``argv`` for None)."""
    defaults = SageConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_corpus_arguments(parser)
    parser.add_argument(
        "--min-df",
        type=int,
        default=defaults.min_df,
        help=f"Documentos mínimos em que o termo aparece (default: {defaults.min_df})",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=defaults.max_features,
        help=f"Tamanho máximo do vocabulário (default: {defaults.max_features})",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=defaults.regularization,
        help=f"Peso da penalidade L1 (default: {defaults.regularization})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Termos extraídos de cada lado (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada da análise SAGE."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return

    config = SageConfig(
        regularization=args.regularization,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    try:
        terms = run_analysis(documents, output_dir, config=config, top_k=args.top_k)
    except SageError as err:
        logger.error(f"Falha na análise SAGE: {type(err).__name__}: {err}")
        return

    if terms.empty:
        return

    for group in CLASS_ORDER:
        top = terms[terms["characteristic_of"] == group.value].head(10)
        preview = ", ".join(top["term"].astype(str))
        logger.info(f"Mais característico de {group.value}: {preview}")


if __name__ == "__main__":
    main()
