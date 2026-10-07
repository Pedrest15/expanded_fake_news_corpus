"""Classificação humano vs máquina: os experimentos sem features linguísticas.

Traz do trabalho anterior
(``noticias_falsas_humano_maquina_semantica/linguistic_features``) os dois
modos que não dependem das tabelas de análise: o vetor de um encoder congelado
(``embedding``) e a sacola de palavras sobre o texto (``tfidf``, ``bow``). Os
estimadores, as grades e o agrupamento por notícia são os mesmos de lá.

Uma diferença deliberada no protocolo, imposta pelo tamanho do corpus. Lá são
5.391 pares e o resultado sai de um teste separado, com a grade ajustada no
treino. Aqui são **20** pares — o lado humano são as mesmas 20 fake news em
todos os geradores, e o pareamento amarra o experimento a esse teto, não
importa quantos modelos entrem. Com 20 grupos, um teste separado teria 4
notícias, e qualquer acurácia viria com granularidade de 12 pontos. Então a
avaliação aqui é **validação cruzada aninhada**: a dobra externa mede, a
interna escolhe a grade, e nenhuma notícia aparece nos dois lados. O relatório
traz média e desvio entre dobras, não um número único.

Uso::

    python -m expanded_fake_news_corpus.classification.runner \\
        --experiment paper_replication --model ollama/qwen3:32b --mode tfidf
    python -m expanded_fake_news_corpus.classification.runner \\
        --mode embedding --encoder norberto --classifier all
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    add_corpus_arguments,
    documents_from_args,
)
from expanded_fake_news_corpus.classification.classifiers import (
    ClassifierSpec,
    build_catalogue,
)
from expanded_fake_news_corpus.classification.encoders import (
    BERT_MODELS,
    DEFAULT_BERT_MODEL,
    DEFAULT_MAX_LENGTH,
    POOLING_STRATEGIES,
    load_or_extract,
    resolve_model_name,
    resolve_pooling,
)
from expanded_fake_news_corpus.classification.pairing import (
    PairedSample,
    build_pairs,
    flatten_pairs,
    truncate_pair,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classification"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "classification" / "cache"

#: Modos que não dependem das tabelas de análise linguística.
MODES = ("embedding", "tfidf", "bow")

DEFAULT_CV_FOLDS = 5
DEFAULT_MAX_FEATURES = 10_000

#: Documentos mínimos por termo na sacola de palavras.
#:
#: NOTA: o trabalho anterior usa 5, calibrado para milhares de documentos. Com
#: 40 documentos, 5 descartaria quase todo o vocabulário, então o padrão aqui é
#: 2 — presente em pelo menos dois documentos.
DEFAULT_MIN_DF = 2

#: Métrica que a busca em grade otimiza, como no trabalho anterior.
SCORING = "f1_weighted"


class ClassificationError(Exception):
    """O experimento não pôde ser montado."""


def truncate_all(
    pairs: Sequence[PairedSample], tokenizer, *, max_length: int
) -> list[PairedSample]:
    """Aplica o truncamento pareado a todos os pares.

    Args:
        pairs: Pares como vieram do corpus
        tokenizer: Tokenizador do encoder
        max_length: Teto de tokens

    Returns:
        Os mesmos pares, com os dois lados do mesmo tamanho
    """
    from dataclasses import replace

    truncated: list[PairedSample] = []
    kept_tokens: list[int] = []
    for pair in pairs:
        human, machine, kept = truncate_pair(
            pair.human_text, pair.machine_text, tokenizer, max_length=max_length
        )
        truncated.append(replace(pair, human_text=human, machine_text=machine))
        kept_tokens.append(kept)

    logger.info(
        f"Truncamento pareado: {min(kept_tokens)}–{max(kept_tokens)} tokens por "
        f"lado (mediana {int(np.median(kept_tokens))})"
    )
    return truncated


def build_pipeline(
    spec: ClassifierSpec,
    mode: str,
    *,
    max_features: int,
    min_df: int,
    memory: str | None = None,
):
    """Monta o pipeline do modo, com o vetorizador dentro.

    O vetorizador fica no pipeline de propósito: ajustá-lo fora da validação
    cruzada deixaria o vocabulário do teste vazar para o treino.

    Args:
        spec: Classificador e grade
        mode: Um de :data:`MODES`
        max_features: Teto de termos da sacola de palavras
        min_df: Documentos mínimos por termo
        memory: Pasta de cache dos transformadores. A busca em grade reajusta
            o pipeline inteiro em cada ponto, mas o vetorizador não depende dos
            hiperparâmetros do classificador — com cache ele é ajustado uma vez
            por dobra em vez de uma por ponto da grade, o que no corpus
            anterior é a diferença entre ~8 e ~2 minutos.

    Returns:
        Tupla ``(pipeline, grade com os prefixos do pipeline)``
    """
    steps: list[tuple[str, Any]] = []
    if mode == "embedding":
        steps.append(("scale", StandardScaler()))
    else:
        vectorizer = (TfidfVectorizer if mode == "tfidf" else CountVectorizer)(
            max_features=max_features, min_df=min_df, ngram_range=(1, 1)
        )
        steps.append(("vectorize", vectorizer))
        # GaussianNB e MLP não aceitam matriz esparsa; com 40 documentos o
        # custo de densificar é irrelevante.
        steps.append(("densify", FunctionTransformer(lambda matrix: matrix.toarray())))
    steps.append(("clf", spec.estimator))

    grid = {f"clf__{key}": values for key, values in spec.param_grid.items()}
    return Pipeline(steps, memory=memory), grid


def nested_cross_validate(
    spec: ClassifierSpec,
    features: Any,
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    mode: str,
    folds: int,
    max_features: int,
    min_df: int,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Avalia o classificador com validação cruzada aninhada.

    A dobra externa mede; a interna escolhe os hiperparâmetros. As duas agrupam
    por notícia, então os dois lados de um par nunca se separam.

    Args:
        spec: Classificador e grade
        features: Matriz de embeddings, ou a lista de textos
        labels: Rótulos, 0 humano e 1 máquina
        groups: Chave de agrupamento por documento
        mode: Um de :data:`MODES`
        folds: Dobras externas; a interna usa uma menos
        max_features: Teto de termos da sacola de palavras
        min_df: Documentos mínimos por termo
        n_jobs: Processos da busca em grade; 1 desliga o paralelismo

    Returns:
        Métricas por dobra, agregado e a matriz de confusão somada
    """
    unique_groups = sorted(set(groups))
    outer_splits = min(folds, len(unique_groups))
    inner_splits = max(2, outer_splits - 1)
    logger.info(
        f"{spec.name}: {spec.grid_size} combinações, GroupKFold externo "
        f"{outer_splits} dobras e interno {inner_splits} ({len(unique_groups)} grupos)"
    )

    group_array = np.asarray(groups)
    indexable = np.asarray(features, dtype=object) if mode != "embedding" else features

    per_fold: list[dict[str, float]] = []
    predictions = np.empty_like(labels)
    chosen: list[dict[str, Any]] = []

    outer = GroupKFold(n_splits=outer_splits)
    for number, (train_index, test_index) in enumerate(
        outer.split(indexable, labels, group_array), start=1
    ):
        pipeline, grid = build_pipeline(
            spec, mode, max_features=max_features, min_df=min_df
        )
        search = GridSearchCV(
            pipeline,
            grid,
            cv=GroupKFold(n_splits=inner_splits),
            scoring=SCORING,
            n_jobs=n_jobs,
            refit=True,
        )
        train_features = (
            indexable[train_index]
            if mode == "embedding"
            else [features[i] for i in train_index]
        )
        test_features = (
            indexable[test_index]
            if mode == "embedding"
            else [features[i] for i in test_index]
        )
        search.fit(train_features, labels[train_index], groups=group_array[train_index])
        fold_prediction = search.predict(test_features)
        predictions[test_index] = fold_prediction

        truth = labels[test_index]
        per_fold.append(
            {
                "fold": number,
                "n_test": int(len(truth)),
                "accuracy": float(accuracy_score(truth, fold_prediction)),
                "f1": float(f1_score(truth, fold_prediction, average="weighted")),
            }
        )
        chosen.append(search.best_params_)
        logger.info(
            f"  dobra {number}/{outer_splits}: n={len(truth)} "
            f"f1={per_fold[-1]['f1']:.3f} acc={per_fold[-1]['accuracy']:.3f}"
        )

    matrix = confusion_matrix(labels, predictions)
    accuracies = [fold["accuracy"] for fold in per_fold]
    f1_scores = [fold["f1"] for fold in per_fold]
    return {
        "per_fold": per_fold,
        "best_params_per_fold": chosen,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "pooled": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, average="weighted")),
            "precision": float(
                precision_score(labels, predictions, average="weighted")
            ),
            "recall": float(recall_score(labels, predictions, average="weighted")),
        },
        "confusion_matrix": {
            "labels": ["human", "machine"],
            "matrix": matrix.tolist(),
        },
    }


def build_features(
    pairs: Sequence[PairedSample],
    *,
    mode: str,
    encoder: str,
    pooling: str,
    max_length: int,
    cache_dir: Path,
) -> tuple[Any, np.ndarray, list[str], list[PairedSample]]:
    """Prepara a matriz de entrada do modo escolhido.

    Args:
        pairs: Pares do corpus
        mode: Um de :data:`MODES`
        encoder: Apelido do encoder (também define o tokenizador do truncamento)
        pooling: ``auto``, ``cls`` ou ``mean``
        max_length: Teto de tokens
        cache_dir: Pasta do cache de embeddings

    Returns:
        Tupla ``(features, rótulos, grupos, pares truncados)``
    """
    model_name = resolve_model_name(encoder)
    from transformers import AutoTokenizer

    # O truncamento pareado vale nos três modos: é o que impede o
    # classificador de decidir por comprimento em vez de por estilo.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    truncated = truncate_all(pairs, tokenizer, max_length=max_length)
    flattened = flatten_pairs(truncated)

    labels = np.array([item.label for item in flattened])
    groups = [item.group for item in flattened]
    texts = [item.text for item in flattened]

    if mode != "embedding":
        return texts, labels, groups, truncated

    features = load_or_extract(
        texts,
        model_name=model_name,
        pooling=resolve_pooling(encoder, pooling),
        max_length=max_length,
        cache_dir=cache_dir,
    )
    logger.info(f"Embeddings {features.shape}")
    return features, labels, groups, truncated


def run_experiment(
    pairs: Sequence[PairedSample],
    *,
    mode: str,
    encoder: str,
    pooling: str,
    max_length: int,
    classifiers: Sequence[str],
    folds: int,
    max_features: int,
    min_df: int,
    cache_dir: Path,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Roda um modo contra os classificadores pedidos.

    Args:
        pairs: Pares do corpus
        mode: Um de :data:`MODES`
        encoder: Apelido do encoder
        pooling: ``auto``, ``cls`` ou ``mean``
        max_length: Teto de tokens do truncamento
        classifiers: Nomes do catálogo
        folds: Dobras externas da validação cruzada
        max_features: Teto de termos da sacola de palavras
        min_df: Documentos mínimos por termo
        cache_dir: Pasta do cache de embeddings
        n_jobs: Processos da busca em grade

    Returns:
        Relatório com a procedência e um resultado por classificador

    Raises:
        ClassificationError: Se um classificador pedido não existir
    """
    catalogue = build_catalogue()
    unknown = [name for name in classifiers if name not in catalogue]
    if unknown:
        raise ClassificationError(
            f"unknown classifier(s) {unknown}; available: {sorted(catalogue)}"
        )

    features, labels, groups, truncated = build_features(
        pairs,
        mode=mode,
        encoder=encoder,
        pooling=pooling,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    results: dict[str, Any] = {}
    for name in classifiers:
        results[name] = nested_cross_validate(
            catalogue[name],
            features,
            labels,
            groups,
            mode=mode,
            folds=folds,
            max_features=max_features,
            min_df=min_df,
            n_jobs=n_jobs,
        )

    models = sorted({pair.model for pair in truncated if pair.model})
    return {
        "mode": mode,
        "encoder": encoder if mode == "embedding" else None,
        "encoder_model": resolve_model_name(encoder),
        "pooling": resolve_pooling(encoder, pooling) if mode == "embedding" else None,
        "max_length": max_length,
        "max_features": max_features if mode != "embedding" else None,
        "min_df": min_df if mode != "embedding" else None,
        "protocol": "nested GroupKFold, grouped by source article",
        "cv_folds": folds,
        "pairs": len(truncated),
        "documents": int(len(labels)),
        "generators": models,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": results,
    }


def report(payload: dict[str, Any]) -> None:
    """Imprime o quadro comparativo dos classificadores."""
    logger.info("-" * 64)
    logger.info(
        f"{payload['mode']} | {payload['pairs']} pares | {payload['cv_folds']} dobras"
    )
    logger.info(f"{'classificador':<22} {'f1 (média±dp)':>18} {'acurácia':>18}")
    ranked = sorted(payload["results"].items(), key=lambda item: -item[1]["f1_mean"])
    for name, result in ranked:
        logger.info(
            f"{name:<22} "
            f"{result['f1_mean']:>10.3f} ±{result['f1_std']:<6.3f} "
            f"{result['accuracy_mean']:>10.3f} ±{result['accuracy_std']:<6.3f}"
        )
    logger.info(
        "NOTA: com 20 notícias de origem, o intervalo de confiança destas "
        "médias é de dezenas de pontos. São números de viabilidade."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="tfidf")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_BERT_MODEL,
        help=(
            f"Encoder do modo embedding, e o tokenizador do truncamento em "
            f"todos os modos. Apelidos: {', '.join(sorted(BERT_MODELS))}"
        ),
    )
    parser.add_argument(
        "--pooling", choices=("auto", *POOLING_STRATEGIES), default="auto"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--classifier",
        action="append",
        dest="classifiers",
        help="Classificador do catálogo (repetível). Sem a opção, todos.",
    )
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Processos da busca em grade. Use 1 se o backend paralelo do "
            "joblib poluir a saída com erros do resource_tracker (acontece no "
            "WSL); o resultado é o mesmo."
        ),
    )
    add_corpus_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do experimento de classificação."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    documents = documents_from_args(args)
    if not documents:
        logger.error("Corpus vazio — verifique os filtros de modelo e rodada")
        return 1

    pairs = build_pairs(documents)
    classifiers = args.classifiers or sorted(build_catalogue())

    payload = run_experiment(
        pairs,
        mode=args.mode,
        encoder=args.encoder,
        pooling=args.pooling,
        max_length=args.max_length,
        classifiers=classifiers,
        folds=args.cv_folds,
        max_features=args.max_features,
        min_df=args.min_df,
        cache_dir=args.cache_dir,
        n_jobs=args.n_jobs,
    )
    report(payload)

    label = args.mode if args.mode != "embedding" else f"{args.mode}_{args.encoder}"
    output = args.output_dir / args.experiment / f"{label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"Gravado {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
