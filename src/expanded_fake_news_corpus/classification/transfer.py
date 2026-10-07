"""Transferência: detector treinado no corpus anterior, testado nos geradores novos.

Responde a pergunta que o nosso corpus sozinho não responde. Com 20 notícias
não há como treinar um classificador; mas há como **testar** um treinado em
outro lugar. O trabalho anterior tem 5.391 pares humano/sabiá-3, e um detector
ajustado neles pergunta:

    um detector de fake news de máquina treinado contra um gerador reconhece os
    textos de outro?

O arranjo é assimétrico de propósito:

* **Treino** — pares humano/sabiá-3 restritos aos ``uid`` de *treino* do
  trabalho anterior, com o truncamento pareado de sempre.
* **Teste, lado máquina** — os textos dos nossos geradores, que nenhum detector
  jamais viu. A métrica é a taxa de detecção (quanto o detector marca como
  máquina), por gerador.
* **Teste, lado humano** — as fake news humanas dos ``uid`` de *teste* do
  trabalho anterior, que o modelo também nunca viu. Dá a taxa de falso
  positivo, sem a qual a taxa de detecção não se interpreta.
* **Referência** — os textos sabiá-3 dos mesmos ``uid`` de teste. É a detecção
  dentro da distribuição de treino, o número contra o qual os nossos geradores
  são lidos.

O lado humano **não** sai do nosso recorte de 20 notícias de propósito: 16
delas estão no treino do trabalho anterior (ver
``resources/prior_splits/README.md``), e usá-las mediria memorização.

Uso::

    python -m expanded_fake_news_corpus.classification.transfer \\
        --mode tfidf --classifier logistic_regression
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from expanded_fake_news_corpus.analysis.documents import (
    EXPERIMENTS,
    HUMAN_COHORT,
    PROJECT_ROOT,
    SABIA3_COHORT,
    Document,
    load_machine_documents,
    load_prior_cohort,
)
from expanded_fake_news_corpus.classification.classifiers import build_catalogue
from expanded_fake_news_corpus.classification.encoders import (
    BERT_MODELS,
    DEFAULT_BERT_MODEL,
    DEFAULT_MAX_LENGTH,
    POOLING_STRATEGIES,
    resolve_model_name,
    resolve_pooling,
)
from expanded_fake_news_corpus.classification.pairing import (
    HUMAN_LABEL,
    MACHINE_LABEL,
    PairedSample,
    build_pairs,
    flatten_pairs,
    truncate_text,
)
from expanded_fake_news_corpus.classification.runner import (
    MODES,
    SCORING,
    build_pipeline,
    truncate_all,
)
from expanded_fake_news_corpus.classification.splits import load_prior_split

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classification"


class TransferError(Exception):
    """O experimento de transferência não pôde ser montado."""


@dataclass(frozen=True)
class TestSet:
    """Um conjunto de teste homogêneo, com o rótulo que se espera dele."""

    name: str
    texts: list[str]
    expected_label: int

    @property
    def size(self) -> int:
        return len(self.texts)


def load_training_pairs(
    train_uids: frozenset[str], *, prior_dir: Path | None
) -> list[PairedSample]:
    """Monta os pares humano/sabiá-3 do treino do trabalho anterior.

    Args:
        train_uids: ``uid`` da parte de treino
        prior_dir: Raiz do corpus anterior; detectada se omitida

    Returns:
        Pares com os dois lados presentes

    Raises:
        TransferError: Se nenhuma das duas coortes carregar
    """
    human = load_prior_cohort(HUMAN_COHORT, prior_dir=prior_dir, uids=train_uids)
    machine = load_prior_cohort(SABIA3_COHORT, prior_dir=prior_dir, uids=train_uids)
    if not human or not machine:
        raise TransferError(
            "prior corpus not found; set FAKEGEN_PRIOR_CORPUS or pass --prior-dir"
        )
    # O carregador marca a coorte sintética como MACHINE, que é o que build_pairs
    # espera; o lado humano já vem como HUMAN.
    pairs = build_pairs([*human, *machine])
    logger.info(f"Treino: {len(pairs)} pares do trabalho anterior")
    return pairs


def _documents_to_texts(
    documents: Sequence[Document], tokenizer, *, budget: int
) -> list[str]:
    """Corta os textos de uma coleção no orçamento de teste."""
    return [
        truncate_text(document.text, tokenizer, budget=budget) for document in documents
    ]


def build_test_sets(
    test_uids: frozenset[str],
    *,
    tokenizer,
    budget: int,
    prior_dir: Path | None,
    experiment: str,
    models: Sequence[str] | None,
) -> list[TestSet]:
    """Monta os conjuntos de teste: controle humano, referência e geradores.

    Args:
        test_uids: ``uid`` da parte de teste do trabalho anterior
        tokenizer: Tokenizador do encoder
        budget: Tokens mantidos em cada texto de teste
        prior_dir: Raiz do corpus anterior
        experiment: Experimento de geração deste repositório
        models: Restringe aos geradores pedidos; todos se omitido

    Returns:
        Um conjunto por coorte avaliada
    """
    sets: list[TestSet] = []

    human = load_prior_cohort(HUMAN_COHORT, prior_dir=prior_dir, uids=test_uids)
    sets.append(
        TestSet(
            "humano (teste do trabalho anterior)",
            _documents_to_texts(human, tokenizer, budget=budget),
            HUMAN_LABEL,
        )
    )

    sabia = load_prior_cohort(SABIA3_COHORT, prior_dir=prior_dir, uids=test_uids)
    sets.append(
        TestSet(
            "sabia-3 (referência, dentro da distribuição)",
            _documents_to_texts(sabia, tokenizer, budget=budget),
            MACHINE_LABEL,
        )
    )

    ours = load_machine_documents(EXPERIMENTS[experiment], models=models)
    by_generator: dict[str, list[Document]] = {}
    for document in ours:
        key = document.model or "<sem modelo>"
        if document.round_name:
            key = f"{key} ({document.round_name})"
        by_generator.setdefault(key, []).append(document)

    for name, documents in sorted(by_generator.items()):
        sets.append(
            TestSet(
                name,
                _documents_to_texts(documents, tokenizer, budget=budget),
                MACHINE_LABEL,
            )
        )

    for test_set in sets:
        logger.info(f"Teste '{test_set.name}': {test_set.size} textos")
    return sets


def train_detector(
    pairs: Sequence[PairedSample],
    *,
    mode: str,
    classifier: str,
    max_features: int,
    min_df: int,
    folds: int,
    n_jobs: int,
) -> tuple[Any, dict[str, Any]]:
    """Ajusta o detector no corpus anterior, com busca em grade agrupada.

    Args:
        pairs: Pares de treino, já truncados
        mode: Um de :data:`MODES`
        classifier: Nome do catálogo
        max_features: Teto de termos da sacola de palavras
        min_df: Documentos mínimos por termo
        folds: Dobras da busca em grade
        n_jobs: Processos da busca

    Returns:
        Tupla ``(detector ajustado, procedência do treino)``

    Raises:
        TransferError: Se o classificador não existir no catálogo
    """
    from sklearn.model_selection import GridSearchCV, GroupKFold

    catalogue = build_catalogue()
    if classifier not in catalogue:
        raise TransferError(
            f"unknown classifier {classifier!r}; available: {sorted(catalogue)}"
        )

    flattened = flatten_pairs(pairs)
    texts = [item.text for item in flattened]
    labels = np.array([item.label for item in flattened])
    groups = np.array([item.group for item in flattened])

    spec = catalogue[classifier]
    pipeline, grid = build_pipeline(
        spec, mode, max_features=max_features, min_df=min_df
    )
    logger.info(
        f"Treinando {classifier} em {len(texts)} documentos "
        f"({spec.grid_size} combinações, {folds} dobras agrupadas)"
    )
    search = GridSearchCV(
        pipeline,
        grid,
        cv=GroupKFold(n_splits=folds),
        scoring=SCORING,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(texts, labels, groups=groups)
    logger.info(f"Grade escolhida: {search.best_params_}")
    logger.info(f"{SCORING} na validação do treino: {search.best_score_:.4f}")

    return search.best_estimator_, {
        "classifier": classifier,
        "documents": len(texts),
        "pairs": len(pairs),
        "best_params": search.best_params_,
        "train_cv_score": float(search.best_score_),
        "cv_folds": folds,
    }


def evaluate(detector: Any, test_sets: Sequence[TestSet]) -> dict[str, Any]:
    """Aplica o detector a cada conjunto e mede a fração marcada como máquina.

    Args:
        detector: Pipeline já ajustado
        test_sets: Conjuntos homogêneos a avaliar

    Returns:
        Um resultado por conjunto, com a fração e o acerto esperado
    """
    results: dict[str, Any] = {}
    for test_set in test_sets:
        if not test_set.texts:
            logger.warning(f"Conjunto vazio, ignorado: {test_set.name}")
            continue
        predictions = detector.predict(test_set.texts)
        machine_share = float(np.mean(np.asarray(predictions) == MACHINE_LABEL))
        correct = (
            machine_share
            if test_set.expected_label == MACHINE_LABEL
            else 1.0 - machine_share
        )
        results[test_set.name] = {
            "texts": test_set.size,
            "expected": "machine" if test_set.expected_label else "human",
            "flagged_as_machine": machine_share,
            "accuracy": correct,
        }
    return results


def report(payload: dict[str, Any]) -> None:
    """Imprime o quadro de transferência."""
    logger.info("-" * 72)
    logger.info(
        f"detector: {payload['training']['classifier']} | modo {payload['mode']} | "
        f"{payload['training']['pairs']} pares de treino"
    )
    logger.info(f"{'conjunto':<46} {'n':>5} {'marcado máquina':>16}")
    for name, result in payload["evaluation"].items():
        logger.info(
            f"{name:<46} {result['texts']:>5} {result['flagged_as_machine']:>15.1%}"
        )
    logger.info(
        "A linha humana é a taxa de falso positivo; a do sabiá-3 é a detecção "
        "dentro da distribuição de treino, contra a qual as outras se leem."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="tfidf")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_BERT_MODEL,
        help=f"Apelidos: {', '.join(sorted(BERT_MODELS))}",
    )
    parser.add_argument(
        "--pooling", choices=("auto", *POOLING_STRATEGIES), default="auto"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--test-tokens",
        type=int,
        help=(
            "Tokens mantidos em cada texto de teste. Sem a opção, a mediana do "
            "truncamento pareado do treino — o teste tem de ter o mesmo "
            "comprimento típico do treino, e igual entre geradores."
        ),
    )
    parser.add_argument("--classifier", default="logistic_regression")
    parser.add_argument(
        "--experiment", choices=sorted(EXPERIMENTS), default="paper_replication"
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=10_000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do experimento de transferência."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    from transformers import AutoTokenizer

    model_name = resolve_model_name(args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    split = load_prior_split()
    pairs = load_training_pairs(split.train, prior_dir=args.prior_dir)
    truncated = truncate_all(pairs, tokenizer, max_length=args.max_length)

    budget = args.test_tokens
    if budget is None:
        lengths = [
            len(tokenizer.encode(pair.human_text, add_special_tokens=False))
            for pair in truncated
        ]
        budget = int(np.median(lengths))
        logger.info(f"Orçamento de teste pela mediana do treino: {budget} tokens")

    detector, training = train_detector(
        truncated,
        mode=args.mode,
        classifier=args.classifier,
        max_features=args.max_features,
        min_df=args.min_df,
        folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )

    test_sets = build_test_sets(
        split.test,
        tokenizer=tokenizer,
        budget=budget,
        prior_dir=args.prior_dir,
        experiment=args.experiment,
        models=args.models,
    )
    payload = {
        "design": "trained on the prior corpus, tested on unseen generators",
        "mode": args.mode,
        "encoder": args.encoder if args.mode == "embedding" else None,
        "pooling": resolve_pooling(args.encoder, args.pooling)
        if args.mode == "embedding"
        else None,
        "max_length": args.max_length,
        "test_tokens": budget,
        "experiment": args.experiment,
        "training": training,
        "evaluation": evaluate(detector, test_sets),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    report(payload)

    output = args.output_dir / args.experiment / f"transfer_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"Gravado {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
