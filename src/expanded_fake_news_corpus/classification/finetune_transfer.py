"""Fine-tuning de BERT no corpus anterior, testado nos geradores novos.

Replica ``linguistic_features/bert_finetuning.py`` do trabalho anterior
(``noticias_falsas_humano_maquina_semantica``): o encoder inteiro é ajustado com
uma cabeça de classificação (``AutoModelForSequenceClassification``, 2 classes),
sobre o mesmo split 80/20 por par, com o truncamento pareado e uma validação
cruzada de ``--cv-folds`` folds **sobre o treino**, agrupada por par. O fold de
maior F1 de validação é o detector avaliado.

O que muda por causa dos LLMs novos é só o teste, como em
:mod:`~expanded_fake_news_corpus.classification.linguistic_transfer`: além do
teste do trabalho anterior (humano contra Sabiá-3), o detector lê os textos dos
geradores publicados, cortados no mesmo orçamento de tokens (ver
:mod:`~expanded_fake_news_corpus.classification.transfer`).

Uso::

    uv run python -m expanded_fake_news_corpus.classification.finetune_transfer \\
        --bert-model bertimbau

    # Teste rápido do encanamento, sem treinar de verdade
    uv run python -m expanded_fake_news_corpus.classification.finetune_transfer \\
        --bert-model bertimbau --limit-pairs 60 --folds-to-run 1 --epochs 1 \\
        --eval-steps 10 --warmup-steps 5

Os classificadores sobre o ``[CLS]`` do encoder ajustado (o
``finetuned_embeddings.py`` de lá) saem do
:mod:`~expanded_fake_news_corpus.classification.linguistic_transfer` apontando
``--encoder`` para o diretório do melhor fold, que o JSON de resultado informa em
``best_model_dir``::

    uv run python -m expanded_fake_news_corpus.classification.linguistic_transfer \\
        --mode embedding --encoder <best_model_dir> --pooling cls \\
        --classifier all --label finetuned_embedding_bertimbau

Precisa de GPU na prática. O ``torch`` e o ``transformers`` só são importados
quando o treino começa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from expanded_fake_news_corpus.analysis.documents import EXPERIMENTS, PROJECT_ROOT
from expanded_fake_news_corpus.classification.encoders import (
    BERT_MODELS,
    DEFAULT_MAX_LENGTH,
)
from expanded_fake_news_corpus.classification.linguistic_transfer import (
    evaluate,
    prior_test_metrics,
)
from expanded_fake_news_corpus.classification.pairing import (
    MACHINE_LABEL,
    PairedSample,
    flatten_pairs,
)
from expanded_fake_news_corpus.classification.runner import truncate_all
from expanded_fake_news_corpus.classification.splits import (
    DEFAULT_SPLIT_DIR,
    load_prior_split,
)
from expanded_fake_news_corpus.classification.transfer import (
    TransferError,
    build_readings,
    build_test_sets,
    load_training_pairs,
    report,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classification"

#: O fine-tuning vale para os encoders com cabeça [CLS]/pooler treinada.
FINETUNE_MODELS = ("bertimbau", "bert-multilingual")

#: Hiperparâmetros do pipeline original, mantidos como estão.
DEFAULTS: dict[str, Any] = {
    "epochs": 3,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "eval_steps": 500,
    "early_stopping_patience": 3,
    "seed": 42,
}

#: Corpus de origem na ordem em que o trabalho anterior lê os arquivos de split.
_SPLIT_ORDER = (
    ("fakebr", "train_fake_br.txt", "fake_br", 0),
    ("faketruebr", "train_fake_true_br.txt", "fake_true_br", 1),
)


@dataclass(frozen=True)
class Sample:
    """Um texto do treino, com o grupo (o par) a que pertence."""

    uid: str
    group: str
    text: str
    label: int


def prior_group(uid: str) -> str:
    """Nome do grupo do par no trabalho anterior: ``<arquivo>__<corpus>``.

    As folds de lá embaralham os *nomes* dos grupos; usar os mesmos nomes dá os
    mesmos folds. O ``.txt`` do FakeTrueBR é 0-based lá e o nosso ``uid`` é
    1-based.

    Args:
        uid: ``uid`` deste projeto (``fakebr:1386``, ``faketruebr:7``)

    Returns:
        O grupo, por exemplo ``1386.txt__fake_br``

    Raises:
        TransferError: Se o ``uid`` não for de um dos dois corpora
    """
    source, _, number = uid.partition(":")
    for name, _, corpus, offset in _SPLIT_ORDER:
        if source == name:
            return f"{int(number) - offset}.txt__{corpus}"
    raise TransferError(f"uid without a prior corpus: {uid!r}")


def prior_order(split_dir: Path | None = None) -> dict[str, int]:
    """Posição de cada ``uid`` de treino nos arquivos de split, na ordem de lá.

    O treino do trabalho anterior percorre os pares nessa ordem, e a ordem das
    amostras decide o embaralhamento do ``Trainer`` (mesma semente, mesmos
    lotes).

    Args:
        split_dir: Pasta dos arquivos; usa a vendorizada se omitida

    Returns:
        ``uid`` -> posição
    """
    root = split_dir or DEFAULT_SPLIT_DIR
    order: dict[str, int] = {}
    for source, filename, _, offset in _SPLIT_ORDER:
        for line in (root / filename).read_text(encoding="utf-8").splitlines():
            stem = line.strip().removesuffix(".txt")
            if stem.isdigit():
                order.setdefault(f"{source}:{int(stem) + offset}", len(order))
    return order


def build_samples(pairs: Sequence[PairedSample], order: dict[str, int]) -> list[Sample]:
    """Expande os pares em duas amostras, na ordem do trabalho anterior.

    Args:
        pairs: Pares já truncados
        order: Saída de :func:`prior_order`

    Returns:
        Humano e depois máquina de cada par
    """
    ordered = sorted(pairs, key=lambda pair: order.get(pair.uid, len(order)))
    return [
        Sample(item.uid, prior_group(item.uid), item.text, item.label)
        for item in flatten_pairs(ordered)
    ]


def grouped_folds(
    groups: Sequence[str], n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds determinísticos agrupados, como em ``bert_finetuning.py``.

    Embaralha os grupos únicos e os distribui em *round-robin*; cada par tem uma
    amostra humana e uma de máquina, então os folds saem balanceados.

    Args:
        groups: Grupo de cada amostra
        n_folds: Número de folds
        seed: Semente do embaralhamento

    Returns:
        ``(índices de treino, índices de validação)`` por fold
    """
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {group: i % n_folds for i, group in enumerate(unique)}
    fold_ids = np.array([fold_of[group] for group in groups])
    return [
        (np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0])
        for fold in range(n_folds)
    ]


class EncodedDataset:
    """``input_ids`` já tokenizados; o *padding* é feito por lote no *collator*."""

    def __init__(
        self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int
    ):
        encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.items = [
            {"input_ids": ids, "attention_mask": mask, "labels": int(label)}
            for ids, mask, label in zip(
                encoded["input_ids"], encoded["attention_mask"], labels, strict=True
            )
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def compute_metrics(eval_pred) -> dict[str, float]:
    """Acurácia e F1/precisão/revocação ponderados, as métricas do trabalho anterior."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predicted = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="weighted"
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def make_training_args(output_dir: Path, hp: dict[str, Any], run_name: str):
    """``TrainingArguments`` do trabalho anterior."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["train_batch_size"],
        per_device_eval_batch_size=hp["eval_batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_steps=hp["warmup_steps"],
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=hp["eval_steps"],
        save_strategy="steps",
        save_steps=hp["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=hp["seed"],
        report_to="none",
        run_name=run_name,
        dataloader_pin_memory=False,
    )


def train_fold(
    model_name: str,
    tokenizer,
    train_dataset: EncodedDataset,
    val_dataset: EncodedDataset,
    output_dir: Path,
    hp: dict[str, Any],
    run_name: str,
) -> tuple[dict[str, float], Path]:
    """Treina um fold a partir dos pesos pré-treinados.

    A cabeça de classificação é nova (pooler, *dropout* e ``Linear(hidden, 2)``);
    o encoder é ajustado por inteiro, nada fica congelado.

    Args:
        model_name: Id do encoder no Hugging Face
        tokenizer: Tokenizador do encoder
        train_dataset: Treino do fold
        val_dataset: Validação do fold
        output_dir: Pasta dos *checkpoints* do fold
        hp: Hiperparâmetros
        run_name: Nome da execução

    Returns:
        As métricas de validação do fold e a pasta do modelo salvo
    """
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, problem_type="single_label_classification"
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(output_dir, hp, run_name),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])
        ],
    )
    trainer.train()
    validation = trainer.evaluate()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return {
        key.removeprefix("eval_"): value for key, value in validation.items()
    }, final_dir


class FineTunedDetector:
    """O modelo ajustado, com a interface ``predict`` que o resto do código espera.

    Os conjuntos de teste chegam como matrizes de uma coluna de texto (o formato
    dos demais modos); :func:`~linguistic_transfer.evaluate` e
    :func:`~linguistic_transfer.prior_test_metrics` calculam as mesmas métricas
    de sempre. As predições são guardadas por conjunto, para não reler os 2.158
    textos do teste anterior duas vezes.
    """

    def __init__(self, model_dir: Path, tokenizer, max_length: int, batch_size: int):
        from transformers import (
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(model_dir.parent / "test_eval"),
                per_device_eval_batch_size=batch_size,
                report_to="none",
                dataloader_pin_memory=False,
            ),
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        self._cache: dict[str, np.ndarray] = {}

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Logits das duas classes, um par por texto."""
        key = hashlib.sha1("\0".join(texts).encode("utf-8")).hexdigest()
        if key not in self._cache:
            dataset = EncodedDataset(
                texts, [0] * len(texts), self.tokenizer, self.max_length
            )
            self._cache[key] = self.trainer.predict(dataset).predictions
        return self._cache[key]

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Probabilidade de ser máquina, uma por texto."""
        logits = self.logits(texts)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (exp / exp.sum(axis=1, keepdims=True))[:, MACHINE_LABEL]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Rótulo previsto de cada linha da matriz de texto (``argmax``, como lá)."""
        texts = [str(row[0]) for row in matrix]
        return np.argmax(self.logits(texts), axis=1)


def default_label(bert_model: str, max_length: int) -> str:
    """Prefixo padrão dos arquivos: o mesmo de ``bert_finetuning.py``."""
    suffix = "" if max_length == DEFAULT_MAX_LENGTH else f"_len{max_length}"
    return f"finetune_{bert_model.replace('-', '_')}{suffix}"


def text_matrix(texts: Sequence[str]) -> np.ndarray:
    """Textos como matriz de uma coluna, o formato dos conjuntos de teste."""
    return np.array(list(texts), dtype=object).reshape(-1, 1)


def hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """Hiperparâmetros efetivos, a partir da linha de comando."""
    return {key: getattr(args, key) for key in DEFAULTS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Treina os folds, escolhe o melhor e lê os geradores com ele."""
    from transformers import AutoTokenizer, set_seed

    hp = hyperparameters(args)
    set_seed(args.seed)
    model_name = BERT_MODELS[args.bert_model]
    label = args.label or default_label(args.bert_model, args.max_length)
    models_dir = args.models_dir / label

    split = load_prior_split()
    train_uids, test_uids = split.train, split.test
    if args.limit_pairs:
        train_uids = frozenset(sorted(train_uids)[: args.limit_pairs])
        test_uids = frozenset(sorted(test_uids)[: args.limit_pairs])
        logger.warning(f"--limit-pairs: só {args.limit_pairs} pares por parte")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pairs = load_training_pairs(train_uids, prior_dir=args.prior_dir)
    samples = build_samples(
        truncate_all(pairs, tokenizer, max_length=args.max_length), prior_order()
    )
    logger.info(f"Treino: {len(samples)} textos ({len(pairs)} pares)")

    groups = [sample.group for sample in samples]
    labels = [sample.label for sample in samples]
    texts = [sample.text for sample in samples]
    fold_results: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(
        grouped_folds(groups, args.cv_folds, args.seed), start=1
    ):
        if args.folds_to_run and fold > args.folds_to_run:
            break
        # A fuga que esse desenho existe para evitar: o par inteiro de um lado só.
        assert not ({groups[i] for i in train_idx} & {groups[i] for i in val_idx}), (
            "pair split across train and validation"
        )
        logger.info(
            f"--- Fold {fold}/{args.cv_folds}: {len(train_idx)} treino / "
            f"{len(val_idx)} validação ---"
        )
        train_dataset = EncodedDataset(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            tokenizer,
            args.max_length,
        )
        val_dataset = EncodedDataset(
            [texts[i] for i in val_idx],
            [labels[i] for i in val_idx],
            tokenizer,
            args.max_length,
        )
        validation, model_dir = train_fold(
            model_name,
            tokenizer,
            train_dataset,
            val_dataset,
            models_dir / f"fold_{fold}",
            hp,
            run_name=f"{label}/fold_{fold}",
        )
        logger.info(
            f"Fold {fold}: validação F1 {validation['f1']:.4f}, "
            f"acurácia {validation['accuracy']:.4f}"
        )
        fold_results.append(
            {"fold": fold, "val_metrics": validation, "model_dir": str(model_dir)}
        )

    best = max(fold_results, key=lambda item: item["val_metrics"]["f1"])
    logger.info(
        f"Melhor fold: {best['fold']} (F1 de validação "
        f"{best['val_metrics']['f1']:.4f}); lendo os conjuntos de teste"
    )

    test_sets = build_test_sets(
        test_uids,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prior_dir=args.prior_dir,
        experiment=args.experiment,
        models=args.models,
    )
    detector = FineTunedDetector(
        Path(best["model_dir"]), tokenizer, args.max_length, hp["eval_batch_size"]
    )
    sets = {
        item.name: (text_matrix(item.texts), item.expected_label) for item in test_sets
    }
    evaluation = evaluate(detector, sets)
    prior_test = prior_test_metrics(detector, sets)
    predictions = {
        item.name: {
            "uids": list(item.uids),
            "machine_probability": [
                round(float(p), 6) for p in detector.probabilities(item.texts)
            ],
        }
        for item in test_sets
    }

    validation_f1 = [item["val_metrics"]["f1"] for item in fold_results]
    payload: dict[str, Any] = {
        "design": "trained on the prior corpus, tested on unseen generators",
        "mode": "finetune",
        "experiment": args.experiment,
        "label": label,
        "bert_model": model_name,
        "max_length": args.max_length,
        "truncation": "paired (shorter member of the human/machine pair)",
        "classification_head": (
            "AutoModelForSequenceClassification, num_labels=2 (BERT pooler + "
            "dropout + Linear(hidden_size, 2), cross-entropy); encoder fully "
            "fine-tuned"
        ),
        "hyperparameters": {
            **hp,
            "optimizer": "adamw_torch",
            "lr_scheduler": "linear",
            "best_model_metric": "f1",
        },
        "limit_pairs": args.limit_pairs,
        "training": {
            "classifier": f"finetune_{args.bert_model}",
            "pairs": len(pairs),
            "documents": len(samples),
            "cv_folds": args.cv_folds,
            "folds_run": len(fold_results),
        },
        "cross_validation": {
            "folds": fold_results,
            "val_f1_mean": float(np.mean(validation_f1)),
            "val_f1_std": float(np.std(validation_f1)),
            "best_fold": best["fold"],
        },
        "best_model_dir": best["model_dir"],
        "evaluation": evaluation,
        "readings": build_readings(evaluation),
        "prior_test": prior_test,
        "predictions": predictions,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
    }
    report(payload)
    if prior_test:
        logger.info(
            f"teste do trabalho anterior ({prior_test['documents']} docs): "
            f"acurácia {prior_test['accuracy']:.4f}, F1 {prior_test['f1']:.4f}, "
            f"precisão {prior_test['precision']:.4f}, "
            f"revocação {prior_test['recall']:.4f}"
        )

    out_dir = args.output_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{label}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"Gravado {output}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--bert-model", choices=FINETUNE_MODELS, default="bertimbau")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--folds-to-run",
        type=int,
        default=None,
        help="Treina só os N primeiros folds (padrão: todos)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument(
        "--train-batch-size", type=int, default=DEFAULTS["train_batch_size"]
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=DEFAULTS["eval_batch_size"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULTS["learning_rate"]
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    parser.add_argument("--eval-steps", type=int, default=DEFAULTS["eval_steps"])
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULTS["early_stopping_patience"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--experiment", choices=sorted(EXPERIMENTS), default="paper_replication"
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "models",
        help="Onde ficam os checkpoints dos folds (grandes, não versionados)",
    )
    parser.add_argument("--label", help="Prefixo do arquivo de saída")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Teste rápido: só os N primeiros pares de cada parte",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do fine-tuning."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
