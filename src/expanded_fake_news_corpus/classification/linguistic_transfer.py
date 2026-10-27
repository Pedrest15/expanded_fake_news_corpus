"""Transferência com features linguísticas e embeddings: detector do corpus anterior.

Versão de :mod:`transfer` para os três modos do trabalho anterior
(``linguistic_features.py --feature-mode``):

* ``linguistic`` — o vetor de features pré-computadas: NILC-Metrix, LIWC, regras
  de dependência, sílabas, UPOS, regras por sentença e bigramas do SAGE;
* ``embedding`` — o vetor do encoder congelado (BERTimbau ou mBERT) sobre o
  texto com truncamento pareado;
* ``combined`` — os dois concatenados;
* ``tfidf`` e ``bow`` — sacola de palavras sobre o texto truncado, juntada às
  features linguísticas padronizadas (``text_enriched.py`` de lá, com a grade B,
  menor), ou só o texto com ``--no-linguistic``.

Aceita um classificador, uma lista ou ``--classifier all`` (os seis do catálogo).

O treino sai de ``data/classification/prior_features.csv``, gerado por
``scripts/export_prior_features.py`` com o código original, e portanto já com o
filtro de comprimento de lá (contagens absolutas e agregadores fora). O teste
monta **as mesmas colunas** para os nossos geradores:

=================  ================================================================
Grupo              De onde vem aqui
=================  ================================================================
``nilc__``         ``data/nilc_metrix/metrics.csv``, texto inteiro
``liwc__``         ``liwc_per_document.csv`` da pasta do gerador
``ud__``           regras da árvore básica (``grammar_rules.sentence_rules``) do
                   CoNLL-U, como contagem da regra sobre o total do documento;
                   as 500 regras são as escolhidas no treino
``syll__``         ``syllables_per_document.csv``
``pos__``          ``pos_per_document.csv``, normalizado pelas 17 UPOS
``psr__``          regras totais sobre sentenças, do mesmo CoNLL-U
``sage__``         contagem dos bigramas do contraste **anterior** no texto
=================  ================================================================

As features linguísticas valem sobre o texto inteiro, como no treino: o
truncamento pareado do trabalho anterior só vale para embedding e BoW/TF-IDF, e o
controle de comprimento daqui é o filtro de colunas. Um gerador sem alguma das
tabelas fica com o grupo vazio, e
o imputador do pipeline o preenche com a mediana do treino — o JSON registra a
cobertura de cada grupo para isso não passar despercebido.

Nos modos de texto, o treino usa os pares do corpus anterior (``--prior-dir`` ou
``FAKEGEN_PRIOR_CORPUS``) e todo texto de teste é cortado no mesmo orçamento de
tokens, como em :mod:`transfer`.

Uso::

    python -m expanded_fake_news_corpus.classification.linguistic_transfer \\
        --mode linguistic --classifier all
    python -m expanded_fake_news_corpus.classification.linguistic_transfer \\
        --mode combined --encoder bert-multilingual --classifier all
    python -m expanded_fake_news_corpus.classification.linguistic_transfer \\
        --mode tfidf --classifier all
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from expanded_fake_news_corpus.analysis.conllu import read_conllu
from expanded_fake_news_corpus.analysis.documents import (
    EXPERIMENTS,
    PROJECT_ROOT,
    Document,
    load_machine_documents,
)
from expanded_fake_news_corpus.analysis.grammar_rules import sentence_rules
from expanded_fake_news_corpus.classification.classifiers import (
    RANDOM_STATE,
    ClassifierSpec,
    build_catalogue,
    build_enriched_catalogue,
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
    HUMAN_LABEL,
    MACHINE_LABEL,
    flatten_pairs,
)
from expanded_fake_news_corpus.classification.runner import SCORING, truncate_all
from expanded_fake_news_corpus.classification.splits import load_prior_split
from expanded_fake_news_corpus.classification.transfer import (
    HUMAN_SET_NAME,
    REFERENCE_SET_NAME,
    TransferError,
    build_readings,
    is_published,
    load_training_pairs,
    report,
)
from expanded_fake_news_corpus.classification.transfer import (
    build_test_sets as build_text_sets,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classification"
DEFAULT_PRIOR_FEATURES = DEFAULT_OUTPUT_DIR / "prior_features.csv"
DEFAULT_NILC_CSV = PROJECT_ROOT / "data" / "nilc_metrix" / "metrics.csv"
DEFAULT_PARSED_ROOT = PROJECT_ROOT / "data" / "parsed"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "data" / "analysis"

#: Modos de entrada do detector.
MODES = ("linguistic", "embedding", "combined", "tfidf", "bow")

#: Modos em que o texto é vetorizado por contagem (BoW/TF-IDF).
TEXT_MODES = ("tfidf", "bow")

#: Coluna do texto na matriz dos modos ``tfidf`` e ``bow``.
TEXT_COLUMN = "text"

#: Grupos de features, na ordem do manifesto do treino.
FEATURE_GROUPS = (
    "nilcmetrics",
    "liwc",
    "enhanced_ud",
    "syllables",
    "pos_tagger",
    "parser_stats",
    "sage_terms",
)

UPOS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

#: Pasta de cada gerador em ``data/analysis/<experimento>`` e ``data/parsed``,
#: pela dupla ``(modelo, rodada)``. A pasta vazia é a raiz do experimento, onde
#: ficou o primeiro gerador (o do lado humano).
GENERATOR_DIRS: dict[tuple[str, str | None], str] = {
    ("openai/gpt-4.1-mini-2025-04-14", None): "",
    ("anthropic/claude-sonnet-4-5-20250929", None): "claude-sonnet-4-5-20250929",
    ("ollama/qwen3:32b", None): "qwen3_32b",
    ("ollama/llama3.1:8b", "rodada2"): "llama3.1_8b_r2",
}

_TOKEN_RE = re.compile(r"[a-záàâãéèêíïóôõöúçñ0-9]+", re.IGNORECASE | re.UNICODE)


class FeatureGroupError(TransferError):
    """Um grupo de features não pôde ser montado."""


def generator_label(model: str | None, round_name: str | None) -> str:
    """Nome do gerador nos resultados — o mesmo de :mod:`transfer`."""
    label = model or "<sem modelo>"
    return f"{label} ({round_name})" if round_name else label


def load_prior_table(
    path: Path, groups: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Lê a tabela de treino exportada e o manifesto das colunas.

    Args:
        path: ``prior_features.csv``
        groups: Grupos de features a manter

    Returns:
        Tupla ``(tabela de (uid, label) -> features, colunas por grupo)``

    Raises:
        TransferError: Se a tabela não existir
    """
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise TransferError(
            f"{path} not found; run scripts/export_prior_features.py with the "
            "prior repository's Python"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    columns = {name: manifest["groups"][name] for name in groups}
    table = pd.read_csv(path).set_index(["uid", "label"])
    selected = [c for cols in columns.values() for c in cols]
    return table[selected], columns


def _by_uid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("uid")


def nilc_features(
    model: str, round_name: str | None, columns: list[str], *, path: Path
) -> pd.DataFrame:
    """Métricas do NILC-Metrix do gerador, com as colunas do treino."""
    frame = pd.read_csv(path)
    frame = frame[(frame["group"] == "machine") & (frame["model"] == model)]
    frame = frame[
        frame["round"].isna() if round_name is None else frame["round"] == round_name
    ]
    renamed = _by_uid(frame).rename(columns=lambda c: f"nilc__{c}")
    return renamed.reindex(columns=columns)


def table_features(
    path: Path, model: str, mapping: dict[str, str], *, prefix: str
) -> pd.DataFrame:
    """Colunas de uma tabela por documento, renomeadas para as do treino."""
    if not path.is_file():
        return pd.DataFrame(columns=[f"{prefix}{c}" for c in mapping.values()])
    frame = pd.read_csv(path)
    frame = frame[(frame["group"] == "machine") & (frame["model"] == model)]
    selected = _by_uid(frame)[list(mapping)]
    return selected.rename(columns={k: f"{prefix}{v}" for k, v in mapping.items()})


def liwc_features(path: Path, model: str, columns: list[str]) -> pd.DataFrame:
    """LIWC do gerador; os nomes de categoria são os do treino sem o prefixo."""
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    frame = _by_uid(frame[(frame["group"] == "machine") & (frame["model"] == model)])
    return frame.rename(columns=lambda c: f"liwc__{c}").reindex(columns=columns)


def pos_features(path: Path, columns: list[str]) -> pd.DataFrame:
    """Frequência de cada UPOS sobre o total de tokens com UPOS."""
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    frame = _by_uid(pd.read_csv(path).query("group == 'machine'"))
    counts = frame[UPOS_TAGS].fillna(0)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return shares.rename(columns=lambda c: f"pos__{c}").reindex(columns=columns)


def rule_features(
    parsed_dir: Path, rule_columns: list[str], stats_columns: list[str]
) -> pd.DataFrame:
    """Regras de dependência e regras por sentença, do CoNLL-U do gerador.

    A frequência de cada uma das regras do treino é a contagem sobre o total de
    regras do documento, e as estatísticas do parser são regras por sentença —
    as mesmas contas de ``load_enhanced_ud`` e ``load_parser_stats``.
    """
    rows: dict[str, dict[str, float]] = {}
    wanted = {c.removeprefix("ud__"): c for c in rule_columns}
    for path in sorted(parsed_dir.glob("*.conllu")):
        if path.name.endswith(".eud.conllu"):
            continue
        uid = path.stem.replace("_", ":", 1)
        sentences = read_conllu(path)
        rules = [rule for sentence in sentences for rule in sentence_rules(sentence)]
        total = len(rules) or 1
        row: dict[str, float] = {column: 0.0 for column in rule_columns}
        for rule, count in Counter(rules).items():
            if rule in wanted:
                row[wanted[rule]] = count / total
        row["psr__rules_per_sentence"] = len(rules) / max(len(sentences), 1)
        rows[uid] = row
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame.reindex(columns=[*rule_columns, *stats_columns])


def sage_features(documents: Sequence[Document], columns: list[str]) -> pd.DataFrame:
    """Contagem, em cada texto, dos bigramas distintivos do contraste anterior.

    Os nomes ``sage__<a>_<b>`` carregam o bigrama, e o sublinhado não pode ser
    parte de um token (a expressão de tokenização o exclui), então a separação é
    exata.
    """
    bigrams = [tuple(c.removeprefix("sage__").split("_")) for c in columns]
    rows: dict[str, list[int]] = {}
    for document in documents:
        tokens = _TOKEN_RE.findall(document.text.lower())
        pairs = Counter(zip(tokens, tokens[1:], strict=False))
        rows[document.uid] = [pairs.get(bigram, 0) for bigram in bigrams]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def build_generator_matrix(
    documents: Sequence[Document],
    model: str,
    round_name: str | None,
    groups: dict[str, list[str]],
    *,
    nilc_csv: Path,
    analysis_dir: Path,
    parsed_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Matriz de features de um gerador, com as colunas e a ordem do treino.

    Returns:
        Tupla ``(matriz uid x feature, cobertura por grupo)``; a cobertura é a
        fração dos documentos que têm o grupo
    """
    blocks: dict[str, pd.DataFrame] = {}
    if "nilcmetrics" in groups:
        blocks["nilcmetrics"] = nilc_features(
            model, round_name, groups["nilcmetrics"], path=nilc_csv
        )
    if "liwc" in groups:
        blocks["liwc"] = liwc_features(
            analysis_dir / "liwc" / "liwc_per_document.csv", model, groups["liwc"]
        )
    if "syllables" in groups:
        blocks["syllables"] = table_features(
            analysis_dir / "syllables" / "syllables_per_document.csv",
            model,
            {
                "syllables_per_sentence": "media_silabas_por_sentenca",
                "syllables_per_word": "media_silabas_por_palavra",
            },
            prefix="syll__",
        ).reindex(columns=groups["syllables"])
    if "pos_tagger" in groups:
        blocks["pos_tagger"] = pos_features(
            analysis_dir / "pos" / "pos_per_document.csv", groups["pos_tagger"]
        )
    if "enhanced_ud" in groups or "parser_stats" in groups:
        rules = rule_features(
            parsed_dir, groups.get("enhanced_ud", []), groups.get("parser_stats", [])
        )
        if "enhanced_ud" in groups:
            blocks["enhanced_ud"] = rules[groups["enhanced_ud"]]
        if "parser_stats" in groups:
            blocks["parser_stats"] = rules[groups["parser_stats"]]
    if "sage_terms" in groups:
        blocks["sage_terms"] = sage_features(documents, groups["sage_terms"])

    uids = sorted({document.uid for document in documents})
    matrix = pd.concat(
        [blocks[name].reindex(uids) for name in groups if name in blocks], axis=1
    )
    coverage = {
        name: float(block.reindex(uids).notna().any(axis=1).mean())
        for name, block in blocks.items()
    }
    return matrix, coverage


def build_linguistic_sets(
    table: pd.DataFrame,
    groups: dict[str, list[str]],
    test_uids: frozenset[str],
    *,
    experiment: str,
    models: Sequence[str] | None,
    nilc_csv: Path,
    analysis_root: Path,
    parsed_root: Path,
) -> tuple[dict[str, tuple[pd.DataFrame, int]], dict[str, dict[str, float]]]:
    """Vetores linguísticos do controle humano, da referência e dos geradores.

    Returns:
        Tupla ``({nome: (matriz indexada por uid, rótulo esperado)}, cobertura
        por gerador)``
    """
    uids = sorted(test_uids)
    sets: dict[str, tuple[pd.DataFrame, int]] = {}
    for name, label in (
        (HUMAN_SET_NAME, HUMAN_LABEL),
        (REFERENCE_SET_NAME, MACHINE_LABEL),
    ):
        index = [(uid, label) for uid in uids if (uid, label) in table.index]
        rows = table.loc[index]
        rows.index = rows.index.get_level_values("uid")
        sets[name] = (rows, label)

    documents = [
        document
        for document in load_machine_documents(EXPERIMENTS[experiment], models=models)
        if is_published(document)
    ]
    by_generator: dict[tuple[str, str | None], list[Document]] = {}
    for document in documents:
        by_generator.setdefault((document.model or "", document.round_name), []).append(
            document
        )

    coverage: dict[str, dict[str, float]] = {}
    for (model, round_name), docs in sorted(by_generator.items()):
        name = generator_label(model, round_name)
        folder = GENERATOR_DIRS.get((model, round_name))
        if folder is None:
            logger.warning(f"Sem pasta de análise para {name}; ignorado")
            continue
        matrix, covered = build_generator_matrix(
            docs,
            model,
            round_name,
            groups,
            nilc_csv=nilc_csv,
            analysis_dir=analysis_root / experiment / folder,
            parsed_dir=parsed_root / folder / experiment / "machine",
        )
        sets[name] = (matrix, MACHINE_LABEL)
        coverage[name] = covered
        gaps = [group for group, share in covered.items() if share < 1.0]
        if gaps:
            logger.warning(
                f"{name}: cobertura incompleta em "
                + ", ".join(f"{g} ({covered[g]:.0%})" for g in gaps)
                + " — o imputador preenche com a mediana do treino"
            )
    return sets, coverage


@dataclass
class Dataset:
    """Matrizes de treino e de teste de um experimento, prontas para ajustar."""

    X_train: Any  # ndarray, ou DataFrame (texto + linguísticas) nos modos de texto
    y_train: np.ndarray
    groups: np.ndarray
    sets: dict[str, tuple[Any, int]]
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    #: Etapa que vem antes do classificador; ``None`` é imputador + padronizador.
    preprocessor: Any = None
    catalogue: dict[str, ClassifierSpec] = field(default_factory=dict)
    #: Nome de cada coluna que chega ao classificador, para a importância.
    feature_names: list[str] = field(default_factory=list)


def uses_linguistic(args: argparse.Namespace) -> bool:
    """Se o modo lê as features linguísticas pré-computadas."""
    if args.mode in TEXT_MODES:
        return not args.no_linguistic
    return args.mode != "embedding"


def catalogue_for(mode: str) -> dict[str, ClassifierSpec]:
    """Grade A (linguístico, embedding, combinado) ou B (BoW/TF-IDF)."""
    return build_enriched_catalogue() if mode in TEXT_MODES else build_catalogue()


def make_vectorizer(kind: str, max_features: int, min_df: int):
    """Vetorizador do ``text_enriched.py`` do trabalho anterior."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    if kind == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            min_df=min_df,
            sublinear_tf=True,
            lowercase=True,
        )
    return CountVectorizer(
        max_features=max_features, ngram_range=(1, 1), min_df=min_df, lowercase=True
    )


def make_preprocessor(
    kind: str, max_features: int, min_df: int, ling_columns: Sequence[str]
):
    """Vetorizador do texto e, se houver, imputador + padronizador das linguísticas.

    Vive dentro do pipeline da busca em grade, então vocabulário, IDF, medianas e
    escala são reajustados só com o treino de cada dobra. A saída é esparsa, na
    ordem texto, linguísticas.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformers: list[tuple[str, Any, Any]] = [
        (TEXT_COLUMN, make_vectorizer(kind, max_features, min_df), TEXT_COLUMN)
    ]
    if ling_columns:
        transformers.append(
            (
                "ling",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(ling_columns),
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def experiment_label(args: argparse.Namespace, groups_wanted: Sequence[str]) -> str:
    """Prefixo dos arquivos de saída: modo, encoder e o que sair do padrão.

    Os modos de texto mantêm os nomes do trabalho anterior
    (``text_tfidf_linguistic``, ``text_tfidf_only``), com o encoder só se não for
    o padrão — nesses modos ele serve apenas ao truncamento.
    """
    if args.label:
        return args.label
    encoder = args.encoder.replace("-", "_")
    if args.mode == "linguistic":
        parts = ["linguistic"]
    elif args.mode in TEXT_MODES:
        parts = ["text", args.mode, "only" if args.no_linguistic else "linguistic"]
        if args.encoder != DEFAULT_BERT_MODEL:
            parts.append(encoder)
    else:
        parts = [args.mode, encoder]
        if resolve_pooling(args.encoder, args.pooling) != resolve_pooling(args.encoder):
            parts.append(args.pooling)
    if args.mode != "linguistic" and args.max_length != DEFAULT_MAX_LENGTH:
        parts.append(f"len{args.max_length}")
    if uses_linguistic(args) and len(groups_wanted) < len(FEATURE_GROUPS):
        parts.append("_".join(groups_wanted))
    return "_".join(parts)


def assemble_dataset(
    args: argparse.Namespace, groups_wanted: Sequence[str], split
) -> Dataset:
    """Monta treino e teste do modo pedido, uma vez para todos os classificadores.

    No modo ``linguistic`` as linhas vêm direto da tabela exportada. Nos modos de
    texto, o treino são os pares do corpus anterior com o truncamento pareado, e o
    teste, os textos cortados no mesmo orçamento de tokens (ver :mod:`transfer`);
    no ``combined`` o vetor linguístico do mesmo ``uid`` e rótulo é concatenado
    ao embedding.
    """
    test_uids = split.test
    train_uids = split.train
    if args.limit_pairs:
        test_uids = frozenset(sorted(test_uids)[: args.limit_pairs])
        train_uids = frozenset(sorted(train_uids)[: args.limit_pairs])

    table = columns = ling_sets = None
    coverage: dict[str, dict[str, float]] = {}
    if uses_linguistic(args):
        table, columns = load_prior_table(args.prior_features, groups_wanted)
        ling_sets, coverage = build_linguistic_sets(
            table,
            columns,
            test_uids,
            experiment=args.experiment,
            models=args.models,
            nilc_csv=args.nilc_csv,
            analysis_root=DEFAULT_ANALYSIS_ROOT,
            parsed_root=DEFAULT_PARSED_ROOT,
        )

    if args.mode == "linguistic":
        assert table is not None and ling_sets is not None
        train = table[table.index.get_level_values("uid").isin(train_uids)]
        return Dataset(
            train.to_numpy(dtype=float),
            train.index.get_level_values("label").to_numpy(),
            train.index.get_level_values("uid").to_numpy(),
            {n: (m.to_numpy(dtype=float), e) for n, (m, e) in ling_sets.items()},
            coverage,
            catalogue=catalogue_for(args.mode),
            feature_names=list(table.columns),
        )

    from transformers import AutoTokenizer

    model_name = resolve_model_name(args.encoder)
    pooling = resolve_pooling(args.encoder, args.pooling)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pairs = sorted(
        load_training_pairs(train_uids, prior_dir=args.prior_dir), key=lambda p: p.uid
    )
    truncated = truncate_all(pairs, tokenizer, max_length=args.max_length)
    flattened = flatten_pairs(truncated)
    y_train = np.array([item.label for item in flattened])
    groups = np.array([item.uid for item in flattened])
    train_texts = [item.text for item in flattened]
    index = pd.MultiIndex.from_arrays([groups, y_train], names=["uid", "label"])

    text_sets = build_text_sets(
        test_uids,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prior_dir=args.prior_dir,
        experiment=args.experiment,
        models=args.models,
    )
    extra: dict[str, Any] = {
        "encoder": args.encoder,
        "encoder_model": model_name,
        "max_length": args.max_length,
        "test_truncation": "paired",
    }

    if args.mode in TEXT_MODES:
        ling_columns = list(table.columns) if table is not None else []
        X_train = pd.DataFrame({TEXT_COLUMN: train_texts})
        if table is not None:
            X_train = pd.concat(
                [X_train, table.reindex(index).reset_index(drop=True)], axis=1
            )
        sets: dict[str, tuple[Any, int]] = {}
        for test_set in text_sets:
            if not test_set.texts:
                continue
            frame = pd.DataFrame({TEXT_COLUMN: list(test_set.texts)})
            if ling_sets is not None:
                if test_set.name not in ling_sets:
                    logger.warning(f"Sem features linguísticas para {test_set.name}")
                    continue
                linguistic = ling_sets[test_set.name][0].reindex(test_set.uids)
                frame = pd.concat([frame, linguistic.reset_index(drop=True)], axis=1)
            sets[test_set.name] = (frame, test_set.expected_label)
        return Dataset(
            X_train,
            y_train,
            groups,
            sets,
            coverage,
            {
                **extra,
                "vectorizer": args.mode,
                "max_features": args.max_features,
                "min_df": args.min_df,
                "linguistic": not args.no_linguistic,
            },
            make_preprocessor(args.mode, args.max_features, args.min_df, ling_columns),
            catalogue_for(args.mode),
        )

    cache_dir = args.output_dir / "cache"
    X_train = load_or_extract(
        train_texts,
        model_name=model_name,
        pooling=pooling,
        max_length=args.max_length,
        cache_dir=cache_dir,
    )
    if table is not None:
        X_train = np.hstack(
            [table.reindex(index).to_numpy(dtype=float), X_train.astype(float)]
        )

    sets = {}
    for test_set in text_sets:
        if not test_set.texts:
            continue
        matrix = load_or_extract(
            test_set.texts,
            model_name=model_name,
            pooling=pooling,
            max_length=args.max_length,
            cache_dir=cache_dir,
        ).astype(float)
        if ling_sets is not None:
            if test_set.name not in ling_sets:
                logger.warning(f"Sem features linguísticas para {test_set.name}")
                continue
            linguistic = ling_sets[test_set.name][0].reindex(test_set.uids)
            matrix = np.hstack([linguistic.to_numpy(dtype=float), matrix])
        sets[test_set.name] = (matrix, test_set.expected_label)

    linguistic_names = list(table.columns) if table is not None else []
    embedding_names = [
        f"bert_{args.encoder}__{i}"
        for i in range(X_train.shape[1] - len(linguistic_names))
    ]
    return Dataset(
        X_train,
        y_train,
        groups,
        sets,
        coverage,
        {**extra, "pooling": pooling},
        None,
        catalogue_for(args.mode),
        linguistic_names + embedding_names,
    )


def train_detector(
    data: Dataset, *, classifier: str, folds: int, n_jobs: int
) -> tuple[Any, dict[str, Any]]:
    """Ajusta o pré-processamento e o classificador no treino anterior.

    Imputador (mediana), padronizador e, nos modos de texto, o vetorizador ficam
    dentro do pipeline, como lá, para a busca em grade reajustá-los a cada dobra.

    Raises:
        TransferError: Se o classificador não existir no catálogo
    """
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    catalogue = data.catalogue
    if classifier not in catalogue:
        raise TransferError(
            f"unknown classifier {classifier!r}; available: {sorted(catalogue)}"
        )
    spec = catalogue[classifier]
    if data.preprocessor is None:
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    else:
        steps = [("prep", clone(data.preprocessor))]
    pipeline = Pipeline([*steps, ("clf", copy.deepcopy(spec.estimator))])
    grid = {f"clf__{key}": values for key, values in spec.param_grid.items()}
    documents, features = data.X_train.shape
    logger.info(
        f"Treinando {classifier} em {documents} documentos, {features} colunas "
        f"({spec.grid_size} combinações, {folds} dobras agrupadas)"
    )
    search = GridSearchCV(
        pipeline,
        grid,
        cv=GroupKFold(n_splits=folds),
        scoring=SCORING,
        n_jobs=spec.n_jobs if spec.n_jobs is not None else n_jobs,
        refit=True,
    )
    search.fit(data.X_train, data.y_train, groups=data.groups)
    logger.info(f"Grade escolhida: {search.best_params_}")
    logger.info(f"{SCORING} na validação do treino: {search.best_score_:.4f}")
    training = {
        "classifier": classifier,
        "documents": documents,
        "pairs": documents // 2,
        "features": features,
        "best_params": search.best_params_,
        "train_cv_score": float(search.best_score_),
        "cv_folds": folds,
    }
    if data.preprocessor is not None:
        fitted = search.best_estimator_.named_steps["prep"]
        training["vocabulary"] = len(
            fitted.named_transformers_[TEXT_COLUMN].vocabulary_
        )
    return search.best_estimator_, training


def evaluate(detector: Any, sets: dict[str, tuple[np.ndarray, int]]) -> dict[str, Any]:
    """Fração marcada como máquina em cada conjunto (formato de :mod:`transfer`)."""
    results: dict[str, Any] = {}
    for name, (matrix, expected) in sets.items():
        if len(matrix) == 0:
            logger.warning(f"Conjunto vazio, ignorado: {name}")
            continue
        predictions = detector.predict(matrix)
        share = float(np.mean(np.asarray(predictions) == MACHINE_LABEL))
        results[name] = {
            "texts": len(matrix),
            "expected": "machine" if expected == MACHINE_LABEL else "human",
            "flagged_as_machine": share,
            "accuracy": share if expected == MACHINE_LABEL else 1.0 - share,
        }
    return results


def _stack(matrices: Sequence[Any]) -> Any:
    """Empilha matrizes numpy ou DataFrames, conforme o modo."""
    if isinstance(matrices[0], pd.DataFrame):
        return pd.concat(matrices, ignore_index=True)
    return np.vstack(matrices)


def prior_test_metrics(
    detector: Any, sets: dict[str, tuple[Any, int]]
) -> dict[str, Any] | None:
    """Métricas do teste do próprio trabalho anterior, humano contra Sabiá-3.

    São as de ``evaluate`` em ``linguistic_features.py``: acurácia e F1,
    precisão e revocação ponderados, e a matriz de confusão — no mesmo teste
    (os ``uid`` de teste de lá, com o truncamento par a par), para conferir que o
    detector reproduz o experimento original antes de ler os geradores novos.

    Returns:
        As métricas, ou ``None`` se faltar algum dos dois conjuntos
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if HUMAN_SET_NAME not in sets or REFERENCE_SET_NAME not in sets:
        return None
    human, _ = sets[HUMAN_SET_NAME]
    machine, _ = sets[REFERENCE_SET_NAME]
    truth = np.array([HUMAN_LABEL] * len(human) + [MACHINE_LABEL] * len(machine))
    predicted = detector.predict(_stack([human, machine]))
    return {
        "documents": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1": float(f1_score(truth, predicted, average="weighted")),
        "precision": float(precision_score(truth, predicted, average="weighted")),
        "recall": float(recall_score(truth, predicted, average="weighted")),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=[HUMAN_LABEL, MACHINE_LABEL]
        ).tolist(),
        "labels": ["human", "machine"],
    }


def feature_importance(
    detector: Any, feature_names: Sequence[str], top_k: int = 20
) -> list[list[Any]] | None:
    """Importância por impureza/ganho (árvores) ou ``|coef|`` (lineares).

    Como em ``analyze_feature_importance`` do trabalho anterior; ``None`` para os
    classificadores que não expõem nenhum dos dois.
    """
    classifier = detector.named_steps["clf"]
    if hasattr(classifier, "feature_importances_"):
        weights = np.asarray(classifier.feature_importances_)
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        coef = coef.toarray() if hasattr(coef, "toarray") else np.asarray(coef)
        weights = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(weights) != len(feature_names):
        return None
    ranked = sorted(
        zip(feature_names, weights, strict=True), key=lambda x: x[1], reverse=True
    )
    return [[name, float(value)] for name, value in ranked[:top_k]]


def fitted_feature_names(detector: Any, data: Dataset) -> list[str]:
    """Nomes das colunas que chegam ao classificador do detector ajustado."""
    if data.preprocessor is None:
        return data.feature_names
    vectorizer = detector.named_steps["prep"].named_transformers_[TEXT_COLUMN]
    linguistic = [c for c in data.X_train.columns if c != TEXT_COLUMN]
    return [f"text__{w}" for w in vectorizer.get_feature_names_out()] + linguistic


def summarise(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Quadro comparativo dos classificadores de uma mesma bateria."""
    summary: dict[str, Any] = {}
    for classifier, payload in payloads.items():
        llms = payload["readings"]["llms_only"]
        human = payload["readings"]["with_human_control"]
        summary[classifier] = {
            "train_cv_score": payload["training"]["train_cv_score"],
            "detection_rate": llms["detection_rate"],
            "reference_rate": llms["reference_rate"],
            "mean_detection": llms["mean"],
            "false_positive_rate": human["false_positive_rate"],
        }
    return summary


def resolve_classifiers(value: str, mode: str) -> list[str]:
    """``all``, um nome ou vários separados por vírgula."""
    available = list(catalogue_for(mode))
    if value == "all":
        return available
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise TransferError(f"unknown classifiers {unknown}; available: {available}")
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="linguistic")
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        help="Nome, lista separada por vírgula, ou 'all'",
    )
    parser.add_argument(
        "--features",
        default=",".join(FEATURE_GROUPS),
        help=f"Grupos separados por vírgula: {', '.join(FEATURE_GROUPS)}",
    )
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
        "--max-features",
        type=int,
        default=10_000,
        help="Teto de termos do BoW/TF-IDF (modos tfidf e bow)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Documentos mínimos por termo (modos tfidf e bow)",
    )
    parser.add_argument(
        "--no-linguistic",
        action="store_true",
        help="Modos tfidf e bow sem as features linguísticas (só o texto)",
    )
    parser.add_argument(
        "--experiment", choices=sorted(EXPERIMENTS), default="paper_replication"
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--prior-features", type=Path, default=DEFAULT_PRIOR_FEATURES)
    parser.add_argument("--nilc-csv", type=Path, default=DEFAULT_NILC_CSV)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", help="Prefixo dos arquivos de saída")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        help="Só para conferir o encanamento: N primeiros uids de treino e de teste",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Não grava o detector ajustado em <output-dir>/models",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do experimento de transferência."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    groups_wanted = [g.strip() for g in args.features.split(",") if g.strip()]
    unknown = sorted(set(groups_wanted) - set(FEATURE_GROUPS))
    if unknown:
        raise TransferError(f"unknown feature groups: {unknown}")
    classifiers = resolve_classifiers(args.classifier, args.mode)
    label = experiment_label(args, groups_wanted)

    split = load_prior_split()
    data = assemble_dataset(args, groups_wanted, split)
    for name, (matrix, _) in data.sets.items():
        logger.info(f"Teste '{name}': {len(matrix)} textos")

    out_dir = args.output_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    for classifier in classifiers:
        detector, training = train_detector(
            data, classifier=classifier, folds=args.cv_folds, n_jobs=args.n_jobs
        )
        evaluation = evaluate(detector, data.sets)
        prior_test = prior_test_metrics(detector, data.sets)
        importance = feature_importance(
            detector,
            fitted_feature_names(detector, data),
            top_k=30 if args.mode in TEXT_MODES else 20,
        )
        payload = {
            "design": "trained on the prior corpus, tested on unseen generators",
            "mode": args.mode,
            "feature_groups": groups_wanted if uses_linguistic(args) else [],
            **data.extra,
            "experiment": args.experiment,
            "limit_pairs": args.limit_pairs,
            "training": training,
            "coverage": data.coverage,
            "evaluation": evaluation,
            "readings": build_readings(evaluation),
            "prior_test": prior_test,
            "feature_importance": importance,
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": RANDOM_STATE,
        }
        report(payload)
        if prior_test:
            logger.info(
                f"teste do trabalho anterior ({prior_test['documents']} docs): "
                f"acurácia {prior_test['accuracy']:.4f}, F1 {prior_test['f1']:.4f}, "
                f"precisão {prior_test['precision']:.4f}, "
                f"revocação {prior_test['recall']:.4f}"
            )
        payloads[classifier] = payload

        output = out_dir / f"transfer_{label}_{classifier}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(f"Gravado {output}")
        if not args.no_save_models:
            models_dir = args.output_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"detector": detector, "mode": args.mode, **data.extra},
                models_dir / f"{label}_{classifier}.joblib",
            )

    if len(payloads) > 1:
        comparison = out_dir / f"comparison_{label}.json"
        comparison.write_text(
            json.dumps(summarise(payloads), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Gravado {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
