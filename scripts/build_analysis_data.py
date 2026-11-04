#!/usr/bin/env python
"""Monta os dados da página de caracterização para o GitHub Pages.

Lê as tabelas geradas pelos módulos de ``expanded_fake_news_corpus.analysis`` e
condensa tudo num único ``docs/analysis.json``: os resultados de silabação,
diversidade lexical, LIWC, NILC-Metrix, SAGE e Zipf, já no recorte que a página
desenha. O NILC-Metrix só existe para os geradores medidos no servidor do NILC;
nos demais a seção vai marcada como indisponível.

A página publica a replicação do artigo, e um bloco por modelo gerador: o lado
humano é o mesmo em todos (as mesmas 20 fake news humanas), então trocar de
gerador na página troca só o lado máquina. Os geradores publicados estão em
:data:`DEFAULT_GENERATORS`; cada um tem a sua pasta de tabelas, porque uma
rodada nunca sobrescreve a outra. As tabelas completas continuam lá — aqui vai
só o que a página mostra, para o JSON não crescer sem necessidade.

A seção de detecção não pertence a um gerador: lê os resultados dos detectores
treinados no trabalho anterior (``data/classification/paper_replication/``) e
mostra todos os geradores lado a lado, num bloco ``classification`` à parte.

Uso::

    uv run python -m expanded_fake_news_corpus.analysis.syllables \
        --experiment paper_replication   # e os demais módulos
    uv run python scripts/build_analysis_data.py
    uv run python scripts/build_analysis_data.py \
        --generator openai/gpt-4.1-mini-2025-04-14=data/analysis/paper_replication
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Rótulos legíveis das métricas de silabação.
SYLLABLE_LABELS = {
    "syllables_per_word": "Sílabas por palavra",
    "syllables_per_sentence": "Sílabas por sentença",
}

#: Rótulos legíveis das métricas de diversidade lexical.
LEXICAL_LABELS = {
    "mattr": "MATTR (janela de 50 palavras)",
    "type_token_ratio": "Tipo/ocorrência bruta",
}

DATASET_LABELS = {
    "fakebr_human": "Fake.br · humano",
    "fakebr_machine": "Fake.br · máquina",
    "faketruebr_human": "FakeTrueBR · humano",
    "faketruebr_machine": "FakeTrueBR · máquina",
    "human": "Humano",
    "machine": "Máquina",
}

#: Categoria LIWC no formato ``codigo (Nome em Inglês)``.
_LIWC_CATEGORY_RE = re.compile(r"^(\S+)\s*\((.+)\)$")

#: Categorias do NILC-Metrix (``nilc_metrix.METRICS_BY_CATEGORY``) para a página.
NILC_CATEGORY_LABELS = {
    "complexity_descriptive": "Descritivas",
    "complexity_simplicity": "Simplicidade",
    "complexity_lexical_diversity": "Diversidade lexical",
    "complexity_frequency": "Frequência",
    "complexity_readability": "Legibilidade",
    "complexity_psycholinguistic": "Psicolinguísticas",
    "cohesion_referential": "Coesão referencial",
    "cohesion_semantic_lsa": "Coesão semântica (LSA)",
    "cohesion_connectives": "Conectivos",
    "semantics_lexical": "Semântica lexical",
}

#: Critério de sinal forte do repositório de semântica (|δ| e q).
NILC_STRONG_DELTA = 0.330
NILC_STRONG_Q = 0.05

#: Quantos itens de cada lista vão para a página.
TOP_LIWC = 74
TOP_SAGE = 25
TOP_ZIPF = 20

#: Geradores publicados na página, na ordem em que aparecem no seletor, cada um
#: com a pasta das suas tabelas. O primeiro é o que abre por padrão.
#:
#: A pasta do gpt-4.1-mini é a raiz do experimento por ser a primeira rodada;
#: as seguintes ficam em subpastas com o modelo no nome (veja o ``--output-dir``
#: dos módulos de análise) para não sobrescrevê-la.
DEFAULT_GENERATORS: tuple[tuple[str, Path], ...] = (
    (
        "openai/gpt-4.1-mini-2025-04-14",
        Path("data/analysis/paper_replication"),
    ),
    (
        "anthropic/claude-sonnet-4-5-20250929",
        Path("data/analysis/paper_replication/claude-sonnet-4-5-20250929"),
    ),
    (
        "ollama/qwen3:32b",
        Path("data/analysis/paper_replication/qwen3_32b"),
    ),
    (
        "ollama/llama3.1:8b",
        Path("data/analysis/paper_replication/llama3.1_8b_r2"),
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lê um CSV, devolvendo lista vazia se o arquivo não existir."""
    if not path.exists():
        logger.warning(f"Tabela ausente, seção ficará vazia: {path}")
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, *, digits: int | None = None) -> float | None:
    """Converte um campo de CSV em float, tolerando vazio e ``nan``."""
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number:  # nan
        return None
    return round(number, digits) if digits is not None else number


def load_syllables(analysis_dir: Path) -> dict:
    """Resultados da silabação: as duas métricas e os totais por recorte."""
    significance = read_csv(analysis_dir / "syllables" / "syllables_significance.csv")
    by_dataset = read_csv(analysis_dir / "syllables" / "syllables_by_dataset.csv")
    by_group = read_csv(analysis_dir / "syllables" / "syllables_by_group.csv")

    return {
        "metrics": [_comparison_row(row, SYLLABLE_LABELS) for row in significance],
        "by_group": {row["group"]: int(row["documents"]) for row in by_group},
        "by_dataset": [
            {
                "dataset": row["dataset"],
                "label": DATASET_LABELS.get(row["dataset"], row["dataset"]),
                "documents": int(row["documents"]),
                "words": int(row["words"]),
                "syllables_per_word": as_float(
                    row["syllables_per_word_weighted"], digits=3
                ),
            }
            for row in by_dataset
        ],
    }


def load_lexical_diversity(analysis_dir: Path) -> dict:
    """Diversidade lexical: MATTR e razão bruta, com os totais por recorte."""
    folder = analysis_dir / "lexical_diversity"
    significance = read_csv(folder / "lexical_diversity_significance.csv")
    by_dataset = read_csv(folder / "lexical_diversity_by_dataset.csv")

    return {
        "metrics": [_comparison_row(row, LEXICAL_LABELS) for row in significance],
        "by_dataset": [
            {
                "dataset": row["dataset"],
                "label": DATASET_LABELS.get(row["dataset"], row["dataset"]),
                "documents": int(row["documents"]),
                "tokens": int(row["tokens"]),
                "mattr": as_float(row["mattr_mean"], digits=3),
                "type_token_ratio": as_float(row["type_token_ratio_mean"], digits=3),
            }
            for row in by_dataset
        ],
    }


def load_liwc(analysis_dir: Path) -> dict:
    """Categorias LIWC ordenadas por tamanho do efeito, com p bruto e q corrigido."""
    rows = read_csv(analysis_dir / "liwc" / "liwc_significance.csv")
    categories = [_liwc_row(row) for row in rows[:TOP_LIWC]]

    return {
        "categories": categories,
        "total": len(rows),
        "significant_raw": sum(
            1 for row in rows if (as_float(row["u_p_value"]) or 1.0) < 0.05
        ),
        "significant_fdr": sum(
            1 for row in rows if row.get("significant_fdr_005") == "True"
        ),
    }


def load_nilc(analysis_dir: Path) -> dict:
    """Ranking do NILC-Metrix por δ de Cliff, com as ressalvas de cada métrica.

    Returns:
        ``{"available": False}`` quando o gerador não foi medido; senão as 117
        métricas e as contagens de sinais fortes, com e sem as ressalvadas
    """
    path = analysis_dir / "nilc_metrix" / "nilc_metrix_ranked.csv"
    rows = read_csv(path) if path.exists() else []
    if not rows:
        return {"available": False}

    metrics = []
    for row in rows:
        delta = as_float(row["cliffs_delta"], digits=3)
        q_value = as_float(row["q_value"])
        metrics.append(
            {
                "metric": row["metric"],
                "category": NILC_CATEGORY_LABELS.get(row["category"], row["category"]),
                "cliffs_delta": delta,
                "human_median": as_float(row["median_human"], digits=3),
                "machine_median": as_float(row["median_machine"], digits=3),
                "p_value": as_float(row["u_p_value"]),
                "q_value": q_value,
                "magnitude": row["magnitude"],
                "strong": delta is not None
                and abs(delta) >= NILC_STRONG_DELTA
                and q_value is not None
                and q_value < NILC_STRONG_Q,
                "layout_dependent": row["layout_dependent"] == "True",
                "installation_dependent": row.get("installation_dependent") == "True",
            }
        )

    strong = [m for m in metrics if m["strong"]]
    return {
        "available": True,
        "pairs": int(rows[0]["n_machine"]),
        "metrics": metrics,
        "total": len(metrics),
        "strong": len(strong),
        "strong_clean": sum(
            1
            for m in strong
            if not m["layout_dependent"] and not m["installation_dependent"]
        ),
    }


def load_sage(analysis_dir: Path) -> dict:
    """Termos distintivos do SAGE, separados por autoria."""
    rows = read_csv(analysis_dir / "sage" / "sage_distinctive_terms.csv")

    def side(group: str) -> list[dict]:
        return [
            {
                "term": row["term"],
                "distinctiveness": as_float(row["abs_distinctiveness"], digits=3),
            }
            for row in rows
            if row["characteristic_of"] == group
        ][:TOP_SAGE]

    return {"human": side("human"), "machine": side("machine")}


def load_zipf(analysis_dir: Path) -> dict:
    """Palavras mais frequentes sem stopwords e o tamanho de cada recorte."""
    zipf_dir = analysis_dir / "zipf" / "without_stopwords"
    comparison = read_csv(zipf_dir / "top_words_comparison.csv")

    top_words = [
        {
            "rank": int(row["rank"]),
            "human_word": row.get("human_word", ""),
            "human_frequency": int(row["human_frequency"])
            if row.get("human_frequency")
            else 0,
            "machine_word": row.get("machine_word", ""),
            "machine_frequency": (
                int(row["machine_frequency"]) if row.get("machine_frequency") else 0
            ),
        }
        for row in comparison[:TOP_ZIPF]
    ]

    return {"top_words": top_words, "recuts": list(_zipf_recuts(zipf_dir))}


def _zipf_recuts(zipf_dir: Path) -> Iterator[dict]:
    """Totais de ocorrências e de tipos de cada tabela de frequência."""
    for path in sorted(zipf_dir.glob("*_frequencies.csv")):
        dataset = path.name.removesuffix("_frequencies.csv")
        rows = read_csv(path)
        tokens = sum(int(row["frequency"]) for row in rows)
        yield {
            "dataset": dataset,
            "label": DATASET_LABELS.get(dataset, dataset),
            "tokens": tokens,
            "types": len(rows),
            "type_token_ratio": round(len(rows) / tokens, 4) if tokens else 0.0,
        }


def _comparison_row(row: dict[str, str], labels: dict[str, str]) -> dict:
    """Campos comuns a qualquer linha das tabelas de significância."""
    metric = row["metric"]
    return {
        "metric": metric,
        "label": labels.get(metric, metric),
        "human_mean": as_float(row["human_mean"], digits=3),
        "machine_mean": as_float(row["machine_mean"], digits=3),
        "cohens_d": as_float(row["cohens_d"], digits=3),
        "p_value": as_float(row["u_p_value"]),
        "effect_size": row.get("effect_size", ""),
    }


def _liwc_row(row: dict[str, str]) -> dict:
    """Linha de categoria LIWC, com o código separado do nome."""
    entry = _comparison_row(row, {})
    match = _LIWC_CATEGORY_RE.match(entry["metric"])
    entry["label"] = match.group(1) if match else entry["metric"]
    entry["name"] = match.group(2) if match else ""
    entry["q_value"] = as_float(row.get("q_value"))
    entry["significant"] = row.get("significant_fdr_005") == "True"
    return entry


#: Modos do detector, na ordem da tabela: (id, família, rótulo, rótulo dos arquivos).
#: O fine-tuning direto não tem classificador: a decisão sai da cabeça do BERT.
CLASSIFICATION_MODES: tuple[tuple[str, str, str, str], ...] = (
    ("linguistic", "Linguístico", "Features linguísticas", "linguistic"),
    ("embedding_bertimbau", "Embedding congelado", "BERTimbau", "embedding_bertimbau"),
    (
        "embedding_bert_multilingual",
        "Embedding congelado",
        "mBERT",
        "embedding_bert_multilingual",
    ),
    (
        "embedding_norberto",
        "Embedding congelado",
        "NorBERTo (mean pooling)",
        "embedding_norberto",
    ),
    (
        "combined_bertimbau",
        "Combinado",
        "BERTimbau + linguístico",
        "combined_bertimbau",
    ),
    (
        "combined_bert_multilingual",
        "Combinado",
        "mBERT + linguístico",
        "combined_bert_multilingual",
    ),
    (
        "combined_norberto",
        "Combinado",
        "NorBERTo + linguístico",
        "combined_norberto",
    ),
    ("tfidf_linguistic", "Texto", "TF-IDF + linguístico", "text_tfidf_linguistic"),
    ("tfidf_only", "Texto", "TF-IDF", "text_tfidf_only"),
    ("bow_linguistic", "Texto", "BoW + linguístico", "text_bow_linguistic"),
    ("bow_only", "Texto", "BoW", "text_bow_only"),
    (
        "finetune_bertimbau",
        "Fine-tuning",
        "BERTimbau (cabeça do BERT)",
        "finetune_bertimbau",
    ),
    (
        "finetune_bert_multilingual",
        "Fine-tuning",
        "mBERT (cabeça do BERT)",
        "finetune_bert_multilingual",
    ),
    (
        "finetune_norberto",
        "Fine-tuning",
        "NorBERTo (cabeça do ModernBERT)",
        "finetune_norberto",
    ),
    (
        "finetuned_embedding_bertimbau",
        "[CLS] ajustado",
        "BERTimbau",
        "finetuned_embedding_bertimbau",
    ),
    (
        "finetuned_embedding_bert_multilingual",
        "[CLS] ajustado",
        "mBERT",
        "finetuned_embedding_bert_multilingual",
    ),
    (
        "finetuned_embedding_norberto",
        "[CLS] ajustado",
        "NorBERTo (mean pooling)",
        "finetuned_embedding_norberto",
    ),
)

#: Geradores na ordem das colunas: (modelo no JSON, rótulo da coluna).
CLASSIFICATION_GENERATORS: tuple[tuple[str, str], ...] = (
    ("openai/gpt-4.1-mini-2025-04-14", "GPT-4.1-mini"),
    ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
    ("ollama/qwen3:32b", "Qwen3 32B"),
    ("ollama/llama3.1:8b (rodada2)", "Llama 3.1 8B (rodada 2)"),
)

#: O Naive Bayes marca texto demais como máquina (falso positivo de 7 a 19% em
#: vários modos) e infla a detecção; fica fora das médias, mas aparece no detalhe.
CLASSIFICATION_MEAN_EXCLUDES = frozenset({"naive_bayes"})

#: Textos humanos do teste do trabalho anterior (denominador do falso positivo;
#: ex.: 4/1079 = 0,37% no SVM linguístico).
PRIOR_HUMAN_TEST_SIZE = 1079

#: Ablação do XGBoost com features linguísticas, feita fora deste script (os
#: números vêm de execuções avulsas, não dos ``transfer_*.json``). Cada passo
#: tira features do NILC-Metrix antes de treinar; mede o Llama (rodada 2).
XGBOOST_ABLATION: tuple[tuple[str, str, float, float], ...] = (
    ("none", "Nenhuma removida", 1.0, 0.001),
    ("top2", "Sem as 2 principais features", 0.8, 0.004),
    ("top16", "Sem as 16 principais features", 0.5, 0.011),
    ("nilc", "Sem todo o NILC-Metrix", 0.4, 0.021),
)

CLASSIFIER_ORDER = (
    "svm",
    "random_forest",
    "naive_bayes",
    "logistic_regression",
    "mlp",
    "xgboost",
)


def _classification_row(payload: dict) -> dict:
    """Uma execução (um classificador) no formato da página."""
    readings = payload["readings"]
    detection = readings["llms_only"]["detection_rate"]
    reference = readings["llms_only"].get("reference_rate", {})
    return {
        "prior_accuracy": round(payload["prior_test"]["accuracy"], 4),
        "false_positive_rate": round(
            readings["with_human_control"]["false_positive_rate"], 4
        ),
        "reference_rate": round(next(iter(reference.values()), float("nan")), 4),
        "detection": {
            key: round(detection[key], 4) for key, _ in CLASSIFICATION_GENERATORS
        },
    }


def _classification_mean(rows: list[dict]) -> dict:
    """Média simples das execuções, campo a campo."""

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4)

    return {
        "prior_accuracy": mean([r["prior_accuracy"] for r in rows]),
        "false_positive_rate": mean([r["false_positive_rate"] for r in rows]),
        "detection": {
            key: mean([r["detection"][key] for r in rows])
            for key, _ in CLASSIFICATION_GENERATORS
        },
    }


def load_classification(directory: Path) -> dict | None:
    """Resultados dos detectores treinados no trabalho anterior.

    Args:
        directory: Pasta com os ``transfer_*.json`` e ``finetune_*.json``

    Returns:
        O bloco da página, ou ``None`` se a pasta não existir
    """
    if not directory.is_dir():
        logger.warning(f"sem resultados de classificação em {directory}")
        return None

    modes = []
    for mode_id, family, label, stem in CLASSIFICATION_MODES:
        rows: dict[str, dict] = {}
        direct = directory / f"{stem}.json"
        if direct.is_file():
            rows["fine-tuning"] = _classification_row(json.loads(direct.read_text()))
        else:
            for classifier in CLASSIFIER_ORDER:
                path = directory / f"transfer_{stem}_{classifier}.json"
                if path.is_file():
                    rows[classifier] = _classification_row(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
        if not rows:
            logger.warning(f"modo sem resultados: {mode_id}")
            continue
        counted = [
            row
            for name, row in rows.items()
            if name not in CLASSIFICATION_MEAN_EXCLUDES
        ]
        modes.append(
            {
                "id": mode_id,
                "family": family,
                "label": label,
                "classifiers": rows,
                "mean": _classification_mean(counted),
            }
        )
    return {
        "generators": [
            {"key": key, "label": label} for key, label in CLASSIFICATION_GENERATORS
        ],
        "mean_excludes": sorted(CLASSIFICATION_MEAN_EXCLUDES),
        "human_test_size": PRIOR_HUMAN_TEST_SIZE,
        "modes": modes,
        "xgboost_ablation": {
            "label": "Ablação do XGBoost linguístico",
            "generator": "ollama/llama3.1:8b (rodada2)",
            "steps": [
                {"id": i, "label": l, "detection": d, "false_positive_rate": f}
                for i, l, d, f in XGBOOST_ABLATION
            ],
        },
    }


def build_generator(model: str, analysis_dir: Path) -> dict:
    """Junta as cinco seções de um gerador.

    Args:
        model: Modelo em ``provedor/modelo``
        analysis_dir: Pasta com as tabelas daquele gerador

    Returns:
        Bloco pronto para a página de caracterização
    """
    return {
        "model": model,
        "label": model.split("/")[-1],
        "analysis_dir": str(analysis_dir),
        "syllables": load_syllables(analysis_dir),
        "lexical_diversity": load_lexical_diversity(analysis_dir),
        "liwc": load_liwc(analysis_dir),
        "nilc": load_nilc(analysis_dir),
        "sage": load_sage(analysis_dir),
        "zipf": load_zipf(analysis_dir),
    }


def build_payload(
    generators: Sequence[tuple[str, Path]],
    classification_dir: Path | None = None,
) -> dict:
    """Monta o payload da página, um bloco por gerador.

    Args:
        generators: Pares ``(modelo, pasta de tabelas)``, na ordem do seletor
        classification_dir: Pasta dos resultados dos detectores; a seção de
            detecção fica de fora se omitida

    Returns:
        Dados prontos para a página de caracterização
    """
    classification = (
        load_classification(classification_dir) if classification_dir else None
    )
    return {
        "corpus": "FakeGen.BR",
        "experiment": "paper_replication",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generators": [
            build_generator(model, analysis_dir) for model, analysis_dir in generators
        ],
        "classification": classification,
    }


def parse_generator(value: str) -> tuple[str, Path]:
    """Lê um ``--generator`` no formato ``provedor/modelo=pasta``.

    Raises:
        argparse.ArgumentTypeError: Se faltar o ``=`` separando os dois campos
    """
    model, separator, directory = value.partition("=")
    if not separator or not model.strip() or not directory.strip():
        raise argparse.ArgumentTypeError(
            f"esperado 'provedor/modelo=pasta', recebido {value!r}"
        )
    return model.strip(), Path(directory.strip())


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do build."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--generator",
        action="append",
        dest="generators",
        type=parse_generator,
        metavar="MODELO=PASTA",
        help=(
            "Gerador a publicar, repetível, no formato 'provedor/modelo=pasta'. "
            "Sem a opção, os de DEFAULT_GENERATORS."
        ),
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=Path("data/classification/paper_replication"),
        help="Resultados dos detectores (seção de detecção)",
    )
    args = parser.parse_args(argv)

    payload = build_payload(
        args.generators or list(DEFAULT_GENERATORS), args.classification_dir
    )

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    output = args.docs_dir / "analysis.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    for generator in payload["generators"]:
        sage_terms = len(generator["sage"]["human"]) + len(generator["sage"]["machine"])
        logger.info(
            f"{generator['label']}: {len(generator['liwc']['categories'])} "
            f"categorias LIWC, {sage_terms} termos SAGE, "
            f"{len(generator['zipf']['recuts'])} recortes Zipf"
        )
    logger.info(f"{len(payload['generators'])} geradores -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
