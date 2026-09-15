"""Regras de gramática de dependências: produtividade e regras distintivas.

Cada token do CoNLL-U vira uma regra no formato do trabalho anterior (Andrade
et al., PROPOR 2026): a UPOS do núcleo seguida dos dependentes, na ordem
linear, com ``*`` marcando a posição do núcleo e cada dependente escrito como
``UPOS/deprel``::

    VERB(NOUN/nsubj, *, NOUN/obj, PUNCT/punct)

Todo token gera também a regra-folha ``UPOS(*)`` (ou ``*(UPOS)`` se for a
raiz), então uma sentença com ``n`` tokens produz pelo menos ``n`` regras.

Duas perguntas, duas tabelas:

* **Produtividade sintática** — quantas regras (e quantas distintas) cada
  sentença usa. O trabalho anterior achou a máquina com mais regras por
  sentença e menor diversidade; os testes aqui repetem os dele, por sentença
  (todas as sentenças de um grupo juntas) e, adicionalmente, por documento.
* **Regras distintivas** — TF-IDF por documento com as regras como vocabulário,
  e Mann-Whitney + Cohen's d sobre os pesos de cada regra entre os grupos, nas
  duas variantes do trabalho anterior: com repetição (TF real) e sem (só
  presença).

Lê os CoNLL-U de ``data/parsed/<experimento>/`` (ver :mod:`conllu`); não chama
o parser.

Uso::

    python -m expanded_fake_news_corpus.analysis.grammar_rules \\
        --experiment paper_replication
    python -m expanded_fake_news_corpus.analysis.grammar_rules --no-deprel
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

from expanded_fake_news_corpus.analysis.conllu import (
    DEFAULT_PARSED_ROOT,
    ParsedDocument,
    Sentence,
    load_parsed_documents,
)
from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
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
    interpret_effect_size,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "grammar_rules"

#: Mínimo de documentos em que uma regra precisa aparecer para entrar no TF-IDF.
#: O trabalho anterior usava 10 sobre milhares de documentos; com 20 pares o
#: mesmo corte deixaria quase nada.
DEFAULT_MIN_DOCS = 5

DEFAULT_TOP_K = 50

#: Métricas comparadas por documento e por sentença.
DOCUMENT_METRICS = (
    "rules_per_sentence",
    "unique_rules_per_sentence",
    "rule_diversity",
    "words_per_sentence",
)
SENTENCE_METRICS = ("rules", "unique_rules", "words")


@dataclass(frozen=True)
class RuleFormat:
    """Quanto detalhe entra em cada regra."""

    include_upos: bool = True
    include_deprel: bool = True

    def dependent(self, upos: str, deprel: str) -> str:
        if self.include_upos and self.include_deprel:
            return f"{upos}/{deprel}"
        if self.include_upos:
            return f"{upos}/_"
        if self.include_deprel:
            return f"_/{deprel}"
        return "_/_"

    def head(self, upos: str) -> str:
        return upos if self.include_upos else "_"

    @property
    def label(self) -> str:
        return {
            (True, True): "upos+deprel",
            (True, False): "upos",
            (False, True): "deprel",
            (False, False): "shape",
        }[(self.include_upos, self.include_deprel)]


#: Formato do trabalho anterior: UPOS do núcleo e ``UPOS/deprel`` dos dependentes.
FULL_FORMAT = RuleFormat()


def sentence_rules(sentence: Sentence, fmt: RuleFormat = FULL_FORMAT) -> list[str]:
    """Regras de uma sentença, uma por token mais uma por núcleo com dependentes.

    Args:
        sentence: Sentença anotada
        fmt: Formato das regras

    Returns:
        Regras na ordem dos tokens; núcleo com dependentes gera duas (a regra
        com os dependentes e a regra-folha)
    """
    words = sentence.words
    dependents: dict[int, list] = {w.index: [] for w in words}
    for word in words:
        if word.head_index in dependents:
            dependents[word.head_index].append(word)

    rules: list[str] = []
    for word in words:
        is_root = word.head_index == 0
        deps = sorted(dependents[word.index], key=lambda w: w.index)
        if deps:
            parts = (
                [fmt.dependent(d.upos, d.deprel) for d in deps if d.index < word.index]
                + ["*"]
                + [
                    fmt.dependent(d.upos, d.deprel)
                    for d in deps
                    if d.index > word.index
                ]
            )
            rules.append(f"{fmt.head(word.upos)}({', '.join(parts)})")
        rules.append(
            f"*({fmt.head(word.upos)})" if is_root else f"{fmt.head(word.upos)}(*)"
        )
    return rules


#: Função que transforma uma sentença anotada em regras.
RuleExtractor = Callable[[Sentence, RuleFormat], list[str]]


def build_sentence_frame(
    documents: Sequence[ParsedDocument],
    fmt: RuleFormat = FULL_FORMAT,
    *,
    extractor: RuleExtractor = sentence_rules,
) -> pd.DataFrame:
    """Uma linha por sentença: contagens de regras e as regras em si.

    Args:
        documents: Documentos parseados
        fmt: Formato das regras
        extractor: Como tirar as regras da sentença — :func:`sentence_rules`
            (árvore básica) ou :func:`eud_rules.enhanced_rules`
    """
    rows: list[dict[str, object]] = []
    for document in documents:
        for sentence in document.sentences:
            rules = extractor(sentence, fmt)
            rows.append(
                {
                    "uid": document.uid,
                    "sent_id": sentence.sent_id,
                    "source": document.document.source.value,
                    "group": document.group.value,
                    "dataset": document.dataset,
                    "words": len(sentence.words),
                    "rules": len(rules),
                    "unique_rules": len(set(rules)),
                    "rule_list": rules,
                }
            )
    return pd.DataFrame(rows)


def build_document_frame(sentence_frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega as sentenças por documento."""
    rows: list[dict[str, object]] = []
    for (uid, group), block in sentence_frame.groupby(["uid", "group"], sort=False):
        all_rules = [rule for rules in block["rule_list"] for rule in rules]
        rows.append(
            {
                "uid": uid,
                "source": block["source"].iloc[0],
                "group": group,
                "dataset": block["dataset"].iloc[0],
                "sentences": len(block),
                "words": int(block["words"].sum()),
                "rules": len(all_rules),
                "unique_rules": len(set(all_rules)),
                "rules_per_sentence": float(block["rules"].mean()),
                "unique_rules_per_sentence": float(block["unique_rules"].mean()),
                "rule_diversity": len(set(all_rules)) / len(all_rules)
                if all_rules
                else 0.0,
                "words_per_sentence": float(block["words"].mean()),
            }
        )
    return pd.DataFrame(rows)


def rule_frequencies(sentence_frame: pd.DataFrame) -> pd.DataFrame:
    """Contagem de cada regra por grupo, em ocorrências e em documentos."""
    counts: dict[str, Counter] = {g.value: Counter() for g in Group}
    docs: dict[str, Counter] = {g.value: Counter() for g in Group}
    for (_uid, group), block in sentence_frame.groupby(["uid", "group"], sort=False):
        seen: set[str] = set()
        for rules in block["rule_list"]:
            counts[group].update(rules)
            seen.update(rules)
        docs[group].update(seen)

    rules = sorted(set(counts["human"]) | set(counts["machine"]))
    frame = pd.DataFrame(
        {
            "rule": rules,
            "human_count": [counts["human"][r] for r in rules],
            "machine_count": [counts["machine"][r] for r in rules],
            "human_docs": [docs["human"][r] for r in rules],
            "machine_docs": [docs["machine"][r] for r in rules],
        }
    )
    frame["total_count"] = frame["human_count"] + frame["machine_count"]
    return frame.sort_values("total_count", ascending=False).reset_index(drop=True)


def discriminative_rules(
    document_rules: Sequence[tuple[str, str, list[str]]],
    *,
    with_repetition: bool,
    min_docs: int = DEFAULT_MIN_DOCS,
) -> pd.DataFrame:
    """Regras que separam os grupos, pelo peso TF-IDF em cada documento.

    Args:
        document_rules: Triplas ``(uid, grupo, regras do documento)``
        with_repetition: TF real (True) ou binário (False), como no trabalho
            anterior
        min_docs: Descarta regras presentes em menos documentos que isto

    Returns:
        Uma linha por regra, ordenada pela diferença absoluta de médias, com
        p-valor de Mann-Whitney, q-valor (FDR) e Cohen's d (``human - machine``)
    """
    corpus = [
        rules if with_repetition else sorted(set(rules))
        for _, _, rules in document_rules
    ]
    labels = np.array([group for _, group, _ in document_rules])

    vectorizer = TfidfVectorizer(
        analyzer=lambda doc: doc, use_idf=True, norm="l2", smooth_idf=True
    )
    matrix = vectorizer.fit_transform(corpus).toarray()
    names = vectorizer.get_feature_names_out()

    human_idx = np.where(labels == Group.HUMAN.value)[0]
    machine_idx = np.where(labels == Group.MACHINE.value)[0]

    rows: list[dict[str, object]] = []
    for column, rule in enumerate(names):
        human = matrix[human_idx, column]
        machine = matrix[machine_idx, column]
        human_docs = int((human > 0).sum())
        machine_docs = int((machine > 0).sum())
        if human_docs + machine_docs < min_docs:
            continue

        diff = float(human.mean() - machine.mean())
        try:
            u_statistic, p_value = stats.mannwhitneyu(
                human, machine, alternative="two-sided"
            )
        except ValueError:
            u_statistic, p_value = float("nan"), float("nan")
        pooled = np.sqrt((human.std() ** 2 + machine.std() ** 2) / 2)
        cohens_d = diff / pooled if pooled > 0 else 0.0

        rows.append(
            {
                "rule": rule,
                "human_mean": float(human.mean()),
                "machine_mean": float(machine.mean()),
                "difference": diff,
                "abs_difference": abs(diff),
                "human_docs": human_docs,
                "machine_docs": machine_docs,
                "pct_human_docs": 100 * human_docs / len(human_idx),
                "pct_machine_docs": 100 * machine_docs / len(machine_idx),
                "u_statistic": float(u_statistic),
                "u_p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "effect_size": interpret_effect_size(float(cohens_d)),
                "characteristic_of": (
                    "human" if diff > 0 else "machine" if diff < 0 else "neutral"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = add_fdr_correction(frame)
    return frame.sort_values("abs_difference", ascending=False).reset_index(drop=True)


def top_rules(
    frame: pd.DataFrame, group: str, top_k: int = DEFAULT_TOP_K
) -> pd.DataFrame:
    """As regras mais características de um grupo."""
    if frame.empty:
        return frame
    return frame[frame["characteristic_of"] == group].head(top_k).reset_index(drop=True)


def run_analysis(
    documents: Sequence[ParsedDocument],
    output_dir: Path,
    *,
    fmt: RuleFormat = FULL_FORMAT,
    min_docs: int = DEFAULT_MIN_DOCS,
    top_k: int = DEFAULT_TOP_K,
    extractor: RuleExtractor = sentence_rules,
    prefix: str = "grammar_rules",
) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos parseados
        output_dir: Pasta de saída
        fmt: Formato das regras
        min_docs: Documentos mínimos por regra no TF-IDF
        top_k: Regras por grupo nas tabelas de topo
        extractor: Como tirar as regras da sentença (ver
            :func:`build_sentence_frame`)
        prefix: Prefixo dos arquivos gravados

    Returns:
        Tabela por documento
    """
    sentence_frame = build_sentence_frame(documents, fmt, extractor=extractor)
    if sentence_frame.empty:
        logger.warning("Nenhuma sentença parseada — nada a gravar")
        return sentence_frame
    document_frame = build_document_frame(sentence_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        sentence_frame.drop(columns=["rule_list"]),
        output_dir / f"{prefix}_per_sentence.csv",
    )
    _write_csv(document_frame, output_dir / f"{prefix}_per_document.csv")
    _write_csv(
        document_frame.groupby("dataset")
        .agg(
            documents=("uid", "count"),
            sentences=("sentences", "sum"),
            rules=("rules", "sum"),
            rules_per_sentence=("rules_per_sentence", "mean"),
            unique_rules_per_sentence=("unique_rules_per_sentence", "mean"),
            rule_diversity=("rule_diversity", "mean"),
        )
        .reset_index(),
        output_dir / f"{prefix}_by_dataset.csv",
    )

    _write_csv(
        comparisons_to_frame(compare_frame(document_frame, DOCUMENT_METRICS)),
        output_dir / f"{prefix}_significance.csv",
    )
    _write_csv(
        compare_within(document_frame, DOCUMENT_METRICS, split_column="source"),
        output_dir / f"{prefix}_significance_by_source.csv",
    )
    # Por sentença, como no trabalho anterior: cada sentença é uma observação.
    _write_csv(
        comparisons_to_frame(compare_frame(sentence_frame, SENTENCE_METRICS)),
        output_dir / f"{prefix}_sentence_significance.csv",
    )

    _write_csv(
        rule_frequencies(sentence_frame), output_dir / f"{prefix}_frequencies.csv"
    )

    document_rules = [
        (uid, group, [r for rules in block["rule_list"] for r in rules])
        for (uid, group), block in sentence_frame.groupby(["uid", "group"], sort=False)
    ]
    for with_repetition, name in (
        (True, "with_repetition"),
        (False, "without_repetition"),
    ):
        frame = discriminative_rules(
            document_rules, with_repetition=with_repetition, min_docs=min_docs
        )
        _write_csv(frame, output_dir / f"{prefix}_tfidf_{name}.csv")
        for group in (Group.HUMAN, Group.MACHINE):
            _write_csv(
                top_rules(frame, group.value, top_k),
                output_dir / f"{prefix}_top_{group.value}_{name}.csv",
            )

    return document_frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        logger.warning(f"Tabela vazia, arquivo não gerado: {path.name}")
        return
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Gravado {path} ({len(frame)} linhas)")


def build_parser(
    description: str, default_output_dir: Path = DEFAULT_OUTPUT_DIR
) -> argparse.ArgumentParser:
    """Parser compartilhado com :mod:`eud_rules`."""
    parser = argparse.ArgumentParser(description=description)
    add_corpus_arguments(parser)
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=DEFAULT_PARSED_ROOT,
        help=f"Raiz dos CoNLL-U (default: {DEFAULT_PARSED_ROOT})",
    )
    parser.add_argument(
        "--no-upos", action="store_true", help="Regras sem a UPOS dos dependentes"
    )
    parser.add_argument(
        "--no-deprel", action="store_true", help="Regras sem a relação dos dependentes"
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=DEFAULT_MIN_DOCS,
        help=f"Documentos mínimos por regra no TF-IDF (default: {DEFAULT_MIN_DOCS})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Regras por grupo nas tabelas de topo (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Pasta de saída (default: {default_output_dir})",
    )
    return parser


def run_from_args(
    args: argparse.Namespace,
    *,
    default_output_dir: Path,
    extractor: RuleExtractor,
    prefix: str,
    variant: str = "",
) -> pd.DataFrame:
    """Carrega o corpus parseado e roda a análise conforme a linha de comando.

    Compartilhado com :mod:`eud_rules`, que só troca o extrator, o prefixo e a
    variante de CoNLL-U lida.
    """
    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, default_output_dir)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return pd.DataFrame()
    parsed = load_parsed_documents(
        documents, args.experiment, args.parsed_root, variant=variant
    )
    if not parsed:
        logger.warning("Nenhum documento parseado — rode parsing.portparser")
        return pd.DataFrame()

    fmt = RuleFormat(include_upos=not args.no_upos, include_deprel=not args.no_deprel)
    frame = run_analysis(
        parsed,
        output_dir,
        fmt=fmt,
        min_docs=args.min_docs,
        top_k=args.top_k,
        extractor=extractor,
        prefix=prefix,
    )
    if frame.empty:
        return frame

    summary = frame.groupby("group")[list(DOCUMENT_METRICS)].mean()
    for group in (Group.HUMAN, Group.MACHINE):
        if group.value not in summary.index:
            continue
        row = summary.loc[group.value]
        logger.info(
            f"{group.value}: {row['rules_per_sentence']:.2f} regras/sentença, "
            f"{row['unique_rules_per_sentence']:.2f} distintas/sentença, "
            f"diversidade {row['rule_diversity']:.3f}"
        )
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    run_from_args(
        args,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        extractor=sentence_rules,
        prefix="grammar_rules",
    )


if __name__ == "__main__":
    main()
