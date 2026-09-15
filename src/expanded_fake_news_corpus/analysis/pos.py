"""Distribuição de classes gramaticais (UPOS) por autoria.

Repete a análise morfossintática agregada do trabalho anterior (Andrade et
al., PROPOR 2026): a frequência relativa de cada etiqueta UPOS no total de
tokens de cada grupo, a diferença humano − máquina por etiqueta e um teste
qui-quadrado de independência (com V de Cramér) sobre a tabela etiqueta ×
grupo. Lá as etiquetas vinham do Porttagger; aqui, da coluna UPOS dos CoNLL-U
do Portparser v2 (:mod:`conllu`), que anota UPOS com 99,4% de acurácia.

Além do agregado, cada documento vira uma linha com o percentual de cada
etiqueta, e as etiquetas são comparadas uma a uma entre os grupos como as
categorias do LIWC — Mann-Whitney com correção FDR, já que são 17 testes.

Uso::

    python -m expanded_fake_news_corpus.analysis.pos --experiment paper_replication
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from expanded_fake_news_corpus.analysis.conllu import (
    DEFAULT_PARSED_ROOT,
    ParsedDocument,
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
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "pos"

#: As 17 etiquetas universais, na ordem canônica.
UPOS_TAGS: tuple[str, ...] = (
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
)

#: Colunas de identificação que precedem as etiquetas na tabela por documento.
DOCUMENT_COLUMNS = ("uid", "source", "group", "dataset", "tokens")


def count_tags(document: ParsedDocument) -> Counter[str]:
    """Ocorrências de cada UPOS nos tokens sintáticos do documento."""
    return Counter(token.upos for token in document.words())


def build_document_frame(documents: Sequence[ParsedDocument]) -> pd.DataFrame:
    """Uma linha por documento: total de tokens e percentual de cada etiqueta."""
    rows: list[dict[str, object]] = []
    for document in documents:
        counts = count_tags(document)
        total = sum(counts.values())
        if not total:
            logger.warning(f"Documento sem tokens: {document.uid}")
            continue
        rows.append(
            {
                "uid": document.uid,
                "source": document.document.source.value,
                "group": document.group.value,
                "dataset": document.dataset,
                "tokens": total,
                **{tag: 100 * counts.get(tag, 0) / total for tag in UPOS_TAGS},
            }
        )
    return pd.DataFrame(rows)


def pooled_frequencies(documents: Sequence[ParsedDocument]) -> pd.DataFrame:
    """Contagem e frequência relativa de cada etiqueta, por grupo e por recorte.

    É a tabela do trabalho anterior: os tokens de cada conjunto somados, sem
    ponderar por documento. Inclui a diferença ``human − machine`` das
    frequências relativas, positiva quando a etiqueta é mais humana.
    """
    counts: dict[str, Counter[str]] = {}
    for document in documents:
        tags = count_tags(document)
        counts.setdefault(document.group.value, Counter()).update(tags)
        counts.setdefault(document.dataset, Counter()).update(tags)

    frame = pd.DataFrame(
        {
            "tag": UPOS_TAGS,
            **{
                f"{name}_count": [c.get(tag, 0) for tag in UPOS_TAGS]
                for name, c in counts.items()
            },
        }
    )
    for name, counter in counts.items():
        total = sum(counter.values()) or 1
        frame[f"{name}_rel"] = frame[f"{name}_count"] / total
    if {"human_rel", "machine_rel"} <= set(frame.columns):
        frame["difference"] = frame["human_rel"] - frame["machine_rel"]
        frame = frame.reindex(
            frame["difference"].abs().sort_values(ascending=False).index
        )
    return frame.reset_index(drop=True)


def chi_square(documents: Sequence[ParsedDocument]) -> pd.DataFrame:
    """Independência entre etiqueta e autoria na tabela de contingência agregada.

    Uma linha para o corpus todo e uma por corpus de origem; ``cramers_v`` é
    o tamanho do efeito (0 = distribuições iguais).
    """
    rows = [_chi_square_row("all", documents)]
    for source in sorted({d.document.source.value for d in documents}):
        subset = [d for d in documents if d.document.source.value == source]
        rows.append(_chi_square_row(source, subset))
    return pd.DataFrame(rows)


def _chi_square_row(
    label: str, documents: Sequence[ParsedDocument]
) -> dict[str, object]:
    counts = {group: Counter() for group in (Group.HUMAN.value, Group.MACHINE.value)}
    for document in documents:
        counts[document.group.value].update(count_tags(document))
    # Etiquetas ausentes nos dois grupos zerariam uma coluna inteira e
    # invalidariam o teste.
    present = [tag for tag in UPOS_TAGS if any(counts[g].get(tag, 0) for g in counts)]
    table = np.array([[counts[g].get(tag, 0) for tag in present] for g in counts])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    total = table.sum()
    cramers_v = (
        float(np.sqrt(chi2 / (total * (min(table.shape) - 1)))) if total else 0.0
    )
    return {
        "scope": label,
        "tags": len(present),
        "human_tokens": int(table[0].sum()),
        "machine_tokens": int(table[1].sum()),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
    }


def run_analysis(documents: Sequence[ParsedDocument], output_dir: Path) -> pd.DataFrame:
    """Roda a análise completa e grava as tabelas.

    Args:
        documents: Documentos parseados do corpus pareado
        output_dir: Pasta de saída dos CSVs

    Returns:
        Tabela por documento
    """
    frame = build_document_frame(documents)
    if frame.empty:
        logger.warning("Nenhum documento com tokens — nada a gravar")
        return frame

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame, output_dir / "pos_per_document.csv")
    _write_csv(pooled_frequencies(documents), output_dir / "pos_frequencies.csv")
    _write_csv(chi_square(documents), output_dir / "pos_chi_square.csv")

    comparisons = add_fdr_correction(
        comparisons_to_frame(compare_frame(frame, UPOS_TAGS))
    )
    _write_csv(
        comparisons.reindex(
            comparisons["cohens_d"].abs().sort_values(ascending=False).index
        ),
        output_dir / "pos_significance.csv",
    )
    _write_csv(
        compare_within(frame, UPOS_TAGS, split_column="source"),
        output_dir / "pos_significance_by_source.csv",
    )
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        logger.warning(f"Tabela vazia, arquivo não gerado: {path.name}")
        return
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Gravado {path} ({len(frame)} linhas)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos da linha de comando (``sys.argv`` se ``argv`` for None)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_corpus_arguments(parser)
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=DEFAULT_PARSED_ROOT,
        help=f"Raiz dos CoNLL-U (default: {DEFAULT_PARSED_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    documents = documents_from_args(args)
    output_dir = output_dir_from_args(args, DEFAULT_OUTPUT_DIR)
    if not documents:
        logger.warning("Corpus vazio — verifique se a geração já foi executada")
        return
    parsed = load_parsed_documents(documents, args.experiment, args.parsed_root)
    if not parsed:
        logger.warning("Nenhum documento parseado — rode parsing.portparser")
        return

    frame = run_analysis(parsed, output_dir)
    if frame.empty:
        return
    significant = add_fdr_correction(
        comparisons_to_frame(compare_frame(frame, UPOS_TAGS))
    )
    significant = significant[significant["significant_fdr_005"]].sort_values(
        "cohens_d"
    )
    logger.info(
        f"{len(significant)} de {len(UPOS_TAGS)} etiquetas significativas após FDR"
    )
    for row in significant.itertuples():
        side = "human" if row.cohens_d > 0 else "machine"
        logger.info(
            f"  {row.metric}: {row.human_mean:.2f}% vs {row.machine_mean:.2f}% "
            f"(d={row.cohens_d:+.2f}, q={row.q_value:.1e}) -> {side}"
        )


if __name__ == "__main__":
    main()
