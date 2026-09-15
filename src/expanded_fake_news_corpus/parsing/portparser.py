"""Do texto ao CoNLL-U: sentenciar, tokenizar, anotar e pós-processar.

Reproduz a cadeia do trabalho anterior (Andrade et al., PROPOR 2026)::

    texto -> portSentencer -> portTokenizer -> LatinPipe + Portparser_v2_model
          -> pós-processamento do Portparser.v2 (lemas e feats)

O parser carrega um BERTimbau de 1,6 GB, então os documentos são concatenados
e anotados numa chamada só; o resultado é repartido por documento pelo
alinhamento de :mod:`preprocess`. Saída::

    data/parsed/<experimento>/<grupo>/<stem>.conllu     um por documento
    data/parsed/<experimento>/manifest.json             ferramentas e contagens

Uso::

    python -m expanded_fake_news_corpus.parsing.portparser \\
        --experiment paper_replication --model openai/gpt-4.1-mini-2025-04-14
    python -m expanded_fake_news_corpus.parsing.portparser \\
        --experiment paper_replication --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from expanded_fake_news_corpus.analysis.conllu import (
    DEFAULT_PARSED_ROOT,
    conllu_path,
    parse_conllu,
    parsed_dir,
    write_conllu,
)
from expanded_fake_news_corpus.analysis.documents import (
    Document,
    add_corpus_arguments,
    documents_from_args,
)
from expanded_fake_news_corpus.analysis.nltk_resources import (
    TOKENIZER_PACKAGES,
    ensure_nltk_resources,
)
from expanded_fake_news_corpus.parsing.manifest import timestamp, update_manifest
from expanded_fake_news_corpus.parsing.preprocess import (
    AlignmentError,
    align_sentence_counts,
    split_blocks,
)
from expanded_fake_news_corpus.parsing.tools import (
    DEFAULT_TOOLS_DIR,
    ToolError,
    ToolPaths,
    check_tools,
    run_tool,
    tool_versions,
)

logger = logging.getLogger(__name__)

DEFAULT_THREADS = 3

#: Como cada etapa foi chamada, para o manifesto.
PIPELINE_STEPS = (
    "portSentencer -r",
    "portTokenizer -m",
    "latinpipe --load Portparser_v2_model",
    "postprocess.py -l -f",
)


@dataclass(frozen=True)
class ParseOptions:
    """Parâmetros de uma execução do parser."""

    experiment: str
    tools: ToolPaths = ToolPaths()
    parsed_root: Path = DEFAULT_PARSED_ROOT
    threads: int = DEFAULT_THREADS
    force: bool = False
    keep_workdir: bool = False


@dataclass(frozen=True)
class ParseReport:
    """O que uma execução produziu."""

    parsed: int
    skipped: int
    sentences: int
    dropped_sentences: int


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------


def sentencize(document: Document, tools: ToolPaths, workdir: Path) -> list[str]:
    """Segmenta um documento em sentenças com o portSentencer, bloco a bloco.

    ``-r`` normaliza aspas e travessões não padronizados, como no trabalho
    anterior. O programa lê ``abbrev.txt`` do diretório corrente, por isso o
    ``cwd``. Cada bloco de :func:`split_blocks` é sentenciado à parte, para
    que a quebra de linha valha como fronteira.

    Args:
        document: Documento do corpus pareado
        tools: Ferramentas instaladas
        workdir: Pasta temporária para entrada e saída

    Returns:
        Sentenças na ordem do texto

    Raises:
        ToolError: Se o sentenciador falhar ou não devolver nada
    """
    stem = document.uid.replace(":", "_")
    sentences: list[str] = []
    for number, block in enumerate(split_blocks(document.text), start=1):
        source = workdir / f"{stem}.{number}.txt"
        target = workdir / f"{stem}.{number}.sents"
        source.write_text(block + "\n", encoding="utf-8")
        run_tool(
            [sys.executable, "portSent.py", "-o", str(target), "-r", str(source)],
            cwd=tools.sentencer_dir,
        )
        sentences.extend(_non_blank_lines(target))
    if not sentences:
        raise ToolError(f"{document.uid}: sentencer returned no sentences")
    return sentences


def tokenize(sentences_file: Path, target: Path, tools: ToolPaths, log: Path) -> None:
    """Tokeniza um arquivo com uma sentença por linha para CoNLL-U.

    ``-m`` remove pontuação de pareamento sem par; ``-s S0000`` numera as
    sentenças. O portTokenizer carrega um léxico de 1,2 M de entradas, por
    isso uma chamada para o corpus inteiro.
    """
    run_tool(
        [
            sys.executable,
            "portTok.py",
            "-o",
            str(target),
            "-m",
            "-s",
            "S0000",
            str(sentences_file),
        ],
        cwd=tools.tokenizer_dir,
        log=log,
    )


def annotate(
    tokenized: Path, workdir: Path, tools: ToolPaths, log: Path, threads: int
) -> Path:
    """Anota UPOS, lemas, feats e dependências com LatinPipe + Portparser v2.

    Returns:
        O ``<nome>.predicted.conllu`` que o LatinPipe grava em ``--exp``

    Raises:
        ToolError: Se o parser falhar ou não gerar o arquivo
    """
    run_tool(
        [
            str(tools.latinpipe_python),
            "latinpipe_evalatin24.py",
            "--load",
            str(tools.model_dir / "model.weights.h5"),
            "--exp",
            str(workdir),
            "--test",
            str(tokenized),
            "--threads",
            str(threads),
        ],
        cwd=tools.latinpipe_dir,
        log=log,
    )
    predicted = workdir / f"{tokenized.stem}.predicted.conllu"
    if not predicted.is_file():
        raise ToolError(f"LatinPipe did not produce {predicted}")
    return predicted


def postprocess(predicted: Path, target: Path, tools: ToolPaths, log: Path) -> None:
    """Corrige lemas e feats com o pós-processador do Portparser.v2.

    O programa só aceita a forma ``-o SAÍDA ENTRADA`` (lê os argumentos por
    posição) e, sem ``-q``, grava o relatório ``<saída>.rep.tsv`` ao lado.
    """
    run_tool(
        [sys.executable, "postprocess.py", "-o", str(target), str(predicted)],
        cwd=tools.postproc_dir,
        log=log,
    )


def split_by_document(
    annotated: Path,
    documents: Sequence[Document],
    counts: Sequence[int],
    options: ParseOptions,
) -> int:
    """Reparte o CoNLL-U anotado em um arquivo por documento.

    ``sent_id`` vira ``<uid>-<n>``. A ordem e as contagens vêm de
    :func:`align_sentence_counts`; o parser preserva ambas.

    Returns:
        Documentos gravados

    Raises:
        ToolError: Se o total de sentenças não bater com as contagens
    """
    sentences = parse_conllu(annotated.read_text(encoding="utf-8"))
    if len(sentences) != sum(counts):
        raise ToolError(
            f"expected {sum(counts)} annotated sentences, read {len(sentences)}; "
            "the split by document would be misaligned"
        )

    cursor = 0
    for document, count in zip(documents, counts, strict=True):
        chunk = sentences[cursor : cursor + count]
        cursor += count
        for number, sentence in enumerate(chunk, start=1):
            sentence.sent_id = f"{document.uid}-{number}"
            sentence.comments = [
                f"# sent_id = {sentence.sent_id}" if c.startswith("# sent_id") else c
                for c in sentence.comments
            ]
        write_conllu(
            conllu_path(document, options.experiment, options.parsed_root), chunk
        )
    return len(documents)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def parse_corpus(documents: Sequence[Document], options: ParseOptions) -> ParseReport:
    """Parseia os documentos que ainda não têm CoNLL-U e grava o manifesto.

    Args:
        documents: Corpus pareado
        options: Parâmetros da execução

    Returns:
        Contagens da execução

    Raises:
        ToolError: Se alguma ferramenta faltar ou falhar
    """
    check_tools(options.tools)
    ensure_nltk_resources(*TOKENIZER_PACKAGES)

    pending = [d for d in documents if options.force or not _has_conllu(d, options)]
    out_dir = parsed_dir(options.experiment, options.parsed_root)
    logger.info(
        f"{len(documents)} documentos, {len(pending)} a parsear "
        f"({len(documents) - len(pending)} já em {out_dir})"
    )
    if not pending:
        return ParseReport(
            parsed=0, skipped=len(documents), sentences=0, dropped_sentences=0
        )

    workdir = Path(tempfile.mkdtemp(prefix="portparser_"))
    logger.info(f"Pasta de trabalho: {workdir}")
    try:
        report = _run_pipeline(pending, options, workdir)
    except (ToolError, AlignmentError) as err:
        logger.error(
            f"Falha no parsing: {type(err).__name__}: {err} (logs em {workdir})"
        )
        raise ToolError(str(err)) from err

    _copy_postprocess_report(workdir, out_dir)
    _write_manifest(out_dir, options, report)
    if not options.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    logger.info(f"{report.parsed} documentos parseados -> {out_dir}")
    return report


def _run_pipeline(
    pending: Sequence[Document], options: ParseOptions, workdir: Path
) -> ParseReport:
    """Executa as quatro etapas sobre os documentos pendentes, numa passada."""
    per_document = [sentencize(d, options.tools, workdir) for d in pending]
    sentences = [s for block in per_document for s in block]
    logger.info(f"{len(sentences)} sentenças em {len(pending)} documentos")

    sentences_file = workdir / "corpus.sents"
    sentences_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    tokenized = workdir / "corpus.conllu"
    logger.info("Tokenizando (portTokenizer)...")
    tokenize(sentences_file, tokenized, options.tools, workdir / "portTok.log")
    counts = align_sentence_counts(per_document, _tokenized_texts(tokenized))
    dropped = len(sentences) - sum(counts)
    if dropped:
        logger.info(f"{dropped} sentenças sem letras descartadas pelo tokenizador")

    logger.info("Anotando (LatinPipe + Portparser v2)...")
    predicted = annotate(
        tokenized, workdir, options.tools, workdir / "latinpipe.log", options.threads
    )
    logger.info("Pós-processando lemas e feats...")
    annotated = workdir / "corpus.annotated.conllu"
    postprocess(predicted, annotated, options.tools, workdir / "postproc.log")

    parsed = split_by_document(annotated, pending, counts, options)
    return ParseReport(
        parsed=parsed, skipped=0, sentences=sum(counts), dropped_sentences=dropped
    )


def _has_conllu(document: Document, options: ParseOptions) -> bool:
    return conllu_path(document, options.experiment, options.parsed_root).is_file()


def _non_blank_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tokenized_texts(path: Path) -> list[str]:
    """Os ``# text`` de um CoNLL-U, na ordem."""
    return [
        sentence.text for sentence in parse_conllu(path.read_text(encoding="utf-8"))
    ]


def _copy_postprocess_report(workdir: Path, out_dir: Path) -> None:
    """Guarda o relatório de correções do pós-processador ao lado dos CoNLL-U."""
    report = workdir / "corpus.annotated.conllu.rep.tsv"
    if report.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(report, out_dir / "postprocess_report.tsv")


def _write_manifest(out_dir: Path, options: ParseOptions, report: ParseReport) -> None:
    update_manifest(
        out_dir,
        {
            "experiment": options.experiment,
            "parser": "Portparser v2 (LatinPipe + BERTimbau) com pós-processamento",
            "pipeline": list(PIPELINE_STEPS),
            "tools": tool_versions(options.tools),
            "documents": len(list(out_dir.glob("*/*.conllu"))),
            "last_run": {
                "parsed": report.parsed,
                "sentences": report.sentences,
                "dropped_sentences": report.dropped_sentences,
                "finished_at": timestamp(),
            },
        },
    )


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    add_corpus_arguments(parser)
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR)
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=DEFAULT_PARSED_ROOT,
        help=f"Raiz dos CoNLL-U (default: {DEFAULT_PARSED_ROOT})",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS, help="Threads do parser."
    )
    parser.add_argument(
        "--force", action="store_true", help="Reparseia documentos que já têm CoNLL-U."
    )
    parser.add_argument(
        "--keep-workdir", action="store_true", help="Não apaga a pasta temporária."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    options = ParseOptions(
        experiment=args.experiment,
        tools=ToolPaths(args.tools_dir),
        parsed_root=args.parsed_root,
        threads=args.threads,
        force=args.force,
        keep_workdir=args.keep_workdir,
    )

    documents = documents_from_args(args)
    if not documents:
        logger.error("Corpus vazio — verifique se a geração já foi executada")
        return 1
    try:
        parse_corpus(documents, options)
    except ToolError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
