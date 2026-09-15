"""Do CoNLL-U básico ao CoNLL-U com Enhanced Universal Dependencies.

Aplica ao corpus parseado (:mod:`portparser`) o conjunto de regras Grew do
projeto `eud-portugues <https://github.com/alvelvis/eud-portugues>`_,
vendorizado sem alteração em ``resources/eud/conjunto_regras_porttinari.grs``
— o mesmo passo do trabalho anterior sobre discriminação humano/máquina. A
coluna DEPS é zerada antes, como lá, para o Grew partir só das dependências
básicas.

Cada ``<stem>.conllu`` ganha um ``<stem>.eud.conllu`` ao lado; a anotação
básica não é alterada. Precisa do ``grew`` no PATH (https://grew.fr).

Uso::

    python -m expanded_fake_news_corpus.parsing.eud --experiment paper_replication
    python -m expanded_fake_news_corpus.parsing.eud \\
        --experiment paper_replication --force
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from expanded_fake_news_corpus.analysis.conllu import (
    DEFAULT_PARSED_ROOT,
    Sentence,
    conllu_path,
    parse_conllu,
    parsed_dir,
    write_conllu,
)
from expanded_fake_news_corpus.analysis.documents import (
    PROJECT_ROOT,
    Document,
    add_corpus_arguments,
    documents_from_args,
)
from expanded_fake_news_corpus.parsing.manifest import timestamp, update_manifest
from expanded_fake_news_corpus.parsing.tools import ToolError

logger = logging.getLogger(__name__)

DEFAULT_GRS = PROJECT_ROOT / "resources" / "eud" / "conjunto_regras_porttinari.grs"
DEFAULT_STRATEGY = "eud_portuguese"
VARIANT = "eud"


@dataclass(frozen=True)
class EudOptions:
    """Parâmetros de uma execução do Grew."""

    experiment: str
    parsed_root: Path = DEFAULT_PARSED_ROOT
    grs: Path = DEFAULT_GRS
    strategy: str = DEFAULT_STRATEGY
    force: bool = False


@dataclass(frozen=True)
class EudReport:
    """O que uma execução produziu."""

    enriched: int
    skipped: int
    missing: int
    tokens_with_deps: int


def grew_version() -> str:
    """Versão do ``grew`` no PATH.

    Raises:
        ToolError: Se o executável não existir
    """
    if shutil.which("grew") is None:
        raise ToolError(
            "'grew' is not on PATH — install it with opam (https://grew.fr)"
        )
    result = subprocess.run(["grew", "version"], capture_output=True, text=True)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "?"


def clear_deps(sentences: Sequence[Sentence]) -> list[Sentence]:
    """Zera a coluna DEPS de todos os tokens, sem alterar o resto."""
    return [
        replace(
            sentence, tokens=[replace(token, deps="_") for token in sentence.tokens]
        )
        for sentence in sentences
    ]


def apply_eud(source: Path, target: Path, options: EudOptions) -> int:
    """Roda ``grew transform`` num CoNLL-U e grava o resultado.

    Args:
        source: CoNLL-U básico
        target: Onde gravar o CoNLL-U enriquecido
        options: Regras e estratégia

    Returns:
        Tokens que receberam alguma aresta em DEPS

    Raises:
        ToolError: Se o Grew falhar ou não preencher DEPS em token algum
    """
    with tempfile.TemporaryDirectory(prefix="eud_") as tmp:
        cleaned = Path(tmp) / "in.conllu"
        transformed = Path(tmp) / "out.conllu"
        write_conllu(
            cleaned, clear_deps(parse_conllu(source.read_text(encoding="utf-8")))
        )
        _run_grew(cleaned, transformed, options, source.name)
        sentences = parse_conllu(transformed.read_text(encoding="utf-8"))

    enriched = sum(
        1 for sentence in sentences for token in sentence.words if token.deps != "_"
    )
    if not enriched:
        raise ToolError(f"{source.name}: Grew filled DEPS in no token")
    write_conllu(target, sentences)
    return enriched


def _run_grew(source: Path, target: Path, options: EudOptions, label: str) -> None:
    command = [
        "grew",
        "transform",
        "-config",
        "iwpt",
        "-grs",
        str(options.grs),
        "-strat",
        options.strategy,
        "-i",
        str(source),
        "-o",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.is_file():
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-10:])
        raise ToolError(f"grew failed on {label}:\n{tail}")


def enrich_corpus(documents: Sequence[Document], options: EudOptions) -> EudReport:
    """Enriquece com EUD os documentos que ainda não têm ``.eud.conllu``.

    Args:
        documents: Corpus pareado
        options: Parâmetros da execução

    Returns:
        Contagens da execução

    Raises:
        ToolError: Se o Grew ou o conjunto de regras faltarem, ou se um
            documento falhar
    """
    version = grew_version()
    if not options.grs.is_file():
        raise ToolError(f"rule set not found: {options.grs}")

    enriched = skipped = missing = tokens = 0
    for document in documents:
        source = conllu_path(document, options.experiment, options.parsed_root)
        target = conllu_path(
            document, options.experiment, options.parsed_root, variant=VARIANT
        )
        if not source.is_file():
            missing += 1
        elif target.is_file() and not options.force:
            skipped += 1
        else:
            tokens += apply_eud(source, target, options)
            enriched += 1

    if missing:
        logger.warning(
            f"{missing} documentos sem CoNLL-U básico — rode parsing.portparser"
        )
    logger.info(
        f"{enriched} documentos enriquecidos, {skipped} já prontos, "
        f"{tokens} tokens com DEPS"
    )
    if enriched:
        _write_manifest(options, version)
    return EudReport(
        enriched=enriched, skipped=skipped, missing=missing, tokens_with_deps=tokens
    )


def _write_manifest(options: EudOptions, version: str) -> None:
    out_dir = parsed_dir(options.experiment, options.parsed_root)
    grs = options.grs
    update_manifest(
        out_dir,
        {
            "eud": {
                "tool": f"grew {version}",
                "grs": str(grs.relative_to(PROJECT_ROOT))
                if grs.is_relative_to(PROJECT_ROOT)
                else str(grs),
                "grs_sha256": hashlib.sha256(grs.read_bytes()).hexdigest()[:16],
                "strategy": options.strategy,
                "documents": len(list(out_dir.glob(f"*/*.{VARIANT}.conllu"))),
                "finished_at": timestamp(),
            }
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    add_corpus_arguments(parser)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument("--grs", type=Path, default=DEFAULT_GRS)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--force", action="store_true", help="Refaz arquivos já enriquecidos."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    options = EudOptions(
        experiment=args.experiment,
        parsed_root=args.parsed_root,
        grs=args.grs,
        strategy=args.strategy,
        force=args.force,
    )
    try:
        enrich_corpus(documents_from_args(args), options)
    except ToolError as err:
        logger.error(f"Falha no EUD: {type(err).__name__}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
