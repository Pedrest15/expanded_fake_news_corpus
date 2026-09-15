"""Orquestrador das análises linguísticas: roda uma, algumas ou todas.

Este módulo não analisa nada: só conhece o catálogo de análises e chama o
``main`` de cada uma com os argumentos de corpus repassados. Cada módulo
continua executável sozinho e dono das próprias opções; aqui ficam apenas
as opções comuns (``--experiment``, ``--source``, ``--model``).

Para acrescentar uma análise, basta um item em :data:`ANALYSES` — o módulo
precisa expor ``main(argv)`` no padrão dos demais. A ordem do catálogo é a
ordem de execução; as que dependem do corpus parseado ficam por último e são
puladas com aviso quando os CoNLL-U não existem.

Uso::

    python -m expanded_fake_news_corpus.analysis                 # todas
    python -m expanded_fake_news_corpus.analysis --analysis liwc # uma
    python -m expanded_fake_news_corpus.analysis --analysis zipf --analysis sage \\
        --experiment paper_replication --model openai/gpt-4.1-mini-2025-04-14
    python -m expanded_fake_news_corpus.analysis --list
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from expanded_fake_news_corpus.analysis import (
    eud_rules,
    grammar_rules,
    lexical_diversity,
    liwc,
    pos,
    sage,
    syllables,
    zipf,
)
from expanded_fake_news_corpus.analysis.conllu import (
    DEFAULT_PARSED_ROOT,
    parsed_dir,
)
from expanded_fake_news_corpus.analysis.documents import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    Source,
)

logger = logging.getLogger(__name__)

#: Assinatura comum aos ``main`` dos módulos de análise.
AnalysisMain = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class Analysis:
    """Uma análise registrada no catálogo.

    Attributes:
        name: Identificador usado em ``--analysis``
        description: Uma linha, para ``--list``
        main: Ponto de entrada do módulo, que recebe a linha de comando
        requires_parsing: Precisa dos CoNLL-U de ``data/parsed/<experimento>``
    """

    name: str
    description: str
    main: AnalysisMain
    requires_parsing: bool = False


#: Catálogo, na ordem de execução. Acrescente análises novas aqui.
ANALYSES: tuple[Analysis, ...] = (
    Analysis("syllables", "Sílabas por palavra e por sentença", syllables.main),
    Analysis(
        "lexical_diversity", "Diversidade lexical (MATTR)", lexical_diversity.main
    ),
    Analysis("zipf", "Distribuição de frequências e palavras mais comuns", zipf.main),
    Analysis("sage", "Termos distintivos (SAGE)", sage.main),
    Analysis("liwc", "Perfil psicolinguístico (LIWC)", liwc.main),
    Analysis(
        "pos",
        "Distribuição de classes gramaticais (UPOS)",
        pos.main,
        requires_parsing=True,
    ),
    Analysis(
        "grammar_rules",
        "Regras de dependência: produtividade e TF-IDF",
        grammar_rules.main,
        requires_parsing=True,
    ),
    Analysis(
        "eud_rules",
        "Regras das arestas enhanced (EUD)",
        eud_rules.main,
        requires_parsing=True,
    ),
)

ANALYSES_BY_NAME: dict[str, Analysis] = {
    analysis.name: analysis for analysis in ANALYSES
}


@dataclass(frozen=True)
class RunOptions:
    """O que rodar e sobre qual recorte do corpus."""

    names: tuple[str, ...] = ()  # vazio: todas, na ordem do catálogo
    experiment: str = DEFAULT_EXPERIMENT
    sources: tuple[str, ...] = ()
    models: tuple[str, ...] = ()

    def corpus_argv(self) -> list[str]:
        """Argumentos de corpus repassados a cada ``main``."""
        argv = ["--experiment", self.experiment]
        for source in self.sources:
            argv.extend(["--source", source])
        for model in self.models:
            argv.extend(["--model", model])
        return argv


@dataclass
class RunReport:
    """Resultado de uma rodada."""

    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


class UnknownAnalysisError(ValueError):
    """Nome de análise fora do catálogo."""


def select_analyses(names: Sequence[str]) -> list[Analysis]:
    """Resolve os nomes pedidos no catálogo, mantendo a ordem de execução.

    Args:
        names: Nomes de ``--analysis``; vazio significa todas

    Returns:
        Análises a rodar, na ordem do catálogo

    Raises:
        UnknownAnalysisError: Se algum nome não existir
    """
    unknown = sorted(set(names) - set(ANALYSES_BY_NAME))
    if unknown:
        raise UnknownAnalysisError(
            f"unknown analyses: {', '.join(unknown)}; "
            f"available: {', '.join(ANALYSES_BY_NAME)}"
        )
    wanted = set(names) if names else set(ANALYSES_BY_NAME)
    return [analysis for analysis in ANALYSES if analysis.name in wanted]


def run_analyses(options: RunOptions) -> RunReport:
    """Roda as análises pedidas em sequência e resume o que aconteceu.

    Uma análise que falha não interrompe as demais: o erro é registrado com
    o tipo e a rodada segue, para que uma tabela quebrada não custe as
    outras. O código de saída acusa a falha no final.

    Args:
        options: Análises e recorte do corpus

    Returns:
        Quem concluiu, quem foi pulada e quem falhou
    """
    analyses = select_analyses(options.names)
    argv = options.corpus_argv()
    report = RunReport()
    logger.info(
        f"{len(analyses)} análises sobre o experimento {options.experiment!r}: "
        f"{', '.join(a.name for a in analyses)}"
    )

    for analysis in analyses:
        if analysis.requires_parsing and not _has_parsed_corpus(options.experiment):
            logger.warning(
                f"[{analysis.name}] pulada: sem CoNLL-U em "
                f"{parsed_dir(options.experiment)} — rode parsing.portparser"
            )
            report.skipped.append(analysis.name)
            continue
        _run_one(analysis, argv, report)

    logger.info(
        f"Rodada concluída: {len(report.completed)} concluídas, "
        f"{len(report.skipped)} puladas, {len(report.failed)} com falha"
    )
    return report


def _run_one(analysis: Analysis, argv: Sequence[str], report: RunReport) -> None:
    """Executa uma análise, cronometrando e isolando a falha."""
    logger.info(f"[{analysis.name}] iniciando — {analysis.description}")
    started = time.perf_counter()
    try:
        analysis.main(list(argv))
    # NOTA: captura ampla de propósito — o orquestrador não sabe o que cada
    # análise pode lançar e precisa seguir para as demais; a falha não é
    # engolida: fica no relatório e no código de saída.
    except Exception as err:  # noqa: BLE001
        logger.error(f"[{analysis.name}] falhou: {type(err).__name__}: {err}")
        report.failed.append(analysis.name)
        return
    elapsed = time.perf_counter() - started
    logger.info(f"[{analysis.name}] concluída em {elapsed:.1f}s")
    report.completed.append(analysis.name)


def _has_parsed_corpus(experiment: str) -> bool:
    return any(parsed_dir(experiment, DEFAULT_PARSED_ROOT).glob("*/*.conllu"))


def list_analyses() -> str:
    """Catálogo formatado para ``--list``."""
    width = max(len(analysis.name) for analysis in ANALYSES)
    lines = [
        f"  {analysis.name:<{width}}  {analysis.description}"
        + ("  [requer parsing]" if analysis.requires_parsing else "")
        for analysis in ANALYSES
    ]
    return "Análises disponíveis, na ordem de execução:\n" + "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=list_analyses(),
    )
    parser.add_argument(
        "--analysis",
        action="append",
        dest="analyses",
        choices=list(ANALYSES_BY_NAME),
        metavar="NOME",
        help="Análise a rodar (repetível). Sem a opção, roda todas.",
    )
    parser.add_argument("--list", action="store_true", help="Mostra o catálogo e sai.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=DEFAULT_EXPERIMENT,
        help=f"Experimento de geração a analisar (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=[source.value for source in Source],
        help="Restringe a um corpus de origem (repetível).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restringe a um modelo gerador (repetível).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    if args.list:
        logger.info(list_analyses())
        return 0

    options = RunOptions(
        names=tuple(args.analyses or ()),
        experiment=args.experiment,
        sources=tuple(args.sources or ()),
        models=tuple(args.models or ()),
    )
    return run_analyses(options).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
