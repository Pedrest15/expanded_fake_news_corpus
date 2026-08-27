"""Execução de lotes: escrita, contagem e relatório de progresso.

Os dois estágios do pipeline repetem a mesma mecânica — percorrer resultados,
gravar os que deram certo, contar os que falharam, relatar o andamento e
devolver o código de saída. Só variam o tipo do resultado, como resumi-lo na
linha de progresso e os substantivos das mensagens.

Manter isso em um lugar só evita que os estágios divirjam em comportamento
observável: retomada, código de saída e formato do relatório passam a ser
definidos aqui, não em cada comando.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from expanded_fake_news_corpus.corpus import JsonlWriter


@dataclass(frozen=True)
class BatchJob:
    """Descreve um lote a executar.

    Attributes:
        destination: JSONL de saída.
        label: Prefixo das mensagens, normalmente ``provedor/modelo``.
        input_noun: Substantivo do que entra ("notícias", "manchetes").
        output_noun: Substantivo do que sai — nem sempre igual ao de entrada.
        total: Itens que serão de fato processados, já descontada a retomada.
        append: Abre a saída em modo append, para ``--resume``.
        quiet: Silencia o progresso em stderr.
        skipped: Itens pulados pela retomada, só para o cabeçalho.
    """

    destination: Path
    label: str
    input_noun: str
    output_noun: str
    total: int
    append: bool = False
    quiet: bool = False
    skipped: int = 0


@dataclass(frozen=True)
class BatchReport:
    """Contagens de um lote concluído."""

    total: int
    written: int
    failures: int

    @property
    def exit_code(self) -> int:
        """1 apenas quando o lote inteiro falhou.

        Falha isolada não vira erro de processo: em lotes longos, uma recusa
        pontual do provedor não deve marcar a execução como perdida.
        """
        return 1 if self.total and self.failures == self.total else 0


def run_batch(
    job: BatchJob,
    outcomes: Iterable[BaseModel | Exception],
    describe: Callable[[BaseModel], str],
) -> BatchReport:
    """Consome os resultados de um lote, gravando e relatando.

    Args:
        job: Parâmetros do lote.
        outcomes: Resultados na ordem de entrada. Exceções sinalizam falha do
            item e são relatadas sem interromper o lote.
        describe: Resume um resultado para a linha de progresso.

    Returns:
        Contagens do lote.
    """
    _announce(job)

    failures = done = 0
    with JsonlWriter(job.destination, append=job.append) as writer:
        for outcome in outcomes:
            done += 1
            if isinstance(outcome, Exception):
                failures += 1
                _log(job, f"[{done}/{job.total}] falha: {outcome}")
                continue
            writer.write(outcome)
            if not job.quiet:
                _log(job, f"[{done}/{job.total}] {_progress_line(outcome, describe)}")

    report = BatchReport(total=job.total, written=done - failures, failures=failures)
    _log(
        job,
        f"concluído: {report.written} {job.output_noun}, {report.failures} falhas",
    )
    return report


def _announce(job: BatchJob) -> None:
    """Cabeçalho do lote."""
    extra = f" ({job.skipped} já processadas)" if job.skipped else ""
    _log(job, f"{job.total} {job.input_noun}{extra} → {job.destination}")


def _progress_line(outcome: BaseModel, describe: Callable[[BaseModel], str]) -> str:
    """Linha de um item bem-sucedido, com os avisos de qualidade em anexo."""
    warnings = getattr(outcome, "warnings", None) or []
    alerts = f"  [{'; '.join(warnings)}]" if warnings else ""
    source_id = getattr(outcome, "source_id", None) or ""
    return f"{source_id} {describe(outcome)}{alerts}"


def _log(job: BatchJob, message: str) -> None:
    if not job.quiet:
        print(f"[{job.label}] {message}", file=sys.stderr)
