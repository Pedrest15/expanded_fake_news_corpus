"""Estruturas de dados trocadas entre agentes, CLI e arquivos do corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeadlineResult(BaseModel):
    """Resultado da titulação de uma notícia."""

    headline: str = Field(description="Manchete final, já normalizada.")
    rationale: str = Field(
        default="", description="Justificativa do foco escolhido pelo modelo."
    )
    raw_headline: str = Field(
        default="", description="Saída bruta do modelo, antes da limpeza."
    )
    source_id: str | None = Field(
        default=None, description="Identificador da notícia no corpus de origem."
    )
    model: str = Field(default="", description="Modelo usado, em 'provedor/modelo'.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Avisos de qualidade (tamanho fora da faixa, etc.).",
    )


class HeadlineError(Exception):
    """Falha ao gerar uma manchete utilizável."""
