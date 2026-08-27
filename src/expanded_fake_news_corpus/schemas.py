"""Estruturas de dados trocadas entre agentes, CLI e arquivos do corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeadlineResult(BaseModel):
    """Resultado da titulação de uma notícia."""

    headline: str = Field(description="Final headline, already normalized.")
    rationale: str = Field(
        default="", description="Model rationale for the chosen focus."
    )
    raw_headline: str = Field(
        default="", description="Raw model output, before cleaning."
    )
    source_id: str | None = Field(
        default=None, description="Article identifier in the source corpus."
    )
    model: str = Field(default="", description="Model used, as 'provider/model'.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Quality warnings (length out of range, etc.).",
    )


class HeadlineError(Exception):
    """Falha ao gerar uma manchete utilizável."""


class FakeNewsResult(BaseModel):
    """Notícia falsa sintética gerada a partir de uma manchete verdadeira."""

    fake_headline: str = Field(description="Fake headline, already normalized.")
    fake_text: str = Field(description="Fake news body.")
    changes: list[str] = Field(
        default_factory=list,
        description="Disinformation techniques used, one per item.",
    )
    source_id: str | None = Field(
        default=None, description="Article identifier in the source corpus."
    )
    source_headline: str = Field(
        default="", description="True headline used as the seed."
    )
    headline_model: str = Field(
        default="", description="Model that wrote the seed headline."
    )
    genre: str = Field(default="", description="Genre of the seed headline.")
    model: str = Field(default="", description="Model that wrote the fake news.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Quality warnings (length, disclaimer leakage, etc.).",
    )


class FakeNewsError(Exception):
    """Falha ao gerar uma notícia falsa utilizável."""


class FakeNewsWriterResult(BaseModel):
    """Fake news gerada com o prompt e o formato de resposta do artigo base.

    Diferente de :class:`FakeNewsResult`, que separa manchete e corpo, aqui
    ``synthetic_text`` é o bloco ``<syntheticText>`` como o artigo o define — a
    manchete sensacionalista costuma vir na primeira linha dele — e ``changes``
    é o bloco de texto corrido, não uma lista tipada.
    """

    synthetic_text: str = Field(description="Contents of <syntheticText>.")
    changes: str = Field(default="", description="Contents of <changes>.")
    raw_response: str = Field(
        default="", description="Raw model response, before tag extraction."
    )
    source_id: str | None = Field(
        default=None, description="Article identifier in the source corpus."
    )
    source_headline: str = Field(
        default="", description="True headline used as the seed."
    )
    headline_model: str = Field(
        default="", description="Model that wrote the seed headline."
    )
    model: str = Field(default="", description="Model that wrote the fake news.")
    warnings: list[str] = Field(
        default_factory=list, description="Tag extraction warnings."
    )
