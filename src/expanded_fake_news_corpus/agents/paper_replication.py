"""Replicação do método de Silva et al.: notícia verdadeira inteira -> fake news.

Reproduz a geração do artigo base **sem adaptação**: mensagem de sistema e
mensagem de usuário byte a byte como no código publicado, a notícia integral
interpolada ao final, resposta em texto livre delimitada por
``<syntheticText>`` e ``<changes>``. Só o modelo muda.

Não confundir com :mod:`expanded_fake_news_corpus.agents.fake_news_writer`,
que usa o mesmo formato de resposta mas recebe a manchete do estágio 1 e, por
isso, adapta a primeira oração do prompt. Este módulo reaproveita dele o grafo
e o parser das tags — o que difere é apenas a entrada.

Grafo::

    start -> paper_fake_writer -> parse -> end
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from expanded_fake_news_corpus.agents.fake_news_writer import (
    _read,
    build_fake_news_writer_workflow,
    parse_response,
)
from expanded_fake_news_corpus.config import LLMSettings, build_llm
from expanded_fake_news_corpus.corpus import NewsItem
from expanded_fake_news_corpus.prompts import (
    PAPER_ARTICLE_PROMPT,
    PAPER_SYSTEM,
    paper_article_message,
)
from expanded_fake_news_corpus.schemas import FakeNewsError, PaperFakeNewsResult


class PaperReplicationWriter:
    """Gera fake news a partir da notícia inteira, como no artigo base."""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        settings: LLMSettings | None = None,
        prompt: str = PAPER_SYSTEM,
    ) -> None:
        # Sem teto de tokens por padrão: o script dos autores não definia nenhum.
        self._settings = settings or (LLMSettings.from_env() if model is None else None)
        self._model = model or build_llm(self._settings)
        self._prompt = prompt

        timeout = self._settings.timeout if self._settings else None
        retries = self._settings.max_retries if self._settings else 2

        self._graph = build_fake_news_writer_workflow(
            self._model, prompt=prompt, max_retries=retries, timeout=timeout
        ).compile()
        self._async_graph = build_fake_news_writer_workflow(
            self._model,
            prompt=prompt,
            max_retries=retries,
            timeout=timeout,
            mode="async",
        ).compile()

    @property
    def model_name(self) -> str:
        """Modelo em uso, no formato ``provedor/modelo``."""
        return self._settings.model if self._settings else ""

    @property
    def prompt(self) -> str:
        """Prompt de sistema em uso."""
        return self._prompt

    @property
    def prompt_fingerprint(self) -> str:
        """Sistema e template de usuário juntos, para o ``meta.json``."""
        return f"{self._prompt}\n{PAPER_ARTICLE_PROMPT}"

    def _prepare(self, article: str) -> str:
        try:
            return paper_article_message(article)
        except ValueError as exc:
            raise FakeNewsError(str(exc)) from exc

    def _build(
        self, state: Any, article: str, source_id: str | None
    ) -> PaperFakeNewsResult:
        raw = _read(state, "raw_response")
        text, changes, warnings = parse_response(raw)
        if not text:
            raise FakeNewsError("Model returned no synthetic text.")
        if not changes:
            warnings.append("empty <changes> section")
        return PaperFakeNewsResult(
            synthetic_text=text,
            changes=changes,
            raw_response=raw,
            source_id=source_id,
            source_chars=len(article),
            model=self.model_name,
            warnings=warnings,
        )

    def generate(
        self, article: str, *, source_id: str | None = None
    ) -> PaperFakeNewsResult:
        """Gera a fake news de uma notícia verdadeira."""
        message = self._prepare(article)
        state = self._graph.invoke({"headline": message})
        return self._build(state, article, source_id)

    async def agenerate(
        self, article: str, *, source_id: str | None = None
    ) -> PaperFakeNewsResult:
        """Versão assíncrona de :meth:`generate`."""
        message = self._prepare(article)
        state = await self._async_graph.ainvoke({"headline": message})
        return self._build(state, article, source_id)

    def generate_many(self, items: Iterable[NewsItem]) -> Iterator[PaperFakeNewsResult]:
        """Processa pares ``(source_id, texto)`` em sequência."""
        for source_id, text in items:
            yield self.generate(text, source_id=source_id)

    async def agenerate_many(
        self, items: Iterable[NewsItem], *, concurrency: int = 4
    ) -> AsyncIterator[PaperFakeNewsResult | FakeNewsError]:
        """Processa pares ``(source_id, texto)`` com paralelismo limitado."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(source_id: str | None, text: str):
            async with semaphore:
                try:
                    return await self.agenerate(text, source_id=source_id)
                except Exception as exc:  # noqa: BLE001 - falha por item
                    return FakeNewsError(f"{source_id or '<no id>'}: {exc}")

        tasks = [asyncio.create_task(run(sid, text)) for sid, text in items]
        for task in tasks:
            yield await task
