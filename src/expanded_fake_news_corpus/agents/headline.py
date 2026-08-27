"""Agente de titulação: recebe uma notícia e devolve a manchete dela.

Este é o primeiro elo do pipeline do FakeGen.BR. As manchetes produzidas aqui, a
partir de notícias verdadeiras dos corpora Fake.br e FakeTrueBR, são o insumo
para a geração posterior das fake news sintéticas.

Grafo::

    start -> headline_writer -> sanitize -> end

``headline_writer`` é um :class:`~langgraphlib.Agent` com saída estruturada
(``headline`` + ``rationale``); ``sanitize`` é um nó determinístico que limpa a
manchete e preserva a saída bruta para auditoria.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraphlib import Agent, Workflow, create_state

from expanded_fake_news_corpus.config import LLMSettings, build_llm
from expanded_fake_news_corpus.prompts import NEWS, headline_prompt
from expanded_fake_news_corpus.schemas import HeadlineError, HeadlineResult
from expanded_fake_news_corpus.text import (
    clean_headline,
    normalize_news_text,
    word_count,
)

#: Estado que trafega pelo grafo de titulação.
HeadlineState = create_state(
    "HeadlineState",
    include_messages=False,
    news_text=(str, ""),
    headline=(str, ""),
    rationale=(str, ""),
    raw_headline=(str, ""),
)

AGENT_NAME = "headline_writer"

#: Faixa aceitável de palavras e caracteres, conforme as regras do prompt.
MIN_WORDS = 6
MAX_WORDS = 18
MAX_CHARS = 120

DEFAULT_MAX_INPUT_CHARS = 12_000

_WORD_RE = re.compile(r"\w{4,}", re.UNICODE)


def _sanitize_node(state: Any) -> dict[str, str]:
    """Normaliza a manchete gerada, guardando a saída bruta do modelo."""
    raw = getattr(state, "headline", "") or ""
    return {"headline": clean_headline(str(raw)), "raw_headline": str(raw)}


def build_headline_workflow(
    model: BaseChatModel,
    *,
    prompt: str | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
    mode: str = "sync",
) -> Workflow:
    """Monta o workflow de titulação.

    Args:
        model: Chat model já instanciado (ver
            :func:`expanded_fake_news_corpus.config.build_llm`).
        prompt: Prompt de sistema. Se omitido, usa o do gênero notícia.
        max_retries: Tentativas extras em caso de erro do provedor.
        timeout: Tempo limite, em segundos, por chamada ao modelo.
        mode: ``"sync"`` ou ``"async"``, repassado ao :class:`Workflow`.

    Returns:
        Workflow ainda não compilado.
    """
    writer = Agent(
        model=model,
        name=AGENT_NAME,
        prompt=prompt or headline_prompt(NEWS),
        state=HeadlineState,
        input_fields="news_text",
        output_fields=["headline", "rationale"],
        max_retries=max_retries,
        timeout=timeout,
    )

    return Workflow(
        state=HeadlineState,
        agents=[writer],
        nodes={"sanitize": _sanitize_node},
        edges=[
            ("start", AGENT_NAME),
            (AGENT_NAME, "sanitize"),
            ("sanitize", "end"),
        ],
        mode=mode,  # type: ignore[arg-type]
    )


def check_headline(headline: str, news_text: str) -> list[str]:
    """Aplica checagens de qualidade sobre a manchete.

    As checagens não reprovam a manchete: elas viram avisos gravados junto ao
    resultado, para revisão manual e para as estatísticas do corpus.

    Args:
        headline: Manchete já normalizada.
        news_text: Texto da notícia que a originou.

    Returns:
        Lista de avisos (vazia quando tudo está dentro do esperado).
    """
    warnings: list[str] = []

    words = word_count(headline)
    if words < MIN_WORDS:
        warnings.append(f"headline too short ({words} words)")
    elif words > MAX_WORDS:
        warnings.append(f"headline too long ({words} words)")
    if len(headline) > MAX_CHARS:
        warnings.append(f"headline has {len(headline)} characters")

    # Heurística de fidelidade: sinaliza manchetes cujo vocabulário de conteúdo
    # praticamente não aparece na notícia — indício de alucinação.
    source = news_text.casefold()
    tokens = [t.casefold() for t in _WORD_RE.findall(headline)]
    if tokens:
        missing = [t for t in tokens if t not in source]
        if len(missing) > len(tokens) / 2:
            warnings.append(
                "most content words absent from the source text: "
                + ", ".join(missing[:5])
            )

    return warnings


class HeadlineAgent:
    """Interface de alto nível para gerar manchetes.

    Examples:
        >>> agent = HeadlineAgent()  # doctest: +SKIP
        >>> result = agent.generate("O Ministério da Saúde ...")  # doctest: +SKIP
        >>> result.headline  # doctest: +SKIP
        'Ministério da Saúde anuncia ampliação da campanha de vacinação'
    """

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        settings: LLMSettings | None = None,
        genre: str = NEWS,
        prompt: str | None = None,
        max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        """
        Args:
            model: Chat model pronto. Se omitido, é construído de ``settings``.
            settings: Configuração do LLM. Se omitida, é lida do ambiente.
            genre: Gênero do texto de entrada — ``"news"`` (Fake.br) ou
                ``"factcheck"`` (FakeTrueBR). Define o bloco acrescentado ao
                prompt base.
            prompt: Prompt de sistema completo, sobrepondo ``genre``. Use apenas
                para experimentação.
            max_input_chars: Limite de caracteres da notícia enviada ao modelo,
                ou ``None`` para enviar o texto inteiro.
        """
        self._settings = settings or (LLMSettings.from_env() if model is None else None)
        self._model = model or build_llm(self._settings)
        self._genre = genre
        prompt = prompt or headline_prompt(genre)
        self._prompt = prompt
        self._max_input_chars = max_input_chars

        timeout = self._settings.timeout if self._settings else None
        retries = self._settings.max_retries if self._settings else 2

        self._workflow = build_headline_workflow(
            self._model, prompt=prompt, max_retries=retries, timeout=timeout
        )
        self._graph = self._workflow.compile()

        self._async_workflow = build_headline_workflow(
            self._model,
            prompt=prompt,
            max_retries=retries,
            timeout=timeout,
            mode="async",
        )
        self._async_graph = self._async_workflow.compile()

    def _prepare(self, news_text: str) -> str:
        prepared = normalize_news_text(news_text, max_chars=self._max_input_chars)
        if not prepared:
            raise HeadlineError("Empty news text.")
        return prepared

    def _build_result(
        self, state: Any, news_text: str, source_id: str | None
    ) -> HeadlineResult:
        headline = _read(state, "headline")
        if not headline:
            raise HeadlineError(
                "Model returned no usable headline "
                f"(raw output: {_read(state, 'raw_headline')!r})."
            )

        return HeadlineResult(
            headline=headline,
            rationale=_read(state, "rationale"),
            raw_headline=_read(state, "raw_headline"),
            source_id=source_id,
            model=self.model_name,
            warnings=check_headline(headline, news_text),
        )

    @property
    def model_name(self) -> str:
        """Modelo em uso, no formato ``provedor/modelo`` (vazio se desconhecido)."""
        return self._settings.model if self._settings else ""

    @property
    def genre(self) -> str:
        """Gênero configurado para a titulação."""
        return self._genre

    @property
    def prompt_fingerprint(self) -> str:
        """Todo o texto que define o prompt, para registro de procedência."""
        return self._prompt

    @property
    def prompt(self) -> str:
        """Prompt de sistema efetivamente em uso."""
        return self._prompt

    def generate(
        self, news_text: str, *, source_id: str | None = None
    ) -> HeadlineResult:
        """Gera a manchete de uma notícia.

        Args:
            news_text: Corpo da notícia.
            source_id: Identificador da notícia no corpus de origem.

        Returns:
            Resultado com manchete, justificativa e avisos de qualidade.

        Raises:
            HeadlineError: Se o texto de entrada estiver vazio ou se o modelo
                não devolver uma manchete utilizável.
        """
        prepared = self._prepare(news_text)
        state = self._graph.invoke({"news_text": prepared})
        return self._build_result(state, prepared, source_id)

    async def agenerate(
        self, news_text: str, *, source_id: str | None = None
    ) -> HeadlineResult:
        """Versão assíncrona de :meth:`generate`."""
        prepared = self._prepare(news_text)
        state = await self._async_graph.ainvoke({"news_text": prepared})
        return self._build_result(state, prepared, source_id)

    def generate_many(
        self, items: Iterable[tuple[str | None, str]]
    ) -> Iterator[HeadlineResult]:
        """Processa notícias em sequência.

        Args:
            items: Pares ``(source_id, news_text)``.

        Yields:
            Um :class:`HeadlineResult` por notícia.
        """
        for source_id, news_text in items:
            yield self.generate(news_text, source_id=source_id)

    async def agenerate_many(
        self,
        items: Iterable[tuple[str | None, str]],
        *,
        concurrency: int = 4,
    ) -> AsyncIterator[HeadlineResult | HeadlineError]:
        """Processa notícias com paralelismo limitado.

        Falhas individuais são devolvidas como :class:`HeadlineError` em vez de
        interromper o lote — em execuções sobre milhares de notícias, uma
        recusa ou timeout isolado não deve derrubar o processamento.

        Args:
            items: Pares ``(source_id, news_text)``.
            concurrency: Número máximo de chamadas simultâneas ao provedor.

        Yields:
            Resultados na ordem de entrada, ou o erro correspondente.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(
            source_id: str | None, text: str
        ) -> HeadlineResult | HeadlineError:
            async with semaphore:
                try:
                    return await self.agenerate(text, source_id=source_id)
                except Exception as exc:  # noqa: BLE001 - erro por item, não do lote
                    error = HeadlineError(f"{source_id or '<no id>'}: {exc}")
                    return error

        tasks = [asyncio.create_task(run(sid, text)) for sid, text in items]
        for task in tasks:
            yield await task


def _read(state: Any, field: str) -> str:
    """Lê um campo do estado, que pode vir como dict ou modelo Pydantic."""
    value = (
        state.get(field, "") if isinstance(state, dict) else getattr(state, field, "")
    )
    return str(value or "")
