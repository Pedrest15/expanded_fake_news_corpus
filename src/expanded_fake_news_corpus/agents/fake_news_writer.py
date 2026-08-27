"""Gerador de fake news fiel ao artigo base (Silva et al., Figura 1).

Reproduz o método do trabalho anterior: prompt idêntico ao publicado e resposta
em texto livre delimitada pelas tags ``<syntheticText>`` e ``<changes>``, que
são extraídas depois. Não há saída estruturada, faixa de tamanho nem bloco por
gênero — só o que o artigo especifica.

A única adaptação é a entrada: o artigo entregava a notícia verdadeira inteira e
pedia que o LLM a modificasse; aqui a semente é a manchete produzida no estágio 1.

Convive com :mod:`expanded_fake_news_corpus.agents.fake_news`, que é a
variante com saída estruturada, faixas de tamanho calibradas e blocos por
gênero.

Grafo::

    start -> paper_fake_writer -> parse -> end
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import replace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraphlib import Agent, Workflow, create_state

from expanded_fake_news_corpus.config import LLMSettings, build_llm
from expanded_fake_news_corpus.prompts import PAPER_FAKE_PROMPT, PAPER_SYSTEM
from expanded_fake_news_corpus.schemas import FakeNewsError, FakeNewsWriterResult

#: Estado do grafo. ``raw_response`` é campo único de tipo ``str``, o que faz o
#: Agent devolver texto livre em vez de acionar ``with_structured_output``.
FakeNewsWriterState = create_state(
    "FakeNewsWriterState",
    include_messages=False,
    headline=(str, ""),
    raw_response=(str, ""),
    synthetic_text=(str, ""),
    changes=(str, ""),
)

AGENT_NAME = "paper_fake_writer"

DEFAULT_MAX_TOKENS = 2048

_SYNTHETIC_RE = re.compile(
    r"<\s*syntheticText\s*>(.*?)<\s*/\s*syntheticText\s*>", re.IGNORECASE | re.DOTALL
)
_CHANGES_RE = re.compile(
    r"<\s*changes\s*>(.*?)<\s*/\s*changes\s*>", re.IGNORECASE | re.DOTALL
)
#: Abertura sem fechamento — acontece quando a resposta é truncada pelo teto de
#: tokens. Sem este resgate, a amostra inteira seria descartada.
_SYNTHETIC_OPEN_RE = re.compile(r"<\s*syntheticText\s*>(.*)", re.IGNORECASE | re.DOTALL)


def parse_response(raw: str) -> tuple[str, str, list[str]]:
    """Extrai ``syntheticText`` e ``changes`` da resposta em texto livre.

    Args:
        raw: Resposta bruta do modelo.

    Returns:
        Tupla ``(synthetic_text, changes, avisos)``.
    """
    warnings: list[str] = []
    text = ""

    if (m := _SYNTHETIC_RE.search(raw)) is not None:
        text = m.group(1).strip()
    elif (m := _SYNTHETIC_OPEN_RE.search(raw)) is not None:
        text = m.group(1).strip()
        # Se veio changes depois, corta ali para não misturar as seções.
        text = re.split(r"<\s*changes\s*>", text, flags=re.IGNORECASE)[0].strip()
        warnings.append("unclosed <syntheticText> tag")
    else:
        text = raw.strip()
        warnings.append("response without format tags; whole text used")

    changes = ""
    if (m := _CHANGES_RE.search(raw)) is not None:
        changes = m.group(1).strip()
    else:
        parts = re.split(r"<\s*changes\s*>", raw, flags=re.IGNORECASE)
        if len(parts) > 1:
            changes = parts[1].strip()
            warnings.append("unclosed <changes> tag")
        else:
            warnings.append("response without a <changes> section")

    return text, changes, warnings


def _parse_node(state: Any) -> dict[str, str]:
    """Separa as duas seções da resposta bruta."""
    raw = str(getattr(state, "raw_response", "") or "")
    text, changes, _ = parse_response(raw)
    return {"synthetic_text": text, "changes": changes}


def build_fake_news_writer_workflow(
    model: BaseChatModel,
    *,
    prompt: str = PAPER_SYSTEM,
    max_retries: int = 2,
    timeout: float | None = None,
    mode: str = "sync",
) -> Workflow:
    """Monta o workflow do gerador fiel ao artigo."""
    writer = Agent(
        model=model,
        name=AGENT_NAME,
        prompt=prompt,
        state=FakeNewsWriterState,
        input_fields="headline",
        output_fields="raw_response",
        max_retries=max_retries,
        timeout=timeout,
    )
    return Workflow(
        state=FakeNewsWriterState,
        agents=[writer],
        nodes={"parse": _parse_node},
        edges=[
            ("start", AGENT_NAME),
            (AGENT_NAME, "parse"),
            ("parse", "end"),
        ],
        mode=mode,  # type: ignore[arg-type]
    )


class FakeNewsWriter:
    """Gera fake news com o prompt e o formato de resposta do artigo base."""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        settings: LLMSettings | None = None,
        prompt: str = PAPER_SYSTEM,
    ) -> None:
        self._settings = settings or (LLMSettings.from_env() if model is None else None)
        if self._settings is not None and not self._settings.max_tokens:
            self._settings = replace(self._settings, max_tokens=DEFAULT_MAX_TOKENS)
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

    def _prepare(self, headline: str) -> str:
        """Monta a mensagem de usuário como no código do artigo.

        Lá o prompt inteiro vai na mensagem de usuário, com a notícia
        interpolada ao final, e a mensagem de sistema é só a persona curta.
        """
        seed_headline = " ".join((headline or "").split())
        if not seed_headline:
            raise FakeNewsError("Empty seed headline.")
        return f"{PAPER_FAKE_PROMPT}\n{seed_headline}"

    def _build(
        self,
        state: Any,
        headline: str,
        source_id: str | None,
        headline_model: str,
    ) -> FakeNewsWriterResult:
        raw = _read(state, "raw_response")
        text, changes, warnings = parse_response(raw)
        if not text:
            raise FakeNewsError("Model returned no synthetic text.")
        if not changes:
            warnings.append("empty <changes> section")
        return FakeNewsWriterResult(
            synthetic_text=text,
            changes=changes,
            raw_response=raw,
            source_id=source_id,
            source_headline=headline,
            headline_model=headline_model,
            model=self.model_name,
            warnings=warnings,
        )

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
        """Prompt de sistema e de usuário juntos, para registro de procedência.

        Aqui o texto do artigo vai na mensagem de usuário, não na de sistema
        (ver :data:`PAPER_FAKE_PROMPT`), então só ``prompt`` não identificaria a
        execução.
        """
        return f"{self._prompt}\n{PAPER_FAKE_PROMPT}"

    def generate(
        self,
        headline: str,
        *,
        source_id: str | None = None,
        headline_model: str = "",
    ) -> FakeNewsWriterResult:
        """Gera a fake news de uma manchete."""
        message = self._prepare(headline)
        state = self._graph.invoke({"headline": message})
        return self._build(state, headline.strip(), source_id, headline_model)

    async def agenerate(
        self,
        headline: str,
        *,
        source_id: str | None = None,
        headline_model: str = "",
    ) -> FakeNewsWriterResult:
        """Versão assíncrona de :meth:`generate`."""
        message = self._prepare(headline)
        state = await self._async_graph.ainvoke({"headline": message})
        return self._build(state, headline.strip(), source_id, headline_model)

    def generate_many(self, items: Iterable[dict]) -> Iterator[FakeNewsWriterResult]:
        """Processa registros do estágio 1 em sequência."""
        for item in items:
            yield self.generate(
                item.get("headline", ""),
                source_id=item.get("source_id"),
                headline_model=item.get("model", ""),
            )

    async def agenerate_many(
        self, items: Iterable[dict], *, concurrency: int = 4
    ) -> AsyncIterator[FakeNewsWriterResult | FakeNewsError]:
        """Processa registros com paralelismo limitado."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(item: dict) -> FakeNewsWriterResult | FakeNewsError:
            async with semaphore:
                try:
                    return await self.agenerate(
                        item.get("headline", ""),
                        source_id=item.get("source_id"),
                        headline_model=item.get("model", ""),
                    )
                except Exception as exc:  # noqa: BLE001 - falha por item
                    return FakeNewsError(f"{item.get('source_id') or '<no id>'}: {exc}")

        tasks = [asyncio.create_task(run(item)) for item in items]
        for task in tasks:
            yield await task


def _read(state: Any, field: str) -> str:
    value = (
        state.get(field, "") if isinstance(state, dict) else getattr(state, field, "")
    )
    return str(value or "")
