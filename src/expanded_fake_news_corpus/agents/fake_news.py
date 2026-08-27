"""Agente gerador: recebe uma manchete verdadeira e escreve a notícia falsa.

Segundo estágio do pipeline do FakeGen.BR. A entrada é a saída do estágio de
titulação — uma manchete por notícia verdadeira, por modelo —, e a saída é o
corpus sintético propriamente dito.

Grafo::

    start -> fake_writer -> sanitize -> end

``fake_writer`` é um :class:`~langgraphlib.Agent` com saída estruturada
(``fake_headline`` + ``fake_text`` + ``changes``); ``sanitize`` é um nó
determinístico que limpa a manchete falsa e normaliza o corpo.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import replace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraphlib import Agent, Workflow, create_state

from expanded_fake_news_corpus.config import LLMSettings, build_llm
from expanded_fake_news_corpus.prompts import FAKE_WORD_RANGES, NEWS, fake_prompt
from expanded_fake_news_corpus.schemas import FakeNewsError, FakeNewsResult
from expanded_fake_news_corpus.text import clean_headline, word_count

#: Estado que trafega pelo grafo de geração.
FakeNewsState = create_state(
    "FakeNewsState",
    include_messages=False,
    headline=(str, ""),
    fake_headline=(str, ""),
    fake_text=(str, ""),
    changes=(list[str], []),
)

AGENT_NAME = "fake_writer"

#: Limite de tokens da resposta. Sem isso, provedores com teto baixo truncariam
#: a notícia no meio — e um corpo truncado passaria despercebido no JSONL.
DEFAULT_MAX_TOKENS = 2048

#: Marcas de que o modelo inseriu uma ressalva de "conteúdo fictício". O prompt
#: proíbe, mas modelos alinhados às vezes insistem, e uma amostra de treino com
#: aviso embutido é lixo silencioso no corpus.
_DISCLAIMER_RE = re.compile(
    r"\b(fict[íi]ci[oa]|sint[ée]tic[oa]|gerad[oa] por (?:ia|intelig[êe]ncia)|"
    r"n[ãa]o [ée] (?:uma )?not[íi]cia real|apenas para fins de pesquisa|"
    r"este texto (?:foi|é)|desinforma[çc][ãa]o simulada|aviso:|nota do editor)\b",
    re.IGNORECASE,
)


def _sanitize_node(state: Any) -> dict[str, Any]:
    """Normaliza a manchete falsa e o corpo da notícia."""
    raw_headline = str(getattr(state, "fake_headline", "") or "")
    raw_text = str(getattr(state, "fake_text", "") or "")
    text = "\n".join(line.strip() for line in raw_text.strip().splitlines())
    return {
        "fake_headline": clean_headline(raw_headline),
        "fake_text": re.sub(r"\n{3,}", "\n\n", text),
    }


def build_fake_news_workflow(
    model: BaseChatModel,
    *,
    prompt: str | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
    mode: str = "sync",
) -> Workflow:
    """Monta o workflow de geração de notícia falsa.

    Args:
        model: Chat model já instanciado.
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
        prompt=prompt or fake_prompt(NEWS),
        state=FakeNewsState,
        input_fields="headline",
        output_fields=["fake_headline", "fake_text", "changes"],
        max_retries=max_retries,
        timeout=timeout,
    )

    return Workflow(
        state=FakeNewsState,
        agents=[writer],
        nodes={"sanitize": _sanitize_node},
        edges=[
            ("start", AGENT_NAME),
            (AGENT_NAME, "sanitize"),
            ("sanitize", "end"),
        ],
        mode=mode,  # type: ignore[arg-type]
    )


def check_fake_news(
    fake_headline: str, fake_text: str, changes: Sequence[str], *, genre: str = NEWS
) -> list[str]:
    """Aplica checagens de qualidade sobre a notícia falsa gerada.

    Como no estágio de titulação, os avisos não reprovam a amostra: ficam
    gravados no resultado para revisão manual e para as estatísticas do corpus.

    Args:
        fake_headline: Manchete falsa já normalizada.
        fake_text: Corpo da notícia falsa.
        changes: Técnicas relatadas pelo modelo.
        genre: Gênero da manchete de origem, que define a faixa de tamanho.

    Returns:
        Lista de avisos (vazia quando tudo está dentro do esperado).
    """
    warnings: list[str] = []

    minimum, maximum = FAKE_WORD_RANGES.get(genre, FAKE_WORD_RANGES[NEWS])
    words = word_count(fake_text)
    if words < minimum:
        warnings.append(f"body too short ({words} words, minimum {minimum})")
    elif words > maximum:
        warnings.append(f"body too long ({words} words, maximum {maximum})")

    if not fake_headline:
        warnings.append("empty fake headline")
    if not changes:
        warnings.append("model reported no technique")

    if _DISCLAIMER_RE.search(fake_text):
        warnings.append("body contains a fictional-content disclaimer")
    if _DISCLAIMER_RE.search(fake_headline):
        warnings.append("headline contains a fictional-content disclaimer")

    # A primeira linha do corpo repetindo a manchete duplicaria o título no
    # corpus, artefato que o prompt manda evitar.
    first_line = fake_text.strip().splitlines()[0].strip() if fake_text.strip() else ""
    if first_line and first_line.casefold() == fake_headline.casefold():
        warnings.append("body repeats the headline on the first line")

    return warnings


class FakeNewsAgent:
    """Interface de alto nível para gerar notícias falsas a partir de manchetes."""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        settings: LLMSettings | None = None,
        genre: str = NEWS,
        prompt: str | None = None,
    ) -> None:
        """
        Args:
            model: Chat model pronto. Se omitido, é construído de ``settings``.
            settings: Configuração do LLM. Se omitida, é lida do ambiente.
            genre: Gênero da manchete de origem — ``"news"`` ou ``"factcheck"``.
            prompt: Prompt de sistema completo, sobrepondo ``genre``.
        """
        self._settings = settings or (LLMSettings.from_env() if model is None else None)
        if self._settings is not None and not self._settings.max_tokens:
            # Notícia inteira não cabe no teto padrão de vários provedores.
            self._settings = replace(self._settings, max_tokens=DEFAULT_MAX_TOKENS)
        self._model = model or build_llm(self._settings)
        self._genre = genre
        self._prompt = prompt or fake_prompt(genre)

        timeout = self._settings.timeout if self._settings else None
        retries = self._settings.max_retries if self._settings else 2

        self._workflow = build_fake_news_workflow(
            self._model, prompt=self._prompt, max_retries=retries, timeout=timeout
        )
        self._graph = self._workflow.compile()

        self._async_workflow = build_fake_news_workflow(
            self._model,
            prompt=self._prompt,
            max_retries=retries,
            timeout=timeout,
            mode="async",
        )
        self._async_graph = self._async_workflow.compile()

    def _prepare(self, headline: str) -> str:
        seed_headline = " ".join((headline or "").split())
        if not seed_headline:
            raise FakeNewsError("Empty seed headline.")
        return seed_headline

    def _build_result(
        self,
        state: Any,
        headline: str,
        source_id: str | None,
        headline_model: str,
    ) -> FakeNewsResult:
        fake_text = _read_str(state, "fake_text")
        if not fake_text:
            raise FakeNewsError("Model returned no fake news body.")

        fake_headline = _read_str(state, "fake_headline")
        changes = _read_list(state, "changes")

        return FakeNewsResult(
            fake_headline=fake_headline,
            fake_text=fake_text,
            changes=changes,
            source_id=source_id,
            source_headline=headline,
            headline_model=headline_model,
            genre=self._genre,
            model=self.model_name,
            warnings=check_fake_news(
                fake_headline, fake_text, changes, genre=self._genre
            ),
        )

    @property
    def model_name(self) -> str:
        """Modelo em uso, no formato ``provedor/modelo`` (vazio se desconhecido)."""
        return self._settings.model if self._settings else ""

    @property
    def genre(self) -> str:
        """Gênero configurado."""
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
        self,
        headline: str,
        *,
        source_id: str | None = None,
        headline_model: str = "",
    ) -> FakeNewsResult:
        """Gera a notícia falsa de uma manchete.

        Args:
            headline: Manchete verdadeira que serve de semente.
            source_id: Identificador da notícia no corpus de origem.
            headline_model: Modelo que escreveu a manchete, para procedência.

        Returns:
            Resultado com manchete falsa, corpo, técnicas e avisos.

        Raises:
            FakeNewsError: Se a manchete estiver vazia ou o modelo não devolver
                um corpo utilizável.
        """
        seed_headline = self._prepare(headline)
        state = self._graph.invoke({"headline": seed_headline})
        return self._build_result(state, seed_headline, source_id, headline_model)

    async def agenerate(
        self,
        headline: str,
        *,
        source_id: str | None = None,
        headline_model: str = "",
    ) -> FakeNewsResult:
        """Versão assíncrona de :meth:`generate`."""
        seed_headline = self._prepare(headline)
        state = await self._async_graph.ainvoke({"headline": seed_headline})
        return self._build_result(state, seed_headline, source_id, headline_model)

    def generate_many(self, items: Iterable[dict]) -> Iterator[FakeNewsResult]:
        """Processa manchetes em sequência.

        Args:
            items: Registros do estágio 1, com ``headline``, ``source_id`` e
                ``model``.
        """
        for item in items:
            yield self.generate(
                item.get("headline", ""),
                source_id=item.get("source_id"),
                headline_model=item.get("model", ""),
            )

    async def agenerate_many(
        self, items: Iterable[dict], *, concurrency: int = 4
    ) -> AsyncIterator[FakeNewsResult | FakeNewsError]:
        """Processa manchetes com paralelismo limitado.

        Falhas individuais viram :class:`FakeNewsError` em vez de interromper o
        lote, como no estágio de titulação.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(item: dict) -> FakeNewsResult | FakeNewsError:
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


def _read_str(state: Any, field: str) -> str:
    value = (
        state.get(field, "") if isinstance(state, dict) else getattr(state, field, "")
    )
    return str(value or "").strip()


def _read_list(state: Any, field: str) -> list[str]:
    value = (
        state.get(field, []) if isinstance(state, dict) else getattr(state, field, [])
    )
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v).strip() for v in (value or []) if str(v).strip()]
