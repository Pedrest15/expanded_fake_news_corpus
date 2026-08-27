"""FakeGen.BR — geração de corpus de fake news em português brasileiro.

O pipeline parte de notícias verdadeiras dos corpora Fake.br e FakeTrueBR. O
primeiro estágio, implementado aqui, é a titulação: um agente lê a notícia e
escreve a manchete correspondente.
"""

from expanded_fake_news_corpus.agents.headline import (
    HeadlineAgent,
    HeadlineState,
    build_headline_workflow,
)
from expanded_fake_news_corpus.config import LLMSettings, build_llm
from expanded_fake_news_corpus.schemas import HeadlineError, HeadlineResult

__version__ = "0.1.0"

__all__ = [
    "HeadlineAgent",
    "HeadlineState",
    "build_headline_workflow",
    "LLMSettings",
    "build_llm",
    "HeadlineResult",
    "HeadlineError",
]
