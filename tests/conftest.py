"""Utilidades compartilhadas pelos testes."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda


class StubModel:
    """Modelo de mentira que devolve sempre a mesma saída estruturada.

    Reproduz apenas o que o :class:`langgraphlib.Agent` usa quando há saída
    estruturada e nenhuma tool: ``with_structured_output`` e a composição
    ``prompt | modelo``.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[Any] = []

    def with_structured_output(self, schema: type, **_: Any) -> RunnableLambda:
        def run(prompt_value: Any) -> Any:
            self.calls.append(prompt_value)
            return schema(**self.payload)

        return RunnableLambda(run)

    @property
    def last_prompt_text(self) -> str:
        """Texto do último prompt enviado ao modelo."""
        return self.calls[-1].to_string()


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel(
        {
            "headline": "Ministério da Saúde amplia campanha de vacinação em São Paulo",
            "rationale": "O anúncio da ampliação é o fato central do texto.",
        }
    )
