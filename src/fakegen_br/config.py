"""Configuração do LLM usado pelos agentes do FakeGen.BR.

O modelo é declarado sempre no formato ``provedor/modelo`` (ex.: ``ollama/llama3.1``,
``anthropic/claude-sonnet-4-5``, ``openai/gpt-4o-mini``), e a chave de API é
resolvida a partir da variável de ambiente do provedor correspondente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langgraphlib import get_model

DEFAULT_MODEL = "ollama/llama3.1"

#: Provedores previstos para o projeto e a variável de ambiente com a chave.
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "",  # execução local, não exige chave
}


class ConfigError(Exception):
    """Configuração ausente ou inválida."""


def split_model(model: str) -> tuple[str, str]:
    """Divide ``provedor/modelo`` em uma tupla ``(provedor, modelo)``."""
    if "/" not in model:
        raise ConfigError(
            f"Modelo inválido: {model!r}. Use o formato 'provedor/modelo', "
            "por exemplo 'ollama/llama3.1' ou 'anthropic/claude-sonnet-4-5'."
        )
    provider, name = model.split("/", 1)
    return provider.strip().lower(), name.strip()


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Parâmetros de conexão e amostragem do LLM."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = 120.0
    max_retries: int = 2

    @property
    def provider(self) -> str:
        """Provedor extraído de :attr:`model`."""
        return split_model(self.model)[0]

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMSettings:
        """Monta as configurações a partir do ambiente (lendo ``.env`` se existir).

        Variáveis reconhecidas: ``FAKEGEN_MODEL``, ``FAKEGEN_TEMPERATURE``,
        ``FAKEGEN_MAX_TOKENS``, ``FAKEGEN_TIMEOUT``, ``FAKEGEN_MAX_RETRIES``,
        ``OLLAMA_BASE_URL``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.

        Args:
            **overrides: Valores que têm precedência sobre o ambiente. Chaves com
                valor ``None`` são ignoradas (facilita repassar flags de CLI).
        """
        load_dotenv(override=False)

        settings = cls(
            model=os.getenv("FAKEGEN_MODEL", DEFAULT_MODEL),
            temperature=_env_float("FAKEGEN_TEMPERATURE", 0.2),
            max_tokens=_env_int("FAKEGEN_MAX_TOKENS", None),
            base_url=os.getenv("OLLAMA_BASE_URL") or None,
            timeout=_env_float("FAKEGEN_TIMEOUT", 120.0),
            max_retries=_env_int("FAKEGEN_MAX_RETRIES", 2) or 0,
        )
        settings = replace(
            settings, **{k: v for k, v in overrides.items() if v is not None}
        )

        if settings.api_key is None:
            settings = replace(settings, api_key=resolve_api_key(settings.provider))
        return settings


def resolve_api_key(provider: str) -> str:
    """Lê a chave de API do provedor no ambiente.

    Returns:
        A chave encontrada, ou string vazia para provedores locais (Ollama).

    Raises:
        ConfigError: Se o provedor exigir chave e ela não estiver definida.
    """
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        # Provedor fora da lista prevista: LangGraphLib ainda pode suportá-lo,
        # então tentamos a convenção <PROVEDOR>_API_KEY antes de desistir.
        env_var = f"{provider.upper()}_API_KEY"
    if not env_var:
        return ""

    key = os.getenv(env_var, "")
    if not key:
        raise ConfigError(
            f"Provedor {provider!r} exige a variável de ambiente {env_var}. "
            "Defina-a no ambiente ou em um arquivo .env."
        )
    return key


def build_llm(settings: LLMSettings | None = None, **overrides: Any) -> BaseChatModel:
    """Instancia o chat model configurado.

    Args:
        settings: Configuração pronta. Se omitida, é lida do ambiente.
        **overrides: Ajustes pontuais aplicados sobre ``settings``.

    Returns:
        Instância LangChain do modelo, pronta para uso pelos agentes.
    """
    settings = settings or LLMSettings.from_env()
    if overrides:
        settings = replace(
            settings, **{k: v for k, v in overrides.items() if v is not None}
        )

    api_key = settings.api_key
    if api_key is None:
        api_key = resolve_api_key(settings.provider)

    extra: dict[str, Any] = {}
    if settings.provider == "ollama" and settings.base_url:
        extra["base_url"] = settings.base_url

    return get_model(
        settings.model,
        api_key=api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        **extra,
    )


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} deve ser numérico, recebido {raw!r}.") from exc


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} deve ser inteiro, recebido {raw!r}.") from exc
