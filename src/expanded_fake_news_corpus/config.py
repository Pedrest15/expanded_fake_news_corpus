"""Configuração do LLM usado pelos agentes do FakeGen.BR.

O modelo é declarado sempre no formato ``provedor/modelo`` (ex.: ``ollama/llama3.1``,
``anthropic/claude-sonnet-5``, ``openai/gpt-4o-mini``), e a chave de API é
resolvida a partir da variável de ambiente do provedor correspondente.
"""

from __future__ import annotations

import os
import re
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
    "groq": "GROQ_API_KEY",
    "ollama": "",  # execução local, não exige chave
}

#: Parâmetros de amostragem que cada provedor realmente aceita.
#:
#: A cobertura é desigual e é preciso filtrar antes de enviar: ``top_k`` não
#: existe na API da OpenAI (o LangChain o desvia para ``model_kwargs`` com um
#: aviso, e a rejeição só aparece na chamada HTTP) e ``seed`` não existe na API
#: da Anthropic. Só ``temperature`` e ``top_p`` são comuns aos três.
SAMPLING_SUPPORT: dict[str, frozenset[str]] = {
    "anthropic": frozenset({"temperature", "top_p", "top_k"}),
    "openai": frozenset({"temperature", "top_p", "seed"}),
    # A API do Groq é compatível com a da OpenAI: tem top_p e seed, não tem top_k.
    "groq": frozenset({"temperature", "top_p", "seed"}),
    "ollama": frozenset({"temperature", "top_p", "top_k", "seed"}),
}

#: Parâmetros que o ChatGroq não declara como campo e precisa receber via
#: ``model_kwargs`` — passá-los direto funciona, mas emite aviso do LangChain.
_GROQ_VIA_MODEL_KWARGS = frozenset({"top_p", "seed"})

#: Para provedores fora da lista, só o que é universal.
DEFAULT_SAMPLING_SUPPORT = frozenset({"temperature", "top_p"})

#: Modelos da Anthropic que não aceitam parâmetro algum de amostragem: a
#: partir de Opus 4.7 e da geração 5 (Sonnet 5, Opus 5, Fable 5) a API rejeita
#: ``temperature``, ``top_p`` e ``top_k`` com 400 (``"`temperature` is
#: deprecated for this model"``). O LangChain não filtra; o erro só aparece na
#: chamada HTTP.
_ANTHROPIC_NO_SAMPLING_RE = re.compile(r"^claude-(sonnet-5|opus-5|fable-5|opus-4-[78])")


#: Caracteres inaceitáveis em nome de pasta. Tags do Ollama trazem ``:``.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(value: str) -> str:
    """Converte um nome de provedor ou modelo em componente de caminho seguro."""
    return _SLUG_RE.sub("_", value)


def model_path(root, model: str, run_name: str | None = None):
    """Pasta de saída de um modelo.

    ``<root>/<provedor>/<modelo>`` ou, com ``run_name``,
    ``<root>/<run-name>/<provedor>/<modelo>``. O nível extra existe para que o
    mesmo modelo possa ser rodado com configurações diferentes — outra
    temperatura, outro prompt — sem que uma execução apague a anterior.

    Args:
        root: Raiz da saída.
        model: Modelo em ``provedor/modelo``.
        run_name: Nome opcional da execução.

    Returns:
        Caminho da pasta, do mesmo tipo de ``root``.
    """
    provider, name = split_model(model)
    if run_name:
        root = root / model_slug(run_name)
    return root / model_slug(provider) / model_slug(name)


class ConfigError(Exception):
    """Configuração ausente ou inválida."""


def sampling_support(model: str) -> frozenset[str]:
    """Parâmetros de amostragem que ``provedor/modelo`` aceita.

    Parte da tabela por provedor (:data:`SAMPLING_SUPPORT`) e refina por
    modelo onde a API mudou dentro do mesmo provedor.
    """
    provider, name = split_model(model)
    if provider == "anthropic" and _ANTHROPIC_NO_SAMPLING_RE.match(name):
        return frozenset()
    return SAMPLING_SUPPORT.get(provider, DEFAULT_SAMPLING_SUPPORT)


def split_model(model: str) -> tuple[str, str]:
    """Divide ``provedor/modelo`` em uma tupla ``(provedor, modelo)``."""
    if "/" not in model:
        raise ConfigError(
            f"Invalid model string: {model!r}. Expected 'provider/model', "
            "e.g. 'ollama/llama3.1' or 'anthropic/claude-sonnet-4-5'."
        )
    provider, name = model.split("/", 1)
    return provider.strip().lower(), name.strip()


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Parâmetros de conexão e amostragem do LLM."""

    model: str = DEFAULT_MODEL
    #: ``None`` não envia temperatura: o provedor usa o padrão dele.
    temperature: float | None = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
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
        ``FAKEGEN_TOP_P``, ``FAKEGEN_TOP_K``, ``FAKEGEN_SEED``,
        ``FAKEGEN_MAX_TOKENS``, ``FAKEGEN_TIMEOUT``, ``FAKEGEN_MAX_RETRIES``,
        ``OLLAMA_BASE_URL``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.

        Args:
            **overrides: Valores que têm precedência sobre o ambiente. Chaves com
                valor ``None`` são ignoradas (facilita repassar flags de CLI).
        """
        load_dotenv(override=False)

        settings = cls(
            model=os.getenv("FAKEGEN_MODEL", DEFAULT_MODEL),
            temperature=_env_float("FAKEGEN_TEMPERATURE", 0.0),
            top_p=_env_float("FAKEGEN_TOP_P", None),
            top_k=_env_int("FAKEGEN_TOP_K", None),
            seed=_env_int("FAKEGEN_SEED", None),
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
            f"Provider {provider!r} requires the {env_var} environment variable. "
            "Set it in the environment or in a .env file."
        )
    return key


def resolve_sampling(settings: LLMSettings) -> tuple[dict[str, Any], list[str]]:
    """Separa os parâmetros de amostragem entre aplicáveis e descartados.

    Um parâmetro só é enviado quando foi pedido explicitamente (``temperature``
    é, por padrão, mas pode ser desligada com ``None``) **e** o provedor o
    aceita. Os descartados são devolvidos para que
    quem chama registre o fato: numa comparação entre provedores, um ``top_k``
    silenciosamente ignorado em um deles invalidaria o experimento.

    Args:
        settings: Configuração do LLM.

    Returns:
        Tupla ``(aplicados, descartados)``.
    """
    supported = sampling_support(settings.model)
    requested = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "seed": settings.seed,
    }

    applied: dict[str, Any] = {}
    dropped: list[str] = []
    for name, value in requested.items():
        if value is None:
            continue
        if name in supported:
            applied[name] = value
        else:
            dropped.append(name)
    return applied, dropped


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

    applied, _ = resolve_sampling(settings)
    extra: dict[str, Any] = {k: v for k, v in applied.items() if k != "temperature"}
    max_tokens = settings.max_tokens

    if settings.provider == "ollama":
        if settings.base_url:
            extra["base_url"] = settings.base_url
        # ChatOllama não tem max_tokens: aceita o argumento e o ignora em
        # silêncio. O equivalente é num_predict.
        if max_tokens:
            extra["num_predict"] = max_tokens
            max_tokens = None
    elif settings.provider == "groq":
        via_kwargs = {k: extra.pop(k) for k in _GROQ_VIA_MODEL_KWARGS if k in extra}
        if via_kwargs:
            extra["model_kwargs"] = via_kwargs

    # ``get_model`` tem temperatura 0 como padrão; ``None`` precisa ser
    # explícito para que nada seja enviado ao provedor.
    return get_model(
        settings.model,
        api_key=api_key,
        temperature=applied.get("temperature"),
        max_tokens=max_tokens,
        **extra,
    )


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be numeric, got {raw!r}.") from exc


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
