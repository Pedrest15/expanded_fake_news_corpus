import pytest

from fakegen_br.config import (
    ConfigError,
    LLMSettings,
    resolve_api_key,
    split_model,
)


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch):
    for var in (
        "FAKEGEN_MODEL",
        "FAKEGEN_TEMPERATURE",
        "FAKEGEN_MAX_TOKENS",
        "FAKEGEN_TIMEOUT",
        "FAKEGEN_MAX_RETRIES",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_split_model():
    assert split_model("anthropic/claude-sonnet-4-5") == (
        "anthropic",
        "claude-sonnet-4-5",
    )
    assert split_model("Ollama/llama3.1") == ("ollama", "llama3.1")


def test_split_model_invalido():
    with pytest.raises(ConfigError, match="provedor/modelo"):
        split_model("gpt-4o")


def test_ollama_nao_exige_chave():
    assert resolve_api_key("ollama") == ""


def test_provedor_sem_chave_configurada():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        resolve_api_key("anthropic")


def test_from_env_le_variaveis(monkeypatch):
    monkeypatch.setenv("FAKEGEN_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("FAKEGEN_TEMPERATURE", "0.7")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")

    settings = LLMSettings.from_env()
    assert settings.provider == "openai"
    assert settings.temperature == 0.7
    assert settings.api_key == "sk-teste"


def test_from_env_overrides_tem_precedencia(monkeypatch):
    monkeypatch.setenv("FAKEGEN_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    settings = LLMSettings.from_env(model="anthropic/claude-sonnet-4-5")
    assert settings.model == "anthropic/claude-sonnet-4-5"
    assert settings.api_key == "sk-ant"


def test_from_env_ignora_overrides_none(monkeypatch):
    monkeypatch.setenv("FAKEGEN_MODEL", "ollama/llama3.1")
    settings = LLMSettings.from_env(model=None, temperature=None)
    assert settings.model == "ollama/llama3.1"
    assert settings.temperature == 0.2


def test_temperatura_invalida(monkeypatch):
    monkeypatch.setenv("FAKEGEN_TEMPERATURE", "quente")
    with pytest.raises(ConfigError, match="numérico"):
        LLMSettings.from_env()
