import asyncio

import pytest

from fakegen_br.agents.headline import HeadlineAgent, check_headline
from fakegen_br.schemas import HeadlineError

NEWS = (
    "O Ministério da Saúde anunciou nesta terça-feira a ampliação da campanha de "
    "vacinação em São Paulo. Segundo a pasta, mais 2 milhões de doses serão "
    "distribuídas aos municípios do estado até o fim do mês."
)


def test_gera_manchete(stub_model):
    agent = HeadlineAgent(model=stub_model)
    result = agent.generate(NEWS, source_id="42")

    assert result.headline == (
        "Ministério da Saúde amplia campanha de vacinação em São Paulo"
    )
    assert result.rationale.startswith("O anúncio")
    assert result.source_id == "42"
    assert result.warnings == []


def test_prompt_recebe_o_texto_da_noticia(stub_model):
    HeadlineAgent(model=stub_model).generate(NEWS)

    prompt = stub_model.last_prompt_text
    assert "editor de jornalismo brasileiro" in prompt
    assert "2 milhões de doses" in prompt


def test_manchete_e_normalizada(stub_model):
    stub_model.payload["headline"] = '  **"Governo amplia vacinação em São Paulo."**  '
    result = HeadlineAgent(model=stub_model).generate(NEWS)

    assert result.headline == "Governo amplia vacinação em São Paulo"
    assert result.raw_headline == stub_model.payload["headline"]


def test_texto_vazio(stub_model):
    with pytest.raises(HeadlineError, match="vazio"):
        HeadlineAgent(model=stub_model).generate("   \n  ")


def test_modelo_sem_manchete(stub_model):
    stub_model.payload["headline"] = "   "
    with pytest.raises(HeadlineError, match="não devolveu"):
        HeadlineAgent(model=stub_model).generate(NEWS)


def test_truncagem_do_texto_de_entrada(stub_model):
    agent = HeadlineAgent(model=stub_model, max_input_chars=60)
    agent.generate(NEWS)

    assert "2 milhões de doses" not in stub_model.last_prompt_text


def test_geracao_assincrona_em_lote(stub_model):
    agent = HeadlineAgent(model=stub_model)
    items = [("1", NEWS), ("2", NEWS)]

    async def collect():
        return [r async for r in agent.agenerate_many(items, concurrency=2)]

    results = asyncio.run(collect())
    assert [r.source_id for r in results] == ["1", "2"]


def test_falha_de_item_nao_derruba_o_lote(stub_model):
    agent = HeadlineAgent(model=stub_model)
    items = [("1", NEWS), ("2", "  ")]

    async def collect():
        return [r async for r in agent.agenerate_many(items, concurrency=2)]

    ok, failed = asyncio.run(collect())
    assert ok.source_id == "1"
    assert isinstance(failed, HeadlineError)
    assert "2:" in str(failed)


def test_check_headline_avisa_tamanho():
    assert "curta" in " ".join(
        check_headline("Governo anuncia", "Governo anuncia algo")
    )
    longa = " ".join(["palavra"] * 25)
    assert any("longa" in w for w in check_headline(longa, longa))


def test_check_headline_avisa_conteudo_ausente_no_texto():
    warnings = check_headline(
        "Extraterrestres invadem Brasília durante sessão solene", NEWS
    )
    assert any("não aparece no texto" in w for w in warnings)


def test_check_headline_sem_avisos():
    assert (
        check_headline(
            "Ministério da Saúde amplia campanha de vacinação em São Paulo", NEWS
        )
        == []
    )
