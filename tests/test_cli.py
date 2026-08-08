import json

import pytest

from fakegen_br import cli
from fakegen_br.schemas import HeadlineError, HeadlineResult


class FakeAgent:
    """Agente de mentira: titula pelo início do texto e falha em ids marcados."""

    def __init__(self, **_):
        self.seen: list[tuple[str | None, str]] = []

    def generate(self, news_text, *, source_id=None):
        self.seen.append((source_id, news_text))
        if "FALHA" in news_text:
            raise HeadlineError("modelo recusou")
        return HeadlineResult(
            headline=news_text.strip().split("\n")[0][:60],
            source_id=source_id,
            model="stub/stub",
        )

    async def agenerate(self, news_text, *, source_id=None):
        return self.generate(news_text, source_id=source_id)

    async def agenerate_many(self, items, *, concurrency=4):
        for source_id, text in items:
            try:
                yield await self.agenerate(text, source_id=source_id)
            except HeadlineError as exc:
                yield HeadlineError(f"{source_id}: {exc}")


@pytest.fixture(autouse=True)
def agente_falso(monkeypatch):
    criados: list[FakeAgent] = []

    def factory(**kwargs):
        agent = FakeAgent(**kwargs)
        criados.append(agent)
        return agent

    monkeypatch.setattr(cli, "HeadlineAgent", factory)
    monkeypatch.setenv("FAKEGEN_MODEL", "ollama/llama3.1")
    return criados


def test_texto_direto(capsys):
    assert cli.main(["headline", "--text", "Governo anuncia medidas"]) == 0
    assert capsys.readouterr().out.strip() == "Governo anuncia medidas"


def test_saida_json(capsys):
    assert cli.main(["headline", "--text", "Governo anuncia medidas", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["headline"] == "Governo anuncia medidas"


def test_arquivo_usa_o_nome_como_id(tmp_path, capsys):
    noticia = tmp_path / "123.txt"
    noticia.write_text("Prefeitura entrega escola", encoding="utf-8")

    assert cli.main(["headline", "--file", str(noticia), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source_id"] == "123"


def test_lote_em_diretorio(tmp_path):
    entrada = tmp_path / "true"
    entrada.mkdir()
    (entrada / "1.txt").write_text("Primeira notícia", encoding="utf-8")
    (entrada / "2.txt").write_text("Segunda notícia", encoding="utf-8")
    saida = tmp_path / "out.jsonl"

    assert (
        cli.main(["headline", "--input-dir", str(entrada), "--output", str(saida)]) == 0
    )

    linhas = [json.loads(ln) for ln in saida.read_text(encoding="utf-8").splitlines()]
    assert [ln["source_id"] for ln in linhas] == ["1", "2"]
    assert linhas[0]["headline"] == "Primeira notícia"


def test_lote_jsonl_com_limite(tmp_path):
    entrada = tmp_path / "in.jsonl"
    entrada.write_text(
        "\n".join(json.dumps({"id": str(i), "text": f"Notícia {i}"}) for i in range(5)),
        encoding="utf-8",
    )
    saida = tmp_path / "out.jsonl"

    assert (
        cli.main(
            [
                "headline",
                "--input",
                str(entrada),
                "--output",
                str(saida),
                "--limit",
                "2",
            ]
        )
        == 0
    )
    assert len(saida.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_resume_pula_ids_ja_processados(tmp_path):
    entrada = tmp_path / "in.jsonl"
    entrada.write_text(
        "\n".join(json.dumps({"id": str(i), "text": f"Notícia {i}"}) for i in range(3)),
        encoding="utf-8",
    )
    saida = tmp_path / "out.jsonl"
    saida.write_text(
        json.dumps({"headline": "Notícia 0", "source_id": "0"}) + "\n", encoding="utf-8"
    )

    assert (
        cli.main(
            ["headline", "--input", str(entrada), "--output", str(saida), "--resume"]
        )
        == 0
    )

    ids = [json.loads(ln)["source_id"] for ln in saida.read_text().splitlines()]
    assert ids == ["0", "1", "2"]


def test_lote_com_concorrencia(tmp_path):
    entrada = tmp_path / "in.jsonl"
    entrada.write_text(
        "\n".join(json.dumps({"id": str(i), "text": f"Notícia {i}"}) for i in range(4)),
        encoding="utf-8",
    )
    saida = tmp_path / "out.jsonl"

    assert (
        cli.main(
            [
                "headline",
                "--input",
                str(entrada),
                "--output",
                str(saida),
                "--concurrency",
                "3",
            ]
        )
        == 0
    )
    assert len(saida.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_falha_isolada_nao_interrompe_o_lote(tmp_path, capsys):
    entrada = tmp_path / "in.jsonl"
    entrada.write_text(
        json.dumps({"id": "1", "text": "Notícia boa"})
        + "\n"
        + json.dumps({"id": "2", "text": "FALHA aqui"})
        + "\n",
        encoding="utf-8",
    )
    saida = tmp_path / "out.jsonl"

    assert cli.main(["headline", "--input", str(entrada), "--output", str(saida)]) == 0
    assert len(saida.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert "falha" in capsys.readouterr().err


def test_lote_sem_output(tmp_path, capsys):
    entrada = tmp_path / "in.jsonl"
    entrada.write_text(json.dumps({"id": "1", "text": "Notícia"}), encoding="utf-8")

    assert cli.main(["headline", "--input", str(entrada)]) == 1
    assert "--output é obrigatório" in capsys.readouterr().err


def test_extensao_desconhecida(tmp_path, capsys):
    entrada = tmp_path / "in.parquet"
    entrada.write_text("x", encoding="utf-8")

    assert (
        cli.main(["headline", "--input", str(entrada), "--output", str(tmp_path / "o")])
        == 1
    )
    assert "não reconhecida" in capsys.readouterr().err
