import json

import pytest

from fakegen_br.corpus import (
    CorpusError,
    JsonlWriter,
    read_csv,
    read_existing_ids,
    read_jsonl,
    read_text_dir,
)
from fakegen_br.schemas import HeadlineResult


def test_read_jsonl(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"id": "a", "text": "notícia um"}, ensure_ascii=False)
        + "\n\n"
        + json.dumps({"id": "b", "text": "notícia dois"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert list(read_jsonl(path, text_field="text", id_field="id")) == [
        ("a", "notícia um"),
        ("b", "notícia dois"),
    ]


def test_read_jsonl_sem_id_usa_numero_da_linha(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"text": "notícia"}) + "\n", encoding="utf-8")
    assert list(read_jsonl(path, text_field="text", id_field="id")) == [
        ("1", "notícia")
    ]


def test_read_jsonl_campo_ausente(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"corpo": "notícia"}) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="ausente"):
        list(read_jsonl(path, text_field="text", id_field="id"))


def test_read_csv(tmp_path):
    path = tmp_path / "corpus.csv"
    path.write_text("id,text\n1,primeira\n2,segunda\n3,\n", encoding="utf-8")
    assert list(read_csv(path, text_field="text", id_field="id")) == [
        ("1", "primeira"),
        ("2", "segunda"),
    ]


def test_read_csv_cabecalho_invalido(tmp_path):
    path = tmp_path / "corpus.csv"
    path.write_text("id,corpo\n1,primeira\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="cabeçalho"):
        list(read_csv(path, text_field="text", id_field="id"))


def test_read_text_dir_ordena_numericamente(tmp_path):
    for name in ("10", "2", "1"):
        (tmp_path / f"{name}.txt").write_text(f"notícia {name}", encoding="utf-8")
    (tmp_path / "ignorar.md").write_text("outro", encoding="utf-8")

    assert list(read_text_dir(tmp_path)) == [
        ("1", "notícia 1"),
        ("2", "notícia 2"),
        ("10", "notícia 10"),
    ]


def test_read_text_dir_vazio(tmp_path):
    with pytest.raises(CorpusError, match="nenhum arquivo"):
        list(read_text_dir(tmp_path))


def test_writer_e_resume(tmp_path):
    path = tmp_path / "saida" / "out.jsonl"
    with JsonlWriter(path) as writer:
        writer.write(HeadlineResult(headline="Primeira manchete", source_id="1"))

    assert read_existing_ids(path) == {"1"}

    with JsonlWriter(path, append=True) as writer:
        writer.write(HeadlineResult(headline="Segunda manchete", source_id="2"))

    assert read_existing_ids(path) == {"1", "2"}
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_read_existing_ids_arquivo_inexistente(tmp_path):
    assert read_existing_ids(tmp_path / "nao_existe.jsonl") == set()
