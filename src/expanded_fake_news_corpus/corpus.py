"""Leitura das notícias de entrada e escrita incremental dos resultados.

Formatos suportados na entrada:

* **JSONL** — um objeto por linha, com um campo de texto e (opcionalmente) um de id;
* **CSV** — uma notícia por linha, mesmos campos;
* **diretório de .txt** — uma notícia por arquivo, com o id vindo do nome do
  arquivo. É o formato do Fake.br (``full_texts/true/1.txt`` etc.).
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

#: Um item de entrada: ``(source_id, news_text)``.
NewsItem = tuple[str | None, str]


class CorpusError(Exception):
    """Entrada malformada ou campo ausente."""


def read_jsonl(
    path: Path, *, text_field: str, id_field: str | None
) -> Iterator[NewsItem]:
    """Lê notícias de um arquivo JSONL."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{number}: invalid JSON ({exc}).") from exc
            if text_field not in record:
                raise CorpusError(
                    f"{path}:{number}: missing text field {text_field!r}."
                )
            source_id = _read_id(record, id_field, fallback=str(number))
            yield source_id, str(record[text_field])


def read_csv(
    path: Path, *, text_field: str, id_field: str | None
) -> Iterator[NewsItem]:
    """Lê notícias de um CSV com cabeçalho."""
    # Corpora de notícia costumam ter campos longos; o limite padrão do csv estoura.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or text_field not in reader.fieldnames:
            raise CorpusError(
                f"{path}: text field {text_field!r} not in header "
                f"({reader.fieldnames})."
            )
        for number, record in enumerate(reader, start=1):
            text = record.get(text_field) or ""
            if not text.strip():
                continue
            yield _read_id(record, id_field, fallback=str(number)), text


def read_text_dir(path: Path, *, pattern: str = "*.txt") -> Iterator[NewsItem]:
    """Lê notícias de um diretório de arquivos de texto.

    O id de cada notícia é o nome do arquivo sem extensão. Arquivos com nome
    numérico são ordenados numericamente (1.txt, 2.txt, ..., 10.txt).
    """
    files = sorted(path.glob(pattern), key=_sort_key)
    if not files:
        raise CorpusError(f"{path}: no file matches {pattern!r}.")
    for file in files:
        yield file.stem, file.read_text(encoding="utf-8", errors="replace")


def read_headline_records(path: Path) -> Iterator[dict]:
    """Lê os registros do estágio de titulação, para alimentar o estágio 2.

    Diferente de :func:`read_jsonl`, devolve o registro inteiro em vez do par
    ``(id, texto)``: o estágio de geração precisa carregar a procedência —
    sobretudo qual modelo escreveu a manchete de origem.

    Args:
        path: JSONL produzido por ``fakegen headline``.

    Yields:
        Registros com ao menos ``headline``; ``source_id`` e ``model`` quando
        presentes.

    Raises:
        CorpusError: Se uma linha não for JSON válido ou não tiver ``headline``.
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{number}: invalid JSON ({exc}).") from exc
            if not str(record.get("headline") or "").strip():
                raise CorpusError(
                    f"{path}:{number}: missing or empty 'headline' field. "
                    "Stage 2 input is the output of 'fakegen headline'."
                )
            yield record


def read_existing_ids(path: Path, *, id_field: str = "source_id") -> set[str]:
    """Coleta os ids já presentes em um JSONL de saída (para retomar um lote)."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get(id_field)
            if value is not None:
                seen.add(str(value))
    return seen


class JsonlWriter:
    """Escreve resultados em JSONL, com flush a cada linha.

    O flush por linha é intencional: lotes longos sobre os corpora podem ser
    interrompidos, e o que já foi gerado precisa sobreviver no arquivo para que
    a execução seja retomável com ``--resume``.
    """

    def __init__(self, path: Path, *, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, result: BaseModel) -> None:
        """Grava um resultado de qualquer estágio do pipeline."""
        self._handle.write(result.model_dump_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Fecha o arquivo."""
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _read_id(record: dict, id_field: str | None, *, fallback: str) -> str:
    if id_field and record.get(id_field) not in (None, ""):
        return str(record[id_field])
    return fallback


def _sort_key(path: Path) -> tuple[int, str | int]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)
