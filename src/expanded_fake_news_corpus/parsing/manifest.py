"""Proveniência dos CoNLL-U: ``data/parsed/<experimento>/manifest.json``.

Cada etapa (parsing, EUD) grava a sua seção; as demais ficam como estão.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def timestamp() -> str:
    """Instante atual com fuso, no formato dos demais ``meta.json``."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_manifest(parsed_dir: Path) -> dict:
    """Lê o manifesto de um experimento, ou um dicionário vazio se não existir."""
    path = parsed_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(parsed_dir: Path, section: Mapping[str, object]) -> Path:
    """Mescla ``section`` no manifesto do experimento e grava.

    Args:
        parsed_dir: Pasta ``data/parsed/<experimento>``
        section: Chaves a acrescentar ou sobrescrever

    Returns:
        Caminho do manifesto gravado
    """
    manifest = read_manifest(parsed_dir)
    manifest.update(section)
    path = parsed_dir / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
