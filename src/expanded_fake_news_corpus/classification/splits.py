"""Divisão treino/teste do trabalho anterior, lida dos arquivos vendorizados.

O experimento de transferência treina o detector nos ``uid`` de treino do
trabalho anterior e usa os de teste como lado humano da avaliação. Sem respeitar
essa divisão o resultado mediria memorização: **16 das 20 notícias do nosso
recorte estão no treino de lá**, porque a amostra foi sorteada dos mesmos
corpora de origem.

Ver ``resources/prior_splits/README.md`` para a procedência dos arquivos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from expanded_fake_news_corpus.analysis.documents import PROJECT_ROOT, Source

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "resources" / "prior_splits"

#: Arquivo de cada combinação (corpus de origem, parte).
_SPLIT_FILES: dict[tuple[Source, str], str] = {
    (Source.FAKEBR, "train"): "train_fake_br.txt",
    (Source.FAKEBR, "test"): "test_fake_br.txt",
    (Source.FAKETRUEBR, "train"): "train_fake_true_br.txt",
    (Source.FAKETRUEBR, "test"): "test_fake_true_br.txt",
}

#: O ``.txt`` do FakeTrueBR é 0-based lá; o nosso ``uid`` é 1-based.
_INDEX_OFFSET: dict[Source, int] = {Source.FAKEBR: 0, Source.FAKETRUEBR: 1}


class SplitError(Exception):
    """Arquivo de split ausente ou ilegível."""


@dataclass(frozen=True)
class PriorSplit:
    """Os ``uid`` de treino e de teste do trabalho anterior."""

    train: frozenset[str]
    test: frozenset[str]

    def part_of(self, uid: str) -> str | None:
        """Parte a que o ``uid`` pertence, ou None se não estiver em nenhuma."""
        if uid in self.train:
            return "train"
        if uid in self.test:
            return "test"
        return None


def _read_uids(path: Path, source: Source) -> set[str]:
    """Lê um arquivo de split e devolve os ``uid`` no formato deste projeto."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as err:
        raise SplitError(f"prior split file not found: {path}") from err

    offset = _INDEX_OFFSET[source]
    uids: set[str] = set()
    for line in lines:
        stem = line.strip().removesuffix(".txt")
        if not stem.isdigit():
            continue
        uids.add(f"{source.value}:{int(stem) + offset}")
    return uids


def load_prior_split(split_dir: Path | None = None) -> PriorSplit:
    """Carrega a divisão treino/teste do trabalho anterior.

    Args:
        split_dir: Pasta dos arquivos; usa :data:`DEFAULT_SPLIT_DIR` se omitida

    Returns:
        Os conjuntos de ``uid``, já traduzidos para o formato deste projeto

    Raises:
        SplitError: Se algum dos quatro arquivos faltar
    """
    root = split_dir or DEFAULT_SPLIT_DIR
    parts: dict[str, set[str]] = {"train": set(), "test": set()}
    for (source, part), filename in _SPLIT_FILES.items():
        parts[part] |= _read_uids(root / filename, source)

    split = PriorSplit(frozenset(parts["train"]), frozenset(parts["test"]))
    logger.info(
        f"Split do trabalho anterior: {len(split.train)} uid(s) de treino, "
        f"{len(split.test)} de teste"
    )
    return split
