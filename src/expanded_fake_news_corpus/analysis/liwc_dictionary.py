"""Leitura de dicionários LIWC no formato ``.dic``.

O formato tem duas seções separadas por linhas com ``%``: primeiro os pares
``id<TAB>nome da categoria`` (indentados para marcar a hierarquia), depois as
entradas ``palavra<TAB>id<TAB>id...``. Palavras terminadas em ``*`` são
prefixos, e vale o prefixo mais longo que casar.

Os dicionários são proprietários e não acompanham este repositório — veja
https://www.liwc.app/.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CATEGORY_RE = re.compile(r"(\d+)\s+(.+)")

#: O ``.dic`` distribuído costuma vir com BOM.
_DICTIONARY_ENCODING = "utf-8-sig"

#: Tokens de palavra: ``\w`` é unicode-aware, então acentos são preservados.
_TOKEN_RE = re.compile(r"\w+")


class LiwcDictionaryError(Exception):
    """Arquivo ``.dic`` ausente ou malformado."""


@dataclass(frozen=True)
class TextCategorization:
    """Resultado de categorizar um texto inteiro."""

    total_words: int
    matched_words: int
    counts: Mapping[str, int]

    @property
    def coverage(self) -> float:
        """Percentual de palavras reconhecidas pelo dicionário."""
        if self.total_words == 0:
            return 0.0
        return 100 * self.matched_words / self.total_words


class LiwcDictionary:
    """Categorizador psicolinguístico baseado num dicionário LIWC.

    Responsabilidades:
    - Mapear cada palavra para as categorias LIWC a que pertence
    - Contar as ocorrências por categoria num texto
    """

    def __init__(
        self,
        categories: Mapping[int, str],
        exact_matches: Mapping[str, tuple[int, ...]],
        prefix_matches: Mapping[str, tuple[int, ...]],
    ) -> None:
        """Monta o dicionário a partir das seções já parseadas.

        Args:
            categories: Id da categoria -> nome
            exact_matches: Palavra completa -> ids de categoria
            prefix_matches: Prefixo (sem o ``*``) -> ids de categoria
        """
        self._categories = dict(categories)
        self._exact_matches = dict(exact_matches)
        self._prefix_matches = dict(prefix_matches)
        self._word_cache: dict[str, tuple[str, ...]] = {}

    def _category_ids(self, word: str) -> tuple[int, ...] | None:
        """Ids da palavra: casamento exato primeiro, senão o prefixo mais longo."""
        exact = self._exact_matches.get(word)
        if exact is not None:
            return exact

        # NOTA: varrer a lista de prefixos custaria O(n_prefixos) por palavra
        # (são milhares). Testar os prefixos da própria palavra, do mais longo
        # para o mais curto, dá o mesmo resultado em O(len(palavra)).
        for end in range(len(word), 0, -1):
            ids = self._prefix_matches.get(word[:end])
            if ids is not None:
                return ids
        return None

    @property
    def category_names(self) -> list[str]:
        """Nomes das categorias, na ordem dos ids."""
        return [self._categories[key] for key in sorted(self._categories)]

    def categorize_word(self, word: str) -> tuple[str, ...]:
        """Retorna as categorias LIWC de uma palavra.

        Args:
            word: Palavra a categorizar

        Returns:
            Nomes das categorias, possivelmente vazio
        """
        word = word.casefold()
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        ids = self._category_ids(word)
        names = (
            ()
            if ids is None
            else tuple(self._categories[cid] for cid in ids if cid in self._categories)
        )
        self._word_cache[word] = names
        return names

    def categorize_text(self, text: str) -> TextCategorization:
        """Conta as palavras de um texto por categoria LIWC.

        Args:
            text: Texto a categorizar

        Returns:
            Totais de palavras, palavras reconhecidas e contagem por categoria
        """
        tokens = _TOKEN_RE.findall(text.casefold())

        counts: dict[str, int] = {}
        matched_words = 0
        for token in tokens:
            names = self.categorize_word(token)
            if not names:
                continue
            matched_words += 1
            for name in names:
                counts[name] = counts.get(name, 0) + 1

        return TextCategorization(
            total_words=len(tokens),
            matched_words=matched_words,
            counts=counts,
        )


def load_liwc_dictionary(path: Path) -> LiwcDictionary:
    """Lê um dicionário LIWC do disco.

    Args:
        path: Caminho do arquivo ``.dic``

    Returns:
        Dicionário pronto para categorizar

    Raises:
        LiwcDictionaryError: Se o arquivo não existir ou não tiver as duas seções
    """
    try:
        lines = path.read_text(encoding=_DICTIONARY_ENCODING).splitlines()
    except FileNotFoundError as err:
        raise LiwcDictionaryError(
            f"LIWC dictionary not found: {path}. LIWC dictionaries are "
            "proprietary; get the Portuguese one from https://www.liwc.app/"
        ) from err

    separators = [index for index, line in enumerate(lines) if line.strip() == "%"]
    if len(separators) < 2:
        raise LiwcDictionaryError(f"missing '%' section markers: {path}")

    categories = _parse_categories(lines[separators[0] + 1 : separators[1]])
    exact_matches, prefix_matches = _parse_entries(lines[separators[1] + 1 :])

    logger.info(
        f"Dicionário LIWC carregado de {path.name}: {len(categories)} categorias, "
        f"{len(exact_matches)} palavras, {len(prefix_matches)} prefixos"
    )
    return LiwcDictionary(categories, exact_matches, prefix_matches)


def _parse_categories(lines: Sequence[str]) -> dict[int, str]:
    """Lê a seção de categorias, ignorando a indentação da hierarquia."""
    categories: dict[int, str] = {}
    for line in lines:
        match = _CATEGORY_RE.match(line.strip())
        if match:
            categories[int(match.group(1))] = match.group(2).strip()
    return categories


def _parse_entries(
    lines: Sequence[str],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Lê a seção de palavras, separando casamentos exatos de prefixos."""
    exact_matches: dict[str, tuple[int, ...]] = {}
    prefix_matches: dict[str, tuple[int, ...]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "%":
            continue

        parts = stripped.split("\t")
        if len(parts) < 2:
            parts = stripped.split()
        if len(parts) < 2:
            continue

        word = parts[0].strip().casefold()
        category_ids = tuple(
            int(part)
            for part in (piece.strip() for piece in parts[1:])
            if part.isdigit()
        )
        if not category_ids or not word:
            continue

        if word.endswith("*"):
            prefix = word[:-1]
            if prefix:
                prefix_matches[prefix] = category_ids
        else:
            exact_matches[word] = category_ids

    return exact_matches, prefix_matches
