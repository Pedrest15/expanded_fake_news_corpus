"""Download sob demanda dos recursos do NLTK usados nas análises."""

from __future__ import annotations

import logging

import nltk

logger = logging.getLogger(__name__)

#: Nome do pacote no downloader -> caminho usado por ``nltk.data.find``.
NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}

#: NLTK >= 3.9 procura ``punkt_tab``; versões anteriores, ``punkt``.
TOKENIZER_PACKAGES = ("punkt", "punkt_tab")


class UnknownNltkPackageError(Exception):
    """Pacote fora de :data:`NLTK_PACKAGES`."""


def ensure_nltk_resources(*packages: str) -> None:
    """Garante que os recursos do NLTK estejam disponíveis localmente.

    Args:
        *packages: Nomes de pacotes presentes em :data:`NLTK_PACKAGES`

    Raises:
        UnknownNltkPackageError: Se algum pacote não estiver mapeado
    """
    for package in packages:
        lookup_path = NLTK_PACKAGES.get(package)
        if lookup_path is None:
            raise UnknownNltkPackageError(f"unknown NLTK package: {package!r}")

        try:
            nltk.data.find(lookup_path)
            continue
        except LookupError:
            logger.info(f"Baixando recurso do NLTK: {package}")

        if not nltk.download(package, quiet=True):
            logger.warning(f"Falha ao baixar recurso do NLTK: {package}")
