"""Caracterização lexical e sintática do corpus.

Cada módulo compara as notícias falsas sintéticas com as escritas por humanos,
pareadas pelo ``uid`` da notícia de origem. O carregamento dos documentos e os
testes de significância são compartilhados, para que as tabelas dos vários
módulos sejam lidas do mesmo jeito.

Módulos de análise: :mod:`zipf` (distribuição de frequências),
:mod:`syllables` (silabação), :mod:`sage` (termos distintivos) e :mod:`liwc`
(perfil psicolinguístico).
"""

from expanded_fake_news_corpus.analysis.documents import (
    CorpusPaths,
    Document,
    Group,
    Source,
    load_paired_corpus,
)
from expanded_fake_news_corpus.analysis.significance import (
    ComparisonResult,
    add_fdr_correction,
    compare_frame,
    compare_groups,
    compare_within,
    comparisons_to_frame,
)

__all__ = [
    "ComparisonResult",
    "CorpusPaths",
    "Document",
    "Group",
    "Source",
    "add_fdr_correction",
    "compare_frame",
    "compare_groups",
    "compare_within",
    "comparisons_to_frame",
    "load_paired_corpus",
]
