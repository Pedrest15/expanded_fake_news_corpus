"""Regras de dependências *enhanced*: o que o EUD acrescenta à árvore básica.

As Enhanced Universal Dependencies acrescentam à árvore básica arestas que
ela não representa — o sujeito compartilhado por verbos coordenados, o
antecedente de uma relativa, a preposição incorporada à relação
(``nmod:de``). O trabalho anterior sobre discriminação humano/máquina extraiu
regras só dessas arestas novas, e é isso que este módulo repete: para cada
token, as arestas da coluna DEPS que **diferem** de HEAD/DEPREL — outro
núcleo, ou a mesma relação especializada — formam uma segunda árvore, e as
regras saem dela no mesmo formato de :mod:`grammar_rules`::

    VERB(*, NOUN/nsubj)            sujeito propagado a um verbo coordenado
    NOUN(*, PROPN/nmod:de)         relação com a preposição incorporada

Regras-folha só são emitidas para tokens que participam de alguma aresta
nova, então uma sentença sem enriquecimento não gera regra nenhuma.

Lê os ``<stem>.eud.conllu`` de ``data/parsed/<experimento>/``
(:mod:`expanded_fake_news_corpus.parsing.eud`); tabelas, TF-IDF e testes são os de
:mod:`grammar_rules`, com o prefixo ``eud_rules``.

Uso::

    python -m expanded_fake_news_corpus.analysis.eud_rules \\
        --experiment paper_replication
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from expanded_fake_news_corpus.analysis.conllu import Sentence
from expanded_fake_news_corpus.analysis.documents import PROJECT_ROOT
from expanded_fake_news_corpus.analysis.grammar_rules import (
    FULL_FORMAT,
    RuleFormat,
    build_parser,
    run_from_args,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "eud_rules"


def enhanced_edges(sentence: Sentence) -> list[tuple[int, int, str]]:
    """Arestas de DEPS que a árvore básica não tem.

    Returns:
        Triplas ``(dependente, núcleo, relação)``; uma aresta é nova quando o
        núcleo é outro ou a relação difere (``nmod:de`` contra ``nmod``)
    """
    edges: list[tuple[int, int, str]] = []
    for word in sentence.words:
        for head, deprel in word.enhanced_deps():
            if head != word.head_index or deprel != word.deprel:
                edges.append((word.index, head, deprel))
    return edges


def enhanced_rules(sentence: Sentence, fmt: RuleFormat = FULL_FORMAT) -> list[str]:
    """Regras da árvore formada só pelas arestas *enhanced* novas.

    Args:
        sentence: Sentença com a coluna DEPS preenchida
        fmt: Formato das regras

    Returns:
        Uma regra por núcleo com dependentes novos, mais uma regra-folha por
        dependente envolvido; vazio se o EUD não acrescentou nada
    """
    words = {w.index: w for w in sentence.words}
    dependents: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for dependent, head, deprel in enhanced_edges(sentence):
        dependents[head].append((dependent, deprel))

    rules: list[str] = []
    for head_index, deps in sorted(dependents.items()):
        head = words.get(head_index)
        if head is None:  # raiz (0) ou nó vazio: sem UPOS para encabeçar a regra
            continue
        deps.sort()
        parts = (
            [fmt.dependent(words[d].upos, rel) for d, rel in deps if d < head_index]
            + ["*"]
            + [fmt.dependent(words[d].upos, rel) for d, rel in deps if d > head_index]
        )
        rules.append(f"{fmt.head(head.upos)}({', '.join(parts)})")

    for dependent, _head, _deprel in enhanced_edges(sentence):
        word = words[dependent]
        rules.append(
            f"*({fmt.head(word.upos)})"
            if word.head_index == 0
            else f"{fmt.head(word.upos)}(*)"
        )
    return rules


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser(__doc__.splitlines()[0], DEFAULT_OUTPUT_DIR).parse_args(argv)
    run_from_args(
        args,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        extractor=enhanced_rules,
        prefix="eud_rules",
        variant="eud",
    )


if __name__ == "__main__":
    main()
