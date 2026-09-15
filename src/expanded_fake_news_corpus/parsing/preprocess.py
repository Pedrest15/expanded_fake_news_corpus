"""Tratamento do texto antes do sentenciador, e realinhamento depois do tokenizador.

O portSentencer e o portTokenizer foram feitos para texto jornalístico bem
formado. O corpus tem dois desvios que eles não absorvem sozinhos:

* a manchete é a primeira linha e não termina em ponto; como o sentenciador
  ignora quebras de linha, ela seria emendada à primeira frase do corpo;
* o FakeTrueBR humano é distribuído todo em minúsculas, e o sentenciador só
  fecha sentença quando a palavra seguinte é capitalizada — sairia o parágrafo
  inteiro como uma sentença.

:func:`split_blocks` resolve os dois antes de chamar as ferramentas;
:func:`align_sentence_counts` recupera a fronteira entre documentos depois
que o tokenizador descarta ou edita sentenças.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from nltk.tokenize import sent_tokenize

#: Comprimento mínimo do prefixo comparado quando o tokenizador editou a
#: sentença (aspas sem par removidas por ``-m``, por exemplo).
_PREFIX_MATCH_LENGTH = 30

_SENTENCE_END_RE = re.compile("[.!?…][\"'”’)\\]]*$")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zà-ÿ]+")


class AlignmentError(ValueError):
    """A saída do tokenizador não alinha com a entrada do sentenciador."""


def is_lowercased(text: str) -> bool:
    """Indica texto distribuído sem nenhuma maiúscula, como o FakeTrueBR."""
    return not any(char.isupper() for char in text)


def normalize_for_matching(text: str) -> str:
    """Reduz o texto a letras e dígitos em caixa baixa, para comparação."""
    return _NON_ALNUM_RE.sub("", text.casefold())


def split_blocks(text: str) -> list[str]:
    """Divide o texto em blocos que o sentenciador não pode emendar.

    Cada linha é fronteira dura, exceto quando a linha não termina em
    pontuação e a seguinte começa em minúscula: isso é quebra no meio da
    frase (o Fake.br traz algumas), e as duas são coladas — a heurística do
    ``adapt_fake.py`` do trabalho anterior.

    Num texto sem maiúsculas essa heurística não vale (todo início de linha é
    minúsculo) e o sentenciador tampouco funciona; cada linha é então
    pré-segmentada pelo punkt, que quebra em pontuação final sem olhar a
    caixa. O trabalho anterior fez o mesmo para o FakeTrueBR, mas com a
    manchete concatenada ao corpo antes de segmentar; aqui ela fica como
    sentença própria, como no Fake.br.

    Args:
        text: Texto integral do documento

    Returns:
        Blocos com pelo menos uma letra, na ordem do texto
    """
    lowercased = is_lowercased(text)
    blocks: list[str] = []
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        if lowercased:
            blocks.extend(sent_tokenize(line, language="portuguese"))
        elif _is_broken_line_continuation(blocks, line):
            blocks[-1] = f"{blocks[-1]} {line}"
        else:
            blocks.append(line)
    # Bloco sem letra (só pontuação) derruba o sentenciador e seria
    # descartado pelo tokenizador de qualquer forma.
    return [block for block in blocks if any(char.isalpha() for char in block)]


def _is_broken_line_continuation(blocks: Sequence[str], line: str) -> bool:
    """Linha que continua a anterior, partida no meio da frase."""
    return (
        bool(blocks) and not _SENTENCE_END_RE.search(blocks[-1]) and line[0].islower()
    )


def align_sentence_counts(
    per_document: Sequence[Sequence[str]], tokenized_texts: Sequence[str]
) -> list[int]:
    """Conta quantas sentenças de cada documento sobreviveram ao tokenizador.

    O portTokenizer descarta sentenças sem letras e, com ``-m``, apaga
    pontuação sem par — nem a contagem nem o texto batem com a entrada. O
    alinhamento compara os textos sem pontuação, em ordem; o que o
    tokenizador não devolveu conta como descartado.

    Args:
        per_document: Sentenças de cada documento, na ordem enviada
        tokenized_texts: ``# text`` de cada sentença devolvida

    Returns:
        Sentenças tokenizadas por documento, na mesma ordem

    Raises:
        AlignmentError: Se sobrar sentença tokenizada sem origem, ou se um
            documento inteiro tiver sido descartado
    """
    counts: list[int] = []
    cursor = 0
    for sentences in per_document:
        count = 0
        for sentence in sentences:
            if cursor < len(tokenized_texts) and _matches(
                sentence, tokenized_texts[cursor]
            ):
                cursor += 1
                count += 1
        counts.append(count)

    if cursor != len(tokenized_texts):
        raise AlignmentError(
            f"{len(tokenized_texts) - cursor} tokenized sentences do not align "
            "with the sentencer output"
        )
    empty = [index for index, count in enumerate(counts) if count == 0]
    if empty:
        raise AlignmentError(f"documents without any tokenized sentence: {empty}")
    return counts


def _matches(sentence: str, tokenized: str) -> bool:
    """Mesma sentença, tolerando a pontuação que o tokenizador removeu."""
    wanted, got = normalize_for_matching(sentence), normalize_for_matching(tokenized)
    if wanted == got:
        return True
    return len(wanted) >= _PREFIX_MATCH_LENGTH and got.startswith(
        wanted[:_PREFIX_MATCH_LENGTH]
    )
