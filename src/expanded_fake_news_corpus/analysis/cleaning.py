"""Limpezas aplicadas antes da análise, sem tocar no corpus gravado.

Três, independentes: as regras do Fake.br humano (``adapt_fake.py`` do
trabalho anterior), e do lado máquina a remoção de marcação Markdown e do
apêndice em que o modelo comenta as próprias alterações (:func:`strip_appendix`).

Os ``.txt`` do Fake.br trazem lixo de coleta: caracteres fora do teclado
(emoji, espaços não separáveis, bytes de codificação errada), linhas
partidas no meio da frase e espaço antes de pontuação. O trabalho anterior
(Andrade et al., PROPOR 2026) limpou isso antes de analisar — 2.592 dos 3.600
arquivos mudaram — e a comparação com ele pede o mesmo tratamento aqui.

As quatro regras são as de lá, na mesma ordem: lista branca de caracteres,
colagem de linha partida, espaços duplos, espaço antes de pontuação. Duas
diferenças são deliberadas:

* o script original, ao colar uma linha à seguinte, descartava a primeira em
  vez de juntá-la (``continue`` antes de gravar); aqui as duas são juntadas;
* espaços fora da lista branca (o não separável, sobretudo) viram espaço
  comum em vez de sumir — apagá-los colava as palavras vizinhas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Caracteres aceitos — o "teclado padrão" do ``adapt_fake.py``, verbatim.
VALID_CHARACTERS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "àáâãäåçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ"
    " \n\r\t"
    ".,;:!?¡¿()[]{}\"'“”‘’@#$%&*-+=_/|<>^~\\"
)

_INVALID_CHARACTER_RE = re.compile(f"[^{re.escape(VALID_CHARACTERS)}]")
_SENTENCE_END_RE = re.compile(r"[.!?…]$")
_DOUBLE_SPACE_RE = re.compile(r"[ ]{2,}")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,;:!?…])")

# --------------------------------------------------------------------------
# Markdown do lado máquina
# --------------------------------------------------------------------------
#
# Alguns geradores devolvem a notícia formatada — manchete em ``**negrito**``,
# listas, linhas de ``---``. Nenhum dos modelos de API fez isso; apareceu com
# os modelos locais (Qwen3 e DeepSeek-R1). A marcação não é texto de notícia:
# ``**`` seria contado como pontuação na distribuição de UPOS, entraria no
# Zipf e no SAGE como token e inflaria a contagem de PUNCT de um lado só da
# comparação. Os marcadores saem; o texto que eles envolvem fica.

#: ``**negrito**`` e ``__negrito__``, incluindo quando atravessam a linha.
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)

#: ``*itálico*`` e ``_itálico_`` — o marcador precisa colar no texto, para não
#: comer um asterisco solto que faça parte da notícia.
_ITALIC_RE = re.compile(r"(?<![\w*_])([*_])(?=\S)([^*_\n]+?)(?<=\S)\1(?![\w*_])")

#: ``# Título`` no começo da linha.
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)

#: Marcador de item de lista no começo da linha.
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)

#: Linha só de ``---``, ``***`` ou ``___``.
_RULE_LINE_RE = re.compile(r"^[ \t]*([-*_])\1{2,}[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownReport:
    """Quanta marcação foi removida de um texto."""

    bold: int = 0
    italic: int = 0
    headings: int = 0
    bullets: int = 0
    rules: int = 0

    @property
    def total(self) -> int:
        return self.bold + self.italic + self.headings + self.bullets + self.rules

    @property
    def changed(self) -> bool:
        return bool(self.total)


def strip_markdown(text: str) -> tuple[str, MarkdownReport]:
    """Remove marcação Markdown, preservando o texto que ela envolve.

    Args:
        text: Texto da notícia sintética como o modelo devolveu

    Returns:
        O texto sem os marcadores e o relatório do que saiu
    """
    text, rules = _RULE_LINE_RE.subn("", text)
    text, headings = _HEADING_RE.subn("", text)
    text, bullets = _BULLET_RE.subn("", text)
    text, bold = _BOLD_RE.subn(r"\2", text)
    text, italic = _ITALIC_RE.subn(r"\2", text)

    report = MarkdownReport(
        bold=bold, italic=italic, headings=headings, bullets=bullets, rules=rules
    )
    if report.changed:
        text = _DOUBLE_SPACE_RE.sub(" ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, report


#: Frases que abrem o apêndice em que o modelo, depois da notícia, explica o que
#: alterou. O prompt pede as mudanças dentro de ``<changes>``; quando o modelo
#: esquece a tag, elas chegam coladas no corpo. Lista explícita, levantada
#: lendo o corpus (Llama 3.1 8B, ``faketruebr:509`` nas duas rodadas): um
#: padrão genérico pegaria "fake news" e "alterações" ditos dentro da notícia.
APPENDIX_OPENINGS: tuple[str, ...] = (
    "Agora, veja o que mudamos",
    "A notícia apresentada anteriormente foi modificada",
)

_APPENDIX_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(re.escape(opening) for opening in APPENDIX_OPENINGS) + ")",
    re.MULTILINE | re.IGNORECASE,
)


def strip_appendix(text: str) -> tuple[str, int]:
    """Corta o apêndice de metacomentário, da linha que o abre até o fim.

    Args:
        text: Texto da notícia sintética como o modelo devolveu

    Returns:
        O texto sem o apêndice e quantas palavras saíram (0 se não havia)
    """
    match = _APPENDIX_RE.search(text)
    if match is None:
        return text, 0
    return text[: match.start()].rstrip(), len(text[match.start() :].split())


@dataclass(frozen=True)
class CleaningReport:
    """O que a limpeza alterou num texto."""

    removed_characters: int = 0
    joined_lines: int = 0
    normalized_spacing: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.removed_characters or self.joined_lines or self.normalized_spacing
        )


def clean_fakebr_text(text: str) -> tuple[str, CleaningReport]:
    """Aplica ao texto as regras de limpeza do Fake.br.

    Args:
        text: Texto como está no ``.txt`` do corpus

    Returns:
        O texto limpo e o que mudou
    """
    cleaned, removed = _INVALID_CHARACTER_RE.subn(_replacement, text)
    cleaned, joined = _join_broken_lines(cleaned)
    spaced = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", _DOUBLE_SPACE_RE.sub(" ", cleaned))
    return spaced, CleaningReport(
        removed_characters=removed,
        joined_lines=joined,
        normalized_spacing=spaced != cleaned,
    )


def _replacement(match: re.Match[str]) -> str:
    """Espaço exótico vira espaço; qualquer outro caractere inválido some."""
    return " " if match.group().isspace() else ""


def _join_broken_lines(text: str) -> tuple[str, int]:
    """Cola à anterior a linha que continua uma frase partida.

    Uma linha sem pontuação final seguida de outra que começa em minúscula é
    quebra de coleta, não parágrafo.
    """
    lines = text.splitlines()
    joined: list[str] = []
    count = 0
    for line in lines:
        current = line.rstrip()
        continuation = current.lstrip()
        if (
            joined
            and joined[-1]
            and not _SENTENCE_END_RE.search(joined[-1])
            and continuation[:1].islower()
        ):
            joined[-1] = f"{joined[-1]} {continuation}"
            count += 1
        else:
            joined.append(current)
    return "\n".join(joined) + ("\n" if lines else ""), count
