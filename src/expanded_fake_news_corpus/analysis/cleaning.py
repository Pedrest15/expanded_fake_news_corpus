"""Limpeza do Fake.br humano, como no trabalho anterior (``adapt_fake.py``).

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
