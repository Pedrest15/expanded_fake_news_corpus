"""Normalização de texto de entrada e limpeza da manchete gerada."""

from __future__ import annotations

import re

#: Aspas que LLMs costumam colocar em volta da manchete.
_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # “ ”
    ("‘", "’"),  # ‘ ’
    ("«", "»"),  # « »
]

#: Prefixos-rótulo que o modelo às vezes emite antes do texto da manchete.
_LABEL_RE = re.compile(
    r"^\s*(manchete|titulo|título|headline|title|chamada)\s*[:\-–]\s*",
    re.IGNORECASE,
)

_MARKDOWN_RE = re.compile(r"(\*\*|__|\*|_|`|^#{1,6}\s*)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_news_text(text: str, *, max_chars: int | None = 12_000) -> str:
    """Prepara o corpo da notícia para ir ao prompt.

    Remove espaçamento redundante e, opcionalmente, trunca o texto em um limite
    de caracteres para conter custo e estourar menos a janela de contexto. O
    corte é feito na última quebra de parágrafo ou espaço antes do limite, para
    não partir palavras no meio.

    Args:
        text: Texto bruto da notícia.
        max_chars: Limite de caracteres, ou ``None`` para não truncar.

    Returns:
        Texto normalizado.
    """
    cleaned = "\n".join(line.strip() for line in text.strip().splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if max_chars is None or len(cleaned) <= max_chars:
        return cleaned

    head = cleaned[:max_chars]
    cut = max(head.rfind("\n\n"), head.rfind("\n"), head.rfind(" "))
    if cut > max_chars // 2:
        head = head[:cut]
    return head.rstrip()


def clean_headline(raw: str) -> str:
    """Reduz a saída do modelo a uma manchete de uma linha.

    Tira rótulos ("Manchete:"), marcação markdown, aspas envolventes, espaços
    duplicados e ponto final — mantendo ``?`` e ``!``, que são legítimos em
    manchetes.

    Args:
        raw: Texto retornado pelo modelo.

    Returns:
        Manchete limpa (string vazia se não sobrar conteúdo).
    """
    if not raw:
        return ""

    # Fica só com a primeira linha não vazia: alternativas extras são descartadas.
    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    line = _MARKDOWN_RE.sub("", line)
    line = _LABEL_RE.sub("", line)
    line = _WHITESPACE_RE.sub(" ", line).strip()
    line = _strip_wrapping_quotes(line)
    line = line.rstrip(" .;,:…")
    return line.strip()


def _strip_wrapping_quotes(text: str) -> str:
    """Remove pares de aspas que envolvem o texto inteiro."""
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for opening, closing in _QUOTE_PAIRS:
            if text.startswith(opening) and text.endswith(closing):
                text = text[len(opening) : -len(closing)].strip()
                changed = True
                break
    return text


def word_count(text: str) -> int:
    """Conta palavras da manchete (usado nas checagens de qualidade)."""
    return len([token for token in _WHITESPACE_RE.split(text.strip()) if token])
