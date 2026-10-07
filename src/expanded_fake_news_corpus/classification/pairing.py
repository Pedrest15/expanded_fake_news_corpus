"""Pares humano/máquina prontos para classificação.

A tarefa é decidir, para um documento, se a fake news foi escrita por pessoa ou
por modelo. O corpus é pareado por notícia de origem: cada ``uid`` tem uma fake
humana e uma sintética sobre a mesma história.

Duas disciplinas vêm do trabalho de classificação anterior
(``noticias_falsas_humano_maquina_semantica/linguistic_features``) e são o que
impede o experimento de medir a coisa errada:

* **Truncamento pareado.** Os dois lados são cortados no menor número de tokens
  do par. Sem isso o classificador aprende "texto longo = máquina" — e o
  comprimento é justamente a diferença mais grosseira entre os grupos (a
  máquina escreve ~2,2× mais que a pessoa neste corpus).
* **Agrupamento por ``uid``.** Os dois lados de uma notícia nunca caem em
  dobras diferentes da validação cruzada. Sem isso o modelo vê a mesma história
  no treino e no teste e a métrica infla.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from expanded_fake_news_corpus.analysis.documents import Document, Group

logger = logging.getLogger(__name__)

#: Rótulos da classificação binária.
HUMAN_LABEL = 0
MACHINE_LABEL = 1


@dataclass(frozen=True)
class PairedSample:
    """Uma notícia de origem com os dois textos que a disputam."""

    uid: str
    source: str
    human_text: str
    machine_text: str
    model: str | None = None
    round_name: str | None = None

    @property
    def group(self) -> str:
        """Chave de agrupamento da validação cruzada."""
        return self.uid


@dataclass(frozen=True)
class LabelledText:
    """Um documento achatado, do jeito que o classificador consome."""

    uid: str
    source: str
    text: str
    label: int

    @property
    def group(self) -> str:
        """Chave de agrupamento: os dois lados do par compartilham o grupo."""
        return self.uid


class PairingError(Exception):
    """O corpus não pôde ser pareado."""


def build_pairs(documents: Iterable[Document]) -> list[PairedSample]:
    """Monta os pares humano/máquina a partir dos documentos carregados.

    Args:
        documents: Saída de ``load_paired_corpus``, com os dois grupos

    Returns:
        Um par por ``uid`` presente nos dois lados, ordenado por ``uid``

    Raises:
        PairingError: Se nenhum par se formar
    """
    human: dict[str, Document] = {}
    machine: dict[str, Document] = {}
    for document in documents:
        target = human if document.group is Group.HUMAN else machine
        if document.uid in target:
            logger.warning(
                f"Documento repetido em {document.group.value} para "
                f"{document.uid}; mantendo o primeiro"
            )
            continue
        target[document.uid] = document

    pairs = [
        PairedSample(
            uid=uid,
            source=machine[uid].source.value,
            human_text=human[uid].text,
            machine_text=machine[uid].text,
            model=machine[uid].model,
            round_name=machine[uid].round_name,
        )
        for uid in sorted(human.keys() & machine.keys())
    ]

    orphans = (human.keys() | machine.keys()) - (human.keys() & machine.keys())
    if orphans:
        logger.warning(
            f"{len(orphans)} uid(s) sem os dois lados, descartados: "
            f"{', '.join(sorted(orphans)[:5])}"
        )
    if not pairs:
        raise PairingError("no uid has both a human and a machine document")

    logger.info(f"{len(pairs)} pares humano/máquina")
    return pairs


def truncate_pair(
    human_text: str,
    machine_text: str,
    tokenizer,
    *,
    max_length: int,
) -> tuple[str, str, int]:
    """Corta os dois textos no menor número de tokens do par.

    O teto efetivo é ``max_length - 2``, deixando espaço para os tokens
    especiais que o tokenizador acrescenta na codificação final.

    Args:
        human_text: Fake news escrita por pessoa
        machine_text: Fake news gerada por modelo
        tokenizer: Tokenizador do encoder, já carregado
        max_length: Teto de tokens do encoder

    Returns:
        Tupla ``(humano truncado, máquina truncada, tokens mantidos)``
    """
    human_tokens = tokenizer.encode(human_text, add_special_tokens=False)
    machine_tokens = tokenizer.encode(machine_text, add_special_tokens=False)
    kept = min(len(human_tokens), len(machine_tokens), max_length - 2)
    return (
        tokenizer.decode(human_tokens[:kept], skip_special_tokens=True),
        tokenizer.decode(machine_tokens[:kept], skip_special_tokens=True),
        kept,
    )


def flatten_pairs(pairs: Sequence[PairedSample]) -> list[LabelledText]:
    """Achata os pares em documentos rotulados, humano antes de máquina.

    Args:
        pairs: Pares, já truncados se for o caso

    Returns:
        Dois documentos por par, na ordem ``(humano, máquina)``
    """
    flattened: list[LabelledText] = []
    for pair in pairs:
        flattened.append(
            LabelledText(pair.uid, pair.source, pair.human_text, HUMAN_LABEL)
        )
        flattened.append(
            LabelledText(pair.uid, pair.source, pair.machine_text, MACHINE_LABEL)
        )
    return flattened
