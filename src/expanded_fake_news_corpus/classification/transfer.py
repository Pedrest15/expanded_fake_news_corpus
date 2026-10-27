"""Transferência: peças compartilhadas do detector treinado no corpus anterior.

Responde a pergunta que o nosso corpus sozinho não responde. Com 20 notícias
não há como treinar um classificador; mas há como **testar** um treinado em
outro lugar. O trabalho anterior tem 5.391 pares humano/sabiá-3, e um detector
ajustado neles pergunta:

    um detector de fake news de máquina treinado contra um gerador reconhece os
    textos de outro?

O arranjo é assimétrico de propósito:

* **Treino** — pares humano/sabiá-3 restritos aos ``uid`` de *treino* do
  trabalho anterior, com o truncamento pareado de sempre. O Fake.br humano é lido
  sem limpeza (``full_texts/fake``), como no repositório de semântica.
* **Teste, lado máquina** — os textos dos nossos geradores, que nenhum detector
  jamais viu. A métrica é a taxa de detecção (quanto o detector marca como
  máquina), por gerador.
* **Teste, lado humano** — as fake news humanas dos ``uid`` de *teste* do
  trabalho anterior, que o modelo também nunca viu. Dá a taxa de falso
  positivo, sem a qual a taxa de detecção não se interpreta.
* **Referência** — os textos sabiá-3 dos mesmos ``uid`` de teste. É a detecção
  dentro da distribuição de treino, o número contra o qual os nossos geradores
  são lidos.

O truncamento do teste é o do treino, par a par: o humano e o sabiá-3 de cada
``uid`` de teste são cortados no menor dos dois, e o texto de cada gerador é
cortado no menor entre ele e a *fake news* humana do mesmo ``uid`` no corpus
anterior. Assim o comprimento nunca separa as classes, e cada texto entra com
o tamanho que teria no experimento original.

O lado humano **não** sai do nosso recorte de 20 notícias de propósito: 16
delas estão no treino do trabalho anterior (ver
``resources/prior_splits/README.md``), e usá-las mediria memorização. O humano
dessas 20 entra só como medida de comprimento do par do gerador.

Este módulo guarda as peças; o treino e a linha de comando estão em
:mod:`linguistic_transfer`, que cobre todos os modos, inclusive o BoW/TF-IDF sem
features linguísticas (``--mode tfidf --no-linguistic``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from expanded_fake_news_corpus.analysis.documents import (
    EXPERIMENTS,
    HUMAN_COHORT,
    SABIA3_COHORT,
    Document,
    load_machine_documents,
    load_prior_cohort,
)
from expanded_fake_news_corpus.classification.pairing import (
    HUMAN_LABEL,
    MACHINE_LABEL,
    PairedSample,
    build_pairs,
    truncate_pair,
)

logger = logging.getLogger(__name__)

#: Geradores do teste, pela dupla ``(modelo, rodada)``: os que têm dados
#: publicados no site, ou seja, os que geraram texto sem problema. Os demais da
#: pasta de geração (Sonnet 5, GPT-5.1, DeepSeek R1, Tucano) ficaram sem análise.
#: Do Llama 3.1 8B vale a rodada 2: a primeira saiu com apêndice de
#: metacomentário e tem um documento degenerado, e o NILC-Metrix só mediu a segunda.
PUBLISHED_GENERATORS: frozenset[tuple[str, str | None]] = frozenset(
    {
        ("openai/gpt-4.1-mini-2025-04-14", None),
        ("anthropic/claude-sonnet-4-5-20250929", None),
        ("ollama/qwen3:32b", None),
        ("ollama/llama3.1:8b", "rodada2"),
    }
)


def is_published(document: Document) -> bool:
    """Se o gerador do documento é um dos que entram no teste."""
    return (document.model or "", document.round_name) in PUBLISHED_GENERATORS


#: Prefixo do conjunto de referência (Sabiá-3 dos ``uid`` de teste): é lido à
#: parte, porque é o gerador contra o qual o detector foi treinado e não entra na
#: média nem na amplitude dos geradores novos.
REFERENCE_PREFIX = "sabia-3"

#: Nomes dos conjuntos de teste que não são geradores novos.
HUMAN_SET_NAME = "humano (teste do trabalho anterior)"
REFERENCE_SET_NAME = f"{REFERENCE_PREFIX} (referência, dentro da distribuição)"


class TransferError(Exception):
    """O experimento de transferência não pôde ser montado."""


@dataclass(frozen=True)
class TestSet:
    """Um conjunto de teste homogêneo, com o rótulo que se espera dele."""

    name: str
    texts: list[str]
    expected_label: int
    #: ``uid`` de cada texto, na mesma ordem; serve para juntar outras features.
    uids: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.texts)


def load_training_pairs(
    train_uids: frozenset[str], *, prior_dir: Path | None
) -> list[PairedSample]:
    """Monta os pares humano/sabiá-3 do treino do trabalho anterior.

    Args:
        train_uids: ``uid`` da parte de treino
        prior_dir: Raiz do corpus anterior; detectada se omitida

    Returns:
        Pares com os dois lados presentes

    Raises:
        TransferError: Se nenhuma das duas coortes carregar
    """
    human = load_prior_cohort(
        HUMAN_COHORT, prior_dir=prior_dir, uids=train_uids, raw_fakebr=True
    )
    machine = load_prior_cohort(SABIA3_COHORT, prior_dir=prior_dir, uids=train_uids)
    if not human or not machine:
        raise TransferError(
            "prior corpus not found; set FAKEGEN_PRIOR_CORPUS or pass --prior-dir"
        )
    # O carregador marca a coorte sintética como MACHINE, que é o que build_pairs
    # espera; o lado humano já vem como HUMAN.
    pairs = build_pairs([*human, *machine])
    logger.info(f"Treino: {len(pairs)} pares do trabalho anterior")
    return pairs


def build_test_sets(
    test_uids: frozenset[str],
    *,
    tokenizer,
    max_length: int,
    prior_dir: Path | None,
    experiment: str,
    models: Sequence[str] | None,
) -> list[TestSet]:
    """Monta os conjuntos de teste: controle humano, referência e geradores.

    Todo texto é truncado como no treino (:func:`truncate_pair`): o humano e o
    sabiá-3 de teste entre si, e cada gerador contra a *fake news* humana do
    mesmo ``uid`` no corpus anterior.

    Args:
        test_uids: ``uid`` da parte de teste do trabalho anterior
        tokenizer: Tokenizador do encoder
        max_length: Teto de tokens do encoder (o do truncamento pareado)
        prior_dir: Raiz do corpus anterior
        experiment: Experimento de geração deste repositório
        models: Restringe aos geradores pedidos; todos se omitido

    Returns:
        Um conjunto por coorte avaliada
    """
    sets: list[TestSet] = []

    human = {
        document.uid: document
        for document in load_prior_cohort(
            HUMAN_COHORT, prior_dir=prior_dir, uids=test_uids, raw_fakebr=True
        )
    }
    sabia = {
        document.uid: document
        for document in load_prior_cohort(
            SABIA3_COHORT, prior_dir=prior_dir, uids=test_uids
        )
    }
    paired_uids = sorted(human.keys() & sabia.keys())
    human_texts: list[str] = []
    sabia_texts: list[str] = []
    for uid in paired_uids:
        cut_human, cut_sabia, _ = truncate_pair(
            human[uid].text, sabia[uid].text, tokenizer, max_length=max_length
        )
        human_texts.append(cut_human)
        sabia_texts.append(cut_sabia)
    sets.append(TestSet(HUMAN_SET_NAME, human_texts, HUMAN_LABEL, tuple(paired_uids)))
    sets.append(
        TestSet(REFERENCE_SET_NAME, sabia_texts, MACHINE_LABEL, tuple(paired_uids))
    )

    ours = [
        document
        for document in load_machine_documents(EXPERIMENTS[experiment], models=models)
        if is_published(document)
    ]
    counterparts = {
        document.uid: document
        for document in load_prior_cohort(
            HUMAN_COHORT,
            prior_dir=prior_dir,
            uids={document.uid for document in ours},
            raw_fakebr=True,
        )
    }
    by_generator: dict[str, list[Document]] = {}
    for document in ours:
        key = document.model or "<sem modelo>"
        if document.round_name:
            key = f"{key} ({document.round_name})"
        by_generator.setdefault(key, []).append(document)

    for name, documents in sorted(by_generator.items()):
        texts: list[str] = []
        uids: list[str] = []
        for document in documents:
            partner = counterparts.get(document.uid)
            if partner is None:
                logger.warning(
                    f"{name}: sem a fake news humana de {document.uid} no corpus "
                    "anterior; texto ignorado"
                )
                continue
            _, cut, _ = truncate_pair(
                partner.text, document.text, tokenizer, max_length=max_length
            )
            texts.append(cut)
            uids.append(document.uid)
        sets.append(TestSet(name, texts, MACHINE_LABEL, tuple(uids)))

    for test_set in sets:
        logger.info(f"Teste '{test_set.name}': {test_set.size} textos")
    return sets


def build_readings(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Separa as duas leituras do mesmo conjunto de predições.

    A primeira olha só os geradores: é a taxa de detecção, e é o que compara os
    geradores entre si — o comportamento do detector no texto humano é
    constante e não entra nessa comparação. A segunda acrescenta o controle
    humano, que não é objeto do teste mas fixa o ponto de operação: sem ele a
    taxa de detecção de um gerador isolado não tem escala.

    Args:
        evaluation: Saída de ``evaluate`` (em :mod:`linguistic_transfer`)

    Returns:
        As duas leituras, cada uma com o que lhe cabe
    """
    machine = {
        name: result
        for name, result in evaluation.items()
        if result["expected"] == "machine"
    }
    generators = {
        name: result
        for name, result in machine.items()
        if not name.startswith(REFERENCE_PREFIX)
    }
    reference = {
        name: result["flagged_as_machine"]
        for name, result in machine.items()
        if name.startswith(REFERENCE_PREFIX)
    }
    human = {
        name: result
        for name, result in evaluation.items()
        if result["expected"] == "human"
    }

    rates = {name: result["flagged_as_machine"] for name, result in generators.items()}
    llms_only: dict[str, Any] = {
        "detection_rate": rates,
        "reference_rate": reference,
        "mean": float(np.mean(list(rates.values()))) if rates else None,
        "spread": (max(rates.values()) - min(rates.values())) if rates else None,
    }

    with_human: dict[str, Any] = {"false_positive_rate": None, "balanced": {}}
    if human:
        false_positive = float(
            np.mean([result["flagged_as_machine"] for result in human.values()])
        )
        with_human["false_positive_rate"] = false_positive
        with_human["human_texts"] = sum(result["texts"] for result in human.values())
        # Acurácia balanceada do par (gerador, controle humano): tira do número
        # o detector que marcaria tudo como máquina.
        with_human["balanced"] = {
            name: (rate + (1.0 - false_positive)) / 2
            for name, rate in {**rates, **reference}.items()
        }

    return {"llms_only": llms_only, "with_human_control": with_human}


def report(payload: dict[str, Any]) -> None:
    """Imprime o quadro de transferência."""
    logger.info("-" * 72)
    logger.info(
        f"detector: {payload['training']['classifier']} | modo {payload['mode']} | "
        f"{payload['training']['pairs']} pares de treino"
    )
    readings = payload["readings"]

    logger.info("leitura 1 — só os LLMs (taxa de detecção)")
    logger.info(f"  {'gerador':<46} {'n':>5} {'detectado':>11}")
    for name, rate in readings["llms_only"]["detection_rate"].items():
        texts = payload["evaluation"][name]["texts"]
        logger.info(f"  {name:<46} {texts:>5} {rate:>10.1%}")
    for name, rate in readings["llms_only"].get("reference_rate", {}).items():
        texts = payload["evaluation"][name]["texts"]
        logger.info(f"  {name:<46} {texts:>5} {rate:>10.1%}  (referência)")
    spread = readings["llms_only"]["spread"]
    logger.info(
        f"  média {readings['llms_only']['mean']:.1%}, amplitude "
        f"{spread:.1%} entre o melhor e o pior gerador"
    )

    human = readings["with_human_control"]
    if human["false_positive_rate"] is None:
        return
    logger.info("")
    logger.info(
        f"leitura 2 — com o controle humano ({human['human_texts']} textos): "
        f"falso positivo {human['false_positive_rate']:.1%}"
    )
    logger.info(f"  {'gerador':<46} {'acurácia balanceada':>20}")
    for name, value in human["balanced"].items():
        logger.info(f"  {name:<46} {value:>19.1%}")
