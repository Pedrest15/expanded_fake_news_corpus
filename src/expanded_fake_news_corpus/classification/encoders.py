"""Embeddings de encoder congelado, um vetor por documento.

Os encoders e a política de *pooling* são os do trabalho anterior
(``linguistic_features/linguistic_features.py``), incluindo o NorBERTo. O
``torch`` e o ``transformers`` são importados tarde, dentro da classe: quem só
roda os experimentos de TF-IDF não precisa de 2,5 GB de dependência.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Encoders disponíveis, pelo apelido que a linha de comando aceita.
BERT_MODELS: dict[str, str] = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "bert-multilingual": "bert-base-multilingual-cased",
    "norberto": "Itau-Unibanco/NorBERTo-base",
    "norberto-large": "Itau-Unibanco/NorBERTo-large",
}

DEFAULT_BERT_MODEL = "bertimbau"

#: Como transformar a saída do encoder em um vetor por documento.
#:
#: Os modelos da família BERT foram pré-treinados com *next sentence
#: prediction*, então o vetor do ``[CLS]`` é uma representação de sentença e
#: continua o padrão deles. O NorBERTo segue a receita do ModernBERT (só
#: *masked LM*, sem NSP) e a própria configuração dele declara
#: ``classifier_pooling: "mean"`` — não existe ``[CLS]`` treinado para usar.
DEFAULT_POOLING_BY_MODEL: dict[str, str] = {
    "bertimbau": "cls",
    "bert-multilingual": "cls",
    "norberto": "mean",
    "norberto-large": "mean",
}

POOLING_STRATEGIES = ("cls", "mean")

#: Teto de tokens do truncamento pareado. 512 é o limite do BERT; o NorBERTo
#: aceita até 8192, e aí vale medir com contexto maior.
DEFAULT_MAX_LENGTH = 512

DEFAULT_BATCH_SIZE = 16


class EncoderError(Exception):
    """Encoder desconhecido ou *pooling* inválido."""


def resolve_model_name(alias: str) -> str:
    """Traduz o apelido do encoder para o identificador no Hugging Face.

    Args:
        alias: Apelido de :data:`BERT_MODELS`, ou um id completo do Hub

    Returns:
        Identificador aceito pelo ``from_pretrained``
    """
    if alias in BERT_MODELS:
        return BERT_MODELS[alias]
    # Um id do Hub tem a forma ``organização/modelo``; aceitamos direto para
    # não travar um encoder novo atrás de uma entrada na tabela.
    if "/" in alias:
        return alias
    raise EncoderError(
        f"unknown encoder {alias!r}; expected one of {sorted(BERT_MODELS)} "
        "or a full Hugging Face id"
    )


def resolve_pooling(alias: str, pooling: str = "auto") -> str:
    """Resolve ``--pooling auto`` para o padrão do encoder escolhido.

    Args:
        alias: Apelido do encoder
        pooling: ``auto``, ``cls`` ou ``mean``

    Returns:
        A estratégia a aplicar

    Raises:
        EncoderError: Se a estratégia não existir
    """
    if pooling == "auto":
        return DEFAULT_POOLING_BY_MODEL.get(alias, "cls")
    if pooling not in POOLING_STRATEGIES:
        raise EncoderError(
            f"unknown pooling {pooling!r}; expected one of {POOLING_STRATEGIES}"
        )
    return pooling


def cache_key(
    model_name: str, pooling: str, max_length: int, texts: Sequence[str]
) -> str:
    """Chave de cache que muda com qualquer coisa que mude os vetores."""
    digest = hashlib.sha256()
    digest.update(f"{model_name}|{pooling}|{max_length}".encode())
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:24]


class EmbeddingExtractor:
    """Extrai embeddings com os pesos do encoder congelados.

    Responsabilidades:
    - Carregar encoder e tokenizador uma vez
    - Produzir um vetor por documento, com o *pooling* pedido
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """Carrega o encoder.

        Args:
            model_name: Identificador do Hugging Face
            device: ``cuda``, ``cpu``; detectado se omitido
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Carregando encoder {model_name} em {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def _pool(self, hidden: Any, attention_mask: Any, pooling: str) -> Any:
        """Reduz a saída do encoder a um vetor por documento."""
        if pooling == "mean":
            # A média é só sobre os tokens reais: o *padding* não pode diluí-la.
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden[:, 0, :]

    def extract(
        self,
        texts: Sequence[str],
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pooling: str = "cls",
    ) -> np.ndarray:
        """Codifica os textos na ordem em que vieram.

        Args:
            texts: Documentos, já truncados pelo par
            max_length: Teto de tokens
            batch_size: Documentos por lote
            pooling: ``cls`` ou ``mean``

        Returns:
            Matriz ``(len(texts), dimensão do encoder)``

        Raises:
            EncoderError: Se o *pooling* não existir
        """
        if pooling not in POOLING_STRATEGIES:
            raise EncoderError(f"unknown pooling {pooling!r}")

        torch = self._torch
        batches: list[np.ndarray] = []
        total = (len(texts) + batch_size - 1) // batch_size
        for index in range(0, len(texts), batch_size):
            chunk = list(texts[index : index + batch_size])
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
                vectors = self._pool(hidden, inputs["attention_mask"], pooling)
            batches.append(vectors.cpu().numpy())
            number = index // batch_size + 1
            if number % 10 == 0 or number == total:
                logger.info(f"Lote {number}/{total}")

        return np.vstack(batches)


def load_or_extract(
    texts: Sequence[str],
    *,
    model_name: str,
    pooling: str,
    max_length: int,
    cache_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Devolve os embeddings do cache, ou os calcula e guarda.

    Args:
        texts: Documentos na ordem que o classificador espera
        model_name: Identificador do Hugging Face
        pooling: ``cls`` ou ``mean``
        max_length: Teto de tokens
        cache_dir: Pasta dos ``.npy``
        batch_size: Documentos por lote

    Returns:
        Matriz de embeddings
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(model_name, pooling, max_length, texts)}.npy"
    if path.is_file():
        logger.info(f"Embeddings do cache: {path.name}")
        return np.load(path)

    extractor = EmbeddingExtractor(model_name)
    matrix = extractor.extract(
        texts, max_length=max_length, batch_size=batch_size, pooling=pooling
    )
    np.save(path, matrix)
    logger.info(f"Embeddings gravados em {path.name} {matrix.shape}")
    return matrix
