#!/usr/bin/env python
"""Exporta as features linguísticas do trabalho anterior, uma linha por documento.

O detector de transferência com features linguísticas é treinado nas tabelas do
repositório ``noticias_falsas_humano_maquina_semantica``. Em vez de portar os
sete carregadores de lá, este script chama o código original e grava o resultado
com a chave ``uid`` deste projeto, que é o que ``classification`` lê. Assim o
filtro de comprimento (NILC tiers A e B, agregadores do LIWC, contagens brutas do
parser) e as 500 regras mais frequentes saem idênticos ao experimento original.

Roda com o Python **do repositório anterior**, que tem pandas e as dependências
dele::

    ~/Projects/noticias_falsas_humano_maquina_semantica/.venv/bin/python \\
        scripts/export_prior_features.py

Grava ``data/classification/prior_features.csv`` e, ao lado, o manifesto com o
commit do repositório anterior e as colunas de cada grupo.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR_REPO = PROJECT_ROOT.parent / "noticias_falsas_humano_maquina_semantica"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "classification" / "prior_features.csv"

#: Prefixo de coluna -> grupo de features, para o manifesto.
GROUP_PREFIXES = {
    "nilc__": "nilcmetrics",
    "liwc__": "liwc",
    "ud__": "enhanced_ud",
    "syll__": "syllables",
    "pos__": "pos_tagger",
    "psr__": "parser_stats",
    "sage__": "sage_terms",
}

#: O corpus de origem em cada repositório. O FakeTrueBR é 0-based lá e 1-based aqui.
SOURCE_BY_SUBSET = {"fake_br": ("fakebr", 0), "fake_true_br": ("faketruebr", 1)}

logger = logging.getLogger("export_prior_features")


def prior_uid(filename: str, subset: str) -> str:
    """Converte ``(1386.txt, fake_br)`` em ``fakebr:1386``."""
    source, offset = SOURCE_BY_SUBSET[subset]
    return f"{source}:{int(Path(filename).stem) + offset}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--prior-repo", type=Path, default=DEFAULT_PRIOR_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    module_dir = args.prior_repo / "linguistic_features"
    if not (module_dir / "linguistic_features.py").exists():
        logger.error(f"prior repository not found at {args.prior_repo}")
        return 1
    sys.path.insert(0, str(module_dir))
    from linguistic_features import LinguisticFeaturesPipeline  # noqa: PLC0415

    # missing_features='impute' faz o join externo: nenhum documento é perdido por
    # faltar em um grupo, e o transfer imputa com a mediana do treino.
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = LinguisticFeaturesPipeline(
            exp_dir=Path(tmp),
            feature_mode="linguistic",
            missing_features="impute",
        )
        frame = pipeline._load_linguistic_frame()
    if frame is None:
        logger.error("no feature group was loaded")
        return 1

    frame.insert(
        0,
        "uid",
        [prior_uid(f, s) for f, s in zip(frame["filename"], frame["subset_corpus"])],
    )
    frame = frame.drop(columns=["filename", "subset_corpus"])

    duplicated = frame.duplicated(["uid", "label"]).sum()
    if duplicated:
        logger.error(f"{duplicated} duplicated (uid, label) rows")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    groups = {
        name: [c for c in frame.columns if c.startswith(prefix)]
        for prefix, name in GROUP_PREFIXES.items()
    }
    commit = subprocess.run(
        ["git", "-C", str(args.prior_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = {
        "prior_repo_commit": commit,
        "rows": len(frame),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "groups": {name: columns for name, columns in groups.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, columns in groups.items():
        logger.info(f"{name}: {len(columns)} colunas")
    logger.info(f"Gravado {args.output} ({len(frame)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
