"""Instala em ``tools/`` a cadeia de parsing usada por :mod:`portparser`.

Clona portSentencer, portTokenizer e Portparser.v2 (LuceleneL) e o LatinPipe
(ÚFAL), cria o venv do LatinPipe (Python 3.11, torch de CPU — o projeto
principal é 3.13) e baixa o modelo Portparser v2 (1,6 GB) do Google Drive
indicado no README do Portparser.v2. ``tools/`` fica fora do git.

Idempotente: pula o que já existe. Requer ``git`` e ``uv`` no PATH.

Uso::

    python -m expanded_fake_news_corpus.parsing.install_tools
    python -m expanded_fake_news_corpus.parsing.install_tools --tools-dir /outro/lugar
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

from expanded_fake_news_corpus.parsing.tools import (
    DEFAULT_TOOLS_DIR,
    TOOL_REPOSITORIES,
    ToolError,
    ToolPaths,
    check_tools,
)

logger = logging.getLogger(__name__)

#: Portparser_v2_model.zip, conforme o README do Portparser.v2.
MODEL_DRIVE_ID = "1fus6fz3XTUZIVXM58T-ygemCbuFzQnqf"
MODEL_FILES = ("model.weights.h5", "options.json", "mappings.pkl")

LATINPIPE_PYTHON = "3.11"
#: O LatinPipe declara torch com índice CUDA; aqui só CPU, e o gdown é para o
#: download do modelo.
LATINPIPE_PACKAGES = ("keras", "transformers", "ufal.chu-liu-edmonds", "gdown")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def install(paths: ToolPaths) -> None:
    """Instala o que faltar e confere a cadeia ao final.

    Raises:
        ToolError: Se ``git``/``uv`` faltarem, se um passo falhar ou se ao
            final ainda faltar componente
    """
    _require_executables(("git", "uv"))
    paths.root.mkdir(parents=True, exist_ok=True)
    for name, url in TOOL_REPOSITORIES.items():
        clone_repository(url, paths.root / name)
    copy_verb_lexicon(paths)
    create_latinpipe_venv(paths)
    download_model(paths)
    check_tools(paths)
    logger.info(f"Cadeia de parsing pronta em {paths.root}")


def clone_repository(url: str, target: Path) -> None:
    """Clona raso um repositório, se ainda não estiver lá."""
    if (target / ".git").is_dir():
        logger.info(f"[ok] {target.name}")
        return
    _run(["git", "clone", "-q", "--depth", "1", url, str(target)])
    logger.info(f"[clonado] {target.name}")


def copy_verb_lexicon(paths: ToolPaths) -> None:
    """Leva o léxico de verbos ao pós-processador.

    O repositório do Portparser.v2 não traz ``VERB.tsv`` (71 MB); o
    portTokenizer distribui o mesmo arquivo.
    """
    target = paths.postproc_dir / "VERB.tsv"
    if not target.exists():
        shutil.copy(paths.tokenizer_dir / "VERB.tsv", target)
        logger.info("[copiado] postproc/VERB.tsv")


def create_latinpipe_venv(paths: ToolPaths) -> None:
    """Cria o venv do LatinPipe com torch de CPU, se ainda não existir."""
    if paths.latinpipe_python.exists():
        logger.info("[ok] latinpipe/.venv")
        return
    venv = paths.latinpipe_dir / ".venv"
    _run(["uv", "venv", "--python", LATINPIPE_PYTHON, str(venv), "-q"])
    python = str(paths.latinpipe_python)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "-q",
            "torch",
            "--index-url",
            TORCH_CPU_INDEX,
        ]
    )
    _run(["uv", "pip", "install", "--python", python, "-q", *LATINPIPE_PACKAGES])
    logger.info("[criado] latinpipe/.venv")


def download_model(paths: ToolPaths) -> None:
    """Baixa o modelo do Google Drive e extrai só os arquivos necessários."""
    if (paths.model_dir / "model.weights.h5").exists():
        logger.info("[ok] Portparser_v2_model")
        return
    archive = paths.root / "Portparser_v2_model.zip"
    gdown = paths.latinpipe_dir / ".venv" / "bin" / "gdown"
    logger.info("Baixando Portparser_v2_model.zip (1,4 GB)...")
    _run(
        [
            str(gdown),
            f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}",
            "-O",
            str(archive),
        ]
    )
    with zipfile.ZipFile(archive) as zipped:
        for name in MODEL_FILES:
            zipped.extract(f"Portparser_v2_model/{name}", paths.latinpipe_dir)
    archive.unlink()
    logger.info("[baixado] latinpipe/Portparser_v2_model")


def _require_executables(names: Sequence[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise ToolError(f"required executables not on PATH: {', '.join(missing)}")


def _run(command: Sequence[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-10:])
        raise ToolError(f"{command[0]} failed ({result.returncode}):\n{tail}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR)
    args = parser.parse_args(argv)
    try:
        install(ToolPaths(args.tools_dir))
    except ToolError as err:
        logger.error(f"Falha na instalação: {type(err).__name__}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
