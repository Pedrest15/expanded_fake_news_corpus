"""Onde estão as ferramentas externas de parsing e como executá-las.

Tudo mora em ``tools/`` na raiz do projeto, fora do git — são repositórios de
terceiros e um modelo de 1,6 GB. :mod:`install_tools` os coloca lá;
:class:`ToolPaths` sabe onde cada um fica e :func:`check_tools` confere antes
de gastar tempo.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from expanded_fake_news_corpus.analysis.documents import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_TOOLS_DIR = PROJECT_ROOT / "tools"

#: Repositórios clonados em ``tools/``, pelo nome da pasta.
TOOL_REPOSITORIES: dict[str, str] = {
    "portSentencer": "https://github.com/LuceleneL/portSentencer.git",
    "portTokenizer": "https://github.com/LuceleneL/portTokenizer.git",
    "Portparser.v2": "https://github.com/LuceleneL/Portparser.v2.git",
    "latinpipe": "https://github.com/ufal/evalatin2024-latinpipe.git",
}


class ToolError(RuntimeError):
    """Ferramenta externa ausente ou com falha."""


@dataclass(frozen=True)
class ToolPaths:
    """Caminhos da cadeia de parsing a partir da raiz ``tools/``."""

    root: Path = DEFAULT_TOOLS_DIR

    @property
    def sentencer_dir(self) -> Path:
        return self.root / "portSentencer"

    @property
    def tokenizer_dir(self) -> Path:
        return self.root / "portTokenizer"

    @property
    def latinpipe_dir(self) -> Path:
        return self.root / "latinpipe"

    @property
    def latinpipe_python(self) -> Path:
        """Interpretador do venv próprio do LatinPipe (Python 3.11, torch)."""
        return self.latinpipe_dir / ".venv" / "bin" / "python"

    @property
    def model_dir(self) -> Path:
        return self.latinpipe_dir / "Portparser_v2_model"

    @property
    def postproc_dir(self) -> Path:
        return self.root / "Portparser.v2" / "postproc"

    def required_files(self) -> dict[str, Path]:
        """Um arquivo por componente, cuja presença prova a instalação."""
        return {
            "portSentencer": self.sentencer_dir / "portSent.py",
            "portTokenizer": self.tokenizer_dir / "portTok.py",
            "latinpipe": self.latinpipe_dir / "latinpipe_evalatin24.py",
            "latinpipe venv": self.latinpipe_python,
            "Portparser v2 model": self.model_dir / "model.weights.h5",
            "postproc": self.postproc_dir / "postprocess.py",
        }


def check_tools(paths: ToolPaths) -> None:
    """Confere que a cadeia inteira está instalada.

    Raises:
        ToolError: Listando o que falta e como instalar
    """
    missing = [
        f"{name}: {path}"
        for name, path in paths.required_files().items()
        if not path.exists()
    ]
    if missing:
        raise ToolError(
            "missing tools — run 'python -m expanded_fake_news_corpus.parsing."
            "install_tools':\n  " + "\n  ".join(missing)
        )


def run_tool(command: Sequence[str], *, cwd: Path, log: Path | None = None) -> None:
    """Executa uma ferramenta externa, guardando a saída num log.

    Args:
        command: Linha de comando
        cwd: Diretório de trabalho — as ferramentas leem arquivos auxiliares
            do diretório corrente
        log: Arquivo para stdout e stderr, se quiser guardá-los

    Raises:
        ToolError: Se o processo sair com código diferente de zero
    """
    logger.debug(f"[TOOL] {' '.join(str(part) for part in command)} (cwd={cwd.name})")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if log is not None:
        log.write_text(
            f"{result.stdout}\n--- stderr ---\n{result.stderr}", encoding="utf-8"
        )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
        raise ToolError(
            f"{Path(str(command[0])).name} failed ({result.returncode}):\n"
            + "\n".join(tail)
        )


def tool_versions(paths: ToolPaths) -> dict[str, str]:
    """Commit de cada ferramenta clonada e tamanho do modelo, para o manifesto."""
    versions = {name: _git_commit(paths.root / name) for name in TOOL_REPOSITORIES}
    model = paths.model_dir / "model.weights.h5"
    versions["portparser_model_bytes"] = (
        str(model.stat().st_size) if model.exists() else "?"
    )
    return versions


def _git_commit(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
