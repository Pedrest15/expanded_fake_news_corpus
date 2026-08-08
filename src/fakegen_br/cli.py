"""Linha de comando do FakeGen.BR.

Exemplos::

    # uma notícia, texto direto ou arquivo
    fakegen headline --text "O Ministério da Saúde anunciou ..."
    fakegen headline --file noticia.txt --json

    # lote sobre o Fake.br (um .txt por notícia)
    fakegen headline --input-dir Fake.br-Corpus/full_texts/true \\
        --output data/manchetes_fakebr.jsonl --concurrency 4

    # lote sobre um JSONL/CSV
    fakegen headline --input noticias.jsonl --text-field texto --id-field id \\
        --output data/manchetes.jsonl --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from fakegen_br.agents.headline import DEFAULT_MAX_INPUT_CHARS, HeadlineAgent
from fakegen_br.config import ConfigError, LLMSettings
from fakegen_br.corpus import (
    CorpusError,
    JsonlWriter,
    NewsItem,
    read_csv,
    read_existing_ids,
    read_jsonl,
    read_text_dir,
)
from fakegen_br.schemas import HeadlineError, HeadlineResult


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="fakegen",
        description="Ferramentas de geração do corpus FakeGen.BR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    headline = subparsers.add_parser(
        "headline",
        help="Gera a manchete de uma notícia ou de um lote de notícias.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = headline.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Texto da notícia.")
    source.add_argument("--file", type=Path, help="Arquivo com o texto da notícia.")
    source.add_argument("--input", type=Path, help="Arquivo .jsonl ou .csv com o lote.")
    source.add_argument(
        "--input-dir", type=Path, help="Diretório com um .txt por notícia."
    )
    source.add_argument(
        "--stdin", action="store_true", help="Lê o texto da notícia da entrada padrão."
    )

    headline.add_argument(
        "--output", type=Path, help="JSONL de saída (obrigatório em lote)."
    )
    headline.add_argument(
        "--text-field", default="text", help="Campo do texto (jsonl/csv)."
    )
    headline.add_argument("--id-field", default="id", help="Campo do id (jsonl/csv).")
    headline.add_argument(
        "--glob", default="*.txt", help="Padrão de arquivos em --input-dir."
    )
    headline.add_argument("--limit", type=int, help="Processa no máximo N notícias.")
    headline.add_argument(
        "--resume",
        action="store_true",
        help="Pula ids já presentes no --output e escreve em modo append.",
    )
    headline.add_argument(
        "--concurrency", type=int, default=1, help="Chamadas simultâneas ao provedor."
    )

    headline.add_argument("--model", help="Modelo no formato 'provedor/modelo'.")
    headline.add_argument(
        "--temperature", type=float, help="Temperatura de amostragem."
    )
    headline.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help="Trunca a notícia neste número de caracteres (0 desliga).",
    )
    headline.add_argument(
        "--json", action="store_true", help="Imprime o resultado em JSON."
    )
    headline.add_argument("--quiet", action="store_true", help="Silencia o progresso.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do executável ``fakegen``."""
    args = build_parser().parse_args(argv)
    try:
        return _run_headline(args)
    except (ConfigError, CorpusError, HeadlineError, FileNotFoundError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrompido", file=sys.stderr)
        return 130


def _run_headline(args: argparse.Namespace) -> int:
    settings = LLMSettings.from_env(model=args.model, temperature=args.temperature)
    agent = HeadlineAgent(
        settings=settings,
        max_input_chars=args.max_input_chars or None,
    )

    if args.input or args.input_dir:
        return _run_batch(args, agent)
    return _run_single(args, agent)


def _run_single(args: argparse.Namespace, agent: HeadlineAgent) -> int:
    if args.text is not None:
        news_text = args.text
    elif args.file is not None:
        news_text = args.file.read_text(encoding="utf-8", errors="replace")
    else:
        news_text = sys.stdin.read()

    source_id = args.file.stem if args.file is not None else None
    result = agent.generate(news_text, source_id=source_id)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.headline)
        if not args.quiet:
            if result.rationale:
                print(f"\njustificativa: {result.rationale}", file=sys.stderr)
            for warning in result.warnings:
                print(f"aviso: {warning}", file=sys.stderr)

    if args.output:
        with JsonlWriter(args.output) as writer:
            writer.write(result)
    return 0


def _run_batch(args: argparse.Namespace, agent: HeadlineAgent) -> int:
    if not args.output:
        raise CorpusError("--output é obrigatório ao processar um lote.")

    items = list(_load_items(args))
    total = len(items)
    if not args.quiet:
        print(f"{total} notícias a processar → {args.output}", file=sys.stderr)

    failures = 0
    done = 0
    with JsonlWriter(args.output, append=args.resume) as writer:
        for outcome in _process(agent, items, concurrency=args.concurrency):
            done += 1
            if isinstance(outcome, HeadlineResult):
                writer.write(outcome)
                if not args.quiet:
                    _report(done, total, outcome)
            else:
                failures += 1
                print(f"[{done}/{total}] falha: {outcome}", file=sys.stderr)

    if not args.quiet:
        print(
            f"concluído: {done - failures} manchetes, {failures} falhas",
            file=sys.stderr,
        )
    return 1 if failures and failures == total else 0


def _process(
    agent: HeadlineAgent, items: list[NewsItem], *, concurrency: int
) -> Iterator[HeadlineResult | HeadlineError]:
    if concurrency <= 1:
        for source_id, text in items:
            try:
                yield agent.generate(text, source_id=source_id)
            except Exception as exc:  # noqa: BLE001 - falha por item, não do lote
                yield HeadlineError(f"{source_id or '<sem id>'}: {exc}")
        return

    async def collect() -> list[HeadlineResult | HeadlineError]:
        return [
            outcome
            async for outcome in agent.agenerate_many(items, concurrency=concurrency)
        ]

    yield from asyncio.run(collect())


def _load_items(args: argparse.Namespace) -> Iterable[NewsItem]:
    if args.input_dir is not None:
        items: Iterable[NewsItem] = read_text_dir(args.input_dir, pattern=args.glob)
    else:
        path: Path = args.input
        suffix = path.suffix.lower()
        if suffix == ".csv":
            items = read_csv(path, text_field=args.text_field, id_field=args.id_field)
        elif suffix in {".jsonl", ".json", ".ndjson"}:
            items = read_jsonl(path, text_field=args.text_field, id_field=args.id_field)
        else:
            raise CorpusError(
                f"{path}: extensão {suffix!r} não reconhecida (use .jsonl ou .csv)."
            )

    if args.resume and args.output:
        seen = read_existing_ids(args.output)
        if seen:
            items = (item for item in items if item[0] not in seen)

    if args.limit:
        items = _take(items, args.limit)
    return items


def _take(items: Iterable[NewsItem], limit: int) -> Iterator[NewsItem]:
    for index, item in enumerate(items):
        if index >= limit:
            return
        yield item


def _report(done: int, total: int, result: HeadlineResult) -> None:
    suffix = f"  [{'; '.join(result.warnings)}]" if result.warnings else ""
    line = json.dumps(result.headline, ensure_ascii=False)
    print(f"[{done}/{total}] {result.source_id or ''} {line}{suffix}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
