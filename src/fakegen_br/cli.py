"""Linha de comando do FakeGen.BR.

Dois subcomandos, um por estágio do pipeline::

    fakegen headline   notícia verdadeira  ->  manchete
    fakegen fake       manchete            ->  notícia falsa sintética

``--model`` pode ser repetido: a mesma entrada é processada por cada modelo
pedido, e cada execução grava em sua própria pasta::

    <out-dir>/[<run-name>/]<provedor>/<modelo>/<arquivo-de-entrada>.jsonl
    <out-dir>/[<run-name>/]<provedor>/<modelo>/meta.json

Exemplos::

    fakegen headline --text "O Ministério da Saúde anunciou ..."

    fakegen headline --input true-corpus/clean/rounds/round1/FakeBr_true.csv \\
        --genre news --model openai/gpt-4o-mini --out-dir data/round1

    fakegen fake --input data/round1/openai/gpt-4o-mini/FakeBr_true.jsonl \\
        --model openai/gpt-4o-mini --out-dir data/round1-fakes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

from fakegen_br.agents.fake_news import DEFAULT_MAX_TOKENS as FAKE_MAX_TOKENS
from fakegen_br.agents.fake_news import FakeNewsAgent
from fakegen_br.agents.fake_news_writer import FakeNewsWriter
from fakegen_br.agents.headline import DEFAULT_MAX_INPUT_CHARS, HeadlineAgent
from fakegen_br.config import (
    ConfigError,
    LLMSettings,
    model_path,
    resolve_sampling,
    split_model,
)
from fakegen_br.corpus import (
    CorpusError,
    JsonlWriter,
    NewsItem,
    read_csv,
    read_existing_ids,
    read_headline_records,
    read_jsonl,
    read_text_dir,
)
from fakegen_br.prompts import GENRE_BLOCKS, NEWS
from fakegen_br.runner import BatchJob, run_batch
from fakegen_br.schemas import (
    FakeNewsError,
    FakeNewsResult,
    FakeNewsWriterResult,
    HeadlineError,
    HeadlineResult,
)

#: Erros de uso previstos: viram mensagem e código 1, não rastreamento de pilha.
_EXPECTED_ERRORS = (
    ConfigError,
    CorpusError,
    HeadlineError,
    FakeNewsError,
    FileNotFoundError,
)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Adiciona os parâmetros de amostragem, comuns aos dois estágios."""
    parser.add_argument(
        "--temperature", type=float, help="Temperatura de amostragem (padrão 0)."
    )
    parser.add_argument(
        "--top-p", type=float, help="Nucleus sampling. Aceito pelos três provedores."
    )
    parser.add_argument(
        "--top-k", type=int, help="Top-k. A OpenAI não suporta; lá é descartado."
    )
    parser.add_argument(
        "--seed", type=int, help="Semente. A Anthropic não suporta; lá é descartada."
    )


def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    """Adiciona destino, retomada e paralelismo, comuns aos dois estágios."""
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Raiz da saída; cada modelo grava em <out-dir>/<provedor>/<modelo>/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL de saída num caminho exato (só com um --model).",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Nome da execução, inserido em <out-dir>/<run-name>/<provedor>/"
            "<modelo>/. Necessário ao rodar o mesmo modelo com configurações "
            "diferentes: sem ele, a segunda execução sobrescreve a primeira."
        ),
    )
    parser.add_argument("--limit", type=int, help="Processa no máximo N itens.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula ids já presentes na saída e escreve em modo append.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Chamadas simultâneas ao provedor."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="PROVEDOR/MODELO",
        help=(
            "Modelo a usar; repita a opção para rodar a mesma entrada em vários "
            "modelos. Sem a opção, usa FAKEGEN_MODEL do ambiente."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Silencia o progresso.")


def _add_headline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Subcomando ``headline``: notícia verdadeira -> manchete."""
    parser = subparsers.add_parser(
        "headline",
        help="Gera a manchete de uma notícia ou de um lote de notícias.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Texto da notícia.")
    source.add_argument("--file", type=Path, help="Arquivo com o texto da notícia.")
    source.add_argument("--input", type=Path, help="Arquivo .jsonl ou .csv com o lote.")
    source.add_argument(
        "--input-dir", type=Path, help="Diretório com um .txt por notícia."
    )
    source.add_argument(
        "--stdin", action="store_true", help="Lê o texto da notícia da entrada padrão."
    )

    _add_batch_args(parser)
    _add_sampling_args(parser)

    parser.add_argument(
        "--text-field", default="text", help="Campo do texto (jsonl/csv)."
    )
    parser.add_argument("--id-field", default="id", help="Campo do id (jsonl/csv).")
    parser.add_argument(
        "--glob", default="*.txt", help="Padrão de arquivos em --input-dir."
    )
    parser.add_argument(
        "--genre",
        choices=sorted(GENRE_BLOCKS),
        default=NEWS,
        help=(
            "Gênero do texto de entrada: 'news' para jornalismo comum "
            "(Fake.br) ou 'factcheck' para checagem de fatos (FakeTrueBR). "
            "Muda o bloco acrescentado ao prompt base."
        ),
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help="Trunca a notícia neste número de caracteres (0 desliga).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Imprime o resultado em JSON."
    )


def _add_fake_parser(subparsers: argparse._SubParsersAction) -> None:
    """Subcomando ``fake``: manchete -> notícia falsa sintética."""
    parser = subparsers.add_parser(
        "fake",
        help="Gera notícias falsas a partir das manchetes do estágio anterior.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL produzido por 'fakegen headline'.",
    )

    _add_batch_args(parser)
    _add_sampling_args(parser)

    parser.add_argument(
        "--strategy",
        choices=("paper", "estruturada"),
        default="paper",
        help=(
            "'paper' reproduz o prompt e as tags de Silva et al. (padrão); "
            "'estruturada' usa a variante com saída tipada, faixa de tamanho "
            "calibrada e blocos por gênero."
        ),
    )
    parser.add_argument(
        "--genre",
        choices=sorted(GENRE_BLOCKS),
        default=NEWS,
        help="Gênero da manchete de origem (só afeta --strategy estruturada).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=FAKE_MAX_TOKENS,
        help="Teto de tokens da notícia gerada.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="fakegen",
        description="Ferramentas de geração do corpus FakeGen.BR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_headline_parser(subparsers)
    _add_fake_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do executável ``fakegen``."""
    args = build_parser().parse_args(argv)
    try:
        return _run_fake(args) if args.command == "fake" else _run_headline(args)
    except _EXPECTED_ERRORS as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrompido", file=sys.stderr)
        return 130


# --------------------------------------------------------------------------
# Caminhos de saída
# --------------------------------------------------------------------------


def model_dir(root: Path, model: str, run_name: str | None = None) -> Path:
    """Pasta de saída de um modelo. Ver :func:`fakegen_br.config.model_path`."""
    return model_path(root, model, run_name)


def _single_destination(args: argparse.Namespace, settings: LLMSettings) -> Path | None:
    if args.output:
        return args.output
    if args.out_dir:
        return model_dir(args.out_dir, settings.model, args.run_name) / "headline.jsonl"
    return None


def _batch_destination(args: argparse.Namespace, settings: LLMSettings) -> Path:
    if args.out_dir:
        input_dir = getattr(args, "input_dir", None)
        source = input_dir if input_dir is not None else args.input
        stem = source.name if input_dir is not None else source.stem
        return model_dir(args.out_dir, settings.model, args.run_name) / f"{stem}.jsonl"
    return args.output


# --------------------------------------------------------------------------
# Preparação comum aos dois estágios
# --------------------------------------------------------------------------


def _resolve_models(args: argparse.Namespace) -> list[str | None]:
    """Modelos pedidos; ``[None]`` deixa o ambiente decidir."""
    return args.models or [None]


def _check_destination_flags(
    args: argparse.Namespace, models: Sequence[str | None], *, is_batch: bool
) -> None:
    """Valida a combinação de ``--output``, ``--out-dir`` e vários ``--model``.

    Raises:
        CorpusError: Se um arquivo único for pedido para vários modelos, ou se
            um lote não tiver destino.
    """
    if len(models) > 1 and args.output:
        raise CorpusError(
            "--output points at a single file and cannot be combined with several "
            "--model. Use --out-dir, which splits results per provider/model."
        )
    if is_batch and not (args.out_dir or args.output):
        raise CorpusError("Pass --out-dir (or --output) when processing a batch.")


def _settings_for(args: argparse.Namespace, model: str | None) -> LLMSettings:
    """Configuração do LLM para um modelo, com os overrides da linha de comando."""
    return LLMSettings.from_env(
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=getattr(args, "max_tokens", None),
    )


def _warn_dropped(settings: LLMSettings, *, quiet: bool) -> None:
    """Avisa quando um parâmetro pedido não é aceito pelo provedor."""
    _, dropped = resolve_sampling(settings)
    if dropped and not quiet:
        print(
            f"[{settings.model}] aviso: {', '.join(dropped)} não é suportado por "
            f"'{settings.provider}' e não será enviado — as execuções não ficam "
            "equivalentes entre provedores.",
            file=sys.stderr,
        )


def _write_meta(
    args: argparse.Namespace,
    settings: LLMSettings,
    destination: Path,
    *,
    prompt: str,
    total: int,
    failures: int,
) -> None:
    """Grava a procedência da execução ao lado do JSONL."""
    provider, name = split_model(settings.model)
    input_dir = getattr(args, "input_dir", None)
    source = input_dir if input_dir is not None else args.input
    applied, dropped = resolve_sampling(settings)
    meta = {
        "provider": provider,
        "model": name,
        "model_string": settings.model,
        "sampling": applied,
        "sampling_dropped": dropped,
        "max_tokens": settings.max_tokens,
        "max_input_chars": getattr(args, "max_input_chars", None) or None,
        "stage": args.command,
        "run_name": args.run_name,
        "strategy": getattr(args, "strategy", None),
        "genre": args.genre,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "input": str(source),
        "output": str(destination),
        "processed": total,
        "failures": failures,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    # Um meta por lote, não por pasta: o mesmo modelo processa mais de um
    # arquivo de entrada na mesma pasta (um por corpus), e um meta.json único
    # faria a segunda execução apagar a procedência da primeira.
    path = destination.with_suffix(".meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Estágio 1: titulação
# --------------------------------------------------------------------------


def _run_headline(args: argparse.Namespace) -> int:
    """Orquestra a titulação para cada modelo pedido."""
    models = _resolve_models(args)
    is_batch = bool(args.input or args.input_dir)
    _check_destination_flags(args, models, is_batch=is_batch)

    items = list(_load_items(args)) if is_batch else []

    status = 0
    for model in models:
        settings = _settings_for(args, model)
        _warn_dropped(settings, quiet=args.quiet)
        agent = HeadlineAgent(
            settings=settings,
            genre=args.genre,
            max_input_chars=args.max_input_chars or None,
        )
        if is_batch:
            status |= _headline_batch(args, agent, settings, items)
        else:
            status |= _headline_single(args, agent, settings, labelled=len(models) > 1)
    return status


def _headline_single(
    args: argparse.Namespace,
    agent: HeadlineAgent,
    settings: LLMSettings,
    *,
    labelled: bool,
) -> int:
    """Tituladora de uma notícia só, vinda de --text, --file ou --stdin."""
    result = agent.generate(_read_single_text(args), source_id=_single_source_id(args))

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"[{settings.model}] {result.headline}" if labelled else result.headline)
        if not args.quiet:
            if result.rationale:
                print(f"justificativa: {result.rationale}", file=sys.stderr)
            for warning in result.warnings:
                print(f"aviso: {warning}", file=sys.stderr)

    destination = _single_destination(args, settings)
    if destination:
        with JsonlWriter(destination) as writer:
            writer.write(result)
    return 0


def _read_single_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return args.file.read_text(encoding="utf-8", errors="replace")
    return sys.stdin.read()


def _single_source_id(args: argparse.Namespace) -> str | None:
    return args.file.stem if args.file is not None else None


def _headline_batch(
    args: argparse.Namespace,
    agent: HeadlineAgent,
    settings: LLMSettings,
    items: Sequence[NewsItem],
) -> int:
    """Titula um lote de notícias com um modelo."""
    destination = _batch_destination(args, settings)
    pending = _pending_items(items, destination, resume=args.resume)
    job = BatchJob(
        destination=destination,
        label=settings.model,
        input_noun="notícias",
        output_noun="manchetes",
        total=len(pending),
        append=args.resume,
        quiet=args.quiet,
        skipped=len(items) - len(pending),
    )
    report = run_batch(
        job,
        _headline_outcomes(agent, pending, concurrency=args.concurrency),
        lambda r: json.dumps(r.headline, ensure_ascii=False),
    )
    _write_meta(
        args,
        settings,
        destination,
        prompt=agent.prompt_fingerprint,
        total=report.total,
        failures=report.failures,
    )
    return report.exit_code


def _pending_items(
    items: Sequence[NewsItem], destination: Path, *, resume: bool
) -> Sequence[NewsItem]:
    """Remove os itens já gravados na saída deste modelo."""
    if not resume:
        return items
    seen = read_existing_ids(destination)
    return [item for item in items if item[0] not in seen] if seen else items


def _headline_outcomes(
    agent: HeadlineAgent, items: Sequence[NewsItem], *, concurrency: int
) -> Iterator[HeadlineResult | HeadlineError]:
    """Resultados na ordem de entrada, um por notícia."""
    if concurrency > 1:
        yield from _gather(agent.agenerate_many(items, concurrency=concurrency))
        return
    for source_id, text in items:
        try:
            yield agent.generate(text, source_id=source_id)
        except Exception as exc:  # noqa: BLE001 - falha por item, não do lote
            yield HeadlineError(f"{source_id or '<no id>'}: {exc}")


def _load_items(args: argparse.Namespace) -> Iterable[NewsItem]:
    """Lê a entrada do estágio 1, conforme o formato pedido."""
    if args.input_dir is not None:
        items: Iterable[NewsItem] = read_text_dir(args.input_dir, pattern=args.glob)
    else:
        items = _read_tabular(args)
    return _take(items, args.limit) if args.limit else items


def _read_tabular(args: argparse.Namespace) -> Iterable[NewsItem]:
    path: Path = args.input
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path, text_field=args.text_field, id_field=args.id_field)
    if suffix in {".jsonl", ".json", ".ndjson"}:
        return read_jsonl(path, text_field=args.text_field, id_field=args.id_field)
    raise CorpusError(
        f"{path}: unrecognized extension {suffix!r} (use .jsonl or .csv)."
    )


def _take(items: Iterable[NewsItem], limit: int) -> Iterator[NewsItem]:
    for index, item in enumerate(items):
        if index >= limit:
            return
        yield item


# --------------------------------------------------------------------------
# Estágio 2: geração de notícia falsa
# --------------------------------------------------------------------------


def _run_fake(args: argparse.Namespace) -> int:
    """Orquestra a geração para cada modelo pedido."""
    models = _resolve_models(args)
    _check_destination_flags(args, models, is_batch=True)

    records = list(read_headline_records(args.input))
    if args.limit:
        records = records[: args.limit]

    status = 0
    for model in models:
        settings = _settings_for(args, model)
        _warn_dropped(settings, quiet=args.quiet)
        status |= _fake_batch(
            args, _build_fake_agent(args, settings), settings, records
        )
    return status


def _build_fake_agent(
    args: argparse.Namespace, settings: LLMSettings
) -> FakeNewsAgent | FakeNewsWriter:
    """Escolhe o gerador conforme ``--strategy``."""
    if args.strategy == "paper":
        return FakeNewsWriter(settings=settings)
    return FakeNewsAgent(settings=settings, genre=args.genre)


def _fake_batch(
    args: argparse.Namespace,
    agent: FakeNewsAgent | FakeNewsWriter,
    settings: LLMSettings,
    records: Sequence[dict],
) -> int:
    """Gera notícias falsas para um lote de manchetes com um modelo."""
    destination = _batch_destination(args, settings)
    pending = _pending_records(records, destination, resume=args.resume)
    job = BatchJob(
        destination=destination,
        label=settings.model,
        input_noun="manchetes",
        output_noun="notícias",
        total=len(pending),
        append=args.resume,
        quiet=args.quiet,
        skipped=len(records) - len(pending),
    )
    report = run_batch(
        job,
        _fake_outcomes(agent, pending, concurrency=args.concurrency),
        _describe_fake,
    )
    _write_meta(
        args,
        settings,
        destination,
        prompt=agent.prompt_fingerprint,
        total=report.total,
        failures=report.failures,
    )
    return report.exit_code


def _pending_records(
    records: Sequence[dict], destination: Path, *, resume: bool
) -> Sequence[dict]:
    """Remove as manchetes já geradas na saída deste modelo."""
    if not resume:
        return records
    seen = read_existing_ids(destination)
    return (
        [r for r in records if str(r.get("source_id")) not in seen] if seen else records
    )


def _fake_outcomes(
    agent: FakeNewsAgent | FakeNewsWriter,
    records: Sequence[dict],
    *,
    concurrency: int,
) -> Iterator[FakeNewsResult | FakeNewsWriterResult | FakeNewsError]:
    """Resultados na ordem de entrada, um por manchete."""
    if concurrency > 1:
        yield from _gather(agent.agenerate_many(records, concurrency=concurrency))
        return
    for record in records:
        try:
            yield agent.generate(
                record.get("headline", ""),
                source_id=record.get("source_id"),
                headline_model=record.get("model", ""),
            )
        except Exception as exc:  # noqa: BLE001 - falha por item, não do lote
            yield FakeNewsError(f"{record.get('source_id') or '<no id>'}: {exc}")


def _describe_fake(outcome: FakeNewsResult | FakeNewsWriterResult) -> str:
    """Primeira linha do texto gerado, para o relatório de progresso."""
    if isinstance(outcome, FakeNewsResult):
        title = outcome.fake_headline
    else:
        title = outcome.synthetic_text.strip().splitlines()[0]
    return json.dumps(title[:110], ensure_ascii=False)


def _gather(source) -> list:
    """Consome um gerador assíncrono de resultados, preservando a ordem."""

    async def collect() -> list:
        return [outcome async for outcome in source]

    return asyncio.run(collect())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
