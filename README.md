# FakeGen.BR

Building a corpus of AI-generated *fake news* in Brazilian Portuguese, and
measuring how it differs from fake news written by people.

**Project page:** <https://pedrest15.github.io/expanded_fake_news_corpus/> —
corpus browser and
[linguistic characterisation](https://pedrest15.github.io/expanded_fake_news_corpus/analysis.html).

The starting point is the true news of the [Fake.br][fakebr] and
[FakeTrueBR][faketruebr] corpora. Each true article has a human-written fake
counterpart in its source corpus; the LLM-generated fake is paired with it by
identifier, so every comparison is between two fakes about the same story.

What the repository contains:

- **Generation** — two experiments. The main pipeline writes a faithful
  headline for the true article (`fakegen headline`) and then a fake news
  story from that headline (`fakegen fake`). The *paper replication*
  (`fakegen paper`) instead feeds the whole true article to the prompt of
  Silva et al., reproduced verbatim, and only swaps the model.
- **Linguistic analyses** — syllables, lexical diversity (MATTR), Zipf, SAGE,
  LIWC, UPOS distribution, dependency-grammar rules and Enhanced-UD rules, all
  human vs. machine on paired documents, run one at a time or through the
  `fakegen-analysis` orchestrator.
- **Syntactic parsing** — the Portparser v2 chain (LatinPipe + BERTimbau) and
  Grew EUD enrichment, producing the CoNLL-U the rule analyses read.
- **A site** (`docs/`) browsing the generated texts and their characterisation.

Generated text and its provenance live in `corpus/`
([corpus/README.md](corpus/README.md)); analysis tables in `data/analysis/`;
parsed CoNLL-U in `data/parsed/`.

## Pipeline

```
true article ──▶ [headline agent] ──▶ headline ──▶ [fake news writer] ──▶ synthetic fake news
                                                   (paper replication: true article ──▶ [paper prompt] ──▶ synthetic fake news)
```

The first stage is the headline agent, described here; the second stage takes
the headline and writes the fake news with the prompt of Silva et al. adapted
to a headline input (`PAPER_FAKE_PROMPT`, strategy `paper`) or with a
structured-output variant calibrated by genre and length (`estruturada`).

The agent is orchestrated with [LangGraphLib][langgraphlib] and the graph has two
steps:

```
start ──▶ headline_writer ──▶ sanitize ──▶ end
```

- **`headline_writer`** — a LangGraphLib `Agent` with structured output
  (`headline` + `rationale`), guided by the prompt in
  [prompts.py](src/expanded_fake_news_corpus/prompts.py). The prompt is
  deliberately conservative: the headline represents the **true** article, so it
  must be faithful to the text, with no sensationalism — distortion is left to
  the next stage.
- **`sanitize`** — a deterministic node that strips quotes, markdown, labels
  ("Manchete:") and the final period, keeping the model's raw output in
  `raw_headline` for auditing.

After the graph, [`check_headline`](src/expanded_fake_news_corpus/agents/headline.py)
attaches quality warnings to the result (length outside the 6–18 word range, and
headlines whose content vocabulary barely appears in the article — a sign of
hallucination). Warnings do not discard the headline: they are stored in the
JSONL for manual review and corpus statistics.

## Preparing the corpora

The source corpora mix true and fake news, in different formats. The script
below extracts only the true articles and normalises both into a common schema
(`id, text, link, duplicate_rows`, plus `author, category, date` for Fake.br):

```bash
# Fake.br must be downloaded (the full_texts folder is not versioned)
git clone --depth 1 https://github.com/roneysco/Fake.br-Corpus.git
cp -r Fake.br-Corpus/full_texts true-corpus/raw/Fake.br-full_texts

uv run python scripts/prepare_corpora.py
```

| Corpus | Source | True | Duplicates removed | **Total** |
|---|---|---|---|---|
| Fake.br | 3,600 files in `full_texts/true/` | 3,600 | 1 | **3,599** |
| FakeTrueBR | 1,791 (fake, true) pairs | 1,791 | 388 | **1,403** |

**Use Fake.br's `full_texts/`, not `preprocessed/`.** The preprocessed CSV the
repository also ships is lower-cased, unaccented, without punctuation and
without stopwords — headline writing is unfeasible on it, because the model
would have to invent the spelling the text lost. `full_texts/` carries the
article as collected from the website, and the metadata (author, link,
category, date) comes from `true-meta-information/`.

In FakeTrueBR the same true article is paired with more than one fake, hence the
reduction from 1,791 to 1,403. The `duplicate_rows` field keeps the discarded
records and `id` points at the origin (file name in Fake.br, CSV row in
FakeTrueBR), so traceability is preserved. No source file is modified; the
output goes to `true-corpus/clean/`.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Configuration

The model is always declared as `provider/model`. The three supported providers
are **Ollama** (local), **Anthropic** and **OpenAI**:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `FAKEGEN_MODEL` | `ollama/llama3.1`, `anthropic/claude-sonnet-5`, `openai/gpt-4o-mini`, ... |
| `FAKEGEN_TEMPERATURE` | Default `0` |
| `FAKEGEN_TOP_P`, `FAKEGEN_TOP_K`, `FAKEGEN_SEED` | Optional; unset means the provider's default |
| `FAKEGEN_MAX_TOKENS`, `FAKEGEN_TIMEOUT`, `FAKEGEN_MAX_RETRIES` | Optional |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Provider key (Ollama needs none) |
| `OLLAMA_BASE_URL` | Ollama address, if not the default |

Any variable can be overridden on the command line (`--model`, `--temperature`,
`--top-p`, `--top-k`, `--seed`).

### Sampling and reproducibility

The three providers do not expose the same controls:

| | `temperature` | `top_p` | `top_k` | `seed` |
|---|---|---|---|---|
| Anthropic | ✅ | ✅ | ✅ | ❌ |
| OpenAI | ✅ | ✅ | ❌ | ✅ |
| Ollama | ✅ | ✅ | ✅ | ✅ |

So **only `temperature` can be pinned on all three**, and it is what the
comparison relies on: at `temperature=0` decoding is greedy, which makes
`top_k` and `top_p` inert and puts the providers in the most similar regime
possible. That is the project default for headline writing, a fidelity task,
not a creative one. (For the next stage — fake news generation — that default
is wrong: zero temperature would produce formulaic, low-diversity texts.)

Parameters that were not requested **are not sent**: the provider's default
applies, rather than a value we made up. When a parameter is requested but the
provider does not accept it, it is dropped with a warning on `stderr` and
recorded in `sampling_dropped` in `meta.json` — silence here would invalidate
the comparison across providers. The filter is necessary: `ChatOpenAI(top_k=...)`
raises no error, it only diverts the parameter to `model_kwargs`, and the
rejection only surfaces in the HTTP call.

Note that pinning the sampling **does not guarantee identical output** across
runs: hosted APIs have infrastructure non-determinism even at `temperature=0`.
The levers that actually sustain reproducibility are pinning the model snapshot
(instead of a moving alias), versioning the prompt and recording everything —
`meta.json` keeps the effective parameters and a hash of the prompt used.

## Usage

### One article

```bash
uv run fakegen headline --text "O Ministério da Saúde anunciou nesta terça-feira ..."
uv run fakegen headline --file noticia.txt --json
cat noticia.txt | uv run fakegen headline --stdin
```

### Several models in one run

`--model` can be repeated. The same input is processed by each model, and each
one writes to its own folder:

```bash
uv run fakegen headline \
    --input true-corpus/clean/samples/sample_FakeBr.csv \
    --model anthropic/claude-sonnet-5 \
    --model openai/gpt-4o-mini \
    --model ollama/llama3.1 \
    --out-dir data/manchetes
```

```
data/manchetes/
├── anthropic/claude-sonnet-5/sample_FakeBr.jsonl   (+ meta.json)
├── openai/gpt-4o-mini/sample_FakeBr.jsonl          (+ meta.json)
└── ollama/llama3.1/sample_FakeBr.jsonl             (+ meta.json)
```

Each folder's `meta.json` records provider, model, temperature, input, counts
and timestamp — provenance stays next to the result. The model is also written
on every JSONL line. Without `--model`, the environment's `FAKEGEN_MODEL`
applies.

### Batch over a corpus

CSV and JSONL are read by naming the text and identifier fields; the original
Fake.br can also be read straight from its folder of `.txt` files:

```bash
uv run fakegen headline \
    --input true-corpus/clean/FakeBr_true.csv --text-field text --id-field id \
    --model anthropic/claude-sonnet-5 --out-dir data/manchetes \
    --concurrency 4 --resume

uv run fakegen headline \
    --input-dir true-corpus/raw/Fake.br-full_texts/true \
    --model anthropic/claude-sonnet-5 --out-dir data/manchetes --concurrency 4
```

Useful batch options:

- `--concurrency N` — simultaneous calls to the provider (asynchronous
  execution).
- `--resume` — skips ids already present in **that model's** output and
  continues in append mode. The output is flushed after every line, so an
  interrupted run can be resumed without losing what was already generated.
- `--limit N` — processes only the first N articles (handy for calibrating the
  prompt before running the whole corpus).
- `--max-input-chars N` — truncates the article before sending it to the model
  (default 12,000; `0` disables).
- `--output FILE` — exact path, instead of the per-provider layout. Only valid
  with a single `--model`.

Individual failures (model refusal, timeout) are logged to `stderr` and do not
stop the batch.

### Stage 2: fake news from the headlines (`fakegen fake`)

The second stage reads the JSONL written by `fakegen headline` and writes one
fake news story per headline, into the same provider/model layout:

```bash
uv run fakegen fake --input corpus/headlines/round1/openai/gpt-4.1-mini-2025-04-14/FakeBr_true.jsonl \
    --model openai/gpt-4.1-mini-2025-04-14 --out-dir corpus/fake_news/round1
```

`--strategy paper` (default) uses the prompt and `<syntheticText>`/`<changes>`
tags of Silva et al., with the first sentence adapted to take a headline;
`--strategy estruturada` uses typed output, a genre block (`--genre news` for
Fake.br, `factcheck` for FakeTrueBR) and a length range measured on the human
fakes. Rounds, model assignment and every drawn identifier are recorded under
`corpus/rounds/` and `corpus/assignments/`.

### Calibration test (10 articles)

Before running the whole corpus, there is a small, deliberately unbalanced
sample in `true-corpus/clean/samples/`:

- **`sample_FakeBr.csv`** — 3 articles without an embedded title (pure
  generation, in different categories) and 2 that carry the original headline
  on the first line, serving as a reference to compare with what the agent
  wrote.
- **`sample_FakeTrueBr.csv`** — 3 articles in the degraded lower-case text and
  2 that kept the original casing, to measure what the degradation costs.

```bash
uv run fakegen headline --input true-corpus/clean/samples/sample_FakeBr.csv \
    --model anthropic/claude-sonnet-5 --out-dir data/calibracao
uv run fakegen headline --input true-corpus/clean/samples/sample_FakeTrueBr.csv \
    --model anthropic/claude-sonnet-5 --out-dir data/calibracao
```

### Paper replication (`fakegen paper`)

A side experiment to the pipeline: it reproduces the method of Silva et al. as
published — the **whole** true article goes in, and the prompt is the original
Portuguese one, byte for byte (`PAPER_ARTICLE_PROMPT`), with no headline stage.
Only the model changes. The sample is stratified (10 articles from each
corpus, seed 42) and the output goes to `corpus/paper_replication/`. Unlike the
other subcommands, `paper` sends no temperature and no token limit unless they
are passed on the command line (the authors' script set none either). See
[corpus/README.md](corpus/README.md#paper-replication) and the run log in
[corpus/paper_replication/NOTES.md](corpus/paper_replication/NOTES.md).

```bash
uv run python scripts/sample_paper_replication.py --per-source 10 --seed 42
uv run fakegen paper --input true-corpus/clean/paper_replication/FakeBr_true.csv \
    --id-field uid --model openai/gpt-4.1-mini-2025-04-14 \
    --out-dir corpus/paper_replication
```

For the linguistic analyses (`expanded_fake_news_corpus.analysis.*`), the
experiment is selected with `--experiment paper_replication`; the output goes to
`data/analysis/paper_replication/<module>/`, without touching the pipeline's
`data/analysis/<module>/`. The results are summarised in the NOTES.md above.

### Running the analyses

The orchestrator `expanded_fake_news_corpus.analysis` (also installed as
`fakegen-analysis`) runs every analysis in the catalogue in sequence, or only
those requested with `--analysis`; the corpus options are forwarded to each
module, which remains runnable on its own with its own options.

```bash
uv run fakegen-analysis --list                                   # catalogue
uv run fakegen-analysis --experiment paper_replication \
    --model openai/gpt-4.1-mini-2025-04-14                       # all
uv run fakegen-analysis --analysis liwc --analysis sage          # only these
uv run python -m expanded_fake_news_corpus.analysis.liwc --dictionary x.dic  # one, with its own option
```

A failing analysis does not stop the others: the error goes to the log and to
the exit code. Those that depend on the parsed corpus (`grammar_rules`,
`eud_rules`) are skipped with a warning when `data/parsed/<experiment>/` does
not exist. To add an analysis, the module exposes `main(argv)` like the others
and takes one line in the `ANALYSES` catalogue in
[analysis/runner.py](src/expanded_fake_news_corpus/analysis/runner.py).

### Syntactic parsing and dependency rules

The `pos` (UPOS distribution), `grammar_rules` (basic-tree rules) and
`eud_rules` (*enhanced*-edge rules) analyses read CoNLL-U from
`data/parsed/<experiment>/`, produced once by the
chain of the prior work — portSentencer → portTokenizer → LatinPipe with the
Portparser v2 model → post-processing — and enriched with Enhanced UD by Grew:

```bash
M=openai/gpt-4.1-mini-2025-04-14
uv run python -m expanded_fake_news_corpus.parsing.install_tools   # clones the tools into tools/ and downloads the model (1.6 GB)
uv run python -m expanded_fake_news_corpus.parsing.portparser --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.parsing.eud        --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.analysis.grammar_rules --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.analysis.eud_rules     --experiment paper_replication --model $M
```

`tools/` stays out of git (third-party repositories and the model); the chain's
code lives in [parsing/](src/expanded_fake_news_corpus/parsing/), including the
text treatment before the sentencer (`preprocess.py`: line breaks as
boundaries, punkt for the lower-cased FakeTrueBR). The parser runs on CPU (its
own venv, Python 3.11) and takes about 10 minutes for 40 documents; `grew` must
be on the PATH (installed via opam). The EUD rule set comes from
[eud-portugues](https://github.com/alvelvis/eud-portugues), vendored unchanged
in `resources/eud/`.

### Page (GitHub Pages)

`docs/` is published at <https://pedrest15.github.io/expanded_fake_news_corpus/> and shows
**the paper replication**: `index.html` browses the synthetic fake news (only
the link and metadata of the source article, never its text) and
`analysis.html` shows the linguistic characterisation. Both pages carry the
generators that produced the whole sample with no refusal — currently
gpt-4.1-mini and Claude Sonnet 4.5 — one at a time: a *Modelo* filter on the
corpus page, a *Gerador* selector on the characterisation page. The human side
is the same 20 documents everywhere, so the two readings are comparable.

The data comes from two scripts; run them after generating and analysing:

```bash
uv run python scripts/build_site.py \
    --model openai/gpt-4.1-mini-2025-04-14 \
    --model anthropic/claude-sonnet-4-5-20250929
uv run python scripts/build_analysis_data.py   # generators in DEFAULT_GENERATORS
```

`build_site.py` discards refusals (records without the paper's tags), and
`--model` leaves the GPT-5.1 and Claude Sonnet 5 folders out — both refused
most of the sample, so what they did produce is self-selected. A new generator
is added to the corpus page with one more `--model`, and to the
characterisation page with one entry in `DEFAULT_GENERATORS` in
[build_analysis_data.py](scripts/build_analysis_data.py) (model string and the
folder of its tables) or a `--generator modelo=pasta` on the command line. The
headline-pipeline data (round 1) is no longer on the page; it remains in
`corpus/` and `data/analysis/`.

### Output

One JSON line per article:

```json
{
  "headline": "Ministério da Saúde amplia campanha de vacinação em São Paulo",
  "rationale": "O anúncio da ampliação é o fato central do texto.",
  "raw_headline": "\"Ministério da Saúde amplia campanha de vacinação em São Paulo.\"",
  "source_id": "1042",
  "model": "anthropic/claude-sonnet-5",
  "warnings": []
}
```

### As a library

```python
from expanded_fake_news_corpus.agents import HeadlineAgent

agent = HeadlineAgent()                      # configuration from the environment
result = agent.generate(text, source_id="1042")
print(result.headline, result.warnings)
```

## Development

```bash
uv run ruff check src && uv run ruff format --check src
```

## Layout

| File | Contents |
|---|---|
| [agents/headline.py](src/expanded_fake_news_corpus/agents/headline.py) | Headline graph and the `HeadlineAgent` class |
| [prompts.py](src/expanded_fake_news_corpus/prompts.py) | Agent prompts |
| [config.py](src/expanded_fake_news_corpus/config.py) | Provider/model selection and keys |
| [text.py](src/expanded_fake_news_corpus/text.py) | Normalisation of the input text and of the headline |
| [corpus.py](src/expanded_fake_news_corpus/corpus.py) | Reading the corpora and writing the JSONL |
| [cli.py](src/expanded_fake_news_corpus/cli.py) | The `fakegen` command |
| [agents/paper_replication.py](src/expanded_fake_news_corpus/agents/paper_replication.py) | Paper replication: whole article → fake news with the original prompt |
| [scripts/sample_paper_replication.py](scripts/sample_paper_replication.py) | Stratified sample (10 + 10, seed 42) for the replication |
| [analysis/runner.py](src/expanded_fake_news_corpus/analysis/runner.py) | Orchestrator: analysis catalogue and batch execution (`fakegen-analysis`) |
| [analysis/cleaning.py](src/expanded_fake_news_corpus/analysis/cleaning.py) | Fake.br cleaning rules (from the prior work's `adapt_fake.py`) |
| [analysis/conllu.py](src/expanded_fake_news_corpus/analysis/conllu.py) | Reading CoNLL-U and locating the parsed corpus |
| [analysis/pos.py](src/expanded_fake_news_corpus/analysis/pos.py) | UPOS distribution: pooled frequencies, χ²/Cramér's V, per-tag tests |
| [analysis/grammar_rules.py](src/expanded_fake_news_corpus/analysis/grammar_rules.py) | Dependency rules: productivity, frequencies, discriminative TF-IDF |
| [analysis/eud_rules.py](src/expanded_fake_news_corpus/analysis/eud_rules.py) | The same over the *enhanced* edges (EUD) |
| [parsing/preprocess.py](src/expanded_fake_news_corpus/parsing/preprocess.py) | Text treatment before the sentencer and realignment after the tokenizer |
| [parsing/portparser.py](src/expanded_fake_news_corpus/parsing/portparser.py) | Portparser v2 chain → `data/parsed/` |
| [parsing/eud.py](src/expanded_fake_news_corpus/parsing/eud.py) | EUD enrichment with Grew |
| [parsing/install_tools.py](src/expanded_fake_news_corpus/parsing/install_tools.py) | Installation of the tools into `tools/` |

## Notes

- **FakeTrueBR has no published original version.** The corpus is distributed
  only as `FakeTrueBr_corpus.csv`, with the text already lower-cased and missing
  part of its punctuation — there is no folder of raw texts in the official
  repository. Recovering the original would require re-collecting the articles
  from `link_t` (G1, Folha). Until then, the headlines generated for that corpus
  start from degraded text, which is worth recording in the description of
  FakeGen.BR.
- The human Fake.br texts are cleaned before analysis with the rules of the
  prior work's `adapt_fake.py` ([analysis/cleaning.py](src/expanded_fake_news_corpus/analysis/cleaning.py):
  character whitelist, broken lines joined, spacing), minus two defects of the
  original script — it dropped the first line of a joined pair and deleted
  non-breaking spaces. The loader logs how many texts changed.
- Some Fake.br files carry the title on the first line of the text. The prompt
  instructs the model to ignore it and write the headline from the body of the
  article, but this is worth checking in the initial calibration with `--limit`.
- The paper [2025S1_JCBS_fakeNews_LLM_final.pdf](2025S1_JCBS_fakeNews_LLM_final.pdf)
  in this repository is the group's prior work that motivates FakeGen.BR.

[fakebr]: https://github.com/roneysco/Fake.br-Corpus
[faketruebr]: https://github.com/Chavarro/FakeTrueBr
[langgraphlib]: https://github.com/Pedrest15/LangGraphLib
