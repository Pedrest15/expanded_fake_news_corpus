# FakeGen.BR

Data for the FakeGen.BR corpus: AI-generated fake news in Brazilian Portuguese,
seeded by headlines written from human-authored true news.

Everything under this directory is **corpus data and provenance** — the material
that gets published alongside the paper. Pipeline code lives in `../src`, and
the upstream corpora live in `../true-corpus`.

## Pipeline

```
true news  ->  [headline agent]  ->  headline  ->  [fake news writer]  ->  synthetic fake news
```

Source corpora, after extracting the true half and deduplicating:

| Source | Articles | Genre | Notes |
|---|---|---|---|
| Fake.br | 3,599 | `news` | Ordinary journalism, original full texts |
| FakeTrueBR | 1,403 | `factcheck` | Fact-checking articles; the relevant fact is the verdict |

The two genres take different prompt blocks. A fact-check headline states that a
claim is false, so the fake news generated from it asserts that claim as true —
which tends to reconstruct the rumour the agency debunked.

## Contents

```
corpus/
├── rounds/
│   └── manifest.json          seed, sizes and every drawn identifier
├── assignments/
│   └── round1.json            which model generates which block of 20
├── headlines/                 stage 1 output — corpus text
│   └── round1/<provider>/<model>/
│       ├── FakeBr_true.jsonl       + FakeBr_true.meta.json
│       └── FakeTrueBr_true.jsonl   + FakeTrueBr_true.meta.json
├── fake_news/                 stage 2 output, same layout
└── paper_replication/         side experiment, see below
    ├── manifest.json          seed, sizes and drawn identifiers
    └── <provider>/<model>/
        ├── FakeBr_true.jsonl       + FakeBr_true.meta.json
        └── FakeTrueBr_true.jsonl   + FakeTrueBr_true.meta.json
```

Generated text belongs here; `../data/` is scratch space for experiments that
are not part of the corpus.

Every JSONL is paired with a `.meta.json` carrying the run's provenance:
provider, pinned model snapshot, sampling parameters actually applied (and any
the provider dropped), genre, a hash of the prompt, input path, counts and
timestamp.

The per-round CSVs are **not** kept here: they restate the upstream article text
(~1.7 MB per set of three rounds) and are rebuildable from the manifest plus the
source corpora. They are written to `../true-corpus/clean/rounds/` as working
input instead.

### Selection

Articles are drawn from the **pooled** set of both corpora with no per-source
quota, so each round mirrors the corpora's relative sizes (~72% / 28%). Rounds
are disjoint by construction: the pool is shuffled once with a fixed seed and
sliced, so no article can appear twice.

Regenerate with:

```bash
uv run python scripts/sample_rounds.py --rounds 3 --size 100 --seed 42
```

### Identifiers

`id` 4 exists in both corpora, so every article carries a `uid` shaped as
`source:id` — `fakebr:4`, `faketruebr:1542`. For Fake.br the `id` is the file
name under `full_texts/true/`; for FakeTrueBR it is the row number in the
original CSV. Both point at the upstream corpus, not at an intermediate
position, so filtering never breaks traceability.

Generation runs must pass `--id-field uid` so the `source_id` recorded in the
output is the qualified form. A bare numeric id would be ambiguous once both
corpora are pooled.

### Model assignment

Each round of 100 articles is split into contiguous blocks of 20, one per
generating model, following the draw order recorded in `rounds/manifest.json` —
not the order of the round CSVs, which are grouped by source and would give a
block skewed to a single corpus.

Blocks are allocated by position and are stable: re-running the assignment with
a new model preserves the existing allocations and takes the next free block, so
adding a model never redistributes what was already generated.

```bash
uv run python scripts/assign_blocks.py --round 1 --model openai/gpt-4.1-mini-2025-04-14
```

| Round 1 | Block | Articles | Model |
|---|---|---|---|
| positions 0–19 | 0 | 14 Fake.br + 6 FakeTrueBR | `openai/gpt-4.1-mini-2025-04-14` |
| positions 20–99 | 1–4 | 20 each | not assigned yet |

## Paper replication

A side experiment that reproduces the generation method of Silva et al. **as
published**, so its output can be compared with the headline-seeded pipeline
above. It differs from the pipeline in two ways:

* the LLM receives the **full true news article**, not a headline, and is asked
  to modify it (no headline stage);
* the prompt is the authors' original, byte for byte (`PAPER_ARTICLE_PROMPT` in
  `../src/expanded_fake_news_corpus/prompts.py`), including the indentation and
  blank lines of the f-string in their published script. The system message is
  the same short persona. Nothing was adapted; only the model changed.

Selection is **stratified**, unlike the pooled rounds: 10 articles from each
corpus, drawn with one RNG seeded at 42 (Fake.br first, then FakeTrueBR). The
draw shares no article with round 1.

```bash
uv run python scripts/sample_paper_replication.py --per-source 10 --seed 42

fakegen paper --input true-corpus/clean/paper_replication/FakeBr_true.csv \
    --id-field uid --model openai/gpt-4.1-mini-2025-04-14 \
    --out-dir corpus/paper_replication
fakegen paper --input true-corpus/clean/paper_replication/FakeTrueBr_true.csv \
    --id-field uid --model openai/gpt-4.1-mini-2025-04-14 \
    --out-dir corpus/paper_replication
```

| Experiment | Articles | Model | Sampling |
|---|---|---|---|
| paper_replication | 10 Fake.br + 10 FakeTrueBR | `openai/gpt-4.1-mini-2025-04-14` | none sent (provider defaults) |

The authors' script set no sampling parameters (Maritaca defaults), and
`fakegen paper` follows suit: it sends no temperature and no token limit unless
`--temperature`/`--max-tokens` are given explicitly — the project default of
temperature 0 and `FAKEGEN_TEMPERATURE` do not apply to this subcommand. Output
records carry `synthetic_text` and `changes` as in the paper's tags, plus
`source_id`, `source_chars` (length of the article sent) and `raw_response`.

`paper_replication/NOTES.md` is the run log. It records that GPT-5.1
(`gpt-5.1-2025-11-13`) refused 13 of the 20 articles under this prompt and
Claude Sonnet 5 (`anthropic/claude-sonnet-5`) refused 10, which is why the
replication uses gpt-4.1-mini; both outputs are kept as evidence in their own
model folders. Claude Sonnet 4.5 (`anthropic/claude-sonnet-4-5-20250929`)
produced all 20 and is the second full sample.

## Language

Code, identifiers, JSON keys and documentation are in English, for international
publication. **Prompts stay in Brazilian Portuguese** — they are the model-facing
text that produces Brazilian Portuguese news, so they are corpus content rather
than code.

## Prior work

This corpus extends Silva et al., *Fake News Detection in Portuguese Under Large
Language Model-Generated Content*. Their generation prompt is taken from their
published code at https://github.com/renatosvmor/fake-news-llm-ptbr rather than
back-translated from the paper's English rendering, and lives in
`../src/expanded_fake_news_corpus/prompts.py` in two forms: `PAPER_ARTICLE_PROMPT`
(verbatim, full article as input — the paper replication) and
`PAPER_FAKE_PROMPT` (first sentence adapted to take a headline — stage 2 of the
pipeline).
