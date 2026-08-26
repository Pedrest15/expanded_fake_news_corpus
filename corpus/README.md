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
└── headlines/                 stage 1 output — corpus text
    └── round1/<provider>/<model>/
        ├── FakeBr_true.jsonl       + FakeBr_true.meta.json
        └── FakeTrueBr_true.jsonl   + FakeTrueBr_true.meta.json
```

Generated text belongs here; `../data/` is scratch space for experiments that
are not part of the corpus. Stage 2 output will land under `fake_news/` with the
same round/provider/model layout.

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

## Language

Code, identifiers, JSON keys and documentation are in English, for international
publication. **Prompts stay in Brazilian Portuguese** — they are the model-facing
text that produces Brazilian Portuguese news, so they are corpus content rather
than code.

## Prior work

This corpus extends Silva et al., *Fake News Detection in Portuguese Under Large
Language Model-Generated Content*. Their generation prompt is reproduced verbatim
in `../src/fakegen_br/prompts.py` (`PAPER_FAKE_PROMPT`), taken from their
published code at https://github.com/renatosvmor/fake-news-llm-ptbr rather than
back-translated from the paper's English rendering.
