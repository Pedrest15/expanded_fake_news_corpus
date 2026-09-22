# Paper replication — run log

Record of the generation runs for the paper-replication experiment (Silva et
al.'s prompt, verbatim, full true article as input). See `../README.md`
("Paper replication") for the setup and `manifest.json` for the 20 drawn
articles (10 Fake.br + 10 FakeTrueBR, seed 42).

## 2026-09-14 — GPT-5.1 refuses most of the prompt

**Model:** `openai/gpt-5.1-2025-11-13`
**Sampling:** none sent (no temperature, no max_tokens — provider defaults, as
in the authors' script)
**Prompt:** `PAPER_ARTICLE_PROMPT` + system persona, `prompt_sha256` `795f6b38a9560e9c`
**Output:** `openai/gpt-5.1-2025-11-13/` (kept as evidence; not corpus material)

| Corpus | Generated | Refused |
|---|---|---|
| Fake.br | 2 (`fakebr:104`, `fakebr:917`) | 8 |
| FakeTrueBR | 5 (`faketruebr:1773`, `215`, `74`, `230`, `509`) | 5 |
| **Total** | **7 / 20** | **13 / 20 (65%)** |

Refusals are plain-text replies without the `<syntheticText>`/`<changes>`
tags. They are stored as returned, so a refused record has the refusal text in
`synthetic_text` and `raw_response`, an empty `changes`, and `warnings`
containing `response without format tags` and `empty <changes> section`.
Filter on `warnings` to separate them.

Typical wording (all in Portuguese):

> Não posso atender a esse pedido. Mesmo com objetivo acadêmico, transformar
> uma notícia real em uma peça de desinformação atraente e "realista" é
> justamente o […]

> Não posso ajudar a criar ou aprimorar fake news, mesmo para fins acadêmicos
> ou de pesquisa. Posso, no entanto: analisar a notícia original e apontar
> quais […]

The model explicitly rejects the academic framing that the prompt relies on
("mesmo para fins acadêmicos"). Several refusals offer an alternative
(analysing the original article's vulnerabilities to manipulation) instead of
producing the text.

Observations:

- Refusal rate is higher on Fake.br (8/10) than on FakeTrueBR (5/10). The
  Fake.br articles are longer, mostly political, and name real public figures;
  the FakeTrueBR ones are fact-checks whose subject is already a rumour.
- The 7 texts that were generated are long — 650 to 1,750 words — well above
  the ~200–450 words gpt-4.1-mini produced for the same prompt, and far above
  the human fake news in the source corpora.
- No parameters were tuned and no article was retried: with provider-default
  temperature the outcome is non-deterministic, and retrying only the refusals
  would bias the sample.

Conclusion: the prompt that worked on Sabiá-3 in the original paper is not
sufficient on GPT-5.1, whose safety alignment treats "create a realistic fake
news story for research" as a request to decline. GPT-5.1 was therefore not
used for the replication; the run was redone with `gpt-4.1-mini-2025-04-14`,
which had produced 20/20 in an earlier trial (temperature 0, max_tokens 8192;
that output was discarded so the final run uses provider defaults).

## 2026-09-14 — gpt-4.1-mini, provider defaults

**Model:** `openai/gpt-4.1-mini-2025-04-14`
**Sampling:** none sent (no temperature, no max_tokens)
**Output:** `openai/gpt-4.1-mini-2025-04-14/`

| Corpus | Generated | Refused | Words (synthetic text) |
|---|---|---|---|
| Fake.br | 10 | 0 | 344–496 |
| FakeTrueBR | 10 | 0 | 251–403 |

No refusals. Five records (`fakebr:3041`, `1129`, `1006`, `573`,
`faketruebr:70`) carry the warning `unclosed <changes> tag`: the model ended
its reply without the closing tag. The `changes` section is complete in every
case (each response ends on a full closing sentence of the explanation, not
mid-sentence), so nothing was lost — the parser takes everything after
`<changes>`. Counts and timestamps are in the `.meta.json` next to each JSONL.

## 2026-09-14 — linguistic analyses on the gpt-4.1-mini output

All five analysis modules were run on the 20 pairs (synthetic vs. the human
fake news of the same `uid`), restricted to `--model openai/gpt-4.1-mini-2025-04-14`
so the GPT-5.1 refusals kept in the sibling folder are excluded:

```bash
for m in syllables lexical_diversity zipf sage liwc; do
  uv run python -m expanded_fake_news_corpus.analysis.$m \
      --experiment paper_replication --model openai/gpt-4.1-mini-2025-04-14
done
```

Tables and plots: `data/analysis/paper_replication/<module>/`. The pipeline's
tables stay in `data/analysis/<module>/`.

Headline numbers (human vs. machine, n = 20 + 20; d = Cohen's d, positive
means higher on the human side):

| Measure | Human | Machine | d | Pipeline (round 1) for comparison |
|---|---|---|---|---|
| Syllables per word | 2.20 | 2.51 | −2.4 | 2.19 vs 2.51, d = −2.2 |
| Syllables per sentence | 43.6 | 78.2 | −2.2 | 40.9 vs 72.9, d = −2.8 |
| MATTR (window 50) | 0.813 | 0.874 | −2.0 | 0.829 vs 0.857, d = −0.9 |
| Tokens per document (Zipf, with stopwords) | 175 | 372 | — | 181 vs 258 |
| LIWC categories significant at FDR 5% | 10 / 74 | | | 1 / 74 (`focuspast`) |

- Syllables reproduce the pipeline result almost exactly: the machine writes
  longer words and sentences roughly twice as long, in both source corpora.
- The machine texts are ~2.1× longer than the human ones (Silva et al.
  reported 1.8× for Sabiá-3), against ~1.4× in the headline-seeded pipeline —
  giving the model the full article to "modify" makes it keep most of it.
- Lexical diversity: MATTR separates the groups much more strongly here
  (d = −2.0 vs −0.9); the raw type-token ratio does not (it is confounded by
  length, as expected).
- LIWC: `space`, `adj`, `cause`, `work`, `drives`, `power` higher for the
  machine; `pronoun`, `ppron`, `verb`, `auxverb` higher for humans. The
  pipeline run found only `focuspast` (human) significant — with the same
  n, so this is a genuinely stronger stylistic separation, not more power.
- SAGE / Zipf: the machine side is dominated by hedged-attribution and
  institutional vocabulary (`além disso`, `fontes internas`, `especialistas`,
  `afirmam`, `tribunal`, `investigação`, `documentos`); the human side by
  spoken-register items (`vai`, `você`, `só`, `muito`, `hoje`, `que ele`).
  `fontes`/`especialistas`/`afirmam` are top machine terms in both
  experiments — a prompt-independent signature of the model.

The GitHub Pages site (`docs/`) now publishes this experiment only —
`scripts/build_site.py` and `scripts/build_analysis_data.py` read from
`corpus/paper_replication/` and `data/analysis/paper_replication/`; the
headline-pipeline (round 1) data was removed from the page.

## 2026-09-14 — syntactic parsing and dependency-rule analyses

The corpus was parsed with the same chain as the PROPOR 2026 characterisation
work: portSentencer → portTokenizer → LatinPipe with the **Portparser v2**
model (BERTimbau) → Portparser.v2 post-processing of lemmas and features
(`expanded_fake_news_corpus.parsing.install_tools` and `.portparser`). One CoNLL-U per
document lives in `data/parsed/paper_replication/<group>/`, with the tool
commits and model size in `manifest.json`. Enhanced UD was then added with
Grew 1.19 and the vendored `eud-portugues` rule set (`parsing.eud`,
`<stem>.eud.conllu`).

Three preparation details that matter for the numbers:

- Line breaks are hard sentence boundaries. portSentencer ignores them, so the
  headline (first line, no final period) would otherwise be glued to the first
  sentence of the body; lines are only joined when the previous one has no
  final punctuation and the next starts in lower case (the `adapt_fake.py`
  heuristic of the prior work).
- **The FakeTrueBR human side is distributed entirely in lower case**, and
  portSentencer only closes a sentence when the next word is capitalised — on
  that text it produced whole paragraphs as single "sentences" (26 for 10
  documents, ~90 words each), and the lower-case join heuristic glued the
  headline to the body. The prior work built its FakeTrueBR human files from
  the same CSV without treatment and had the same defect. Texts with no upper
  case are now pre-segmented with NLTK punkt (case-insensitive) before the
  sentencer and exempt from the join rule; the FakeTrueBR human side went from
  26 to 80 sentences, in line with Fake.br (21.8 vs 21.4 words per sentence).
- The tokenizer drops sentences without letters (`….`) and, with `-m`, strips
  unpaired quotes; documents are re-split by aligning texts without
  punctuation, not by counting.

Sentences: 224 human, 261 machine.

`grammar_rules` (basic tree, `UPOS(dep/rel, *, dep/rel)` rules; d = human −
machine, Mann-Whitney p):

| Measure | Human | Machine | d | p | Fake.br d | FakeTrueBR d |
|---|---|---|---|---|---|---|
| Rules per sentence | 29.9 | 48.9 | −2.3 | <0.001 | −2.4 | −2.0 |
| Distinct rules per sentence | 16.4 | 23.4 | −2.5 | <0.001 | −2.5 | −2.4 |
| Rule diversity (distinct / total, per document) | 0.269 | 0.207 | +1.8 | <0.001 | +1.9 | +2.0 |
| Words per sentence | 21.6 | 35.2 | −2.2 | <0.001 | −2.4 | −2.0 |

This reproduces the PROPOR finding on a different generator (gpt-4.1-mini
instead of Sabiá-3) and on both source corpora: the machine packs more rules
into each sentence and repeats them more.

TF-IDF over rules (154 rules present in ≥ 5 documents): 27 significant after
FDR — 25 on the machine side, 2 on the human side (the root leaves `*(VERB)`
and `*(NOUN)`, i.e. more sentences per document, hence more roots). The
strongest machine rule is the adjective leaf `ADJ(*)` (d = −2.9), the "more
adjectival modifiers" result of the prior work; then `NOUN(ADP/case, *,
ADJ/amod)`, `NOUN(ADP/case, *, VERB/acl)`, `VERB(PRON/nsubj, *, VERB/xcomp)`,
`VERB(NOUN/nsubj, *, VERB/ccomp, PUNCT/punct)`.

`eud_rules` (rules from enhanced edges that differ from the basic tree — case
markers folded into the relation, propagated subjects, `ref` in relatives):
9.0 vs 15.3 enhanced rules per sentence (d = −1.9), diversity 0.383 vs 0.300
(d = +1.4), consistent across sources. 70 rules tested, 10 significant, all
machine: `NOUN(*, PRON/ref)` (relative clauses on nouns), `ADV(*, NOUN/obl:de)`
("além de", "apesar de"), `NOUN(*, NOUN/nmod:sobre)`, `VERB(*, NOUN/obj)`.

With n = 20 + 20 these are calibration numbers for the pipeline, not
publishable effects; the prior results were computed on thousands of pairs.

## 2026-09-15 — Claude Sonnet 5 refuses half of the prompt

**Model:** `anthropic/claude-sonnet-5` (Anthropic API, through `langchain-anthropic`)
**Sampling:** none sent (no temperature, no max_tokens — provider defaults, as
in the authors' script). Note that Sonnet 5 *rejects* `temperature`/`top_p`/
`top_k` with HTTP 400, so "provider defaults" is the only option on this model
(`config.sampling_support` now drops them with a warning instead of failing
the request). Adaptive thinking is on by default on this model and was left
as is; the response's `thinking` block comes back empty (`display` omitted)
and only the `text` block is stored.
**Prompt:** `PAPER_ARTICLE_PROMPT` + system persona, `prompt_sha256` `795f6b38a9560e9c`
**Output:** `anthropic/claude-sonnet-5/` (kept as evidence; not corpus material
until the refusals are dealt with)

| Corpus | Generated | Refused | Words (synthetic text) |
|---|---|---|---|
| Fake.br | 5 (`fakebr:1129`, `917`, `573`, `3020`, `421`) | 5 (`2623`, `458`, `104`, `3041`, `1006`) | 386–575 |
| FakeTrueBR | 5 (`faketruebr:1773`, `215`, `1587`, `509`, `542`) | 5 (`1429`, `1043`, `74`, `70`, `230`) | 286–438 |
| **Total** | **10 / 20** | **10 / 20 (50%)** | |

Refusals have the same shape as GPT-5.1's: plain text without the
`<syntheticText>`/`<changes>` tags, stored as returned, with `warnings`
containing `response without format tags` (filter on it). Typical wording:

> Não posso ajudar a criar essa desinformação, mesmo em contexto
> declaradamente acadêmico. […]

> Não posso criar essa desinformação, mesmo em contexto de pesquisa. Explico o
> porquê: gerar uma notícia falsa completa e […]

Every refusal names the academic framing explicitly and rejects it, as
GPT-5.1 did; most go on to explain the reasoning and offer an alternative
(analysing the article, producing an annotated example, etc.).

Observations:

- Unlike GPT-5.1 (8/10 Fake.br vs 5/10 FakeTrueBR), the refusal rate is the
  same on both corpora. The overlap with GPT-5.1's decisions is partial:
  `fakebr:917`, `faketruebr:1773`, `215`, `509` were generated by both;
  `fakebr:104` was generated by GPT-5.1 and refused here; `fakebr:1129`,
  `573`, `3020`, `421`, `faketruebr:1587`, `542` the other way round.
- The generated texts are 286–575 words: in the range of gpt-4.1-mini
  (251–496), not of GPT-5.1 (650–1,750).
- The 10 generated records parse cleanly — no unclosed-tag warnings, non-empty
  `<changes>` in every case.
- No article was retried, for the reason given in the GPT-5.1 entry.

Conclusion: as with GPT-5.1, the verbatim prompt does not get a full sample
out of Claude Sonnet 5. The 10 generated texts are usable as a partial sample
if a comparison across generators is wanted, but a 50% refusal rate makes the
sample self-selected (the model decides which stories it will falsify), so
they are not used for the replication tables.

Reproduce:

```bash
for f in FakeBr_true FakeTrueBr_true; do
  uv run fakegen paper --input true-corpus/clean/paper_replication/$f.csv \
      --id-field uid --model anthropic/claude-sonnet-5 \
      --out-dir corpus/paper_replication --concurrency 4
done
```

## 2026-09-15 — Claude Sonnet 4.5, provider defaults: 20/20

**Model:** `anthropic/claude-sonnet-4-5-20250929` (dated snapshot, as with
gpt-4.1-mini)
**Sampling:** none sent (no temperature, no max_tokens; `langchain-anthropic`
fills `max_tokens` with the model's 64k output cap, which is never reached).
No extended thinking on this model by default.
**Prompt:** `PAPER_ARTICLE_PROMPT` + system persona, `prompt_sha256` `795f6b38a9560e9c`
**Output:** `anthropic/claude-sonnet-4-5-20250929/`

| Corpus | Generated | Refused | Words (synthetic text) |
|---|---|---|---|
| Fake.br | 10 | 0 | 274–761 |
| FakeTrueBR | 10 | 0 | 197–529 |

No refusals and no parser warnings: every reply carries both tags, closed,
with a non-empty `<changes>` section. This is the second generator (after
gpt-4.1-mini) to give a full sample under the verbatim prompt, and the first
from Anthropic — the refusals on Sonnet 5 (previous entry) are therefore a
matter of the newer model's alignment, not of the prompt or the API path.

Length: the Fake.br side is somewhat more spread than gpt-4.1-mini's
(274–761 vs 344–496 words; `fakebr:1129` and `573` above 700), the
FakeTrueBR side is in the same range (197–529 vs 251–403). The `changes`
sections are numbered lists of the manipulations applied ("Título
sensacionalista", "Adição de teoria conspiratória", …), like gpt-4.1-mini's.

The analyses were not rerun on this output yet; when they are, restrict them
with `--model anthropic/claude-sonnet-4-5-20250929` so the Sonnet 5 and
GPT-5.1 refusal folders stay out.

Reproduce:

```bash
for f in FakeBr_true FakeTrueBr_true; do
  uv run fakegen paper --input true-corpus/clean/paper_replication/$f.csv \
      --id-field uid --model anthropic/claude-sonnet-4-5-20250929 \
      --out-dir corpus/paper_replication --concurrency 4
done
```

## 2026-09-22 — analyses on the Claude Sonnet 4.5 output

All eight modules were run on the 20 pairs of the
`anthropic/claude-sonnet-4-5-20250929` run. To keep the gpt-4.1-mini tables
(which feed `docs/`) intact, this run writes to sibling paths with the model
in the name:

```bash
M=anthropic/claude-sonnet-4-5-20250929
OUT=data/analysis/paper_replication/claude-sonnet-4-5-20250929
R=data/parsed/claude-sonnet-4-5-20250929

for a in syllables lexical_diversity zipf sage liwc; do
  uv run python -m expanded_fake_news_corpus.analysis.$a \
      --experiment paper_replication --model $M --output-dir $OUT/$a
done

# the human side is generator-independent: its CoNLL-U was copied from
# data/parsed/paper_replication/human, so only the 20 machine texts were parsed
uv run python -m expanded_fake_news_corpus.parsing.portparser \
    --experiment paper_replication --model $M --parsed-root $R
uv run python -m expanded_fake_news_corpus.parsing.eud \
    --experiment paper_replication --model $M --parsed-root $R

for a in pos grammar_rules eud_rules; do
  uv run python -m expanded_fake_news_corpus.analysis.$a \
      --experiment paper_replication --model $M --parsed-root $R --output-dir $OUT/$a
done
```

Sentences: 224 human, **405** machine (gpt-4.1-mini: 261). Same total length
(~2.2× the human side, 7,713 tokens vs 7,442), split into many more, shorter
sentences.

Headline numbers, with the gpt-4.1-mini column for comparison (d = Cohen's d,
negative means higher on the machine side; the human column is identical in
both, same 20 texts):

| Measure | Human | Sonnet 4.5 | d | gpt-4.1-mini | d |
|---|---|---|---|---|---|
| Syllables per word | 2.20 | 2.43 | −1.88 | 2.51 | −2.38 |
| Syllables per sentence | 43.6 | 55.5 | −0.81 | 78.2 | −2.19 |
| MATTR (window 50) | 0.813 | 0.873 | −2.04 | 0.874 | −2.02 |
| Words per sentence | 21.6 | 24.4 | −0.45 | 35.2 | −2.21 |
| Rules per sentence | 29.9 | 33.9 | −0.46 | 48.9 | −2.27 |
| Distinct rules per sentence | 16.4 | 18.3 | −0.64 | 23.4 | −2.50 |
| Rule diversity (per document) | 0.269 | 0.216 | +1.43 | 0.207 | +1.80 |
| EUD rules per sentence | 9.0 | 10.1 | −0.33 | 15.4 | −1.95 |
| EUD rule diversity | 0.383 | 0.317 | +1.01 | 0.300 | +1.38 |
| ADJ (% of tokens) | 4.2 | 6.7 | −1.57 | 8.9 | −2.71 |
| LIWC categories significant (p < 0.05) | | 17 / 74 | | 19 / 74 | |
| — of those, surviving FDR | | 0 / 74 | | 10 / 74 | |
| Rule TF-IDF significant after FDR | | 15 / 155 (all machine) | | 27 / 154 (25 machine) | |
| EUD rule TF-IDF significant | | 7 / 77 (all machine) | | 10 / 70 (all machine) | |

The picture is the same one, weaker almost everywhere — with one exception.

- **Sentence length is where the two generators diverge.** Sonnet 4.5 writes
  24-word sentences against gpt-4.1-mini's 35 (human: 21.6). Every measure
  that is a function of sentence length follows it down: syllables per
  sentence, rules per sentence, EUD rules per sentence all drop from d ≈ −2
  to d ≈ −0.3/−0.8, and words per sentence and rules per sentence no longer
  reach significance on their own (p = 0.06, Mann-Whitney). The PROPOR
  finding "the machine packs more rules into each sentence" does **not**
  reproduce on this generator.
- **Rule diversity survives**: 0.269 vs 0.216 per document (d = +1.43,
  p = 1.4e-04), and the EUD version too (d = +1.01). This is the repetition
  result, and it is independent of sentence length — Sonnet 4.5 repeats its
  constructions almost as much as gpt-4.1-mini does, in shorter sentences.
- **MATTR is unchanged**: 0.873 vs 0.874, d = −2.04 vs −2.02. The strongest
  and most generator-stable separation in the whole set.
- **ADJ is still the single significant UPOS** (4.2% vs 6.7%, d = −1.57,
  q = 1e-03), matching `ADJ(*)` as the top machine rule in the TF-IDF (d =
  −1.94) — but at roughly half the effect. The pooled χ² is also weaker
  (Cramér's V 0.074 vs 0.122), and, as with gpt-4.1-mini, FakeTrueBR
  separates better than Fake.br (0.158 vs 0.065).
- **LIWC depends entirely on which criterion is used.** By the raw p < 0.05
  of the mirrored pipeline (see the entry below on the criterion), the two
  generators are nearly indistinguishable: 17 categories here against 19 for
  gpt-4.1-mini, the same families on both lists (`adj`, `cause`, `work`,
  `space`, `ppron`, `power`, `informal`, `relativ`). Under FDR, gpt-4.1-mini
  keeps 10 and Sonnet 4.5 keeps none — its p-values sit just the wrong side
  of the BH line (smallest q: `adj` 0.062, `informal` 0.081,
  `cause`/`work`/`compare`/`tentat` 0.132). So "LIWC separates the groups for
  one generator and not the other" is an artefact of the threshold, not a
  difference in the signal: the effects point the same way in both, a little
  smaller here (`adj` −1.26 vs −1.39).
- **The rules that remain significant are largely the same ones**: `ADJ(*)`
  first in both, then the nominal-modification family (`NOUN(ADP/case, *,
  ADJ/amod, NOUN/nmod)`, `VERB(PRON/nsubj, *, VERB/xcomp)`), and, on the
  enhanced side, `NOUN(*, PRON/ref)` (relative clauses) first in both,
  followed by `ADV(*, NOUN/obl:de)` and `NOUN(*, NOUN/nmod:sobre)`. Sonnet
  4.5 adds a proper-noun group the other run did not have (`PROPN(*,
  PUNCT/punct)` d = −1.84, `*(PROPN)` d = −1.09, `NOUN(*, PROPN/nmod:em)`):
  it names more institutions and places.
- **SAGE** shows the clearest qualitative difference between the two
  generators. gpt-4.1-mini's machine side was institutional-hedged
  (`além disso`, `tribunal`, `fontes internas`, `investigação`); Sonnet 4.5's
  is tabloid (`urgente`, `revelam`, `documentos obtidos`, `com
  exclusividade`, `compartilhe`, `antes que`). `especialistas` is a top
  machine term in both, as it was for GPT-5.1 — the prompt-independent
  signature noted above survives a change of vendor.

Reading: with n = 20 these remain calibration numbers. What they suggest is
that the two robust effects across generators are **lexical diversity
(MATTR)** and **rule repetition (rule diversity)**, while everything driven
by sentence length is a property of the particular generator and not of
machine-written fake news as such.

## 2026-09-22 — LIWC now mirrors the semantics repository

`analysis/liwc.py` no longer uses the shared `significance.py` for the
per-category comparison. It reproduces `liwc/analyzer.py` of
`noticias_falsas_humano_maquina_semantica`, so that the LIWC numbers of the
two studies are computed and read the same way. What changed:

| | before | now |
|---|---|---|
| Normality / parametric test | Shapiro-Wilk + t-test reported | not computed |
| Rank test | Mann-Whitney (two-sided) | unchanged |
| Cohen's d | pooled SD weighted by `n-1` (`ddof=1`) | `sqrt((σ_h² + σ_m²) / 2)` with `ddof=0` |
| Headline criterion | `q < 0.05` (Benjamini-Hochberg) | `p < 0.05` (`significant_005`, plus `significant_001`) |
| New columns | — | `difference`, `abs_difference`, `characteristic_of` |

Scoring was already identical on both sides and was not touched: same `.dic`
parsing (two `%` sections, exact match before longest prefix), same `\w+`
tokenisation over the lower-cased text, same per-document percentage
`100 × count / total_words`, same `dictionary_coverage`, same skipping of
empty documents. The mirrored statistics were checked term by term against a
literal re-implementation of `liwc/analyzer.py` — means, SDs, difference, U,
p, d and the two flags agree exactly.

Two deliberate departures, both outside the method:

- **Column names stay ours** (`metric`, `human_*`, `machine_*`,
  `u_statistic`, `u_p_value`) instead of theirs (`category`, `mean_human`,
  `mean_llm`, `mann_whitney_u`, `p_value`), so the LIWC table keeps the same
  shape as the other seven modules and `scripts/build_analysis_data.py` keeps
  working. `machine` is used where they write `llm`, including in
  `characteristic_of`.
- **`q_value` and `significant_fdr_005` stay in the table** as trailing
  informative columns — the page in `docs/` publishes "X with p < 0.05 · Y
  after FDR correction" — but they are no longer the criterion.

Effect on the existing tables: the p-values and the ranking are unchanged
(same test, same ordering by |d|); every |d| grew by exactly
`sqrt(n / (n-1))` = 2.60% with n = 20, which moves no category across a
Cohen threshold. What changes is the reported headline: 19 / 74 for
gpt-4.1-mini and 17 / 74 for Sonnet 4.5, against 10 and 0 under FDR.

With n = 20 and 74 tests, ~4 of those categories are expected false
positives; the earlier entries' FDR counts remain in the tables for whoever
wants the conservative reading.

Regenerated: `data/analysis/paper_replication/liwc/` (gpt-4.1-mini) and
`data/analysis/paper_replication/claude-sonnet-4-5-20250929/liwc/`, plus
`docs/analysis.json`. The page still highlights the FDR-surviving rows.

## 2026-09-22 — the site now publishes both generators

`docs/` carries the two runs that produced the whole sample without refusing —
gpt-4.1-mini and Claude Sonnet 4.5 — one at a time:

- **`index.html`** (corpus browser): `build_site.py` is now given both models,
  so the *Modelo* filter has two entries and the browser holds 40 records (20
  source articles × 2 generators). The top panel used to be computed at build
  time and no longer matched the filters once there was more than one
  generator; it is now recomputed in the page from the visible records, so
  picking a model updates the counts and the median lengths with it. The human
  medians still come from the build — they are a property of the source
  corpora, not of the selection.
- **`analysis.html`** (characterisation): `analysis.json` changed shape. It
  used to hold one set of sections; it now holds `generators[]`, one block per
  model with its own `syllables`, `lexical_diversity`, `liwc`, `sage` and
  `zipf`, and the page has a *Gerador* selector that re-renders everything.
  The human side is identical in every block, so switching generator switches
  only the machine side.

`build_analysis_data.py` takes the published generators from
`DEFAULT_GENERATORS` (model string plus the folder of its tables), overridable
with a repeatable `--generator modelo=pasta`. Adding a third generator is one
line there plus one more `--model` on `build_site.py`.

The GPT-5.1 and Claude Sonnet 5 folders stay off the site, as before: with 65%
and 50% refusal rates, what they produced is a sample the model selected for
itself.

Both pages were exercised in a headless DOM against the real `docs/*.json`
(selector options, re-render on switch, no leftover rows, panel following the
filter).

## 2026-09-15 — Fake.br cleaning, results regenerated, POS distribution

**Cleaning.** The human Fake.br side now goes through the four rules of the
prior work's `adapt_fake.py` before any analysis
(`expanded_fake_news_corpus.analysis.cleaning`): characters outside a keyboard
whitelist removed, lines broken mid-sentence joined, double spaces collapsed,
space before punctuation removed. Two deliberate departures from the original
script: it dropped the first line when "joining" a broken pair (a `continue`
before the write — text was lost in the 699 Fake.br files it flagged), and it
deleted non-breaking spaces, gluing the neighbouring words; here lines are
joined and exotic whitespace becomes a space. On the 10 Fake.br human texts of
this sample the effect is tiny — 10 characters removed (mis-decoded quotes in
`fakebr:104`), no joins, 9 texts with spacing normalised — and every table
above kept its values after regeneration (parsing, EUD and all analyses were
rerun; the significance tables are identical to three decimals). The cleaning
matters for the full corpus, where 72% of the files change.

**POS distribution** (`analysis/pos.py`, new in the `fakegen-analysis`
catalogue, reading UPOS from the Portparser CoNLL-U as the prior work read
Porttagger tags): the pooled tag × authorship table is not independent
(χ² = 198, 15 d.f., p < 0.001) but the effect is small (Cramér's V = 0.12;
0.12 Fake.br, 0.18 FakeTrueBR). Per document, only **ADJ** survives FDR: 4.2%
of tokens in human texts vs 8.9% in machine texts (d = −2.7) — the same
adjective signal as `ADJ(*)` in the rule analysis and `adj` in LIWC. The
largest pooled differences after ADJ are PRON (4.0% vs 1.9%, human),
PUNCT (11.2% vs 9.5%, human), PROPN (5.5% vs 6.8%, machine) and NUM
(2.3% vs 1.1%, human), none significant per document at n = 20.
