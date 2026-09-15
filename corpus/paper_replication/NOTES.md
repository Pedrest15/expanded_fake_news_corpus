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
