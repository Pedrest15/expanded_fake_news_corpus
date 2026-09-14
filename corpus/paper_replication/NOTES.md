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
