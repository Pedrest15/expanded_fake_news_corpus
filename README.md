# FakeGen.BR

Construção de um corpus de *fake news* em português brasileiro geradas por IA.

**Página do projeto:** <https://github.com/Pedrest15/expanded_fake_news_corpus> — navegador do
corpus e [caracterização linguística](https://pedrest15.github.io/fakegen_br/analysis.html).

O ponto de partida são notícias verdadeiras dos corpora [Fake.br][fakebr] e
[FakeTrueBR][faketruebr]. A partir das manchetes dessas notícias verdadeiras são
geradas as notícias falsas sintéticas que comporão o FakeGen.BR.

Este repositório implementa, por enquanto, o **primeiro estágio do pipeline**: um
agente de titulação que lê uma notícia e escreve a manchete correspondente.

## Pipeline

```
notícia verdadeira ──▶ [agente de titulação] ──▶ manchete ──▶ (próximo estágio: geração da fake news)
```

O agente é orquestrado com [LangGraphLib][langgraphlib] e o grafo tem duas etapas:

```
start ──▶ headline_writer ──▶ sanitize ──▶ end
```

- **`headline_writer`** — `Agent` da LangGraphLib com saída estruturada
  (`headline` + `rationale`), guiado pelo prompt de
  [prompts.py](src/fakegen_br/prompts.py). O prompt é deliberadamente
  conservador: a manchete representa a notícia **verdadeira**, então precisa ser
  fiel ao texto, sem sensacionalismo — a distorção fica para o estágio seguinte.
- **`sanitize`** — nó determinístico que remove aspas, markdown, rótulos
  ("Manchete:") e ponto final, preservando a saída bruta do modelo em
  `raw_headline` para auditoria.

Depois do grafo, [`check_headline`](src/fakegen_br/agents/headline.py) anota
avisos de qualidade no resultado (tamanho fora da faixa de 6–18 palavras, e
manchetes cujo vocabulário de conteúdo quase não aparece na notícia — indício de
alucinação). Os avisos não descartam a manchete: ficam gravados no JSONL para
revisão manual e estatísticas do corpus.

## Preparação dos corpora

Os corpora de origem misturam notícias verdadeiras e falsas, em formatos
diferentes. O script abaixo extrai só as verdadeiras e normaliza os dois num
esquema comum (`id, text, link, duplicate_rows`, mais `author, category, date`
no Fake.br):

```bash
# o Fake.br precisa ser baixado (a pasta full_texts não vai para o git)
git clone --depth 1 https://github.com/roneysco/Fake.br-Corpus.git
cp -r Fake.br-Corpus/full_texts true-corpus/raw/Fake.br-full_texts

uv run python scripts/prepare_corpora.py
```

| Corpus | Origem | Verdadeiras | Duplicatas removidas | **Total** |
|---|---|---|---|---|
| Fake.br | 3.600 arquivos em `full_texts/true/` | 3.600 | 1 | **3.599** |
| FakeTrueBR | 1.791 pares (fake, true) | 1.791 | 388 | **1.403** |

**Use o `full_texts/` do Fake.br, não o `preprocessed/`.** O CSV pré-processado
que o repositório também distribui está em minúsculas, sem acentos, sem
pontuação e sem *stopwords* — nele a titulação é inviável, porque o modelo
teria que inventar a grafia que o texto perdeu. O `full_texts/` traz a notícia
como foi coletada do site, e os metadados (autor, link, categoria, data) vêm de
`true-meta-information/`.

No FakeTrueBR a mesma notícia verdadeira aparece pareada com mais de uma fake,
daí a redução de 1.791 para 1.403. O campo `duplicate_rows` guarda os registros
descartados e `id` aponta a origem (nome do arquivo no Fake.br, linha do CSV no
FakeTrueBR), então a rastreabilidade é preservada. Nenhum arquivo de origem é
alterado; a saída vai para `true-corpus/clean/`.

## Instalação

Requer Python 3.13+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Configuração

O modelo é declarado sempre como `provedor/modelo`. Os três provedores previstos
são **Ollama** (local), **Anthropic** e **OpenAI**:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `FAKEGEN_MODEL` | `ollama/llama3.1`, `anthropic/claude-sonnet-5`, `openai/gpt-4o-mini`, ... |
| `FAKEGEN_TEMPERATURE` | Padrão `0` |
| `FAKEGEN_TOP_P`, `FAKEGEN_TOP_K`, `FAKEGEN_SEED` | Opcionais; sem valor, vale o padrão do provedor |
| `FAKEGEN_MAX_TOKENS`, `FAKEGEN_TIMEOUT`, `FAKEGEN_MAX_RETRIES` | Opcionais |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Chave do provedor (Ollama dispensa) |
| `OLLAMA_BASE_URL` | Endereço do Ollama, se não for o padrão |

Qualquer variável pode ser sobrescrita na linha de comando (`--model`,
`--temperature`, `--top-p`, `--top-k`, `--seed`).

### Amostragem e reprodutibilidade

Os três provedores não expõem os mesmos controles:

| | `temperature` | `top_p` | `top_k` | `seed` |
|---|---|---|---|---|
| Anthropic | ✅ | ✅ | ✅ | ❌ |
| OpenAI | ✅ | ✅ | ❌ | ✅ |
| Ollama | ✅ | ✅ | ✅ | ✅ |

Por isso **só `temperature` é travável nos três**, e é nela que a comparação se
apoia: em `temperature=0` a decodificação é gulosa, o que torna `top_k` e
`top_p` inertes e deixa os provedores no regime mais parecido possível. É o
padrão do projeto para a titulação, que é tarefa de fidelidade, não de
criatividade. (Para o estágio seguinte — a geração das fake news — esse padrão
está errado: temperatura zero produziria textos formulaicos e pouco diversos.)

Os parâmetros não pedidos **não são enviados**: vale o padrão do provedor, em
vez de um valor inventado por nós. Quando um parâmetro é pedido mas o provedor
não o aceita, ele é descartado com aviso em `stderr` e registrado em
`sampling_dropped` no `meta.json` — silêncio aqui invalidaria a comparação entre
provedores. O filtro é necessário: `ChatOpenAI(top_k=...)` não dá erro, apenas
desvia o parâmetro para `model_kwargs`, e a rejeição só aparece na chamada HTTP.

Note que travar a amostragem **não garante saída idêntica** entre execuções: APIs
hospedadas têm não-determinismo de infraestrutura mesmo em `temperature=0`. As
alavancas que de fato sustentam a reprodutibilidade são fixar o snapshot do
modelo (em vez de um alias móvel), versionar o prompt e registrar tudo — o
`meta.json` guarda os parâmetros efetivos e um hash do prompt usado.

## Uso

### Uma notícia

```bash
uv run fakegen headline --text "O Ministério da Saúde anunciou nesta terça-feira ..."
uv run fakegen headline --file noticia.txt --json
cat noticia.txt | uv run fakegen headline --stdin
```

### Vários modelos na mesma execução

`--model` pode ser repetido. A mesma entrada é processada por cada modelo, e
cada um grava na sua própria pasta:

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

O `meta.json` de cada pasta registra provedor, modelo, temperatura, entrada,
quantidades e horário — a procedência fica junto do resultado. O modelo também
vai gravado em cada linha do JSONL. Sem `--model`, vale o `FAKEGEN_MODEL` do
ambiente.

### Lote sobre um corpus

CSV e JSONL são lidos indicando os campos de texto e de identificador; o Fake.br
original também pode ser lido direto da pasta de `.txt`:

```bash
uv run fakegen headline \
    --input true-corpus/clean/FakeBr_true.csv --text-field text --id-field id \
    --model anthropic/claude-sonnet-5 --out-dir data/manchetes \
    --concurrency 4 --resume

uv run fakegen headline \
    --input-dir true-corpus/raw/Fake.br-full_texts/true \
    --model anthropic/claude-sonnet-5 --out-dir data/manchetes --concurrency 4
```

Opções úteis em lote:

- `--concurrency N` — chamadas simultâneas ao provedor (execução assíncrona).
- `--resume` — pula ids já presentes na saída **daquele modelo** e continua em
  modo *append*. A saída é gravada com `flush` a cada linha, então uma execução
  interrompida pode ser retomada sem perder o que já foi gerado.
- `--limit N` — processa apenas as N primeiras notícias (útil para calibrar o
  prompt antes de rodar o corpus inteiro).
- `--max-input-chars N` — trunca a notícia antes de enviá-la ao modelo (padrão
  12.000; `0` desliga).
- `--output ARQUIVO` — caminho exato, em vez da estrutura por provedor. Só vale
  com um único `--model`.

Falhas individuais (recusa do modelo, timeout) são registradas em `stderr` e não
interrompem o lote.

### Teste de calibração (10 notícias)

Antes de rodar o corpus inteiro, há uma amostra pequena e deliberadamente
desbalanceada em `true-corpus/clean/samples/`:

- **`sample_FakeBr.csv`** — 3 notícias sem título embutido (geração pura, em
  categorias diferentes) e 2 que trazem a manchete original na primeira linha,
  servindo de gabarito para comparar com o que o agente escreveu.
- **`sample_FakeTrueBr.csv`** — 3 notícias no texto minúsculo degradado e 2 das
  que preservaram a caixa original, para medir quanto a degradação custa.

```bash
uv run fakegen headline --input true-corpus/clean/samples/sample_FakeBr.csv \
    --model anthropic/claude-sonnet-5 --out-dir data/calibracao
uv run fakegen headline --input true-corpus/clean/samples/sample_FakeTrueBr.csv \
    --model anthropic/claude-sonnet-5 --out-dir data/calibracao
```

### Replicação do artigo (`fakegen paper`)

Experimento paralelo ao pipeline: reproduz o método de Silva et al. como
publicado — a notícia verdadeira **inteira** entra, e o prompt é o original em
português, byte a byte (`PAPER_ARTICLE_PROMPT`), sem estágio de manchete. Só o
modelo muda. A amostra é estratificada (10 notícias de cada corpus, seed 42) e
a saída vai para `corpus/paper_replication/`. Diferente dos outros subcomandos,
`paper` não envia temperatura nem teto de tokens a menos que sejam passados na
linha de comando (o script dos autores também não definia). Ver
[corpus/README.md](corpus/README.md#paper-replication) e o registro das
execuções em [corpus/paper_replication/NOTES.md](corpus/paper_replication/NOTES.md).

```bash
uv run python scripts/sample_paper_replication.py --per-source 10 --seed 42
uv run fakegen paper --input true-corpus/clean/paper_replication/FakeBr_true.csv \
    --id-field uid --model openai/gpt-4.1-mini-2025-04-14 \
    --out-dir corpus/paper_replication
```

Para as análises linguísticas (`expanded_fake_news_corpus.analysis.*`), o
recorte é escolhido com `--experiment paper_replication`; a saída vai para
`data/analysis/paper_replication/<módulo>/`, sem tocar em `data/analysis/<módulo>/`
do pipeline. Os resultados estão resumidos no NOTES.md acima.

### Rodar as análises

O orquestrador `expanded_fake_news_corpus.analysis` (também instalado como
`fakegen-analysis`) roda todas as análises do catálogo em sequência, ou só as
pedidas com `--analysis`; as opções de corpus são repassadas a cada módulo, que
continua executável sozinho com as próprias opções.

```bash
uv run fakegen-analysis --list                                   # catálogo
uv run fakegen-analysis --experiment paper_replication \
    --model openai/gpt-4.1-mini-2025-04-14                       # todas
uv run fakegen-analysis --analysis liwc --analysis sage          # só estas
uv run python -m expanded_fake_news_corpus.analysis.liwc --dictionary x.dic  # uma, com opção própria
```

Uma análise que falha não interrompe as demais: o erro fica no log e no
código de saída. As que dependem do corpus parseado (`grammar_rules`,
`eud_rules`) são puladas com aviso quando `data/parsed/<experimento>/` não
existe. Para acrescentar uma análise, o módulo expõe `main(argv)` como os
outros e entra numa linha do catálogo `ANALYSES` em
[analysis/runner.py](src/expanded_fake_news_corpus/analysis/runner.py).

### Parsing sintático e regras de dependência

As análises `grammar_rules` (regras da árvore básica) e `eud_rules` (regras
das arestas *enhanced*) leem CoNLL-U de `data/parsed/<experimento>/`, produzido
uma vez pela cadeia do trabalho anterior — portSentencer → portTokenizer →
LatinPipe com o modelo Portparser v2 → pós-processamento — e enriquecido com
Enhanced UD pelo Grew:

```bash
M=openai/gpt-4.1-mini-2025-04-14
uv run python -m expanded_fake_news_corpus.parsing.install_tools   # clona as ferramentas em tools/ e baixa o modelo (1,6 GB)
uv run python -m expanded_fake_news_corpus.parsing.portparser --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.parsing.eud        --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.analysis.grammar_rules --experiment paper_replication --model $M
uv run python -m expanded_fake_news_corpus.analysis.eud_rules     --experiment paper_replication --model $M
```

`tools/` fica fora do git (repositórios de terceiros e o modelo); o código da
cadeia está em [parsing/](src/expanded_fake_news_corpus/parsing/), incluindo o
tratamento do texto antes do sentenciador (`preprocess.py`: quebra de linha
como fronteira, punkt para o FakeTrueBR sem maiúsculas). O parser roda em CPU
(venv próprio, Python 3.11) e leva uns 10 minutos para 40 documentos; o `grew`
precisa estar no PATH (instalado via opam). O conjunto de regras EUD vem de
[eud-portugues](https://github.com/alvelvis/eud-portugues), vendorizado sem
alteração em `resources/eud/`.

### Página (GitHub Pages)

`docs/` é publicado em <https://pedrest15.github.io/fakegen_br/> e mostra **a
replicação do artigo**: `index.html` navega pelas 20 fake
news sintéticas (só link e metadados da notícia de origem, nunca o texto) e
`analysis.html` mostra a caracterização linguística. Os dados vêm de dois
scripts; rode-os depois de gerar e analisar:

```bash
uv run python scripts/build_site.py --model openai/gpt-4.1-mini-2025-04-14
uv run python scripts/build_analysis_data.py   # lê data/analysis/paper_replication/
```

`build_site.py` descarta recusas (registros sem as tags do artigo), e o
`--model` deixa de fora a pasta do GPT-5.1. Os dados do pipeline por manchete
(round 1) não estão mais na página; continuam em `corpus/` e `data/analysis/`.

### Saída

Uma linha JSON por notícia:

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

### Como biblioteca

```python
from fakegen_br import HeadlineAgent

agent = HeadlineAgent()                      # configuração vinda do ambiente
result = agent.generate(texto, source_id="1042")
print(result.headline, result.warnings)
```

## Desenvolvimento

```bash
uv run ruff check src && uv run ruff format --check src
```

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| [agents/headline.py](src/fakegen_br/agents/headline.py) | Grafo de titulação e a classe `HeadlineAgent` |
| [prompts.py](src/fakegen_br/prompts.py) | Prompt do agente |
| [config.py](src/fakegen_br/config.py) | Seleção de provedor/modelo e chaves |
| [text.py](src/fakegen_br/text.py) | Normalização do texto de entrada e da manchete |
| [corpus.py](src/fakegen_br/corpus.py) | Leitura dos corpora e escrita do JSONL |
| [cli.py](src/fakegen_br/cli.py) | Comando `fakegen` |
| [agents/paper_replication.py](src/expanded_fake_news_corpus/agents/paper_replication.py) | Replicação do artigo: notícia inteira → fake news com o prompt original |
| [scripts/sample_paper_replication.py](scripts/sample_paper_replication.py) | Amostra estratificada (10 + 10, seed 42) da replicação |
| [analysis/runner.py](src/expanded_fake_news_corpus/analysis/runner.py) | Orquestrador: catálogo das análises e execução em lote (`fakegen-analysis`) |
| [analysis/conllu.py](src/expanded_fake_news_corpus/analysis/conllu.py) | Leitura dos CoNLL-U e localização do corpus parseado |
| [analysis/grammar_rules.py](src/expanded_fake_news_corpus/analysis/grammar_rules.py) | Regras de dependência: produtividade, frequências, TF-IDF discriminativo |
| [analysis/eud_rules.py](src/expanded_fake_news_corpus/analysis/eud_rules.py) | O mesmo sobre as arestas *enhanced* (EUD) |
| [parsing/preprocess.py](src/expanded_fake_news_corpus/parsing/preprocess.py) | Tratamento do texto antes do sentenciador e realinhamento após o tokenizador |
| [parsing/portparser.py](src/expanded_fake_news_corpus/parsing/portparser.py) | Cadeia Portparser v2 → `data/parsed/` |
| [parsing/eud.py](src/expanded_fake_news_corpus/parsing/eud.py) | Enriquecimento EUD com Grew |
| [parsing/install_tools.py](src/expanded_fake_news_corpus/parsing/install_tools.py) | Instalação das ferramentas em `tools/` |

## Notas

- **O FakeTrueBR não tem versão original publicada.** O corpus é distribuído
  apenas como `FakeTrueBr_corpus.csv`, com o texto já em minúsculas e sem parte
  da pontuação — não há pasta de textos brutos no repositório oficial. Recuperar
  o original exigiria recoletar os artigos a partir de `link_t` (G1, Folha). Até
  lá, as manchetes geradas para esse corpus partem de um texto degradado, o que
  vale registrar na descrição do FakeGen.BR.
- Alguns arquivos do Fake.br trazem o título na primeira linha do texto. O
  prompt instrui o modelo a ignorá-lo e titular a partir do corpo da notícia,
  mas vale conferir isso na calibração inicial com `--limit`.
- O artigo [2025S1_JCBS_fakeNews_LLM_final.pdf](2025S1_JCBS_fakeNews_LLM_final.pdf)
  neste repositório é o trabalho anterior do grupo que motiva o FakeGen.BR.

[fakebr]: https://github.com/roneysco/Fake.br-Corpus
[faketruebr]: https://github.com/Chavarro/FakeTrueBr
[langgraphlib]: https://github.com/Pedrest15/LangGraphLib
