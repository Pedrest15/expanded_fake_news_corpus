# FakeGen.BR

Construção de um corpus de *fake news* em português brasileiro geradas por IA.

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
uv run pytest          # testes (usam um modelo de mentira, não chamam provedor)
uv run ruff check src tests
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
