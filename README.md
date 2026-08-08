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
| `FAKEGEN_MODEL` | `ollama/llama3.1`, `anthropic/claude-sonnet-4-5`, `openai/gpt-4o-mini`, ... |
| `FAKEGEN_TEMPERATURE` | Padrão `0.2` |
| `FAKEGEN_MAX_TOKENS`, `FAKEGEN_TIMEOUT`, `FAKEGEN_MAX_RETRIES` | Opcionais |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Chave do provedor (Ollama dispensa) |
| `OLLAMA_BASE_URL` | Endereço do Ollama, se não for o padrão |

Qualquer variável pode ser sobrescrita na linha de comando com `--model` e
`--temperature`.

## Uso

### Uma notícia

```bash
uv run fakegen headline --text "O Ministério da Saúde anunciou nesta terça-feira ..."
uv run fakegen headline --file noticia.txt --json
cat noticia.txt | uv run fakegen headline --stdin
```

### Lote sobre um corpus

O Fake.br distribui uma notícia por arquivo `.txt`:

```bash
uv run fakegen headline \
    --input-dir Fake.br-Corpus/full_texts/true \
    --output data/manchetes_fakebr.jsonl \
    --concurrency 4
```

JSONL e CSV também são aceitos, indicando os campos de texto e de identificador:

```bash
uv run fakegen headline \
    --input faketruebr.jsonl --text-field texto --id-field id \
    --output data/manchetes_faketruebr.jsonl \
    --concurrency 4 --resume
```

Opções úteis em lote:

- `--concurrency N` — chamadas simultâneas ao provedor (execução assíncrona).
- `--resume` — pula ids já presentes no `--output` e continua em modo *append*.
  A saída é gravada com `flush` a cada linha, então uma execução interrompida
  pode ser retomada sem perder o que já foi gerado.
- `--limit N` — processa apenas as N primeiras notícias (útil para calibrar o
  prompt antes de rodar o corpus inteiro).
- `--max-input-chars N` — trunca a notícia antes de enviá-la ao modelo (padrão
  12.000; `0` desliga).

Falhas individuais (recusa do modelo, timeout) são registradas em `stderr` e não
interrompem o lote.

### Saída

Uma linha JSON por notícia:

```json
{
  "headline": "Ministério da Saúde amplia campanha de vacinação em São Paulo",
  "rationale": "O anúncio da ampliação é o fato central do texto.",
  "raw_headline": "\"Ministério da Saúde amplia campanha de vacinação em São Paulo.\"",
  "source_id": "1042",
  "model": "anthropic/claude-sonnet-4-5",
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

- Alguns arquivos do Fake.br trazem o título na primeira linha do texto. O
  prompt instrui o modelo a ignorá-lo e titular a partir do corpo da notícia,
  mas vale conferir isso na calibração inicial com `--limit`.
- O artigo [2025S1_JCBS_fakeNews_LLM_final.pdf](2025S1_JCBS_fakeNews_LLM_final.pdf)
  neste repositório é o trabalho anterior do grupo que motiva o FakeGen.BR.

[fakebr]: https://github.com/roneysco/Fake.br-Corpus
[faketruebr]: https://github.com/Chavarro/FakeTrueBr
[langgraphlib]: https://github.com/Pedrest15/LangGraphLib
