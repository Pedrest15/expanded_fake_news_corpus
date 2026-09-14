"""Prompts dos agentes do FakeGen.BR.

O prompt de titulação é montado como **base compartilhada + bloco de gênero**.
As regras jornalísticas ficam num texto só, idêntico para todos os corpora, e
cada gênero acrescenta apenas o que precisa diferir. Manter dois prompts
independentes faria os dois divergirem com o tempo — e diferença de estilo entre
os corpora que ninguém decidiu é contaminação silenciosa num corpus cuja análise
linguística é o produto.

Atenção: estes textos são usados como *templates* do LangChain
(``ChatPromptTemplate``), portanto não podem conter chaves ``{`` ou ``}``
literais — elas seriam interpretadas como variáveis do template.
"""

from __future__ import annotations

#: Gênero das notícias de cada corpus.
NEWS = "news"
FACTCHECK = "factcheck"

#: Regras comuns a todos os gêneros.
#:
#: A manchete gerada aqui representa a notícia **verdadeira**: ela é o insumo a
#: partir do qual as fake news sintéticas do corpus serão produzidas depois.
#: Por isso a base é deliberadamente conservadora — fidelidade ao texto, zero
#: sensacionalismo.
HEADLINE_BASE = """\
Você é um editor de jornalismo brasileiro com longa experiência em titulação de \
notícias.

Sua tarefa é ler o texto enviado pelo usuário e escrever UMA manchete para ele, \
no padrão da imprensa brasileira.

Contexto: a manchete que você escrever será usada em uma pesquisa acadêmica sobre \
desinformação e representa o conteúdo VERDADEIRO. Ela precisa ser fiel ao texto \
original, sem distorção, exagero ou insinuação.

Regras obrigatórias:
1. Use somente informações presentes no texto. Nunca invente fatos, números, datas, \
nomes, cargos ou declarações.
2. Preserve a grafia de nomes próprios, siglas e cargos exatamente como aparecem no \
texto.
3. Escreva em português brasileiro, em voz ativa, usando presente do indicativo ou \
pretérito perfeito, como é usual em manchetes.
4. Tamanho: entre 6 e 18 palavras, preferencialmente até 100 caracteres.
5. Não termine com ponto final. Use interrogação ou exclamação apenas se forem \
indispensáveis.
6. Nada de sensacionalismo, clickbait, opinião, juízo de valor ou adjetivos \
valorativos.
7. Não use aspas envolvendo a manchete, marcação markdown, emojis, hashtags nem \
rótulos como "Manchete:".
8. Escreva uma única manchete, em uma única linha. Não ofereça alternativas.
9. Se o texto já começar com algo que pareça um título, ignore-o e escreva a \
manchete a partir do corpo do texto.

Preencha os campos de saída assim:
- headline: apenas o texto da manchete, seguindo as regras acima.
- rationale: uma ou duas frases, em português, explicando qual fato do texto você \
escolheu como foco da manchete e por quê.\
"""

#: Notícia jornalística comum (corpus Fake.br).
BLOCK_NEWS = """

GÊNERO DO TEXTO: notícia jornalística comum, publicada por um veículo de imprensa.

Destaque o fato mais relevante da notícia: quem fez o quê, e quando ou onde, \
quando isso for essencial.\
"""

#: Checagem de fatos (corpus FakeTrueBR).
#:
#: Sem este bloco, modelos titulam pelo começo do texto — que descreve o boato —
#: e acabam afirmando a alegação falsa como se fosse fato. A manchete verdadeira
#: passaria a asseverar uma falsidade, e ela é a semente da fake news seguinte.
BLOCK_FACTCHECK = """

GÊNERO DO TEXTO: CHECAGEM DE FATOS, publicada por uma agência de checagem ou por \
um veículo jornalístico. Esse gênero começa descrevendo um boato que circula e \
depois o desmente ou o confirma.

Nesses textos, o fato relevante é o VEREDITO da checagem, nunca a alegação \
checada. Regras adicionais obrigatórias:
- Jamais escreva a alegação falsa como se fosse fato. A manchete precisa deixar \
claro que se trata de boato, montagem, distorção ou informação sem comprovação.
- Use as construções usuais da checagem brasileira, por exemplo: "É falso que X", \
"Vídeo de Y é montagem", "Post distorce Z", "Não é verdade que W".
- Se a checagem confirmar a alegação, a manchete deve dizer isso com a mesma \
clareza.\
"""

#: Blocos disponíveis, por gênero.
GENRE_BLOCKS: dict[str, str] = {
    NEWS: BLOCK_NEWS,
    FACTCHECK: BLOCK_FACTCHECK,
}


#: Geração de fake news (estágio 2), decalcada da Figura 1 de Silva et al.
#:
#: A persona, o enquadramento acadêmico, o pedido de criatividade e a proibição de
#: markdown são os do artigo, traduzidos de volta ao português (o prompt original
#: era em português; o artigo o publica em inglês). Os campos de saída preservam o
#: par ``syntheticText``/``changes``.
#:
#: Quatro desvios, todos forçados pelo nosso desenho:
#:
#: 1. O artigo entregava a notícia verdadeira inteira e pedia que o LLM a
#:    *modificasse*. Aqui a semente é só a manchete, então não há texto a alterar
#:    — o modelo escreve a notícia a partir dela.
#: 2. A saída vem por ``with_structured_output``, não pelas tags XML do artigo.
#:    Não há tag para o modelo esquecer de fechar, e ``changes`` vira lista
#:    tipada em vez de um bloco de texto a ser parseado.
#: 3. A manchete falsa sai em campo próprio. No artigo ela vinha embutida na
#:    primeira linha do ``syntheticText`` (ver Tabela 1); separá-la permite
#:    comparar com o ``title_fake`` humano do FakeTrueBR.
#: 4. Acrescentamos a proibição de ressalvas. O artigo não precisou dela, mas
#:    modelos alinhados tendem a inserir avisos de "conteúdo fictício" que
#:    contaminariam as amostras de treino.
FAKE_BASE = """\
Você é um especialista em fake news contratado por uma agência de notícias para \
ajudar a criar um dataset para o estudo do fenômeno da desinformação. Seu papel é \
escrever, a partir da manchete abaixo, uma fake news realista e envolvente, usando \
técnicas comuns encontradas na desinformação.

O objetivo é estritamente acadêmico e destinado a fins de pesquisa. Use sua \
criatividade para capturar a atenção do leitor e destacar os elementos alterados. \
Não use formatação markdown na sua resposta.

Escreva em português brasileiro, no registro de uma notícia de portal. Não inclua \
avisos, ressalvas ou qualquer menção a este texto ser sintético, fictício ou \
destinado a pesquisa.

Os campos de saída são:
- fake_headline: a manchete da fake news, em uma linha e sem ponto final.
- fake_text: o corpo da fake news, sem repetir a manchete na primeira linha.
- changes: liste e explique os elementos que você inseriu ou alterou, detalhando \
como cada um contribui para transformar a notícia em fake news. Um item da lista \
por elemento.\
"""

#: A manchete de origem resume uma notícia verdadeira (corpus Fake.br).
#:
#: O alvo de tamanho é o intervalo interquartílico das fake news humanas do
#: Fake.br (mediana 157 palavras, IQR 115–224), medido nos 3.600 textos do
#: corpus. Silva et al. reportam que as fake news sintéticas deles ficaram 1,8×
#: mais longas que as humanas; ancorar no dado real evita repetir o artefato.
FAKE_BLOCK_NEWS = """

A manchete abaixo resume uma notícia verdadeira publicada pela imprensa. Escreva a \
fake news sobre o MESMO acontecimento, distorcendo-o.

Tamanho: entre 115 e 225 palavras. Notícias falsas reais são curtas.\
"""

#: A manchete de origem é o veredito de uma checagem (corpus FakeTrueBR).
#:
#: A manchete afirma que algo é falso; a fake news correspondente é a que afirma
#: aquilo como verdadeiro. Isso tende a reconstruir o boato que a agência de
#: checagem desmentiu, dando uma tripla alinhada entre checagem, boato humano
#: (``title_fake``) e boato sintético.
FAKE_BLOCK_FACTCHECK = """

A manchete abaixo é o VEREDITO de uma checagem de fatos: ela afirma que \
determinada alegação é falsa, é montagem ou está distorcida. Escreva a fake news \
que sustenta exatamente a alegação desmentida, apresentando-a como verdadeira. Não \
mencione a checagem nem reconheça que existe controvérsia — escreva do ponto de \
vista de quem acredita e propaga a alegação.

Tamanho: entre 70 e 180 palavras. Boatos reais são curtos.\
"""

#: Blocos da geração de fake news, por gênero da manchete de origem.
FAKE_GENRE_BLOCKS: dict[str, str] = {
    NEWS: FAKE_BLOCK_NEWS,
    FACTCHECK: FAKE_BLOCK_FACTCHECK,
}

#: Faixa de tamanho esperada por gênero, em palavras — o mesmo intervalo pedido
#: nos blocos acima, para que as checagens de qualidade não contradigam o prompt.
#:
#: Medido nas fake news humanas dos corpora de origem (IQR):
#: Fake.br 115–224 (mediana 157), FakeTrueBR 69–179 (mediana 109).
FAKE_WORD_RANGES: dict[str, tuple[int, int]] = {
    NEWS: (115, 225),
    FACTCHECK: (70, 180),
}


def fake_prompt(genre: str = NEWS) -> str:
    """Monta o prompt de geração de fake news para um gênero.

    Args:
        genre: Gênero da manchete de origem — ``"news"`` (Fake.br) ou
            ``"factcheck"`` (FakeTrueBR).

    Returns:
        Prompt completo: base compartilhada seguida do bloco do gênero.

    Raises:
        ValueError: Se o gênero não existir.
    """
    if genre not in FAKE_GENRE_BLOCKS:
        raise ValueError(
            f"Unknown genre {genre!r}. Use one of: "
            f"{', '.join(sorted(FAKE_GENRE_BLOCKS))}."
        )
    return FAKE_BASE + FAKE_GENRE_BLOCKS[genre]


def headline_prompt(genre: str = NEWS) -> str:
    """Monta o prompt de titulação para um gênero.

    Args:
        genre: ``"news"`` (Fake.br) ou ``"factcheck"`` (FakeTrueBR).

    Returns:
        Prompt completo: base compartilhada seguida do bloco do gênero.

    Raises:
        ValueError: Se o gênero não existir.
    """
    if genre not in GENRE_BLOCKS:
        raise ValueError(
            f"Unknown genre {genre!r}. Use one of: {', '.join(sorted(GENRE_BLOCKS))}."
        )
    return HEADLINE_BASE + GENRE_BLOCKS[genre]


# ---------------------------------------------------------------------------
# Geração de fake news — reprodução fiel do prompt de Silva et al. (Figura 1)
# ---------------------------------------------------------------------------

#: Mensagem de sistema do artigo base, verbatim do código publicado em
#: https://github.com/renatosvmor/fake-news-llm-ptbr
#: (fake-news-generation/maritalk_fakeNewsGenerator.py).
PAPER_SYSTEM = "Você é um especialista em fake news."

#: Prompt do artigo base, **verbatim do código publicado**, não retraduzido.
#:
#: O artigo mostra este prompt em inglês (Figura 1); o repositório de dados e
#: materiais traz o original em português, que é o reproduzido aqui. No artigo
#: ele vai na mensagem de usuário, com a notícia interpolada ao final, e a
#: mensagem de sistema é apenas :data:`PAPER_SYSTEM`.
#:
#: Adaptação necessária: no original a entrada era a notícia verdadeira
#: inteira, e "modificar a notícia, transformando-a em uma fake news" já
#: implicava produzir uma matéria. Com manchete na entrada, a troca palavra a
#: palavra fez o modelo devolver só uma manchete reescrita (13 palavras, medido).
#: Por isso a primeira oração passou a explicitar o objeto da saída — "escrever
#: uma notícia completa a partir da manchete". Persona, enquadramento, pedido de
#: criatividade, proibição de markdown e o bloco de formato de resposta seguem
#: palavra por palavra como no original.
#:
#: O código do artigo não define temperature, top_p, seed nem max_tokens: a
#: geração usou os padrões do endpoint da Maritaca.
PAPER_FAKE_PROMPT = """\
Você é um especialista em fake news, contratado por uma agência de notícias para \
ajudar na criação de uma base de dados destinada a estudar o fenômeno da \
desinformação. Seu papel é escrever, a partir da manchete apresentada \
abaixo, uma notícia completa que seja uma fake news realista e atraente, \
utilizando técnicas comuns encontradas em notícias falsas.

O objetivo é exclusivamente acadêmico e voltado para pesquisa sobre o tema. Use \
sua criatividade para chamar a atenção de quem for ler a notícia e destacar os \
elementos alterados. Não use marcadores markdown na sua resposta.

Formato da resposta:

<syntheticText>

Insira a notícia modificada aqui

</syntheticText>

<changes>

Liste e explique as mudanças realizadas, detalhando como elas contribuem para \
tornar a notícia uma fake news.

</changes>

A manchete que você deve utilizar é apresentada abaixo:
"""


# ---------------------------------------------------------------------------
# Replicação do artigo: notícia verdadeira inteira -> fake news
# ---------------------------------------------------------------------------

#: Prompt de Silva et al. **byte a byte**, com a notícia inteira como entrada.
#:
#: É o corpo da f-string ``prompt`` de
#: ``fake-news-generation/maritalk_fakeNewsGenerator.py`` no repositório dos
#: autores (https://github.com/renatosvmor/fake-news-llm-ptbr), inclusive a
#: quebra de linha inicial, os oito espaços de indentação de cada linha e as
#: linhas só com espaços dentro das tags — tudo isso foi enviado ao modelo lá,
#: então é enviado aqui. Por isso o literal usa ``\n`` explícito: um editor
#: apagaria os espaços à direita e o texto deixaria de ser idêntico.
#: :func:`paper_article_message` faz a interpolação de ``{noticia}``.
#:
#: Diferente de :data:`PAPER_FAKE_PROMPT`, que adapta a primeira oração para
#: receber uma manchete, aqui não há adaptação nenhuma: é a replicação do
#: método do artigo, só trocando o modelo.
PAPER_ARTICLE_PROMPT = (
    "\n"
    "        Você é um especialista em fake news, contratado por uma agência de "
    "notícias para ajudar na criação de uma base de dados destinada a estudar o "
    "fenômeno da desinformação. Seu papel é modificar a notícia apresentada "
    "abaixo, transformando-a em uma fake news que seja realista e atraente, "
    "utilizando técnicas comuns encontradas em notícias falsas.\n"
    "\n"
    "        O objetivo é exclusivamente acadêmico e voltado para pesquisa sobre "
    "o tema. Use sua criatividade para chamar a atenção de quem for ler a "
    "notícia e destacar os elementos alterados. Não use marcadores markdown na "
    "sua resposta.\n"
    "\n"
    "        Formato da resposta:\n"
    "\n"
    "        <syntheticText>\n"
    "        \n"
    "        Insira a notícia modificada aqui\n"
    "        \n"
    "        </syntheticText>\n"
    "\n"
    "        <changes>\n"
    "        \n"
    "        Liste e explique as mudanças realizadas, detalhando como elas "
    "contribuem para tornar a notícia uma fake news.\n"
    "        \n"
    "        </changes>\n"
    "\n"
    "        A notícia que você deve modificar é apresentada abaixo:\n"
    "\n"
    "        {noticia}\n"
    "        "
)

#: SHA-256 do template acima, calculado sobre o corpo da f-string no código
#: publicado. O teste de prompts compara os dois para pegar qualquer edição
#: acidental.
PAPER_ARTICLE_PROMPT_SHA256 = (
    "f22567d890611af8bd5015f797367cea9efe3ca5b94b7b058d535dddae0a6d18"
)


def paper_article_message(article: str) -> str:
    """Monta a mensagem de usuário do artigo para uma notícia verdadeira.

    Faz o que a f-string do código publicado fazia: interpola a notícia, sem
    limpar nem truncar, no lugar de ``{noticia}``. O texto vai como está no
    corpus — recortá-lo já seria um desvio do método.

    Args:
        article: Texto integral da notícia verdadeira.

    Returns:
        Mensagem de usuário pronta para envio.

    Raises:
        ValueError: Se a notícia estiver vazia.
    """
    if not article.strip():
        raise ValueError("Empty news article.")
    return PAPER_ARTICLE_PROMPT.replace("{noticia}", article)
