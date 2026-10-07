# Splits do trabalho anterior

Os quatro arquivos de divisão treino/teste de
`noticias_falsas_humano_maquina_semantica/linguistic_features`, copiados
verbatim. Cada linha é o nome do `.txt` no corpus do trabalho anterior
(`Fake.br-Corpus-master/full_texts/fake_br_clean` e `FakeTrue.Br-main/fake`,
com a contraparte sintética de mesmo nome em `fake-news-llm-ptbr-main`).

| Arquivo | Linhas |
|---|---|
| `train_fake_br.txt` | 2.879 |
| `test_fake_br.txt` | 719 |
| `train_fake_true_br.txt` | 1.431 |
| `test_fake_true_br.txt` | 358 |

Estão aqui para que o experimento de transferência seja reproduzível a partir
deste repositório: o detector é treinado nos uids de treino **de lá**, e os
textos humanos de teste saem dos uids de teste **de lá**, que o modelo nunca
viu.

É por isso que os splits importam. Nossa amostra de 20 notícias foi sorteada
dos mesmos corpora de origem, e **16 das 20 caem no treino do trabalho
anterior** — usá-las como lado humano do teste mediria memorização, não
detecção. O lado humano vem do conjunto de teste deles; o lado máquina vem dos
nossos geradores, que nenhum detector jamais viu.

A numeração do FakeTrueBR é 0-based lá e 1-based no nosso `uid`
(`_PRIOR_INDEX_OFFSET` em `analysis/documents.py` faz o ajuste).
