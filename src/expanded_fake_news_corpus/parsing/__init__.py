"""Parsing sintático do corpus pareado e enriquecimento com Enhanced UD.

A cadeia é a do trabalho anterior (Andrade et al., PROPOR 2026): portSentencer
-> portTokenizer -> LatinPipe com o modelo Portparser v2 -> pós-processamento
de lemas e feats -> Grew com as regras do eud-portugues. As ferramentas são
externas, ficam em ``tools/`` (fora do git; :mod:`install_tools` as instala) e
só :mod:`portparser` e :mod:`eud` as chamam. Tudo o mais lê os CoNLL-U que
elas produzem, em ``data/parsed/<experimento>/`` (ver
:mod:`expanded_fake_news_corpus.analysis.conllu`).

Módulos:

* :mod:`preprocess` — tratamento do texto antes do sentenciador;
* :mod:`tools` — onde estão as ferramentas e como executá-las;
* :mod:`manifest` — proveniência gravada ao lado dos CoNLL-U;
* :mod:`portparser` — do texto ao CoNLL-U (``python -m ...parsing.portparser``);
* :mod:`eud` — do CoNLL-U ao CoNLL-U com DEPS (``python -m ...parsing.eud``);
* :mod:`install_tools` — instalação em ``tools/``
  (``python -m ...parsing.install_tools``).
"""

from expanded_fake_news_corpus.parsing.tools import ToolError, ToolPaths

__all__ = ["ToolError", "ToolPaths"]
