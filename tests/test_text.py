from fakegen_br.text import clean_headline, normalize_news_text, word_count


def test_clean_headline_remove_aspas_e_ponto_final():
    assert clean_headline('"Governo anuncia novo pacote de medidas."') == (
        "Governo anuncia novo pacote de medidas"
    )


def test_clean_headline_remove_aspas_tipograficas_e_markdown():
    assert clean_headline("**“Prefeitura entrega nova escola no centro”**") == (
        "Prefeitura entrega nova escola no centro"
    )


def test_clean_headline_remove_rotulo():
    assert clean_headline("Manchete: STF julga ação sobre porte de armas") == (
        "STF julga ação sobre porte de armas"
    )
    assert clean_headline("Título - Câmara aprova projeto") == "Câmara aprova projeto"


def test_clean_headline_usa_apenas_a_primeira_linha_nao_vazia():
    raw = "\n\nSenado aprova reforma\nAlternativa: Senado aprova texto"
    assert clean_headline(raw) == "Senado aprova reforma"


def test_clean_headline_preserva_interrogacao():
    assert clean_headline("Vacina é segura?") == "Vacina é segura?"


def test_clean_headline_vazio():
    assert clean_headline("") == ""
    assert clean_headline("  \n  ") == ""


def test_normalize_news_text_colapsa_linhas_em_branco():
    assert normalize_news_text("  a  \n\n\n\n   b  ") == "a\n\nb"


def test_normalize_news_text_trunca_em_fronteira_de_palavra():
    text = " ".join(["palavra"] * 100)
    truncated = normalize_news_text(text, max_chars=50)
    assert len(truncated) <= 50
    assert not truncated.endswith("pal")
    assert truncated.split()[-1] == "palavra"


def test_normalize_news_text_sem_truncagem():
    text = "x" * 100
    assert normalize_news_text(text, max_chars=None) == text


def test_word_count():
    assert word_count("  três  palavras aqui ") == 3
    assert word_count("") == 0
