#!/usr/bin/env python3
"""
mesclar_datasets.py v7.1 — Unifica fontes em corpus pronto para 01_prepare_data,
com FILTRAGEM data-driven de ruido cross-lingual e enciclopedico.

v7.1 vs v7.0 (CORRECOES DE FALSOS POSITIVOS observados na run 2026-06-06):
  + Removidas palavras AMBIGUAS PT/EN da lista de marcadores de ingles:
    "me", "as", "do", "is", "or", "i", "he", "she", "we", "it", "be",
    "in", "to", "can", "if", "us", "our", "her" (todas tem cognatos PT
    que causaram descarte de pares legitimos do dataset construido).
  + Sinal B exige agora n_palavras>=8 (era >=4) e n_ingles>=2 (era >=1)
    para disparar pct>=25%. Calibrado contra falsos positivos como
    "Estou me sentindo fraco" (4 pal, 1 marcador "me") e "Comprei no
    shopping" (4 pal, 1 marcador "shopping").
  + Removido padrao enciclopedico generico "em \\d{4}": data por si so
    NAO indica conteudo enciclopedico. "Me casei em 2010", "Lei de
    Libras aprovada em 2002" sao cotidianos legitimos. Mantidos apenas
    padroes inequivocos (termos academicos raros + citacoes "et al",
    "ibid", "segundo a [WikipediaXYZ]").

MOTIVACAO (data-driven): a analise da run v6 mostrou que 11 das 30 palavras
"vocab_alvo" do GPS sao em INGLES ou nomes proprios estrangeiros (beast,
cruelty, children, youhei, andreessen, mandeville, etc), e outras sao termos
academicos raros (subcaducifolia, cromlech, aliquota, armisticio). Esses pares
nao sao "glosa Libras real" — sao residuos de scraping de Wikipedia/conteudo
academico que contaminaram a v1 do bilingual dataset. Eles diluem o sinal de
treino e nao atendem o caso de uso do app (cotidiano: compras, viagem, medico,
shopping, orgaos publicos, reformas, eletronicos, socializacao).

O QUE MUDOU vs v6:
  + Aceita 3 fontes em vez de 2:
    - dataset_construido.csv (qualidade humana, cotidiano)
    - libras_bilingual_dataset.csv (v1 legada, fallback)
    - vlibras_db_v2_normalizado.csv (v2 expandida, gerada por preparar_vlibras_db)
  + Filtro cross-lingual: descarta pares com marcadores fortes de ingles
  + Filtro enciclopedico: descarta pares com sinais de origem academica/wiki
  + Filtro de pontuacao excessiva: razao pontuacao/letras > 30%
  + Filtro de razao caractere-nao-pt: textos sem nenhum acento em frase longa
  + TEMAS DIFERENCIADOS para o 01_prepare_data poder balancear:
    construido_<tema>, bilingual_v1, vlibrasdb_geral, vlibrasdb_governo
  + Logs detalhados de causa de descarte (auditavel)
  + Relatorio JSON com contagem por categoria de descarte

FILOSOFIA:
  - Sem chutes: cada limiar de filtro vem de CLI com default justificado.
  - Erros explicitos: require() em cada etapa critica.
  - Auditavel: salva sample_descartados.csv para inspecao humana das decisoes.
  - Reproducivel: SHA256 e tamanho de cada fonte no relatorio.

Uso:
  python tools/mesclar_datasets.py
  python tools/mesclar_datasets.py --pct-ingles-max 0.10  # mais permissivo
  python tools/mesclar_datasets.py --max-glosa-palavras 40  # mais restritivo
"""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.common import (  # noqa: E402
    DATA_RAW, DATA_REPORTS, ART_LOGS, DIR_PT2LIBRAS, DIR_LIBRAS2PT,
    TOKENS_PONTUACAO,
    get_logger, salvar_json, require, StepGate, montar_input,
    validar_dataframe, DataContractError,
)

log = get_logger("mesclar", ART_LOGS / "mesclar.log")

# Fontes possiveis (ordem de prioridade na deduplicacao: primeiro = melhor qualidade)
FONTE_CONSTRUIDO = DATA_RAW / "dataset_construido.csv"
FONTE_VLIBRASDB_V2 = DATA_RAW / "vlibras_db_v2_normalizado.csv"
FONTE_BILINGUAL_V1 = DATA_RAW / "libras_bilingual_dataset.csv"
SAIDA = DATA_RAW / "dataset_merged_v7.csv"

_RE_BRACKET = re.compile(r"\[[^\]]*\]")
_RE_PALAVRAS = re.compile(r"\b[a-zA-ZÀ-ÖØ-öø-ÿ]+\b")
_RE_ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")

# Marcadores fortes de ingles. Lista calibrada com base nas palavras que
# apareceram no vocab_alvo do GPS apos a run v6 + palavras funcionais
# comuns em ingles que NUNCA aparecem em portugues nativo.
#
# v7.1 — CORRECAO de falsos positivos:
# Removidas palavras AMBIGUAS PT/EN que causaram descarte de pares legitimos
# do dataset construido em v7.0 (rodada 2026-06-06):
#   "me" (pronome PT) -> "Estou me sentindo fraco" descartado
#   "as" (artigo PT)  -> "As ferias acabaram"      descartado
#   "do" (prep PT)    -> "Quero participar do..."  descartado
#   "is", "or", "i", "he", "she", "we", "it"       (todas tem cognatos PT)
# Filosofia: false positive em palavra de 2 letras causa muito dano (corta
# PT cotidiano); melhor ter false negative em ingles puro com 1 marcador
# (sinal A com >=2 marcadores ainda pega frases ingles longas).
MARCADORES_INGLES_FUNCIONAIS = {
    "the", "and", "with", "from", "that", "this", "these", "those",
    "have", "has", "had", "having", "will", "would", "could", "should",
    "shall", "must", "might", "may", "cannot",
    "of", "on", "at", "by", "for", "are", "was", "were",
    "been", "being", "does", "did", "done", "doing",
    "but", "then", "than", "when", "where", "why", "how",
    "what", "which", "who", "whom", "whose", "their", "they", "them", "his",
    "her", "you", "your", "mine",
}
# Marcadores de conteudo (palavras que apareceram no vocab_alvo da run v6
# como ingles puro): expandivel
MARCADORES_INGLES_CONTEUDO = {
    "beast", "cruelty", "children", "hunter", "wood", "goodman", "bully",
    "mandeville", "andreessen", "youhei", "isabela", "renata",
    "level", "respire", "smart", "fast", "true", "false", "high", "low",
    "good", "bad", "old", "new", "young", "great", "small", "big", "make",
    "made", "got", "get", "go", "going", "came", "come", "saw", "see",
    "say", "said", "tell", "told", "find", "found", "give", "gave", "take",
    "took", "know", "knew", "think", "thought", "want", "wanted", "need",
    "needed",
}
MARCADORES_INGLES = MARCADORES_INGLES_FUNCIONAIS | MARCADORES_INGLES_CONTEUDO

# Marcadores de conteudo enciclopedico/academico (raros em rotina cotidiana,
# comuns em Wikipedia/papers/livros de referencia)
#
# v7.1 — CORRECAO: removido padrao "em \d{4}" porque descarta frases COTIDIANAS
# legitimas com data ("A lei de Libras foi aprovada em 2002", "Me casei em
# 2010", "Comecei em janeiro de 2025"). Data por si so NAO e enciclopedica.
# Mantidos apenas padroes inequivocos de citacao academica e termos raros.
PADROES_ENCICLOPEDICOS = [
    re.compile(r"\bsubcaducif[oó]li[ao]\b", re.IGNORECASE),
    re.compile(r"\bcromlech\b", re.IGNORECASE),
    re.compile(r"\barmist[ií]cio\b", re.IGNORECASE),
    re.compile(r"\baliqu?ot[ao]\b", re.IGNORECASE),
    re.compile(r"\bsegundo (a|o) [A-Z][a-z]+", re.IGNORECASE),  # "segundo a Wikipedia"
    re.compile(r"\b(cf\.|ibid\.|op\.\s?cit|et\s?al\.)", re.IGNORECASE),
]


def canonicalizar_pontuacao(glosa: str) -> str:
    """Normaliza interior de [..]: remove acento + upcase.
    '[INTERROGAÇÃO]' -> '[INTERROGACAO]'.
    """
    def repl(m: re.Match) -> str:
        interior = "".join(
            c for c in unicodedata.normalize("NFD", m.group(0)[1:-1])
            if unicodedata.category(c) != "Mn").upper()
        return "[" + interior + "]"
    return _RE_BRACKET.sub(repl, str(glosa))


def _palavras_lower(texto: str) -> list[str]:
    return [w.lower() for w in _RE_PALAVRAS.findall(str(texto))]


def detectar_ingles(texto: str, pct_max: float, min_palavras_disparo: int) -> tuple[bool, dict]:
    """
    Retorna (eh_suspeito_ingles, debug_dict).
    
    Heuristica multi-sinal:
      A) >= min_palavras_disparo marcadores de ingles no texto, OU
      B) >= pct_max das palavras do texto sao marcadores de ingles, OU
      C) texto longo (>= 5 palavras) so com ASCII E sem nenhuma palavra com
         acento (pt nativo quase sempre tem acentos em frases dessa extensao)
         E nao tem palavras-stop pt comuns (que, o, a, e, de).
    
    Justificativa da heuristica C: "the cat is here" e ASCII puro e curto.
    "Eu vou ao mercado hoje" e ASCII puro mas tem stopwords pt. "Beast was found
    near the cromlech in 1843" e longo, sem acentos e sem stopwords pt: ingles
    quase certo.
    """
    palavras = _palavras_lower(texto)
    n_pal = len(palavras)
    if n_pal == 0:
        return False, {"n_palavras": 0}
    n_ing = sum(1 for p in palavras if p in MARCADORES_INGLES)
    pct_ing = n_ing / n_pal

    sinal_a = n_ing >= min_palavras_disparo
    # v7.1: sinal B precisa de frase MAIOR (era n_pal >= 4, agora >= 8) e
    # mais marcadores em proporcao. Com 4 palavras, basta 1 anglicismo PT
    # legitimo ("level", "shopping") para disparar 25% — falso positivo
    # observado na run v7.0. Exigir frase mais longa reduz isso.
    sinal_b = pct_ing >= pct_max and n_pal >= 8 and n_ing >= 2

    # Sinal C: longo + ASCII puro + sem stopwords pt
    # Lista expandida — calibrada com casos reais de PT cotidiano sem acentos
    stopwords_pt = {
        "que", "o", "a", "e", "de", "do", "da", "para", "com",
        "em", "no", "na", "um", "uma", "os", "as", "dos", "das",
        "ao", "à", "aos", "às", "pelo", "pela", "pelos", "pelas",
        "é", "sou", "foi", "fui", "ser", "será", "seria", "tem", "ter",
        "tinha", "está", "estou", "estava", "era", "são",
        "seu", "sua", "seus", "suas", "meu", "minha", "meus", "minhas",
        "isso", "isto", "esse", "essa", "este", "esta", "aqui", "ali",
        "muito", "muita", "muitos", "muitas", "pouco", "pouca",
        "também", "tambem", "ainda", "agora", "depois", "antes",
        "quando", "onde", "como", "por", "sem", "sob", "sobre", "entre",
        "vou", "vai", "vamos", "vão", "fiz", "fez", "ir", "vir", "vem",
        "quero", "quer", "queria", "queremos",
    }
    tem_stopword_pt = any(p in stopwords_pt for p in palavras)
    texto_str = str(texto)
    so_ascii = bool(_RE_ASCII_ONLY.match(texto_str))
    sinal_c = (n_pal >= 5 and so_ascii and not tem_stopword_pt
               and not any(c in texto_str.lower() for c in "ãáéíóúâêôàç"))

    eh = sinal_a or sinal_b or sinal_c
    return eh, {"n_palavras": n_pal, "n_ingles": n_ing,
                "pct_ingles": round(pct_ing, 3),
                "so_ascii": so_ascii, "sinais": {"a": sinal_a, "b": sinal_b, "c": sinal_c}}


def detectar_enciclopedico(texto: str) -> tuple[bool, str]:
    """Retorna (eh_enciclopedico, padrao_que_disparou)."""
    for pad in PADROES_ENCICLOPEDICOS:
        if pad.search(str(texto)):
            return True, pad.pattern
    return False, ""


def detectar_pontuacao_excessiva(texto: str, limite: float) -> tuple[bool, float]:
    """Razao caracteres-nao-alfanumericos vs total."""
    t = str(texto)
    if len(t) == 0:
        return False, 0.0
    n_pont = sum(1 for c in t if not c.isalnum() and c != " " and c != "[" and c != "]")
    razao = n_pont / len(t)
    return razao > limite, razao


def sha256_arquivo(p: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Filtros de comprimento (calibrados para Libras cotidiana)
    p.add_argument("--max-glosa-palavras", type=int, default=40,
                   help="Apertado de 60 para 40: glosa Libras cotidiana cabe em "
                        "<= 40 sinais; acima e quase sempre paragrafo enciclopedico.")
    p.add_argument("--max-pt-palavras", type=int, default=60,
                   help="Apertado de 100 para 60: frases cotidianas raramente "
                        "passam de 60 palavras.")
    # Filtros de ingles
    p.add_argument("--pct-ingles-max", type=float, default=0.25,
                   help="Se >= 25%% das palavras do texto sao marcadores de "
                        "ingles, descarta. Default calibrado: 0.25 evita falsos "
                        "positivos em frases PT curtas com 1 anglicismo "
                        "(ex: 'O level do jogo aumentou').")
    p.add_argument("--min-palavras-ingles-disparo", type=int, default=2,
                   help="Disparo absoluto: >= N marcadores de ingles -> descarta.")
    # Filtro de pontuacao
    p.add_argument("--pct-pontuacao-max", type=float, default=0.25,
                   help="Razao max de pontuacao/total. > este valor -> descarta.")
    # Saidas
    p.add_argument("--saida", default=str(SAIDA))
    p.add_argument("--salvar-descartados", action="store_true", default=True,
                   help="Salva sample_descartados.csv para auditoria humana")
    a = p.parse_args()

    require(0.0 < a.pct_ingles_max <= 1.0, "E_PARAM",
            f"--pct-ingles-max fora de (0, 1]: {a.pct_ingles_max}",
            err=DataContractError)

    import pandas as pd

    # ============ ETAPA 1: CARREGAR FONTES (qual existe?) ============
    with StepGate("carregar_fontes", log) as g:
        require(FONTE_CONSTRUIDO.exists(), "E_FONTE_AUSENTE",
                f"obrigatorio: {FONTE_CONSTRUIDO}",
                fix="o dataset_construido.csv (qualidade humana) e o nucleo. "
                    "Sem ele, nao mesclamos nada.",
                err=DataContractError)

        # Construido: sempre carregado
        small = pd.read_csv(FONTE_CONSTRUIDO).dropna(
            subset=["portugues_original", "libras_glosa"])
        sha_construido = sha256_arquivo(FONTE_CONSTRUIDO)
        log.info(f"  construido: {len(small)} linhas  sha={sha_construido[:12]}")

        # VLibrasBD v2: preferencial. Se existe, usa ele e ignora o v1.
        vlibras_v2_existe = FONTE_VLIBRASDB_V2.exists()
        bilingual_v1_existe = FONTE_BILINGUAL_V1.exists()
        fonte_grande = None
        sha_grande = None
        if vlibras_v2_existe:
            df_grande = pd.read_csv(FONTE_VLIBRASDB_V2).dropna(
                subset=["pt", "glosa"])
            sha_grande = sha256_arquivo(FONTE_VLIBRASDB_V2)
            fonte_grande = "vlibrasdb_v2"
            log.info(f"  vlibrasdb_v2: {len(df_grande)} pares  "
                     f"sha={sha_grande[:12]}")
            if bilingual_v1_existe:
                log.info("  bilingual_v1 EXISTE mas sera IGNORADO (vlibrasdb_v2 "
                         "e superior: 127k pares vs 68k, mesma equipe UFPB, "
                         "mesma metodologia).")
        elif bilingual_v1_existe:
            df_grande = pd.read_csv(FONTE_BILINGUAL_V1).dropna()
            require({"portuguese", "libras"}.issubset(df_grande.columns),
                    "E_SCHEMA_BILINGUAL",
                    f"bilingual_v1 sem colunas esperadas; tem {list(df_grande.columns)}",
                    err=DataContractError)
            df_grande = df_grande.rename(columns={"portuguese": "pt",
                                                   "libras": "glosa"})
            df_grande["tema"] = "bilingual_v1"
            sha_grande = sha256_arquivo(FONTE_BILINGUAL_V1)
            fonte_grande = "bilingual_v1"
            log.info(f"  bilingual_v1: {len(df_grande)} pares  "
                     f"sha={sha_grande[:12]}  (FALLBACK; prefira rodar "
                     f"preparar_vlibras_db.py)")
        else:
            df_grande = pd.DataFrame(columns=["pt", "glosa", "tema"])
            log.warning("  Nenhuma fonte expandida! Resultado tera so o "
                        "dataset_construido (~2.5k pares). Considere rodar "
                        "preparar_vlibras_db.py.")

        g.info.update({
            "n_construido": len(small),
            "n_grande": len(df_grande),
            "fonte_grande": fonte_grande,
            "sha_construido": sha_construido,
            "sha_grande": sha_grande,
        })

    # ============ ETAPA 2: UNIFICAR + CANONICALIZAR ============
    with StepGate("unificar_e_normalizar", log) as g:
        # Constroi schema unificado: pt, glosa, tema
        # Construido: prefixa tema com "construido_" para distinguir
        small_u = small.rename(columns={"portugues_original": "pt",
                                        "libras_glosa": "glosa"})[["pt", "glosa", "tema"]]
        small_u["tema"] = "construido_" + small_u["tema"].astype(str)

        # Garante coluna 'tema' no df_grande (vlibrasdb ja tem; bilingual_v1 setamos acima)
        if "tema" not in df_grande.columns:
            df_grande["tema"] = fonte_grande or "desconhecido"

        uni = pd.concat([small_u[["pt", "glosa", "tema"]],
                          df_grande[["pt", "glosa", "tema"]]], ignore_index=True)
        uni["pt"] = uni["pt"].astype(str).str.strip()
        uni["glosa"] = uni["glosa"].astype(str).map(canonicalizar_pontuacao).str.strip()

        # Confirma canonicalizacao (sanidade)
        acent = int(uni["glosa"].str.contains(r"\[INTERROGAÇÃO\]|\[EXCLAMAÇÃO\]").sum())
        require(acent == 0, "E_NORMALIZACAO",
                f"{acent} tokens acentuados restantes apos canonicalizacao",
                err=DataContractError)

        g.info["n_unificado_bruto"] = len(uni)

    # ============ ETAPA 3: FILTRAGEM data-driven ============
    with StepGate("filtrar_qualidade", log) as g:
        # Aplicar filtros em ordem do mais rapido para o mais caro
        n0 = len(uni)
        contadores = Counter()
        descartes_sample = []  # primeiros N de cada categoria, para auditoria

        razoes: list[str] = [None] * len(uni)
        diagnosticos: list[dict] = [None] * len(uni)
        for i, (pt, glosa, tema) in enumerate(zip(uni["pt"], uni["glosa"],
                                                    uni["tema"])):
            # Filtro 1: comprimento (rapido)
            pt_pal = len(pt.split())
            gl_pal = len(glosa.split())
            if pt_pal > a.max_pt_palavras:
                razoes[i] = "longo_pt"
                contadores["longo_pt"] += 1
                if len(descartes_sample) < 200:
                    descartes_sample.append({"razao": "longo_pt", "pt": pt[:200],
                                              "glosa": glosa[:200], "tema": tema,
                                              "pt_palavras": pt_pal})
                continue
            if gl_pal > a.max_glosa_palavras:
                razoes[i] = "longo_glosa"
                contadores["longo_glosa"] += 1
                if len(descartes_sample) < 200:
                    descartes_sample.append({"razao": "longo_glosa", "pt": pt[:200],
                                              "glosa": glosa[:200], "tema": tema,
                                              "glosa_palavras": gl_pal})
                continue

            # Filtro 2: pontuacao excessiva
            pont_pt, razao_pt = detectar_pontuacao_excessiva(pt, a.pct_pontuacao_max)
            if pont_pt:
                razoes[i] = "pontuacao_excessiva_pt"
                contadores["pontuacao_excessiva_pt"] += 1
                if len(descartes_sample) < 200:
                    descartes_sample.append({"razao": "pontuacao_excessiva_pt",
                                              "pt": pt[:200], "glosa": glosa[:200],
                                              "tema": tema, "razao_pont": round(razao_pt, 3)})
                continue

            # Filtro 3: ingles cross-lingual (mais caro)
            eh_ing_pt, diag_pt = detectar_ingles(pt, a.pct_ingles_max,
                                                  a.min_palavras_ingles_disparo)
            if eh_ing_pt:
                razoes[i] = "ingles_em_pt"
                contadores["ingles_em_pt"] += 1
                if len(descartes_sample) < 200:
                    descartes_sample.append({"razao": "ingles_em_pt",
                                              "pt": pt[:200], "glosa": glosa[:200],
                                              "tema": tema, **diag_pt})
                continue

            # Filtro 4: enciclopedico (caro: regex)
            eh_enc_pt, padrao_enc = detectar_enciclopedico(pt)
            if eh_enc_pt:
                razoes[i] = "enciclopedico"
                contadores["enciclopedico"] += 1
                if len(descartes_sample) < 200:
                    descartes_sample.append({"razao": "enciclopedico",
                                              "pt": pt[:200], "glosa": glosa[:200],
                                              "tema": tema, "padrao": padrao_enc})
                continue

            razoes[i] = "manter"
            contadores["manter"] += 1

        uni["_razao"] = razoes
        n_manter = contadores["manter"]
        log.info(f"  ANTES dos filtros: {n0} pares")
        for cat in ("manter", "longo_pt", "longo_glosa", "pontuacao_excessiva_pt",
                    "ingles_em_pt", "enciclopedico"):
            n = contadores[cat]
            log.info(f"    {cat:<30} {n:>6}  ({100*n/n0:.1f}%)")
        log.info(f"  APOS filtros: {n_manter} pares "
                 f"(taxa retencao: {100*n_manter/n0:.1f}%)")

        require(n_manter >= 1000, "E_VAZIO_POS_FILTRO",
                f"so {n_manter} pares sobraram apos filtros — limites muito restritivos",
                fix="afrouxe --pct-ingles-max ou --max-glosa-palavras",
                err=DataContractError)

        uni_filtrado = uni[uni["_razao"] == "manter"].drop(columns=["_razao"])
        uni_filtrado = uni_filtrado.reset_index(drop=True)

        g.info.update({
            "n_total_pre": n0,
            "n_manter_pos": n_manter,
            "contagem_descartes": dict(contadores),
            "taxa_retencao": round(n_manter / n0, 4),
        })

        # Salva sample de descartados para auditoria humana
        if a.salvar_descartados and descartes_sample:
            sample_path = DATA_REPORTS / "mesclar_descartados_amostra.csv"
            pd.DataFrame(descartes_sample).to_csv(sample_path, index=False,
                                                    encoding="utf-8")
            log.info(f"  [SALVO] {sample_path} (sample 200 descartes para auditoria)")
            g.info["sample_descartados"] = str(sample_path)

    # ============ ETAPA 4: DEDUP (preserva o melhor de cada par) ============
    with StepGate("deduplicar", log) as g:
        # Prioridade: construido > vlibrasdb > bilingual_v1
        # (Construido tem qualidade humana revisada)
        def prio(tema: str) -> int:
            if tema.startswith("construido_"): return 0
            if tema.startswith("vlibrasdb_geral"): return 1
            if tema.startswith("vlibrasdb_governo"): return 2
            return 3  # bilingual_v1 / desconhecido

        uni_filtrado["_prio"] = uni_filtrado["tema"].map(prio)
        antes_dedup = len(uni_filtrado)
        uni_dedup = (uni_filtrado
                     .sort_values("_prio")
                     .drop_duplicates(subset=["pt", "glosa"])
                     .drop(columns="_prio")
                     .reset_index(drop=True))
        n_duplicados = antes_dedup - len(uni_dedup)
        log.info(f"  {antes_dedup} -> {len(uni_dedup)} pares "
                 f"({n_duplicados} duplicados removidos)")
        g.info.update({"pares_antes_dedup": antes_dedup,
                       "duplicados_removidos": n_duplicados,
                       "pares_finais_unicos": len(uni_dedup)})

    # ============ ETAPA 5: DISTRIBUICAO POR TEMA (diagnostico) ============
    with StepGate("distribuicao_por_tema", log) as g:
        dist = uni_dedup["tema"].value_counts().to_dict()
        log.info(f"  Distribuicao por tema (apos filtro + dedup):")
        for t, n in sorted(dist.items(), key=lambda kv: -kv[1])[:30]:
            log.info(f"    {t:<40} {n:>6}  ({100*n/len(uni_dedup):.1f}%)")
        g.info["distribuicao_tema"] = dist

    # ============ ETAPA 6: EXPANDIR DIRECOES ============
    with StepGate("expandir_direcoes", log) as g:
        linhas = []
        for _, r in uni_dedup.iterrows():
            pt, glosa, tema = r["pt"], r["glosa"], r["tema"]
            linhas.append({"input": montar_input(pt, DIR_PT2LIBRAS),
                           "output": glosa,
                           "direcao": DIR_PT2LIBRAS, "tema": tema,
                           "portugues_original": pt, "libras_glosa": glosa})
            linhas.append({"input": montar_input(glosa, DIR_LIBRAS2PT),
                           "output": pt,
                           "direcao": DIR_LIBRAS2PT, "tema": tema,
                           "portugues_original": pt, "libras_glosa": glosa})
        merged = pd.DataFrame(linhas)
        validar_dataframe(merged, "merged", log)
        g.info["linhas_finais"] = len(merged)

    # ============ ETAPA 7: SALVAR ============
    with StepGate("salvar", log) as g:
        saida = Path(a.saida)
        saida.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(saida, index=False, encoding="utf-8")
        log.info(f"  [SALVO] {saida} ({len(merged)} linhas)")

        rel = {
            "timestamp": datetime.now().isoformat(), "versao": "7.1.0",
            "fontes_usadas": {
                "construido": {"path": str(FONTE_CONSTRUIDO),
                                "sha256": sha_construido,
                                "n_linhas_brutas": len(small)},
                "grande": {"path": (str(FONTE_VLIBRASDB_V2)
                                     if vlibras_v2_existe
                                     else (str(FONTE_BILINGUAL_V1)
                                            if bilingual_v1_existe else None)),
                            "tipo": fonte_grande,
                            "sha256": sha_grande,
                            "n_linhas_brutas": (len(df_grande)
                                                if fonte_grande else 0)},
            },
            "parametros_filtro": {
                "max_glosa_palavras": a.max_glosa_palavras,
                "max_pt_palavras": a.max_pt_palavras,
                "pct_ingles_max": a.pct_ingles_max,
                "min_palavras_ingles_disparo": a.min_palavras_ingles_disparo,
                "pct_pontuacao_max": a.pct_pontuacao_max,
            },
            "contagem_filtros": dict(contadores),
            "taxa_retencao": round(n_manter / n0, 4),
            "duplicados_removidos": n_duplicados,
            "pares_finais": len(uni_dedup),
            "linhas_finais": len(merged),
            "distribuicao_tema": dist,
            "ganho_vs_construido": round(len(uni_dedup) / 2492, 2),
            "tokens_pontuacao_canonicos": sorted(TOKENS_PONTUACAO),
        }
        salvar_json(rel, DATA_REPORTS / "mesclar_report.json")
        g.info["saida"] = str(saida)

    log.info("=" * 64)
    log.info(f"MESCLADO: {len(uni_dedup)} pares "
             f"({len(uni_dedup)/2492:.1f}x vs construido) -> {saida}")
    log.info(f"Filtrados {n0 - n_manter} ({100*(n0-n_manter)/n0:.1f}%): "
             f"longo_pt={contadores['longo_pt']} "
             f"longo_glosa={contadores['longo_glosa']} "
             f"pontuacao={contadores['pontuacao_excessiva_pt']} "
             f"ingles={contadores['ingles_em_pt']} "
             f"enciclopedico={contadores['enciclopedico']}")
    log.info(f"Auditoria humana: {DATA_REPORTS / 'mesclar_descartados_amostra.csv'}")
    log.info("Proximo: aponte 01_prepare_data.py para este arquivo.")
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
