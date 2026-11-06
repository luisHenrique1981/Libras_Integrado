#!/usr/bin/env python3
"""
04_evaluate.py v7 — Avaliacao com GPS MULTI-VERSAO de proxima acao.

FILOSOFIA: a avaliacao final NAO PODE ser apenas constatacao. Tem que apontar
o que fazer a seguir. Este script integra:
  1. Eval batched no hold-out (mantido da v5/v6)
  2. Categorizacao dos erros (espelha tools/analise_erros.py)
  3. Mineracao do vocabulario raro alvo (que palavras o modelo falha + sao raras
     no treino -> alvos prioritarios para coleta de dados nova)
  4. Comparacao com TODO o historico, nao so o anterior (v7)
  5. GPS: arvore de decisao baseada em medidas, emitindo:
       diagnostico + acao_prioritaria + razao + trajetoria + melhor_historico

MUDANCAS v7 vs v6:
  + GPS analisa TRAJETORIA completa, nao so delta vs run anterior:
    - delta_vs_anterior (mantido)
    - delta_vs_melhor_historico (NOVO): identifica se voltamos atras
    - delta_vs_baseline (NOVO): progresso total desde o inicio
  + Novo diagnostico 'ABAIXO DA MELHOR HISTORICA': melhorou vs anterior mas
    ainda esta abaixo do melhor que ja tivemos
  + Novo diagnostico 'NOVO MELHOR HISTORICO': estabeleceu novo recorde
  + PLATEAU agora distingue '2 runs' de 'PROLONGADO (3+)' com acoes diferentes
  + Trajetoria detalhada salva no GPS json + impressa no log
  + Flag --inserir-historico-retroativo: insere v5 (chrf 79.35/57.96) no
    historico para que o GPS compare automaticamente com baseline pre-v6.
    Idempotente (so insere se v5 ainda nao consta).

Saidas:
  data/reports/evaluate_report.json        (run atual)
  data/reports/historico_avaliacoes.json   (append-only, com v5 retroativo)
  data/reports/gps_recomendacao.json       (proxima acao + trajetoria)

Uso:
  python scripts/04_evaluate.py --save-errors
  python scripts/04_evaluate.py --save-errors --inserir-historico-retroativo
  python scripts/04_evaluate.py --model-dir artifacts/models/libras_v7
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.common import (  # noqa: E402
    TEST_CSV, TRAIN_CSV, MAX_INPUT, MAX_TARGET, ART_MODELS, ART_LOGS, DATA_REPORTS,
    get_logger, salvar_json, require, StepGate, validar_dataframe,
    calcular_metricas, DataContractError,
)

log = get_logger("evaluate", ART_LOGS / "evaluate.log")

HISTORICO_PATH = DATA_REPORTS / "historico_avaliacoes.json"
GPS_PATH = DATA_REPORTS / "gps_recomendacao.json"

# Stopwords PT compactas: suficientes para limpar mineracao de vocab dos erros.
STOPWORDS = {
    "que", "para", "dos", "das", "uma", "uns", "umas", "isso", "isto", "aquilo",
    "quais", "como", "quando", "onde", "pela", "pelo", "com", "ser", "ter", "estar",
    "ir", "vir", "fazer", "voce", "nao", "sim", "mas", "mais", "menos", "muito",
    "muita", "tao", "ainda", "tambem", "porque", "este", "esta", "esses", "essas",
    "neste", "nesta", "naquele", "naquela", "ele", "ela", "eles", "elas", "seu",
    "sua", "seus", "suas", "meu", "minha", "meus", "minhas", "nosso", "nossa",
}
RE_PALAVRA = re.compile(r"\b\w{4,}\b", re.UNICODE)
_RE_BRACKET = re.compile(r"\[[^\]]*\]")


def _detectar_modelo_padrao() -> Path:
    """Encontra o diretorio de modelo mais recente em artifacts/models/libras_v*.
    Fallback: libras_v7 (a ser produzido pelo 03_train v7). Se quiser usar
    versao especifica, passe --model-dir explicito.
    """
    cands = sorted(ART_MODELS.glob("libras_v*"),
                   key=lambda p: p.stat().st_mtime if p.exists() else 0,
                   reverse=True)
    return cands[0] if cands else ART_MODELS / "libras_v7"


def carregar_modelo(model_dir: Path, device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    require(model_dir.exists(), "E_MODELO_AUSENTE",
            f"nao encontrado: {model_dir}", fix="rode 03_train.py primeiro")
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(device)
        log.info(f"Modelo merged carregado: {model_dir}")
        return model, tok
    except Exception:
        from peft import PeftModel, PeftConfig
        pc = PeftConfig.from_pretrained(str(model_dir))
        base_id = pc.base_model_name_or_path
        if base_id is None:
            raise DataContractError(
                code="E_PEFT_SEM_BASE",
                msg=f"PeftConfig em {model_dir} sem base_model_name_or_path",
                fix="re-treine com 03_train.py ou aponte para o modelo merged")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_id)
        model = cast(Any, PeftModel.from_pretrained(base, str(model_dir))).to(device)
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        log.info("Modelo LoRA (nao-merged) carregado")
        return model, tok


def avaliar(model, tok, df, device, batch_size: int = 32) -> dict:
    import torch
    resultados = {}
    for direcao in sorted(df["direcao"].unique()):
        sub = df[df["direcao"] == direcao].reset_index(drop=True)
        log.info(f"Avaliando {direcao}: {len(sub)} pares (batch={batch_size})")
        inputs = [str(x) for x in sub["input"]]
        refs = [str(x).strip() for x in sub["output"]]
        preds: list[str] = []
        for i in range(0, len(inputs), batch_size):
            lote = inputs[i:i + batch_size]
            try:
                enc = tok(lote, return_tensors="pt", padding=True,
                          max_length=MAX_INPUT, truncation=True).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_TARGET, num_beams=4)
                preds.extend(tok.batch_decode(out, skip_special_tokens=True))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for txt in lote:
                    try:
                        e1 = tok(txt, return_tensors="pt", max_length=MAX_INPUT,
                                 truncation=True).to(device)
                        with torch.no_grad():
                            o1 = model.generate(**e1, max_new_tokens=MAX_TARGET, num_beams=4)
                        preds.append(tok.decode(o1[0], skip_special_tokens=True))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        preds.append("[OOM]")
            if (i // batch_size) % 5 == 0:
                log.info(f"  {min(i + batch_size, len(inputs))}/{len(inputs)}")
        preds = [p.strip() for p in preds]
        erros = [{"direcao": direcao, "input": inp, "esperado": r, "obtido": p}
                 for inp, r, p in zip(inputs, refs, preds) if p.upper() != r.upper()]
        m = calcular_metricas(preds, refs)
        m["n"] = len(sub); m["n_erros"] = len(erros)
        m["erros_amostra"] = erros[:20]
        m["_erros_todos"] = erros
        resultados[direcao] = m
        log.info(f"  [{direcao}] chrF={m['chrf']:.2f} BLEU4={m['bleu4']:.2f} "
                 f"EM={m['exact_match']:.2f}% (n={len(sub)})")
    return resultados


# ─────────────── CATEGORIZACAO DE ERROS (espelha tools/analise_erros.py) ──────
def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm_brackets(s: str) -> str:
    return _RE_BRACKET.sub(
        lambda m: "[" + _sem_acento(m.group(0)[1:-1]).upper() + "]", s)


def _strip_pont(s: str) -> str:
    out = s
    for t in ("[PONTO]", "[INTERROGACAO]", "[EXCLAMACAO]"):
        out = out.replace(t, "")
    return re.sub(r"\s+", " ", out).strip()


def categorizar(esperado: str, obtido: str, chrf_par: float, limiar_quase: float) -> str:
    e, o = esperado.strip(), obtido.strip()
    if e.upper() == o.upper():
        return "sem_erro"
    ne, no = _norm_brackets(e).upper(), _norm_brackets(o).upper()
    if ne == no:
        return "token_especial"
    if _strip_pont(ne) == _strip_pont(no):
        return "pontuacao"
    if chrf_par >= limiar_quase:
        return "quase_acerto"
    return "erro_real"


def categorizar_todos(erros: list[dict], limiar_quase: float) -> dict:
    import sacrebleu as sb
    cats: Counter = Counter()
    exemplos: dict[str, list] = {}
    erros_reais: list[dict] = []
    for e in erros:
        esp, obt = str(e["esperado"]), str(e["obtido"])
        chrf_par = sb.sentence_chrf(obt, [esp], word_order=2).score
        cat = categorizar(esp, obt, chrf_par, limiar_quase)
        cats[cat] += 1
        exemplos.setdefault(cat, [])
        if len(exemplos[cat]) < 6:
            exemplos[cat].append({"esperado": esp, "obtido": obt,
                                  "chrf_par": round(chrf_par, 1)})
        if cat == "erro_real":
            erros_reais.append({**e, "chrf_par": chrf_par})
    total = max(sum(cats.values()), 1)
    return {
        "n_erros": sum(cats.values()),
        "categorias": {k: {"n": v, "pct": round(100 * v / total, 1)}
                       for k, v in cats.most_common()},
        "exemplos": exemplos,
        "_erros_reais": erros_reais,  # consumido pela mineracao
    }


# ─────────────── MINERACAO DE VOCABULARIO ALVO ────────────────────────────────
def palavras_de(s: str) -> list[str]:
    return [w.lower() for w in RE_PALAVRA.findall(str(s))
            if w.lower() not in STOPWORDS]


def minerar_vocab_alvo(erros_reais: list[dict], train_csv: Path,
                        top_n: int = 30) -> list[dict]:
    """Identifica palavras raras no treino que aparecem nos erros_reais.
    score = n_em_erros / (n_no_treino + 1) -- maior = mais subrepresentada
    no treino e mais problematica no eval. Ranking sugere alvos de coleta.
    """
    import pandas as pd
    df_tr = pd.read_csv(train_csv).dropna(subset=["output"])
    freq_treino: Counter = Counter()
    for txt in df_tr["output"]:
        for w in palavras_de(txt):
            freq_treino[w] += 1
    n_palavras: Counter = Counter()
    n_ocorrencias_erros: Counter = Counter()
    for e in erros_reais:
        palavras_unicas_esperado = set(palavras_de(e["esperado"]))
        palavras_unicas_obtido = set(palavras_de(e["obtido"]))
        # palavras-alvo: o que ESTAVA esperado mas o modelo errou
        para_alvejar = palavras_unicas_esperado - palavras_unicas_obtido
        for w in para_alvejar:
            n_palavras[w] += 1
        for w in palavras_unicas_esperado:
            n_ocorrencias_erros[w] += 1
    pontuados = []
    for w, n_err in n_palavras.most_common():
        n_tr = freq_treino.get(w, 0)
        score = n_err / (n_tr + 1)
        if n_err >= 2:  # filtra ruido (palavras unicas em 1 erro nao priorizam)
            pontuados.append({"palavra": w, "n_erros": n_err, "n_treino": n_tr,
                              "score_raridade": round(score, 3)})
    pontuados.sort(key=lambda x: -x["score_raridade"])
    return pontuados[:top_n]


# ─────────────── HISTORICO ────────────────────────────────────────────────────
def carregar_historico() -> list[dict]:
    if not HISTORICO_PATH.exists():
        return []
    try:
        return json.loads(HISTORICO_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning(f"historico_avaliacoes.json corrompido; iniciando novo")
        return []


def adicionar_historico(historico: list[dict], entrada: dict) -> list[dict]:
    historico.append(entrada)
    HISTORICO_PATH.write_text(
        json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8")
    return historico


# ─────────────── GPS: ARVORE DE DECISAO ───────────────────────────────────────
def gerar_gps(metricas: dict, breakdown: dict, vocab_alvo: list[dict],
              historico: list[dict], limiares: dict) -> dict:
    """
    Decide a proxima acao com base em medidas, nao em opiniao. v7: analise
    MULTI-VERSAO — compara contra TODO o historico, identifica trajetoria
    (melhora/regressao/estagnacao), aponta melhor versao historica e
    quantifica delta vs ela tambem.

    Estrutura do historico esperado (lista append-only, mais novo NO FIM):
      [
        {timestamp, model_dir, chrf_pt2libras, chrf_libras2pt, chrf_medio,
         bleu4_*, n_teste, diagnostico},
        ...
      ]
    """
    p2l = metricas["pt2libras"]["chrf"]
    l2p = metricas["libras2pt"]["chrf"]
    chrf_medio = (p2l + l2p) / 2
    gap = p2l - l2p

    # ---- Trajetoria: comparacoes vs anterior, melhor historico, primeiro ----
    delta_anterior = None
    delta_melhor = None
    delta_baseline = None
    melhor_historico = None
    info_anterior = None
    info_baseline = None
    trajetoria = []

    if historico:
        # Anterior: ultimo registro
        info_anterior = historico[-1]
        delta_anterior = chrf_medio - info_anterior.get("chrf_medio", chrf_medio)
        # Melhor historico: maior chrf_medio ate agora
        melhor_historico = max(historico,
                                key=lambda h: h.get("chrf_medio", 0.0))
        delta_melhor = chrf_medio - melhor_historico.get("chrf_medio", 0.0)
        # Baseline: primeiro registro
        info_baseline = historico[0]
        delta_baseline = chrf_medio - info_baseline.get("chrf_medio", chrf_medio)
        # Trajetoria: serie cronologica resumida
        for i, h in enumerate(historico):
            trajetoria.append({
                "i": i,
                "model_dir": Path(h.get("model_dir", "?")).name,
                "chrf_medio": round(h.get("chrf_medio", 0.0), 2),
                "chrf_p2l": round(h.get("chrf_pt2libras", 0.0), 2),
                "chrf_l2p": round(h.get("chrf_libras2pt", 0.0), 2),
                "diagnostico": h.get("diagnostico", "?"),
            })
        # Inclui a entrada ATUAL no fim da trajetoria (so para visualizacao;
        # nao e salva ainda neste momento)
        trajetoria.append({
            "i": len(historico),
            "model_dir": "ATUAL",
            "chrf_medio": round(chrf_medio, 2),
            "chrf_p2l": round(p2l, 2),
            "chrf_l2p": round(l2p, 2),
            "diagnostico": "(em avaliacao)",
        })

    # Categorias dominantes
    cats_p2l = breakdown["pt2libras"]["categorias"]
    cats_l2p = breakdown["libras2pt"]["categorias"]
    pct_quase_p2l = cats_p2l.get("quase_acerto", {}).get("pct", 0)
    pct_quase_l2p = cats_l2p.get("quase_acerto", {}).get("pct", 0)
    pct_real_p2l = cats_p2l.get("erro_real", {}).get("pct", 0)
    pct_real_l2p = cats_l2p.get("erro_real", {}).get("pct", 0)

    # Bloco comum em qualquer veredito
    contexto_multi = {
        "chrf_medio_atual": round(chrf_medio, 2),
        "delta_vs_anterior": (round(delta_anterior, 2)
                                if delta_anterior is not None else None),
        "delta_vs_melhor_historico": (round(delta_melhor, 2)
                                        if delta_melhor is not None else None),
        "delta_vs_baseline": (round(delta_baseline, 2)
                                if delta_baseline is not None else None),
        "melhor_historico": ({
            "model_dir": Path(melhor_historico.get("model_dir", "?")).name,
            "chrf_medio": round(melhor_historico.get("chrf_medio", 0.0), 2),
            "timestamp": melhor_historico.get("timestamp", "?"),
        } if melhor_historico else None),
        "trajetoria": trajetoria,
        "n_historico": len(historico),
    }

    # ---- Arvore de decisao (avaliada em ordem de prioridade) ----
    if not historico:
        return {
            "diagnostico": "BASELINE",
            "acao_prioritaria": ("Registro inicial. Rode outra iteracao apos "
                                 "ajustar config. Comparacoes inteligentes "
                                 "comecam a partir da 2a run."),
            "razao": "primeira execucao registrada; sem historico para comparar",
            "estamos_no_teto": False,
            "delta_chrf": None,
            **contexto_multi,
        }

    # REGRESSAO vs ANTERIOR (caso mais urgente)
    if delta_anterior is not None and delta_anterior < -limiares["delta_significativo"]:
        return {
            "diagnostico": "REGRESSAO vs run anterior",
            "acao_prioritaria": (f"INVESTIGAR. Comparar config_campeao.json e "
                                 f"prepare_report.json desta run "
                                 f"({Path(metricas.get('_model_dir','?')).name}) "
                                 f"com a anterior "
                                 f"({Path(info_anterior.get('model_dir','?')).name}). "
                                 f"Verificar se houve mudanca de corpus, filtro, ou "
                                 f"config sem benchmark adequado. NAO rode 03_train "
                                 f"de novo ate identificar a causa."),
            "razao": (f"chrF medio caiu {-delta_anterior:.2f} pontos vs run "
                      f"anterior ({info_anterior.get('chrf_medio', 0):.2f})"),
            "estamos_no_teto": False,
            "delta_chrf": round(delta_anterior, 2),
            **contexto_multi,
        }

    # REGRESSAO vs MELHOR HISTORICO (mesmo se melhorou vs anterior)
    # Caso classico: depois de uma regressao, a "recuperacao" e parcial
    if delta_melhor is not None and delta_melhor < -limiares["delta_significativo"]:
        return {
            "diagnostico": "ABAIXO DA MELHOR HISTORICA",
            "acao_prioritaria": (
                f"O melhor modelo historico foi "
                f"{Path(melhor_historico.get('model_dir','?')).name} com "
                f"chrF={melhor_historico.get('chrf_medio',0):.2f}. Atual "
                f"esta {-delta_melhor:.2f} pontos abaixo. Analise o "
                f"prepare_report e config_campeao DAQUELA run para identificar "
                f"o que mudou. Considere voltar para aquela config ou aquele "
                f"corpus."),
            "razao": (f"chrF medio {chrf_medio:.2f} < melhor historico "
                      f"{melhor_historico.get('chrf_medio',0):.2f} "
                      f"(delta {delta_melhor:+.2f})"),
            "estamos_no_teto": False,
            "delta_chrf": round(delta_anterior, 2) if delta_anterior is not None else None,
            **contexto_multi,
        }

    # PLATEAU (3+ runs consecutivas sem mudanca significativa)
    if (delta_anterior is not None and abs(delta_anterior) < limiares["delta_significativo"]
            and len(historico) >= 2):
        # Verificar se as 2 anteriores tambem estavam no plateau
        plateau_recorrente = False
        if len(historico) >= 2:
            antes = historico[-2].get("chrf_medio", 0.0)
            anterior = historico[-1].get("chrf_medio", 0.0)
            plateau_recorrente = abs(anterior - antes) < limiares["delta_significativo"]
        if plateau_recorrente:
            return {
                "diagnostico": "PLATEAU PROLONGADO (3+ runs)",
                "acao_prioritaria": (
                    "3+ runs no mesmo patamar. Mudancas incrementais nao movem o "
                    "ponteiro. Tente UMA mudanca grande (uma de cada vez para "
                    "isolar): (a) DOBRAR DADOS (preparar_vlibras_db v3, outras "
                    "fontes), (b) lora_r=64 fixo no config, (c) modelo base "
                    "diferente (t5-large), (d) reformular filtros do "
                    "mesclar_datasets."),
                "razao": (f"3+ runs com chrF medio em [{anterior-1:.1f}, "
                          f"{anterior+1:.1f}]"),
                "estamos_no_teto": False,
                "delta_chrf": round(delta_anterior, 2),
                **contexto_multi,
            }
        return {
            "diagnostico": "PLATEAU (2 runs)",
            "acao_prioritaria": ("Mudancas incrementais nao movem mais o ponteiro. "
                                 "Tente UMA mudanca grande: (a) dobrar dados, "
                                 "(b) lora_r=64, ou (c) re-HPO com espaco ampliado. "
                                 "Isolar a variavel."),
            "razao": (f"delta {delta_anterior:+.2f} chrF < "
                      f"{limiares['delta_significativo']}; 2 runs no mesmo patamar"),
            "estamos_no_teto": False,
            "delta_chrf": round(delta_anterior, 2),
            **contexto_multi,
        }

    if pct_quase_p2l >= limiares["quase_saturacao"] and p2l >= 70:
        return {
            "diagnostico": "TETO DA METRICA em pt2libras",
            "acao_prioritaria": ("pt2libras saturou para a metrica de referencia "
                                 "unica. Acoes: (1) avaliacao multi-referencia "
                                 "(2-3 PT validas por glosa), (2) avaliacao humana "
                                 "em amostra; foco do treino agora em libras2pt."),
            "razao": (f"{pct_quase_p2l:.0f}% dos erros em pt2libras sao "
                      f"quase-acertos (chrF par >= 60); chrF p2l={p2l:.1f}"),
            "estamos_no_teto": True,
            "delta_chrf": (round(delta_anterior, 2)
                            if delta_anterior is not None else None),
            **contexto_multi,
        }

    if pct_real_l2p >= limiares["erro_real_dominante"] or pct_real_p2l >= 35:
        alvo = vocab_alvo[:15]
        return {
            "diagnostico": "COBERTURA DE DADOS INSUFICIENTE",
            "acao_prioritaria": (
                f"Coletar 200-500 frases novas contendo as palavras de "
                f"'vocabulario_alvo' (top 15 priorizadas por raridade no treino). "
                f"Foco maior na direcao libras2pt (erro_real {pct_real_l2p:.0f}%). "
                f"Apos coletar, rode mesclar_datasets + 01 + 02 + 03 + 04."),
            "razao": (f"erro_real domina: libras2pt {pct_real_l2p:.0f}%, "
                      f"pt2libras {pct_real_p2l:.0f}%. Padrao tipico de "
                      f"alucinacao OOV/morfologica em palavras subrepresentadas."),
            "palavras_alvo_top5": [p["palavra"] for p in alvo[:5]],
            "estamos_no_teto": False,
            "delta_chrf": (round(delta_anterior, 2)
                            if delta_anterior is not None else None),
            **contexto_multi,
        }

    if abs(gap) >= limiares["gap_direcional"]:
        direcao_fraca = "libras2pt" if gap > 0 else "pt2libras"
        return {
            "diagnostico": "ASSIMETRIA DIRECIONAL",
            "acao_prioritaria": (
                f"Direcao {direcao_fraca} esta {abs(gap):.1f} chrF abaixo da "
                f"forte. Considere: (1) oversample de exemplos da direcao "
                f"fraca durante treino, (2) avaliar se {direcao_fraca} tem "
                f"componente intrinseco (libras2pt e 1-para-muitos por "
                f"design), (3) coletar dados que ENFATIZEM {direcao_fraca}."),
            "razao": f"gap de {abs(gap):.1f} chrF entre direcoes",
            "estamos_no_teto": False,
            "delta_chrf": (round(delta_anterior, 2)
                            if delta_anterior is not None else None),
            **contexto_multi,
        }

    if chrf_medio < limiares["regime_baixo"]:
        return {
            "diagnostico": "REGIME INSUFICIENTE",
            "acao_prioritaria": (
                "Modelo subajustado. Acoes: (1) Re-HPO com lora_r ate 64 e "
                "lora_alpha ate 128, (2) verificar se dados tem ruido excessivo "
                "(inspecione mesclar_descartados_amostra.csv), (3) verificar lr "
                "no log do trial campeao (muito alto/baixo aparece como "
                "instabilidade)."),
            "razao": (f"chrF medio {chrf_medio:.1f} abaixo de "
                      f"{limiares['regime_baixo']}"),
            "estamos_no_teto": False,
            "delta_chrf": (round(delta_anterior, 2)
                            if delta_anterior is not None else None),
            **contexto_multi,
        }

    # MELHORIA NOVA (passou do anterior E do melhor historico)
    if (delta_anterior is not None and delta_anterior > limiares["delta_significativo"]
            and (delta_melhor is None or delta_melhor > 0)):
        return {
            "diagnostico": "NOVO MELHOR HISTORICO",
            "acao_prioritaria": (
                f"Esta run estabeleceu novo melhor (chrF medio {chrf_medio:.2f}, "
                f"+{delta_anterior:.2f} vs anterior). "
                f"Considere: (1) consolidar fazendo eval em conjuntos externos, "
                f"(2) seguir coletando dados nas areas subcobertas, (3) tentar "
                f"otimizacoes adicionais (lora_r, dropout, mais epocas)."),
            "razao": (f"chrF medio {chrf_medio:.2f} > anterior "
                      f"{info_anterior.get('chrf_medio',0):.2f}"),
            "estamos_no_teto": False,
            "delta_chrf": round(delta_anterior, 2),
            **contexto_multi,
        }

    return {
        "diagnostico": "PROGRESSO SAUDAVEL",
        "acao_prioritaria": (
            "Continuar a iteracao: coletar mais dados das areas subcobertas "
            "(ver vocabulario_alvo), rodar mesclar + 01 + 02 + 03 + 04 de novo."),
        "razao": f"chrF medio {chrf_medio:.1f}, sem padrao patologico",
        "estamos_no_teto": False,
        "delta_chrf": (round(delta_anterior, 2)
                        if delta_anterior is not None else None),
        **contexto_multi,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None,
                   help="Padrao: detecta o libras_v* mais recente em artifacts/models/")
    p.add_argument("--save-errors", action="store_true",
                   help="exporta data/reports/evaluate_erros.csv")
    p.add_argument("--limiar-quase", type=float, default=60.0,
                   help="chrF par-a-par acima do qual erro vira quase_acerto")
    p.add_argument("--limiar-erro-real-dominante", type=float, default=50.0,
                   help="%% erro_real para ativar a recomendacao COBERTURA DE DADOS")
    p.add_argument("--limiar-quase-saturacao", type=float, default=70.0,
                   help="%% quase_acerto para ativar TETO DA METRICA")
    p.add_argument("--limiar-gap-direcional", type=float, default=15.0,
                   help="diferenca de chrF entre direcoes para ativar ASSIMETRIA")
    p.add_argument("--limiar-regime-baixo", type=float, default=40.0,
                   help="chrF medio abaixo do qual diagnostica REGIME INSUFICIENTE")
    p.add_argument("--delta-significativo", type=float, default=1.0,
                   help="chrF de melhoria minima entre runs para nao ser PLATEAU/REGRESSAO")
    p.add_argument("--inserir-historico-retroativo", action="store_true",
                   help="ANTES de avaliar, insere entrada retroativa v5 no historico "
                        "(se nao existir). Util na 1a run da v7 para que o GPS "
                        "compare automaticamente com o baseline v5. Idempotente.")
    a = p.parse_args()

    # ============ INSERCAO RETROATIVA v5 (idempotente) ============
    if a.inserir_historico_retroativo:
        hist = carregar_historico()
        ja_tem_v5 = any("v5" in str(h.get("model_dir", "")).lower()
                         or "MANUAL_RETROATIVO_v5" == h.get("diagnostico", "")
                         for h in hist)
        if ja_tem_v5:
            log.info("v5 ja consta no historico; pulando insercao retroativa")
        else:
            v5_entry = {
                "timestamp": "2026-06-02T18:51:31",
                "model_dir": "v5_baseline_retroativo",
                "chrf_pt2libras": 79.35,
                "chrf_libras2pt": 57.96,
                "chrf_medio": 68.66,
                "bleu4_pt2libras": 68.39,
                "bleu4_libras2pt": 35.88,
                "n_teste": 750,
                "diagnostico": "MANUAL_RETROATIVO_v5",
            }
            hist.insert(0, v5_entry)
            HISTORICO_PATH.parent.mkdir(parents=True, exist_ok=True)
            import json
            HISTORICO_PATH.write_text(
                json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info(f"Entrada v5 inserida em {HISTORICO_PATH} "
                     f"(agora {len(hist)} entradas)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    model_dir = Path(a.model_dir) if a.model_dir else _detectar_modelo_padrao()
    log.info(f"Modelo: {model_dir}")

    with StepGate("dados_teste", log) as g:
        require(TEST_CSV.exists(), "E_TESTE_AUSENTE",
                f"teste nao encontrado: {TEST_CSV}", fix="rode 01_prepare_data.py")
        import pandas as pd
        df = pd.read_csv(TEST_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
        validar_dataframe(df, "teste", log)
        g.info["n"] = len(df)

    with StepGate("carregar_modelo", log):
        model, tok = carregar_modelo(model_dir, device)

    with StepGate("avaliar", log) as g:
        t0 = time.time()
        res = avaliar(model, tok, df, device)
        g.info["duracao_s"] = round(time.time() - t0, 1)

    # ─── Salvar erros em CSV (consumo do analise_erros.py legacy) ─────────────
    todos_erros = []
    for r in res.values():
        todos_erros.extend(r.get("_erros_todos", []))
    if a.save_errors and todos_erros:
        import pandas as pd
        ep = DATA_REPORTS / "evaluate_erros.csv"
        pd.DataFrame(todos_erros).to_csv(ep, index=False, encoding="utf-8")
        log.info(f"[SALVO] {ep} ({len(todos_erros)} erros)")

    # ─── Categorizar erros por direcao ────────────────────────────────────────
    with StepGate("categorizar_erros", log) as g:
        breakdown: dict = {}
        erros_reais_concat: list[dict] = []
        for direcao, r in res.items():
            erros = r.get("_erros_todos", [])
            bd = categorizar_todos(erros, a.limiar_quase)
            erros_reais_concat.extend(bd.pop("_erros_reais"))
            breakdown[direcao] = bd
            log.info(f"  [{direcao}] {bd['n_erros']} erros: " +
                     " | ".join(f"{k}={v['pct']:.0f}%"
                                for k, v in bd['categorias'].items()))
        g.info["categorias"] = {d: b["categorias"] for d, b in breakdown.items()}

    # ─── Minerar vocabulario alvo ─────────────────────────────────────────────
    with StepGate("minerar_vocab_alvo", log) as g:
        vocab_alvo = minerar_vocab_alvo(erros_reais_concat, TRAIN_CSV, top_n=30)
        log.info(f"  Top-10 palavras priorizadas para coleta:")
        for v in vocab_alvo[:10]:
            log.info(f"    {v['palavra']:<20} n_erros={v['n_erros']:>3} "
                     f"n_treino={v['n_treino']:>4} score={v['score_raridade']}")
        g.info["top10_alvo"] = vocab_alvo[:10]

    # ─── GPS ──────────────────────────────────────────────────────────────────
    with StepGate("gps_proxima_acao", log) as g:
        # Limpa erros_todos do dict de metricas antes de persistir
        metricas_simples = {}
        for d, r in res.items():
            metricas_simples[d] = {k: v for k, v in r.items() if k != "_erros_todos"}
        chrf_medio = (metricas_simples["pt2libras"]["chrf"]
                      + metricas_simples["libras2pt"]["chrf"]) / 2
        limiares = {
            "delta_significativo": a.delta_significativo,
            "quase_saturacao": a.limiar_quase_saturacao,
            "erro_real_dominante": a.limiar_erro_real_dominante,
            "gap_direcional": a.limiar_gap_direcional,
            "regime_baixo": a.limiar_regime_baixo,
        }
        historico = carregar_historico()
        gps = gerar_gps(metricas_simples, breakdown, vocab_alvo, historico, limiares)
        gps["timestamp"] = datetime.now().isoformat()
        gps["limiares_usados"] = limiares
        gps["chrf_medio"] = round(chrf_medio, 2)
        gps["assimetria_direcoes"] = round(
            metricas_simples["pt2libras"]["chrf"]
            - metricas_simples["libras2pt"]["chrf"], 2)
        gps["vocabulario_alvo"] = vocab_alvo[:30]
        salvar_json(gps, GPS_PATH)
        # Append no historico (so o essencial)
        adicionar_historico(historico, {
            "timestamp": gps["timestamp"],
            "model_dir": str(model_dir),
            "chrf_pt2libras": metricas_simples["pt2libras"]["chrf"],
            "chrf_libras2pt": metricas_simples["libras2pt"]["chrf"],
            "chrf_medio": chrf_medio,
            "bleu4_pt2libras": metricas_simples["pt2libras"]["bleu4"],
            "bleu4_libras2pt": metricas_simples["libras2pt"]["bleu4"],
            "n_teste": int(res["pt2libras"]["n"] + res["libras2pt"]["n"]),
            "diagnostico": gps["diagnostico"],
        })
        g.info["diagnostico"] = gps["diagnostico"]
        g.info["acao"] = gps["acao_prioritaria"]

    # Persiste o evaluate_report
    salvar_json({"model_dir": str(model_dir), "por_direcao": metricas_simples,
                 "breakdown_erros": breakdown},
                DATA_REPORTS / "evaluate_report.json")

    # ─── LOG FINAL: o GPS em formato legivel + trajetoria multi-versao ─────
    log.info("=" * 64)
    log.info("METRICAS")
    for d, r in metricas_simples.items():
        nivel = ("ADEQUADO" if r["chrf"] >= 45
                 else ("ACEITAVEL" if r["chrf"] >= 30 else "BAIXO"))
        log.info(f"  [{d}] chrF={r['chrf']:.2f} BLEU4={r['bleu4']:.2f} -> {nivel}")
    log.info(f"  chrF medio: {chrf_medio:.2f}  |  "
             f"delta vs anterior: {gps.get('delta_chrf')}  |  "
             f"delta vs melhor historico: "
             f"{gps.get('delta_vs_melhor_historico')}")
    # Trajetoria multi-versao
    if gps.get("trajetoria"):
        log.info("-" * 64)
        log.info("TRAJETORIA (multi-versao):")
        for t in gps["trajetoria"]:
            marca = " <- ATUAL" if t["model_dir"] == "ATUAL" else ""
            log.info(f"  [{t['i']}] {t['model_dir']:<35} "
                     f"chrF_medio={t['chrf_medio']:>6.2f} "
                     f"(p2l={t['chrf_p2l']:>5.2f} l2p={t['chrf_l2p']:>5.2f}) "
                     f"{t['diagnostico']}{marca}")
        if gps.get("melhor_historico"):
            mh = gps["melhor_historico"]
            log.info(f"  MELHOR HISTORICO: {mh['model_dir']} "
                     f"chrF_medio={mh['chrf_medio']:.2f}")
    log.info("=" * 64)
    log.info(f"GPS: {gps['diagnostico']}")
    log.info(f"  RAZAO: {gps['razao']}")
    log.info(f"  ACAO : {gps['acao_prioritaria']}")
    if vocab_alvo:
        log.info(f"  TOP-5 PALAVRAS-ALVO: " +
                 ", ".join(v["palavra"] for v in vocab_alvo[:5]))
    log.info("=" * 64)
    log.info(f"[SALVO] {GPS_PATH}")
    log.info(f"[SALVO] {HISTORICO_PATH} (entrada {len(historico)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
