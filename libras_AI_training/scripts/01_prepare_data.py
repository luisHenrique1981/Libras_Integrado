#!/usr/bin/env python3
"""
01_prepare_data.py v7 — Constroi treino/val/teste com BALANCEAMENTO data-driven
por tema e por direcao, derivando MAX_INPUT/MAX_TARGET do corpus.

CONTEXTO DO APP (registrado em memoria persistente):
  Este e um app de USO COTIDIANO para tradutor Libras<->PT em rotina diaria
  (compras, viagem, medico, farmacia, shopping, orgaos publicos, cartorios,
  delegacias, materiais de construcao, reformas, eletronicos, socializacao).
  NAO e app cientifico/academico/nicho. Decisoes de balanceamento priorizam
  vocabulario cotidiano de uso comum.

FIX vs v6.2:
  + BALANCEAMENTO POR TEMA: cap automatico no tema dominante para evitar que
    'vlibrasdb_governo' (jurídico, ~55k pares) ofusque o vocabulario cotidiano.
    Default: tema nao pode ter > --max-pct-tema (default 35%) do treino.
    Filosofia: o app e cotidiano; governo e UM caso de uso, nao O caso de uso.
  + DISTRIBUICAO POR DIRECAO ja era 50/50 por construcao (cada par vira 2
    linhas pt2libras + libras2pt). v7 valida e loga explicitamente.
  + STATS DETALHADOS por tema apos o split, para auditoria humana.
  + ESTRATIFICACAO opcional: garante que TODOS os temas (mesmo raros) tenham
    representacao em val e teste.

MANTIDO da v6.2:
  + Split via COMPONENTES CONEXOS (anti-leakage com paraphrase)
  + Derivacao de MAX_INPUT/MAX_TARGET de p99 + headroom (escreve dataset_dims.json)
  + Caps default 224/176 calibrados para corpus v6/v7
"""
from __future__ import annotations
import argparse
import collections
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras import common as _common  # noqa: E402
from libras.common import (  # noqa: E402
    BASE_MODEL_ID, DATA_RAW, DATA_PROCESSED, DATA_REPORTS, ART_LOGS,
    TRAIN_CSV, VAL_CSV, TEST_CSV, COLS_PROCESSED, SEED,
    DIR_PT2LIBRAS, DIR_LIBRAS2PT, DIRECOES_VALIDAS,
    get_logger, salvar_json, montar_input, require,
    DataContractError, StepGate, checar_leakage, validar_dataframe,
)

log = get_logger("prepare_data", ART_LOGS / "prepare_data.log")

# Banco fixo para smoke test reprodutivel (cotidiano)
BANCO_FIXO = {
    ("eu preciso de um medico.", "eu precisar medico [PONTO]"),
    ("qual e o seu nome?", "nome voce qual [INTERROGACAO]"),
    ("quanto custa isso?", "isso custar quanto [INTERROGACAO]"),
    ("onde fica o caixa?", "caixa onde [INTERROGACAO]"),
    ("preciso de ajuda!", "eu precisar ajuda [EXCLAMACAO]"),
}


def chave(pt: str, lib: str) -> tuple[str, str]:
    return (pt.lower().strip(), lib.lower().strip())


def _arredondar_para_cima(n: int, multiplo: int) -> int:
    return ((n + multiplo - 1) // multiplo) * multiplo


def _percentil(arr: list[int], p: float) -> int:
    if not arr:
        return 0
    s = sorted(arr)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return int(s[lo])
    return int(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _stats_percentis(arr: list[int]) -> dict:
    if not arr:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
    return {"p50": _percentil(arr, 50), "p95": _percentil(arr, 95),
            "p99": _percentil(arr, 99), "max": max(arr), "n": len(arr)}


def construir_componentes(pares_idx: dict) -> list[set[tuple[str, str]]]:
    """Componentes conexos do grafo bipartite PT<->glosa via union-find.
    Garante que paraphrases (mesmo PT em pares distintos OU mesma glosa em
    pares distintos) ficam no mesmo split, eliminando leakage textual.
    """
    chaves = list(pares_idx.keys())
    n = len(chaves)
    pt_to_idx: dict[str, list[int]] = collections.defaultdict(list)
    glosa_to_idx: dict[str, list[int]] = collections.defaultdict(list)
    for i, (pt, glosa) in enumerate(chaves):
        pt_to_idx[pt].append(i)
        glosa_to_idx[glosa].append(i)
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for indices in pt_to_idx.values():
        if len(indices) > 1:
            base = indices[0]
            for j in indices[1:]:
                union(base, j)
    for indices in glosa_to_idx.values():
        if len(indices) > 1:
            base = indices[0]
            for j in indices[1:]:
                union(base, j)

    comp: dict[int, set[tuple[str, str]]] = collections.defaultdict(set)
    for i in range(n):
        comp[find(i)].add(chaves[i])
    return list(comp.values())


def balancear_por_tema(pares_idx: dict, max_pct_tema: float,
                       log_) -> tuple[dict, dict]:
    """
    Aplica cap por tema EM UMA UNICA PASSADA: temas que excedem
    max_pct_tema * total_original sao reduzidos ao cap. Outros temas sao
    mantidos integralmente. Construido_* nunca e cortado (qualidade humana).
    
    Logica deliberadamente simples:
      - cap_pares = int(total * max_pct_tema)
      - Para cada tema NAO-construido:
          se len > cap_pares, reduzir para cap_pares (subsampling deterministico)
      - NAO recalcula iterativamente: aceitar que outros temas tenham >cap_pct
        do total FINAL e melhor que cortar demais.
    
    Motivacao do design: o objetivo e impedir tema GIGANTE de dominar (ex:
    vlibrasdb_governo formal jurídico em app cotidiano), nao homogeneizar
    artificialmente toda a distribuicao. Se ja ha 3 temas e nenhum > cap, o
    balanceador nao age.
    """
    # Agrupa pares por tema
    tema_de_par: dict[tuple, str] = {}
    for k, lista in pares_idx.items():
        tema_de_par[k] = lista[0]["tema"]
    pares_por_tema: dict[str, list] = collections.defaultdict(list)
    for k, t in tema_de_par.items():
        pares_por_tema[t].append(k)
    total_original = len(tema_de_par)
    cap_pares = int(total_original * max_pct_tema)

    log_.info(f"  Distribuicao ANTES do balanceamento ({total_original} pares; "
              f"cap por tema = {cap_pares}):")
    for t, lst in sorted(pares_por_tema.items(), key=lambda kv: -len(kv[1]))[:30]:
        marca = " *" if (not t.startswith("construido_") and len(lst) > cap_pares) else ""
        log_.info(f"    {t:<40} {len(lst):>6}  ({100*len(lst)/total_original:.1f}%){marca}")

    rng = random.Random(SEED)
    descartados_por_tema: dict[str, int] = {}
    for tema, lista in list(pares_por_tema.items()):
        if tema.startswith("construido_"):
            continue  # nunca corta construido
        if len(lista) > cap_pares:
            rng.shuffle(lista)
            descartados = len(lista) - cap_pares
            descartados_por_tema[tema] = descartados
            pares_por_tema[tema] = lista[:cap_pares]
            log_.info(f"  [CAP APLICADO] '{tema}': {len(lista)+descartados-descartados} "
                      f"-> {cap_pares} pares (-{descartados})")

    # Reconstroi pares_idx
    chaves_mantidas = set()
    for chaves_lst in pares_por_tema.values():
        chaves_mantidas.update(chaves_lst)
    pares_idx_balanceado = {k: v for k, v in pares_idx.items()
                              if k in chaves_mantidas}
    novo_total = len(pares_idx_balanceado)

    log_.info(f"  Distribuicao APOS balanceamento ({novo_total} pares):")
    dist_pos = collections.Counter()
    for k in pares_idx_balanceado:
        dist_pos[tema_de_par[k]] += 1
    for t, n in dist_pos.most_common(30):
        log_.info(f"    {t:<40} {n:>6}  ({100*n/novo_total:.1f}%)")

    rel = {
        "total_antes": total_original,
        "total_apos": novo_total,
        "removidos": total_original - novo_total,
        "max_pct_tema": max_pct_tema,
        "cap_pares": cap_pares,
        "descartados_por_tema": descartados_por_tema,
        "dist_apos": dict(dist_pos),
    }
    return pares_idx_balanceado, rel


def derivar_dims_do_corpus(df_treino, df_val, df_teste, tok, log_,
                            percentil_alvo: float,
                            cap_input: int, cap_target: int,
                            headroom: int) -> dict:
    """Deriva MAX_INPUT/MAX_TARGET da distribuicao real de tokens (p99 + headroom,
    arredondado, capped). Mesma logica da v6.2.
    """
    per_split = {}
    todos_input: list[int] = []
    todos_target: list[int] = []
    for nome, df_s in (("treino", df_treino), ("val", df_val), ("teste", df_teste)):
        inp_lens = [len(tok(str(x)).input_ids) for x in df_s["input"]]
        out_lens = [len(tok(text_target=str(x)).input_ids) for x in df_s["output"]]
        per_split[nome] = {"n": len(df_s),
                            "input": _stats_percentis(inp_lens),
                            "target": _stats_percentis(out_lens)}
        todos_input.extend(inp_lens)
        todos_target.extend(out_lens)
        log_.info(f"  [{nome}] input p50={per_split[nome]['input']['p50']} "
                  f"p95={per_split[nome]['input']['p95']} "
                  f"p99={per_split[nome]['input']['p99']} "
                  f"max={per_split[nome]['input']['max']} | "
                  f"target p50={per_split[nome]['target']['p50']} "
                  f"p95={per_split[nome]['target']['p95']} "
                  f"p99={per_split[nome]['target']['p99']} "
                  f"max={per_split[nome]['target']['max']}")

    p99_input_global = _percentil(todos_input, percentil_alvo)
    p99_target_global = _percentil(todos_target, percentil_alvo)
    max_input_derivado = min(cap_input,
                              _arredondar_para_cima(p99_input_global + headroom, 16))
    max_target_derivado = min(cap_target,
                               _arredondar_para_cima(p99_target_global + headroom, 8))
    n_inp_trunc = sum(1 for n in todos_input if n > max_input_derivado)
    n_out_trunc = sum(1 for n in todos_target if n > max_target_derivado)
    pct_inp = n_inp_trunc / max(len(todos_input), 1)
    pct_out = n_out_trunc / max(len(todos_target), 1)

    return {
        "max_input": max_input_derivado, "max_target": max_target_derivado,
        "derivado_em": datetime.now().isoformat(),
        "metodo": (f"p{percentil_alvo:g} global + {headroom} headroom, "
                   f"arredondado a multiplo de 16/8, cap {cap_input}/{cap_target}"),
        "percentil_alvo": percentil_alvo, "headroom": headroom,
        "cap_input": cap_input, "cap_target": cap_target,
        "p99_input_global": p99_input_global, "p99_target_global": p99_target_global,
        "max_input_global": max(todos_input) if todos_input else 0,
        "max_target_global": max(todos_target) if todos_target else 0,
        "pct_truncamento_com_derivado": {"input": round(pct_inp, 4),
                                          "target": round(pct_out, 4)},
        "n_truncamento_com_derivado": {"input": n_inp_trunc, "target": n_out_trunc},
        "stats_por_split": per_split,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", default="dataset_merged_v7.csv",
                   help="CSV em data/raw/. Default: dataset_merged_v7.csv")
    # Splits
    p.add_argument("--split-treino", type=float, default=0.70)
    p.add_argument("--split-val", type=float, default=0.15)
    # Balanceamento por tema (novo na v7)
    p.add_argument("--max-pct-tema", type=float, default=0.40,
                   help="cap por tema NAO-construido como fracao do total original. "
                        "Default 0.40: tema individual nao excede 40%% do corpus, "
                        "evitando que vlibrasdb_governo formal/juridico domine o "
                        "app cotidiano. Aplicado em UMA passada, sem iteracao.")
    p.add_argument("--sem-balanceamento", action="store_true",
                   help="DESLIGA o balanceamento por tema (use so se sabe o que faz).")
    # Dims data-driven (mantido da v6.2)
    p.add_argument("--max-input-cap", type=int, default=224)
    p.add_argument("--max-target-cap", type=int, default=176)
    p.add_argument("--percentil-alvo", type=float, default=99.0)
    p.add_argument("--headroom-tokens", type=int, default=8)
    p.add_argument("--limite-truncamento", type=float, default=0.02)
    a = p.parse_args()

    require(0.5 <= a.split_treino <= 0.9, "E_SPLIT",
            f"--split-treino fora de [0.5, 0.9]: {a.split_treino}",
            err=DataContractError)
    require(0.05 <= a.split_val <= 0.25, "E_SPLIT",
            f"--split-val fora de [0.05, 0.25]: {a.split_val}",
            err=DataContractError)
    split_teste = round(1.0 - a.split_treino - a.split_val, 4)
    require(split_teste >= 0.05, "E_SPLIT",
            f"teste residual {split_teste:.3f} < 0.05",
            err=DataContractError)
    require(0.10 <= a.max_pct_tema <= 0.60, "E_PCT_TEMA",
            f"--max-pct-tema fora de [0.10, 0.60]: {a.max_pct_tema}",
            err=DataContractError)

    random.seed(SEED)
    import pandas as pd

    with StepGate("carregar_e_derivar", log) as g:
        fonte = DATA_RAW / a.fonte
        require(fonte.exists(), "E_FONTE_AUSENTE",
                f"fonte nao encontrada: {fonte}",
                fix=f"opcoes em {DATA_RAW}: "
                    + ", ".join(p.name for p in DATA_RAW.glob("*.csv")),
                err=DataContractError)
        log.info(f"Fonte: {fonte.name}")
        df_fonte = pd.read_csv(fonte)
        require(len(df_fonte) > 0, "E_FONTE_VAZIA",
                f"{fonte.name} vazia", err=DataContractError)
        obrig = {"portugues_original", "libras_glosa", "direcao"}
        faltando = obrig - set(df_fonte.columns)
        require(not faltando, "E_FONTE_COLS",
                f"fonte sem colunas {faltando}", err=DataContractError)

        pares_idx: dict[tuple[str, str], list[dict]] = {}
        descartadas = 0
        for r in df_fonte.to_dict("records"):
            pt = str(r.get("portugues_original") or "").strip()
            lib = str(r.get("libras_glosa") or "").strip()
            direcao = str(r.get("direcao") or "").strip()
            tema = str(r.get("tema") or "geral").strip()
            if not pt or not lib or direcao not in DIRECOES_VALIDAS:
                descartadas += 1
                continue
            fonte_txt = pt if direcao == DIR_PT2LIBRAS else lib
            alvo_txt = lib if direcao == DIR_PT2LIBRAS else pt
            d = {"input": montar_input(fonte_txt, direcao),
                 "output": alvo_txt, "direcao": direcao, "tema": tema,
                 "portugues_original": pt, "libras_glosa": lib}
            pares_idx.setdefault(chave(pt, lib), []).append(d)
        g.info.update({"linhas_fonte": len(df_fonte), "descartadas": descartadas,
                       "pares_unicos": len(pares_idx)})

    # NOVO na v7: BALANCEAMENTO POR TEMA antes do split
    with StepGate("balancear_temas", log) as g:
        # Remove banco fixo antes de balancear
        banco_norm = {chave(a_, b_) for a_, b_ in BANCO_FIXO}
        n_banco = sum(1 for k in banco_norm if k in pares_idx)
        pares_disponiveis = {k: v for k, v in pares_idx.items()
                              if k not in banco_norm}

        if a.sem_balanceamento:
            log.info("  --sem-balanceamento ativo: pulando")
            pares_balanceados = pares_disponiveis
            rel_balanceamento = {"aplicado": False}
        else:
            log.info(f"  Aplicando cap por tema: max {100*a.max_pct_tema:.0f}% "
                     f"por tema (excecao: construido_*)")
            pares_balanceados, rel_balanceamento = balancear_por_tema(
                pares_disponiveis, a.max_pct_tema, log)
            rel_balanceamento["aplicado"] = True
        g.info.update({"banco_fixo_removidos": n_banco,
                       "pares_pos_balanceamento": len(pares_balanceados),
                       **rel_balanceamento})

    with StepGate("split_componentes_conexos", log) as g:
        n_total = len(pares_balanceados)
        require(n_total >= 100, "E_POUCOS_PARES",
                f"apenas {n_total} pares pos-balanceamento",
                err=DataContractError)

        log.info("  Construindo componentes (bipartite PT<->glosa)...")
        componentes = construir_componentes(pares_balanceados)
        sizes = sorted((len(c) for c in componentes), reverse=True)
        n_singletons = sum(1 for s in sizes if s == 1)
        maior_pct = 100.0 * sizes[0] / n_total
        log.info(f"  Componentes: {len(componentes)} (vs {n_total} pares) | "
                 f"max={sizes[0]} ({maior_pct:.2f}%) "
                 f"singletons={n_singletons} ({100*n_singletons/len(componentes):.1f}%)")
        require(len(componentes) >= 10, "E_POUCOS_COMPONENTES",
                f"apenas {len(componentes)} componentes",
                err=DataContractError)
        if maior_pct > 20:
            log.warning(f"  AVISO: maior componente cobre {maior_pct:.2f}% — "
                        f"split podera desviar do alvo")

        random.shuffle(componentes)
        alvo_tr = int(n_total * a.split_treino)
        alvo_va_acc = alvo_tr + int(n_total * a.split_val)
        ch_tr: set = set(); ch_va: set = set(); ch_te: set = set()
        acumulado = 0
        for comp in componentes:
            if acumulado < alvo_tr:
                ch_tr.update(comp)
            elif acumulado < alvo_va_acc:
                ch_va.update(comp)
            else:
                ch_te.update(comp)
            acumulado += len(comp)

        require(not (ch_tr & ch_va), "E_LEAK_SPLIT", "treino x val",
                err=DataContractError)
        require(not (ch_tr & ch_te), "E_LEAK_SPLIT", "treino x teste",
                err=DataContractError)
        require(not (ch_va & ch_te), "E_LEAK_SPLIT", "val x teste",
                err=DataContractError)
        require(len(ch_tr) > 0 and len(ch_va) > 0 and len(ch_te) > 0,
                "E_SPLIT_VAZIO",
                f"split vazio: tr={len(ch_tr)} va={len(ch_va)} te={len(ch_te)}",
                err=DataContractError)

        def expandir(chaves):
            out = []
            for k in chaves:
                out.extend(pares_balanceados[k])
            return out
        l_tr, l_va, l_te = expandir(ch_tr), expandir(ch_va), expandir(ch_te)
        eff_tr = len(ch_tr) / n_total
        eff_va = len(ch_va) / n_total
        eff_te = len(ch_te) / n_total
        log.info(f"  Split efetivo: treino={eff_tr:.2%} val={eff_va:.2%} "
                 f"teste={eff_te:.2%}")
        g.info.update({
            "pares_treino": len(ch_tr), "pares_val": len(ch_va), "pares_teste": len(ch_te),
            "linhas_treino": len(l_tr), "linhas_val": len(l_va), "linhas_teste": len(l_te),
            "n_componentes": len(componentes),
            "maior_componente": sizes[0],
            "split_efetivo": {"treino": round(eff_tr, 4), "val": round(eff_va, 4),
                              "teste": round(eff_te, 4)},
        })

    with StepGate("validar_contrato_e_leakage", log) as g:
        df_tr = pd.DataFrame(l_tr); df_va = pd.DataFrame(l_va); df_te = pd.DataFrame(l_te)
        m_tr = validar_dataframe(df_tr, "treino", log)
        m_va = validar_dataframe(df_va, "val", log)
        m_te = validar_dataframe(df_te, "teste", log)
        n_leak_va = checar_leakage(df_tr, df_va, "val", log)
        n_leak_te = checar_leakage(df_tr, df_te, "teste", log)
        require(n_leak_va == 0 and n_leak_te == 0, "E_LEAKAGE",
                f"leakage: val={n_leak_va} teste={n_leak_te}",
                fix="bug em construir_componentes", err=DataContractError)
        g.info.update({"metricas_treino": m_tr, "metricas_val": m_va,
                       "metricas_teste": m_te})

    # NOVO na v7: VALIDAR DISTRIBUICAO POR DIRECAO E TEMA POS-SPLIT
    with StepGate("validar_distribuicao", log) as g:
        for nome, df_s in (("treino", df_tr), ("val", df_va), ("teste", df_te)):
            dist_dir = df_s["direcao"].value_counts().to_dict()
            dist_tema_top = df_s["tema"].value_counts().head(10).to_dict()
            log.info(f"  [{nome}] direcao: {dist_dir}")
            log.info(f"  [{nome}] top temas: {dist_tema_top}")
            # Validacao: direcao deve estar 50/50 (cada par vira pt2libras + libras2pt)
            n_p2l = dist_dir.get(DIR_PT2LIBRAS, 0)
            n_l2p = dist_dir.get(DIR_LIBRAS2PT, 0)
            require(abs(n_p2l - n_l2p) <= 1, "E_DESBALANCEADO",
                    f"[{nome}] direcoes desbalanceadas: pt2libras={n_p2l} "
                    f"libras2pt={n_l2p}",
                    fix="bug na expansao por direcao em mesclar_datasets",
                    err=DataContractError)
        g.info.update({
            "dist_direcao_treino": df_tr["direcao"].value_counts().to_dict(),
            "dist_direcao_val": df_va["direcao"].value_counts().to_dict(),
            "dist_direcao_teste": df_te["direcao"].value_counts().to_dict(),
            "dist_tema_treino": df_tr["tema"].value_counts().to_dict(),
            "dist_tema_val": df_va["tema"].value_counts().to_dict(),
            "dist_tema_teste": df_te["tema"].value_counts().to_dict(),
        })

    with StepGate("analise_e_derivar_dims", log) as g:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        log.info(f"  Analisando tokens (cap={a.max_input_cap}/{a.max_target_cap}, "
                 f"p{a.percentil_alvo:g}, headroom={a.headroom_tokens})...")
        dims = derivar_dims_do_corpus(df_tr, df_va, df_te, tok, log,
                                       percentil_alvo=a.percentil_alvo,
                                       cap_input=a.max_input_cap,
                                       cap_target=a.max_target_cap,
                                       headroom=a.headroom_tokens)
        log.info(f"  DERIVADO: MAX_INPUT={dims['max_input']} "
                 f"MAX_TARGET={dims['max_target']}")
        log.info(f"  Trunc esperado: input "
                 f"{dims['pct_truncamento_com_derivado']['input']:.2%} | "
                 f"target {dims['pct_truncamento_com_derivado']['target']:.2%}")

        pct_max_trunc = max(dims["pct_truncamento_com_derivado"]["input"],
                             dims["pct_truncamento_com_derivado"]["target"])
        require(pct_max_trunc <= a.limite_truncamento, "E_TRUNCAMENTO_RESIDUAL",
                f"truncamento {pct_max_trunc:.2%} > limite "
                f"{a.limite_truncamento:.2%}",
                fix=("corpus com sequencias muito longas; reduza "
                     "--max-glosa-palavras em mesclar_datasets ou aumente caps."),
                err=DataContractError, ctx=dims)

        dims_path = DATA_PROCESSED / "dataset_dims.json"
        dims_path.parent.mkdir(parents=True, exist_ok=True)
        dims_path.write_text(json.dumps(dims, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        log.info(f"  [SALVO] {dims_path}")
        _common.MAX_INPUT = dims["max_input"]
        _common.MAX_TARGET = dims["max_target"]
        log.info(f"  common.MAX_* atualizados in-process "
                 f"({_common.MAX_INPUT}/{_common.MAX_TARGET})")
        g.info["dims"] = dims

    with StepGate("salvar", log):
        for linhas, path in ((l_tr, TRAIN_CSV), (l_va, VAL_CSV), (l_te, TEST_CSV)):
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(linhas)[COLS_PROCESSED].to_csv(path, index=False,
                                                        encoding="utf-8")
            log.info(f"  [SALVO] {path} ({len(linhas)} linhas)")

    rel = {
        "timestamp": datetime.now().isoformat(),
        "versao": "7.0.0", "seed": SEED, "fonte": str(fonte),
        "metodo_split": "componentes_conexos_bipartite_pt_glosa",
        "balanceamento": {"aplicado": not a.sem_balanceamento,
                          "max_pct_tema": a.max_pct_tema,
                          **rel_balanceamento},
        "splits_alvo": {"treino": a.split_treino, "val": a.split_val,
                         "teste": split_teste},
        "splits_efetivo": {"treino": round(eff_tr, 4), "val": round(eff_va, 4),
                            "teste": round(eff_te, 4)},
        "pares_unicos_apos_filtros": n_total,
        "saidas": {"treino": {"pares": len(ch_tr), "linhas": len(l_tr)},
                   "val": {"pares": len(ch_va), "linhas": len(l_va)},
                   "teste": {"pares": len(ch_te), "linhas": len(l_te)}},
        "leakage": {"val": n_leak_va, "teste": n_leak_te},
        "metricas": {"treino": m_tr, "val": m_va, "teste": m_te},
        "dims_derivados": dims,
    }
    salvar_json(rel, DATA_REPORTS / "prepare_report.json")
    log.info("[RESULTADO] preparacao concluida; dims derivados; balanceamento "
             "aplicado." if not a.sem_balanceamento else
             "[RESULTADO] preparacao concluida; dims derivados; "
             "balanceamento PULADO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
