#!/usr/bin/env python3
"""
analise_erros.py — Categoriza os erros de evaluate_erros.csv para decidir,
COM DADOS, onde investir: correcao de dados, mais dados, ou aceitar que o erro
e intrinseco (parafrase valida sub-penalizada por metrica de referencia unica).

Pre-requisito: rode antes
  python scripts/04_evaluate.py --save-errors
que gera data/reports/evaluate_erros.csv.

Categorias (mutuamente exclusivas, avaliadas em ordem):
  token_especial : a UNICA divergencia e acento/grafia de [PONTO]/[INTERROGACAO]/
                   [EXCLAMACAO] (ex.: '[INTERROGAÇÃO]' vs '[INTERROGACAO]').
                   -> CORRIGIVEL: registrar tokens atomicos no tokenizer + retreinar.
  pontuacao      : igual apos remover tokens de pontuacao (sobra/falta de [PONTO]).
                   -> CORRIGIVEL: normalizar pontuacao na construcao do dataset.
  quase_acerto   : chrF par-a-par >= limiar; provavel parafrase valida ou diferenca
                   de 1 palavra (ex.: 'Preciso de ajuda' vs 'Eu preciso de ajuda').
                   -> PARCIALMENTE INTRINSECO (mapeamento 1-para-muitos); metrica de
                   referencia unica subestima a qualidade real.
  erro_real      : baixa similaridade; erro de traducao genuino.
                   -> precisa de MAIS/MELHOR dado para a direcao afetada.

Uso:
  python tools/analise_erros.py
  python tools/analise_erros.py --limiar-quase 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERROS_CSV = ROOT / "data" / "reports" / "evaluate_erros.csv"
OUT_JSON = ROOT / "data" / "reports" / "analise_erros.json"

TOKENS_PONT = ["[PONTO]", "[INTERROGACAO]", "[EXCLAMACAO]"]
_RE_BRACKET = re.compile(r"\[[^\]]*\]")


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm_tokens_especiais(s: str) -> str:
    # normaliza o interior de qualquer [..]: remove acento + upcase
    # -> '[INTERROGAÇÃO]' e '[INTERROGACAO]' colapsam no mesmo texto
    return _RE_BRACKET.sub(lambda m: "[" + _sem_acento(m.group(0)[1:-1]).upper() + "]", s)


def _strip_pont(s: str) -> str:
    out = s
    for t in TOKENS_PONT:
        out = out.replace(t, "")
    return re.sub(r"\s+", " ", out).strip()


def categorizar(esperado: str, obtido: str, chrf_par: float, limiar_quase: float) -> str:
    e, o = esperado.strip(), obtido.strip()
    if e.upper() == o.upper():
        return "sem_erro"  # nao deveria estar no CSV; seguranca
    ne, no = _norm_tokens_especiais(e).upper(), _norm_tokens_especiais(o).upper()
    if ne == no:
        return "token_especial"
    if _strip_pont(ne) == _strip_pont(no):
        return "pontuacao"
    if chrf_par >= limiar_quase:
        return "quase_acerto"
    return "erro_real"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ERROS_CSV))
    p.add_argument("--limiar-quase", type=float, default=60.0,
                   help="chrF par-a-par acima do qual o erro vira quase-acerto")
    a = p.parse_args()

    import pandas as pd
    import sacrebleu as sb

    csv_path = Path(a.csv)
    if not csv_path.exists():
        print(f"[ERRO] {csv_path} nao existe. Rode antes:\n"
              f"  python scripts/04_evaluate.py --save-errors", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    for col in ("direcao", "esperado", "obtido"):
        if col not in df.columns:
            print(f"[ERRO] coluna ausente no CSV: {col} | tem: {list(df.columns)}",
                  file=sys.stderr)
            return 1

    relatorio: dict = {"limiar_quase": a.limiar_quase, "por_direcao": {}}
    for direcao in sorted(df["direcao"].unique()):
        sub = df[df["direcao"] == direcao]
        cats: Counter = Counter()
        exemplos: dict[str, list] = {}
        for _, row in sub.iterrows():
            esp, obt = str(row["esperado"]), str(row["obtido"])
            chrf_par = sb.sentence_chrf(obt, [esp], word_order=2).score
            cat = categorizar(esp, obt, chrf_par, a.limiar_quase)
            cats[cat] += 1
            exemplos.setdefault(cat, [])
            if len(exemplos[cat]) < 6:
                exemplos[cat].append({"esperado": esp, "obtido": obt,
                                      "chrf_par": round(chrf_par, 1)})
        total = max(sum(cats.values()), 1)
        relatorio["por_direcao"][direcao] = {
            "n_erros": sum(cats.values()),
            "categorias": {k: {"n": v, "pct": round(100 * v / total, 1)}
                           for k, v in cats.most_common()},
            "exemplos": exemplos,
        }
        print(f"\n[{direcao}] {sum(cats.values())} erros:")
        for k, v in cats.most_common():
            print(f"  {k:<16} {v:>4} ({100 * v / total:.1f}%)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[SALVO] {OUT_JSON}")
    print("Interpretacao: token_especial+pontuacao = corrigivel por dados/tokenizer; "
          "quase_acerto = em parte intrinseco; erro_real = precisa de mais/melhor dado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
