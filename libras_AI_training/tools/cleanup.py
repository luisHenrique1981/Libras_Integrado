#!/usr/bin/env python3
"""
cleanup.py — Organiza a pasta de treino MOVENDO lixo para uma lixeira datada.

Principios de seguranca:
  - NUNCA deleta. Apenas MOVE para _arquivo_morto_<data>/{lixo,arquivo,modelos_antigos}.
  - Dry-run por PADRAO. So move com --aplicar.
  - Lista de protegidos NUNCA e tocada (venv, datasets atuais, fonte, git).
  - Gera manifesto JSON com origem->destino (reversivel: basta mover de volta).
  - Idempotente: re-rodar nao quebra (lixeira ja existente e reutilizada).

Uso:
  python tools/cleanup.py --raiz /home/bruno/Projetos/treinamento_modelo_IA          # dry-run
  python tools/cleanup.py --raiz /home/bruno/Projetos/treinamento_modelo_IA --aplicar
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Protegidos: jamais movidos (defesa em profundidade) ──────────────────────
PROTEGIDOS_EXATOS = {
    ".venv_pytorch", ".git", ".gitignore", "libras_AI_training",
    # datasets atuais (fonte e processados v5)
    "dataset_treino_v5.csv", "dataset_val_v5.csv", "dataset_teste_v5.csv",
    "dataset_construido_v4.csv", "libras_bilingual_dataset.csv",
    # builder de dataset (insubstituivel sem ele)
    "construir_dataset_v4.py",
}
PROTEGIDOS_PREFIXO = (".venv", ".git")

# ── Classificacao por padrao ─────────────────────────────────────────────────
# LIXO: regeneravel ou claramente obsoleto
RE_LIXO = [
    re.compile(r".*\.bak\d*$"),                 # backups .bak, .bak2
    re.compile(r".*_backup.*\.py$"),
    re.compile(r".*\.pyc$"),
    re.compile(r"^__pycache__$"),
    re.compile(r".*_fixed\.py$"),               # variantes "fixed" intermediarias
    re.compile(r".*_full_fixed\.py$"),
    re.compile(r"^bohb_debug\.log$"),
    re.compile(r"^bohb_search\.log$"),          # superseded por bohb_search_v4.log
]
# Versoes antigas: _vN quando existe _vM (M>N) do mesmo prefixo e extensao.
# Generico (pega v4 quando ha v5), mas so mexe se houver versao estritamente maior.
RE_VERSAO_ANTIGA = re.compile(r"^(.*?)_v(\d+)\b.*\.(py|db|csv|json|log|md)$")
# Modelos antigos (diretorios)
DIRS_MODELO_ANTIGO = {"libras_model_ptt5", "libras_model_ptt5_v3", "libras_model_mt5",
                      "Primeiro"}
# Arquivo (manter mas tirar da raiz): relatorios e logs historicos
RE_ARQUIVO = [
    re.compile(r".*\.log$"),
    re.compile(r"^relatorio.*\.(json|html)$"),
    re.compile(r"^preparar_report.*\.json$"),
    re.compile(r"^build_(report|errors).*"),
    re.compile(r"^auditoria.*\.log$"),
    re.compile(r"^dataset_corrigido.*\.csv$"),
    re.compile(r"^check_env_report\.json$"),
    re.compile(r".*\.bak$"),
]


def existe_versao_mais_nova(nome: str, todos: set[str]) -> bool:
    m = RE_VERSAO_ANTIGA.match(nome)
    if not m:
        return False
    prefixo, ver, ext = m.group(1), int(m.group(2)), m.group(3)
    for cand in todos:
        cm = re.match(rf"^{re.escape(prefixo)}_v(\d+)\b.*\.{ext}$", cand)
        if cm and int(cm.group(1)) > ver:
            return True
    return False


def classificar(nome: str, eh_dir: bool, todos: set[str]) -> str | None:
    """Retorna subpasta da lixeira ou None (manter na raiz)."""
    if nome in PROTEGIDOS_EXATOS or nome.startswith(PROTEGIDOS_PREFIXO):
        return None
    if nome.startswith("_arquivo_morto_"):
        return None
    if eh_dir and nome in DIRS_MODELO_ANTIGO:
        return "modelos_antigos"
    if any(rx.match(nome) for rx in RE_LIXO):
        return "lixo"
    if existe_versao_mais_nova(nome, todos):
        return "versoes_antigas"
    if any(rx.match(nome) for rx in RE_ARQUIVO):
        return "arquivo"
    return None  # desconhecido -> mantem (conservador)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", required=True)
    p.add_argument("--aplicar", action="store_true",
                   help="sem esta flag e dry-run (nada e movido)")
    a = p.parse_args()

    raiz = Path(a.raiz).expanduser().resolve()
    if not raiz.is_dir():
        print(f"ERRO: raiz nao e diretorio: {raiz}", file=sys.stderr)
        return 1

    todos = {e.name for e in raiz.iterdir()}
    lixeira = raiz / f"_arquivo_morto_{datetime.now():%Y%m%d}"

    plano: list[tuple[Path, Path, str]] = []
    for e in sorted(raiz.iterdir()):
        cat = classificar(e.name, e.is_dir(), todos)
        if cat is None:
            continue
        destino = lixeira / cat / e.name
        plano.append((e, destino, cat))

    modo = "APLICANDO" if a.aplicar else "DRY-RUN (use --aplicar para mover)"
    print(f"== cleanup [{modo}] ==")
    print(f"Raiz: {raiz}")
    print(f"Lixeira: {lixeira}")
    print(f"Itens a mover: {len(plano)}\n")
    by_cat: dict[str, int] = {}
    for origem, _, cat in plano:
        by_cat[cat] = by_cat.get(cat, 0) + 1
        print(f"  [{cat:16}] {origem.name}")
    print("\nResumo:", {k: v for k, v in sorted(by_cat.items())})

    manifesto = {"timestamp": datetime.now().isoformat(), "raiz": str(raiz),
                 "lixeira": str(lixeira), "aplicado": a.aplicar, "itens": []}

    if a.aplicar:
        for origem, destino, cat in plano:
            destino.parent.mkdir(parents=True, exist_ok=True)
            if destino.exists():  # evita colisao
                destino = destino.with_name(destino.name + f"_{datetime.now():%H%M%S}")
            shutil.move(str(origem), str(destino))
            manifesto["itens"].append({"de": str(origem), "para": str(destino), "cat": cat})
        man_path = lixeira / "MANIFESTO.json"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n[OK] {len(plano)} itens movidos. Manifesto: {man_path}")
        print("Para reverter um item: mova de volta do destino para a origem (ver manifesto).")
    else:
        print("\nNada movido (dry-run). Revise a lista e rode com --aplicar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
