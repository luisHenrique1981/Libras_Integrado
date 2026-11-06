#!/usr/bin/env python3
"""
preparar_vlibras_db.py — Valida e normaliza o VLibrasBD v2 (127.349 pares,
UFPB, Mendeley DOI 10.17632/ryj88ckjww.2) para o schema canonico do projeto.

POR QUE EXISTE: o libras_bilingual_dataset.csv que voce ja tem (68k pares) e
a VERSAO 1 do mesmo corpus. A versao 2 publicada em mai/2025 tem 127.349 pares
(+87%), incluindo 55.047 pares de conteudo do governo federal e flag
is_government_source que permite balanceamento por dominio. Mesma equipe (UFPB),
mesma metodologia (10 interpretes Libras profissionais), mesma licenca
(CC BY-SA 4.0). Esta e a maior alavanca de dados disponivel publicamente.

POR QUE DOWNLOAD MANUAL: a API do Mendeley exige OAuth, o que tornaria o
script fragil (token-dependent, falha silenciosa em rotacao). Download
manual via navegador e mais robusto e auditavel. Voce baixa uma vez e o
arquivo fica no disco; reproducibilidade total via hash SHA256.

INSTRUCOES:
  1. Abrir https://data.mendeley.com/datasets/ryj88ckjww/2 no navegador
  2. Clicar "Download All" (baixa um .zip)
  3. Extrair, achar o arquivo CSV (~30MB)
  4. Mover para data/raw/vlibras_db_v2.csv
  5. Rodar: python tools/preparar_vlibras_db.py

O que esse script faz:
  - Valida que o arquivo existe e tem o schema esperado (4 colunas)
  - Calcula SHA256 (auditoria)
  - Normaliza para schema canonico (pt, glosa, tema) com tema diferenciando
    governo (is_government_source=True) vs geral (False)
  - Salva em data/raw/vlibras_db_v2_normalizado.csv para o mesclar_datasets
  - Gera relatorio com estatisticas
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.common import (  # noqa: E402
    DATA_RAW, DATA_REPORTS, ART_LOGS,
    get_logger, salvar_json, require, StepGate, DataContractError,
)

log = get_logger("preparar_vlibras_db", ART_LOGS / "preparar_vlibras_db.log")

URL_DATASET = "https://data.mendeley.com/datasets/ryj88ckjww/2"
ENTRADA_PADRAO = DATA_RAW / "vlibras_db_v2.csv"
SAIDA_PADRAO = DATA_RAW / "vlibras_db_v2_normalizado.csv"


def imprimir_instrucoes_download() -> None:
    print("=" * 70)
    print("  VLibrasBD v2 NAO ENCONTRADO em " + str(ENTRADA_PADRAO))
    print("=" * 70)
    print()
    print("  Para baixar (uma unica vez, ~30MB):")
    print(f"  1. Abra: {URL_DATASET}")
    print("  2. Clique 'Download All' (baixa um .zip)")
    print("  3. Extraia, achar o arquivo .csv")
    print("  4. Mover para: " + str(ENTRADA_PADRAO))
    print("  5. Re-rodar este script")
    print()
    print("  Citacao: Lima M., Franca D., Silva D. et al. pt-br2libras-gloss.")
    print("           Mendeley Data, V2 (2025). DOI: 10.17632/ryj88ckjww.2")
    print("  Licenca: CC BY-SA 4.0")
    print("=" * 70)


def sha256_arquivo(p: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", default=str(ENTRADA_PADRAO),
                   help=f"caminho do CSV baixado. Default: {ENTRADA_PADRAO}")
    p.add_argument("--saida", default=str(SAIDA_PADRAO),
                   help=f"caminho do CSV normalizado. Default: {SAIDA_PADRAO}")
    p.add_argument("--abrir-navegador", action="store_true",
                   help="abre o URL de download no navegador padrao do sistema "
                        "se o arquivo nao existir")
    a = p.parse_args()

    entrada = Path(a.entrada)
    saida = Path(a.saida)

    # Defesa de existencia: arquivo precisa estar baixado manualmente
    if not entrada.exists():
        imprimir_instrucoes_download()
        if a.abrir_navegador:
            try:
                webbrowser.open(URL_DATASET)
                log.info(f"Navegador aberto em: {URL_DATASET}")
            except Exception as e:
                log.warning(f"nao consegui abrir o navegador: {e}")
        return 2

    import pandas as pd

    with StepGate("validar_entrada", log) as g:
        log.info(f"Entrada: {entrada}")
        sha = sha256_arquivo(entrada)
        size_mb = entrada.stat().st_size / (1024 * 1024)
        log.info(f"  SHA256: {sha[:16]}... ({size_mb:.1f} MB)")
        df = pd.read_csv(entrada)
        cols_esperadas = {"pt-br", "libras-gloss", "is_government_source"}
        cols_atuais = set(df.columns)
        faltando = cols_esperadas - cols_atuais
        require(not faltando, "E_SCHEMA_VLIBRASDB",
                f"colunas ausentes: {faltando}",
                fix=("o arquivo nao parece ser o VLibrasBD v2. "
                     f"Esperado: {cols_esperadas}. Encontrado: {cols_atuais}"),
                err=DataContractError)
        n_linhas_brutas = len(df)
        # Esperado: 127349 pares pelo paper. Tolera diferenca pequena (atualizacoes)
        require(120_000 <= n_linhas_brutas <= 140_000,
                "E_TAMANHO_INESPERADO",
                f"{n_linhas_brutas} linhas; esperado ~127349 +- 10000",
                fix="arquivo pode estar corrompido ou ser versao diferente",
                err=DataContractError)
        n_nulos_pt = int(df["pt-br"].isna().sum())
        n_nulos_glosa = int(df["libras-gloss"].isna().sum())
        log.info(f"  linhas: {n_linhas_brutas}  nulos pt: {n_nulos_pt}  "
                 f"nulos glosa: {n_nulos_glosa}")
        g.info.update({"sha256": sha, "tamanho_mb": round(size_mb, 2),
                       "n_linhas_brutas": n_linhas_brutas,
                       "n_nulos_pt": n_nulos_pt, "n_nulos_glosa": n_nulos_glosa})

    with StepGate("normalizar", log) as g:
        # Remove linhas com null em campos criticos
        antes = len(df)
        df = df.dropna(subset=["pt-br", "libras-gloss"]).reset_index(drop=True)
        descartadas_nulo = antes - len(df)

        # Normaliza tipos: is_government_source pode vir como string "True"/"False"
        # ou bool ou int 0/1. Padroniza para bool real.
        col_gov = df["is_government_source"]
        if col_gov.dtype == "object":
            df["is_government_source"] = col_gov.astype(str).str.lower().isin(
                {"true", "1", "yes", "sim", "verdadeiro"})
        elif pd.api.types.is_numeric_dtype(col_gov):
            df["is_government_source"] = col_gov.astype(bool)
        # else: ja bool

        # Strip whitespace nos textos
        df["pt-br"] = df["pt-br"].astype(str).str.strip()
        df["libras-gloss"] = df["libras-gloss"].astype(str).str.strip()
        antes = len(df)
        df = df[(df["pt-br"].str.len() > 0) & (df["libras-gloss"].str.len() > 0)]
        df = df.reset_index(drop=True)
        descartadas_vazias = antes - len(df)

        # Schema canonico do projeto: pt, glosa, tema
        # O tema diferencia governo (jurídico/formal) vs geral (cotidiano)
        df_normalizado = pd.DataFrame({
            "pt": df["pt-br"].values,
            "glosa": df["libras-gloss"].values,
            "tema": df["is_government_source"].map(
                lambda b: "vlibrasdb_governo" if b else "vlibrasdb_geral"),
        })

        n_governo = int((df_normalizado["tema"] == "vlibrasdb_governo").sum())
        n_geral = int((df_normalizado["tema"] == "vlibrasdb_geral").sum())
        log.info(f"  governo: {n_governo}  geral: {n_geral}  "
                 f"total: {len(df_normalizado)}")
        log.info(f"  descartadas: nulos={descartadas_nulo} "
                 f"vazias={descartadas_vazias}")
        g.info.update({"n_governo": n_governo, "n_geral": n_geral,
                       "n_normalizado": len(df_normalizado),
                       "descartadas_nulos": descartadas_nulo,
                       "descartadas_vazias": descartadas_vazias})

    with StepGate("salvar", log) as g:
        saida.parent.mkdir(parents=True, exist_ok=True)
        df_normalizado.to_csv(saida, index=False, encoding="utf-8")
        log.info(f"  [SALVO] {saida} ({len(df_normalizado)} pares)")
        rel = {
            "timestamp": datetime.now().isoformat(),
            "versao": "1.0.0",
            "fonte": str(entrada),
            "fonte_sha256": sha,
            "fonte_url": URL_DATASET,
            "citacao": ("Lima M., Franca D., Silva D. et al. "
                        "pt-br2libras-gloss. Mendeley Data, V2 (2025). "
                        "DOI: 10.17632/ryj88ckjww.2"),
            "licenca": "CC BY-SA 4.0",
            "saida": str(saida),
            "n_pares": len(df_normalizado),
            "distribuicao_tema": {
                "vlibrasdb_governo": n_governo,
                "vlibrasdb_geral": n_geral,
            },
            "ganho_vs_v1": round(len(df_normalizado) / 68029, 2),
        }
        salvar_json(rel, DATA_REPORTS / "preparar_vlibras_db_report.json")
        g.info["saida"] = str(saida)

    log.info("=" * 64)
    log.info(f"PRONTO: {len(df_normalizado)} pares normalizados em {saida}")
    log.info(f"Ganho vs libras_bilingual_dataset v1: "
             f"{len(df_normalizado)/68029:.2f}x")
    log.info("Proximo: python tools/mesclar_datasets.py")
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
