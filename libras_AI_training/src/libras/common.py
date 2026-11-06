#!/usr/bin/env python3
"""
libras.common — backbone do pipeline Libras <-> Portugues.

Concentra TUDO que e compartilhado entre as etapas:
  - Paths canonicos (raiz do projeto resolvida de forma robusta)
  - Logging estruturado (arquivo + stdout, line-buffered)
  - StepGate: context manager que cronometra, audita e falha de forma explicita
  - Hierarquia de erros tipados com codigo + remediacao (auditavel)
  - require(): assert auditavel (nunca usa assert nu, que some com python -O)
  - Contratos de dados (validacao de colunas, NaN, leakage, formato de glosa)
  - PREFIXOS DE TAREFA canonicos -- ponto unico de verdade para evitar o bug
    de treino-vs-inferencia com formatos divergentes.

Decisao de engenharia: um unico ponto de verdade para prefixo e contrato.
Qualquer divergencia entre etapas (que foi a causa do smoke test 0/18) passa
a ser impossivel por construcao -- todas as etapas importam daqui.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATHS CANONICOS
# ─────────────────────────────────────────────────────────────────────────────
# common.py vive em <raiz>/src/libras/common.py -> raiz = parents[2]
ROOT: Path = Path(__file__).resolve().parents[2]

DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
DATA_REPORTS: Path = ROOT / "data" / "reports"
ART_MODELS: Path = ROOT / "artifacts" / "models"
ART_CONFIGS: Path = ROOT / "artifacts" / "configs"
ART_LOGS: Path = ROOT / "artifacts" / "logs"
ART_STUDIES: Path = ROOT / "artifacts" / "studies"
ART_AUDIT: Path = ROOT / "artifacts" / "audit"

# Datasets processados (contrato fixo de nomes)
TRAIN_CSV: Path = DATA_PROCESSED / "treino.csv"
VAL_CSV: Path = DATA_PROCESSED / "val.csv"
TEST_CSV: Path = DATA_PROCESSED / "teste.csv"

# Modelo base
BASE_MODEL_ID = "unicamp-dl/ptt5-base-portuguese-vocab"

# Limites de tokenizacao (contrato compartilhado).
# DATA-DRIVEN: carregados de data/processed/dataset_dims.json que o
# 01_prepare_data deriva do corpus efetivo (p99 + headroom, capped).
# Fallback conservador (128/64) se o JSON ainda nao existe (primeira run)
# ou esta corrompido. Validacao de range previne valores absurdos.
#
# Motivacao: corpus diferentes tem distribuicoes de comprimento muito
# diferentes. Hardcoded 128/64 era adequado para v5 (corpus construido,
# sequencias curtas) mas truncava 9.5%+ dos targets no corpus v6 mesclado.
# A unica forma de evitar truncamento ou padding excessivo e DERIVAR.
def _carregar_dims_iniciais() -> tuple[int, int]:
    dims_path = DATA_PROCESSED / "dataset_dims.json"
    if dims_path.exists():
        try:
            d = json.loads(dims_path.read_text(encoding="utf-8"))
            mi, mt = int(d["max_input"]), int(d["max_target"])
            # Sanidade: rejeita valores absurdos (corrupcao do JSON)
            if 32 <= mi <= 512 and 16 <= mt <= 256:
                return mi, mt
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            pass
    return 128, 64  # fallback se 01_prepare_data nao rodou ainda

MAX_INPUT, MAX_TARGET = _carregar_dims_iniciais()
SEED = 42

# Colunas obrigatorias do dataset processado
COLS_PROCESSED = [
    "input",
    "output",
    "direcao",
    "tema",
    "portugues_original",
    "libras_glosa",
]

# Direcoes validas
DIR_PT2LIBRAS = "pt2libras"
DIR_LIBRAS2PT = "libras2pt"
DIRECOES_VALIDAS = {DIR_PT2LIBRAS, DIR_LIBRAS2PT}

# ─────────────────────────────────────────────────────────────────────────────
# PREFIXOS DE TAREFA  (PONTO UNICO DE VERDADE)
# ─────────────────────────────────────────────────────────────────────────────
# T5 e multi-task via prefixo. Sem prefixo explicito, um modelo BIDIRECIONAL
# precisa inferir a direcao pelo formato do input -- desperdicio de capacidade
# num dataset pequeno. Pior: se o prefixo aparece no smoke test mas nao no
# treino, o modelo recebe entrada fora-de-distribuicao e degenera (copia input).
# Solucao: prefixo canonico aplicado AQUI e em lugar nenhum mais.
PREFIXO_PT2LIBRAS = "traduzir portugues para libras: "
PREFIXO_LIBRAS2PT = "traduzir libras para portugues: "

TOKENS_PONTUACAO = {"[PONTO]", "[INTERROGACAO]", "[EXCLAMACAO]"}


def montar_input(texto_fonte: str, direcao: str) -> str:
    """Aplica o prefixo de tarefa canonico. Unico lugar que faz isso."""
    texto_fonte = (texto_fonte or "").strip()
    if direcao == DIR_PT2LIBRAS:
        return PREFIXO_PT2LIBRAS + texto_fonte
    if direcao == DIR_LIBRAS2PT:
        return PREFIXO_LIBRAS2PT + texto_fonte
    raise DataContractError(
        code="E_DIR_INVALIDA",
        msg=f"direcao desconhecida ao montar input: {direcao!r}",
        fix=f"direcao deve ser um de {sorted(DIRECOES_VALIDAS)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ERROS TIPADOS E AUDITAVEIS
# ─────────────────────────────────────────────────────────────────────────────
class LibrasError(Exception):
    """Base. Sempre carrega codigo + remediacao para auditoria."""

    def __init__(self, code: str, msg: str, fix: str = "", ctx: Optional[dict] = None):
        self.code = code
        self.msg = msg
        self.fix = fix
        self.ctx = ctx or {}
        super().__init__(f"[{code}] {msg}" + (f" | CORRECAO: {fix}" if fix else ""))

    def as_dict(self) -> dict:
        return {"code": self.code, "msg": self.msg, "fix": self.fix, "ctx": self.ctx}


class EnvironmentError_(LibrasError):
    """Problema de ambiente (deps, GPU, disco). Bloqueia execucao."""


class DataContractError(LibrasError):
    """Dataset viola contrato (colunas, NaN, leakage, formato)."""


class StepError(LibrasError):
    """Falha dentro de uma etapa cronometrada."""


def require(
    cond: bool,
    code: str,
    msg: str,
    fix: str = "",
    ctx: Optional[dict] = None,
    err: type[LibrasError] = LibrasError,
) -> None:
    """
    Assert auditavel. NUNCA usar `assert` no pipeline: `python -O` o remove,
    o que silenciaria validacoes de seguranca. require() levanta erro tipado
    sempre, independente de flags do interpretador.
    """
    if not cond:
        raise err(code=code, msg=msg, fix=fix, ctx=ctx)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(nome: str, arquivo: Optional[Path] = None) -> logging.Logger:
    """Logger line-buffered para arquivo + stdout. Idempotente."""
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, "reconfigure", None)  # None se nao for TextIOWrapper
        if callable(_reconf):
            try:
                _reconf(line_buffering=True)
            except (ValueError, OSError):
                pass  # stream fechado/redirecionado -- nao critico
    log = logging.getLogger(nome)
    if log.handlers:  # ja configurado
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if arquivo is not None:
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(arquivo, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    return log


# ─────────────────────────────────────────────────────────────────────────────
# STEP GATE  (verificacao de etapas + auditoria)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepRecord:
    etapa: str
    inicio: str
    fim: str = ""
    duracao_s: float = -1.0
    status: str = "iniciado"  # iniciado | ok | falha
    erro: dict = field(default_factory=dict)


class StepGate:
    """
    Context manager que:
      1. Loga inicio com timestamp.
      2. Cronometra.
      3. Em sucesso: loga duracao.
      4. Em falha: escreve registro de auditoria estruturado em artifacts/audit/
         e re-levanta como StepError com remediacao, parando o pipeline.

    Uso:
        with StepGate("carregar_dados", log) as g:
            ...
            g.info["n_linhas"] = 123   # metadados anexados ao registro
    """

    def __init__(
        self,
        etapa: str,
        log: logging.Logger,
        audit_dir: Path = ART_AUDIT,
        fatal: bool = True,
    ):
        self.etapa = etapa
        self.log = log
        self.audit_dir = audit_dir
        self.fatal = fatal
        self.info: dict[str, Any] = {}
        self._t0 = 0.0
        self.record: Optional[StepRecord] = None

    def __enter__(self) -> "StepGate":
        self._t0 = time.time()
        self.record = StepRecord(
            etapa=self.etapa, inicio=datetime.now(timezone.utc).isoformat()
        )
        self.log.info(f"[ETAPA] >>> {self.etapa}")
        return self

    def __exit__(self, et, ev, tb) -> bool:
        assert self.record is not None
        self.record.duracao_s = round(time.time() - self._t0, 2)
        self.record.fim = datetime.now(timezone.utc).isoformat()
        if et is None:
            self.record.status = "ok"
            self.log.info(
                f"[ETAPA] <<< {self.etapa} OK ({self.record.duracao_s}s) {self.info}"
            )
            return False

        # falha
        self.record.status = "falha"
        if isinstance(ev, LibrasError):
            self.record.erro = ev.as_dict()
        else:
            self.record.erro = {
                "code": "E_EXCECAO_NAO_TIPADA",
                "msg": f"{type(ev).__name__}: {ev}",
                "fix": "inspecione o traceback no registro de auditoria",
                "ctx": {"traceback": traceback.format_exc()},
            }
        self._escrever_auditoria()
        self.log.error(
            f"[ETAPA] <<< {self.etapa} FALHOU: {self.record.erro['code']} "
            f"-- {self.record.erro['msg']}"
        )
        if self.fatal:
            # converte qualquer excecao em StepError tipado e para o pipeline
            raise StepError(
                code=self.record.erro["code"],
                msg=f"etapa '{self.etapa}' falhou: {self.record.erro['msg']}",
                fix=self.record.erro.get("fix", ""),
                ctx={"audit": str(self._audit_path()), **self.info},
            ) from ev
        return True  # suprime se nao-fatal

    def _audit_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.audit_dir / f"audit_{self.etapa}_{ts}.json"

    def _escrever_auditoria(self) -> None:
        if self.record is None:  # invariante: nunca deveria, mas auditamos a anomalia
            self.log.warning("auditoria sem StepRecord (record=None); nada a gravar")
            return
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            payload = asdict(self.record)
            payload["info"] = self.info
            self._audit_path().write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:  # auditoria nunca deve mascarar o erro original
            self.log.warning(f"falha ao escrever auditoria: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# IO JSON robusto
# ─────────────────────────────────────────────────────────────────────────────
def salvar_json(dados: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    tmp.replace(path)  # escrita atomica: evita JSON truncado em caso de crash


def carregar_json(path: Path) -> Any:
    require(
        path.exists(),
        "E_JSON_AUSENTE",
        f"arquivo nao encontrado: {path}",
        fix="verifique o caminho ou rode a etapa anterior",
    )
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# CONTRATOS DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: Any) -> str:
    import re

    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def validar_dataframe(df, nome: str, log: logging.Logger) -> dict:
    """
    Valida UM dataframe processado contra o contrato. Levanta DataContractError
    no primeiro problema critico. Retorna dict de metricas.
    """
    import pandas as pd  # noqa: F401  (garante dependencia presente)

    require(
        len(df) > 0,
        "E_DS_VAZIO",
        f"dataset '{nome}' vazio",
        fix="rode 01_prepare_data.py",
        err=DataContractError,
    )

    faltando = [c for c in COLS_PROCESSED if c not in df.columns]
    require(
        not faltando,
        "E_COLS_AUSENTES",
        f"dataset '{nome}' sem colunas: {faltando}",
        fix=f"esperado: {COLS_PROCESSED}",
        err=DataContractError,
        ctx={"colunas_presentes": list(df.columns)},
    )

    n_nan = int(df[["input", "output"]].isna().any(axis=1).sum())
    require(
        n_nan == 0,
        "E_NAN",
        f"dataset '{nome}' tem {n_nan} linhas com input/output NaN",
        fix="dropna na preparacao",
        err=DataContractError,
    )

    # Direcao valida
    dirs = set(df["direcao"].unique())
    invalidas = dirs - DIRECOES_VALIDAS
    require(
        not invalidas,
        "E_DIR_INVALIDA",
        f"dataset '{nome}' com direcoes invalidas: {invalidas}",
        fix=f"direcoes validas: {sorted(DIRECOES_VALIDAS)}",
        err=DataContractError,
    )

    # Prefixo de tarefa presente no input (contrato critico anti-bug-do-smoke)
    sem_prefixo = df[
        ~df["input"].astype(str).str.startswith((PREFIXO_PT2LIBRAS, PREFIXO_LIBRAS2PT))
    ]
    require(
        len(sem_prefixo) == 0,
        "E_SEM_PREFIXO",
        f"dataset '{nome}': {len(sem_prefixo)} inputs sem prefixo de tarefa canonico",
        fix="rode 01_prepare_data.py que injeta o prefixo via montar_input()",
        err=DataContractError,
        ctx={
            "exemplo": (
                str(sem_prefixo["input"].iloc[0])[:80] if len(sem_prefixo) else ""
            )
        },
    )

    # Glosa: alvo de pt2libras deve ter token de pontuacao
    p2l = df[df["direcao"] == DIR_PT2LIBRAS]
    if len(p2l) > 0:
        sem_pont = p2l[
            ~p2l["output"]
            .astype(str)
            .apply(lambda g: any(t in g for t in TOKENS_PONTUACAO))
        ]
        pct = len(sem_pont) / len(p2l)
        if pct > 0.05:
            log.warning(f"[{nome}] {pct:.1%} das glosas-alvo sem token de pontuacao")

    return {
        "n_linhas": len(df),
        "n_pt2libras": int((df["direcao"] == DIR_PT2LIBRAS).sum()),
        "n_libras2pt": int((df["direcao"] == DIR_LIBRAS2PT).sum()),
    }


def checar_leakage(df_treino, df_outro, nome_outro: str, log: logging.Logger) -> int:
    """
    Verifica leakage em nivel de PAR (portugues_original + libras_glosa),
    nao em nivel de linha -- um par bidirecional aparece como 2 linhas e nao
    pode ser tratado como duplicata. Retorna numero de pares vazados.
    """

    def chaves(df):
        return set(
            zip(df["portugues_original"].map(_norm), df["libras_glosa"].map(_norm))
        )

    vazados = chaves(df_treino) & chaves(df_outro)
    if vazados:
        log.error(
            f"[LEAKAGE] {len(vazados)} pares de '{nome_outro}' presentes no treino"
        )
    else:
        log.info(f"[OK] sem leakage entre treino e '{nome_outro}'")
    return len(vazados)


def auditar_truncamento(df, tokenizer, log: logging.Logger) -> dict:
    """
    Mede quantos exemplos seriam SILENCIOSAMENTE truncados por MAX_INPUT/MAX_TARGET.
    Truncamento silencioso corrompe labels e degrada metricas sem aviso.
    """
    inp_lens = [len(tokenizer(str(x)).input_ids) for x in df["input"]]
    out_lens = [len(tokenizer(text_target=str(x)).input_ids) for x in df["output"]]
    n_inp_trunc = sum(1 for n in inp_lens if n > MAX_INPUT)
    n_out_trunc = sum(1 for n in out_lens if n > MAX_TARGET)
    res = {
        "max_input_tokens": max(inp_lens) if inp_lens else 0,
        "max_target_tokens": max(out_lens) if out_lens else 0,
        "p95_input": (
            int(sorted(inp_lens)[int(0.95 * len(inp_lens))]) if inp_lens else 0
        ),
        "p95_target": (
            int(sorted(out_lens)[int(0.95 * len(out_lens))]) if out_lens else 0
        ),
        "n_input_truncados": n_inp_trunc,
        "n_target_truncados": n_out_trunc,
        "pct_input_truncados": round(n_inp_trunc / max(len(inp_lens), 1), 4),
        "pct_target_truncados": round(n_out_trunc / max(len(out_lens), 1), 4),
    }
    if n_inp_trunc or n_out_trunc:
        log.warning(
            f"[TRUNCAMENTO] input>{MAX_INPUT}: {n_inp_trunc} | "
            f"target>{MAX_TARGET}: {n_out_trunc} (corrompe labels)"
        )
    return res


# ─────────────────────────────────────────────────────────────────────────────
# METRICAS  (ponto unico de verdade -- usado por HPO, train e eval)
# ─────────────────────────────────────────────────────────────────────────────
def calcular_metricas(predicoes: list[str], referencias: list[str]) -> dict:
    """
    chrF++ e a metrica PRIMARIA. BLEU-4 sobre glosa curta (~6 tokens) e
    patologicamente baixo e instavel: um erro de 1 palavra destroi varios
    4-gramas e a brevity penalty pune sequencias curtas. chrF opera em
    n-gramas de caractere -> estavel para sequencias curtas. Padrao em pesquisa
    de traducao de lingua de sinais.
    """
    import sacrebleu as sb

    pares = [(p, r) for p, r in zip(predicoes, referencias) if p.strip() and r.strip()]
    pct_empty = 1.0 - len(pares) / max(len(predicoes), 1)
    if not pares:
        return {
            "chrf": 0.0,
            "bleu4": 0.0,
            "exact_match": 0.0,
            "pct_empty": 1.0,
            "avg_pred_len": 0.0,
            "avg_ref_len": 0.0,
        }
    pc = [p for p, _ in pares]
    rc = [r for _, r in pares]
    try:
        chrf = sb.corpus_chrf(pc, [rc], word_order=2).score  # chrF++
    except Exception:
        chrf = 0.0
    try:
        bleu4 = sb.corpus_bleu(pc, [rc]).score
    except Exception:
        bleu4 = 0.0
    em = 100.0 * sum(p.strip() == r.strip() for p, r in pares) / len(pares)
    apl = sum(len(p.split()) for p in pc) / len(pc)
    arl = sum(len(r.split()) for r in rc) / len(rc)
    return {
        "chrf": round(chrf, 4),
        "bleu4": round(bleu4, 4),
        "exact_match": round(em, 4),
        "pct_empty": round(pct_empty, 4),
        "avg_pred_len": round(apl, 2),
        "avg_ref_len": round(arl, 2),
    }


def detectar_device():
    """
    Retorna (device, dtype, grad_ckpt). Falha EXPLICITAMENTE se cair em CPU
    sem o usuario ter pedido -- 30 epocas em CPU levam dias, e um erro silencioso.
    """
    import torch

    if not torch.cuda.is_available():
        raise EnvironmentError_(
            code="E_SEM_CUDA",
            msg="CUDA indisponivel -- treino cairia em CPU (dias de duracao)",
            fix="ative a venv com PyTorch CUDA; verifique nvidia-smi e "
            "torch.cuda.is_available(); use --permitir-cpu para forcar",
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    maj = props.major
    dtype = torch.bfloat16 if maj >= 8 else torch.float16
    grad_ckpt = vram_gb < 10.0
    return "cuda", dtype, grad_ckpt
