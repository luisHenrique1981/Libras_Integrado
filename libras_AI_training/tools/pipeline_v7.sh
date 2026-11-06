#!/usr/bin/env bash
# ==============================================================================
# pipeline_v7.sh — Pipeline Libras v7. Deixe rodando e volte depois.
# ==============================================================================
#
# DIFERENCAS vs v6:
#   + Etapa 00:  preparar_vlibras_db.py    (idempotente; pula se ja normalizado)
#   + Etapa 00b: mesclar_datasets.py       (idempotente; pula se merged_v7 existe)
#   + 01-04 mantidos com novos scripts v7 (dims data-driven + balanceamento +
#     HPO refinado + dataloader paralelo + GPS multi-versao)
#   + Validacao de versao adaptada aos sinais v7
#   + --inserir-historico-retroativo v5 (idempotente; so insere uma vez)
#   + RESULTADO_v7.txt mostra TRAJETORIA MULTI-VERSAO (todas as runs)
#   + --reiniciar tambem limpa dataset_merged_v*.csv velho
#
# Sequencia: [00 vlibras] -> [00b mesclar] -> 01 -> 02 -> 03 -> 04 -> analise
# Paralelo:  monitor.py em background (HPO e treino com tags distintos)
# Logs:      artifacts/logs/pipeline_<TS>.log  (terminal + arquivo simultaneamente)
# Resumo:    RESULTADO_v7.txt na raiz (cole na conversa ao voltar)
#
# ── TMUX VS QUEDA DE ENERGIA ──────────────────────────────────────────────────
#  tmux protege contra DESCONEXAO DE TERMINAL, nao contra queda de energia.
#  Se o PC desligar, o processo morre. A solucao aqui sao os CHECKPOINTS:
#  cada etapa grava um arquivo .done ao concluir. Se o script for re-executado,
#  etapas ja concluidas sao puladas automaticamente.
#  Para recomecar do zero: bash tools/pipeline_v7.sh --reiniciar
#
# ── COMO EXECUTAR ─────────────────────────────────────────────────────────────
#
#  PRE-REQUISITO MANUAL (uma vez): baixar VLibrasBD v2 do Mendeley:
#    https://data.mendeley.com/datasets/ryj88ckjww/2
#    Clicar "Download All" (.zip ~30MB), extrair, e mover o CSV para:
#    ${PROJ_DIR}/data/raw/vlibras_db_v2.csv
#    (se nao baixar, pipeline cai no fallback bilingual_v1 com aviso)
#
#  Forma 1 — sessao atual (terminal deve ficar aberto):
#    cd ~/Projetos/treinamento_modelo_IA/libras_AI_training
#    bash tools/pipeline_v7.sh
#
#  Forma 2 — tmux (recomendado; sobrevive fechamento de terminal):
#    tmux new -s libras
#    bash tools/pipeline_v7.sh
#    Ctrl+B  D                        # destaca (terminal pode fechar)
#    tmux attach -t libras            # reconecta quando voltar
#
#  Forma 3 — nohup (sobrevive fechamento; nao permite reconectar ao vivo):
#    nohup bash tools/pipeline_v7.sh > nohup_pipeline.out 2>&1 &
#    echo "PID=$!"
#    tail -f nohup_pipeline.out
#
# ── OPCOES ────────────────────────────────────────────────────────────────────
#
#  --reiniciar          Limpa checkpoints + merged_v*.csv e recomeca do zero
#  --skip-hpo           Pula 02_hpo (usa config_campeao.json existente)
#  --skip-mesclar       Pula 00 e 00b (usa dataset_merged_v7.csv existente)
#  --steps-hpo N        Steps por trial no HPO (default 4000)
#  --timeout N          Budget total do HPO em segundos (default 43200 = 12h)
#  --monitor-interval N Segundos entre amostras do monitor (default 2)
#  --so-validar         Apenas valida ambiente e abort sem rodar (dry-run)
#  -h | --help          Esta mensagem
#
#  Exemplos:
#    bash tools/pipeline_v7.sh --steps-hpo 500 --timeout 1800   # smoke fluxo
#    bash tools/pipeline_v7.sh --skip-mesclar                    # apenas re-treina
#    bash tools/pipeline_v7.sh --skip-hpo                        # 03+04 apenas
#    bash tools/pipeline_v7.sh --reiniciar                       # tudo do zero
#    bash tools/pipeline_v7.sh --so-validar                      # checagem pre-run
# ==============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROJ_DIR="${HOME}/Projetos/treinamento_modelo_IA/libras_AI_training"
REINICIAR="false"
SKIP_HPO="false"
SKIP_MESCLAR="false"
STEPS_HPO="4000"
HPO_TIMEOUT="43200"   # 12h (v7 espaco maior)
MON_INTERVAL="2"      # segundos entre amostras
SO_VALIDAR="false"

# ── Parse CLI ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reiniciar)        REINICIAR="true" ;;
        --skip-hpo)         SKIP_HPO="true" ;;
        --skip-mesclar)     SKIP_MESCLAR="true" ;;
        --steps-hpo)        STEPS_HPO="$2";    shift ;;
        --timeout)          HPO_TIMEOUT="$2";  shift ;;
        --monitor-interval) MON_INTERVAL="$2"; shift ;;
        --so-validar)       SO_VALIDAR="true" ;;
        -h|--help)
            grep '^#' "$0" | head -75 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERRO: argumento desconhecido: '$1'"
            echo "Use -h para ver as opcoes."
            exit 1 ;;
    esac
    shift
done

# ── Validacao de existencia do projeto ────────────────────────────────────────
if [[ ! -d "${PROJ_DIR}" ]]; then
    echo "ERRO: PROJ_DIR nao existe: ${PROJ_DIR}"
    echo "Edite a variavel PROJ_DIR no inicio do script."
    exit 1
fi

# ── Deteccao e validacao do venv ──────────────────────────────────────────────
PYTHON=""
for candidate in ".venv_pytorch" ".venv"; do
    _dir="${PROJ_DIR}/${candidate}"
    if [[ -f "${_dir}/bin/python" && -f "${_dir}/bin/activate" ]]; then
        PYTHON="${_dir}/bin/python"
        VENV_ACT="${_dir}/bin/activate"
        VENV_DIR="${_dir}"
        break
    fi
done

if [[ -z "${PYTHON}" ]]; then
    echo "ERRO: nenhum venv encontrado. Procurei em:"
    echo "  ${PROJ_DIR}/.venv_pytorch/"
    echo "  ${PROJ_DIR}/.venv/"
    echo "Crie/ative o venv correto antes de executar o pipeline."
    exit 1
fi

# ── Log: tudo vai para terminal E para arquivo simultaneamente ────────────────
LOG_DIR="${PROJ_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"
TS_INICIO="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/pipeline_${TS_INICIO}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

# ── Helpers ───────────────────────────────────────────────────────────────────
ts()  { date '+%Y-%m-%dT%H:%M:%S'; }
sep() { printf '%0.s─' $(seq 1 70); echo; }

fmt_elapsed() {
    local s=$1
    printf '%02dh%02dm%02ds' $((s / 3600)) $(( (s % 3600) / 60 )) $((s % 60))
}

log_etapa() { sep; echo "[$(ts)] ▶  $*"; sep; }

# notify-send no KDE Plasma/Wayland via D-Bus da sessao do usuario.
notificar() {
    local urgency="$1" title="$2" msg="$3"
    command -v notify-send &>/dev/null || return 0
    notify-send -u "${urgency}" -t 15000 -a "Libras Pipeline" \
                "${title}" "${msg}" 2>/dev/null || true
}

# ── Sistema de checkpoints ────────────────────────────────────────────────────
CKPT_DIR="${PROJ_DIR}/artifacts/pipeline_state"
mkdir -p "${CKPT_DIR}"

done_existe() { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

marcar_done() {
    local nome="$1"
    touch "${CKPT_DIR}/${nome}.done"
    echo "[$(ts)] ✓ Checkpoint gravado: ${nome}"
}

if [[ "${REINICIAR}" == "true" ]]; then
    echo "[$(ts)] --reiniciar: limpando estado anterior..."
    # Apaga checkpoints
    rm -f "${CKPT_DIR}"/*.done 2>/dev/null || true
    # v7: apaga TAMBEM dataset_merged_v*.csv (sem isso, o mesclar pula e usa
    # corpus velho, mascarando re-aplicacao de filtros)
    rm -f "${PROJ_DIR}/data/raw/dataset_merged_v"*.csv 2>/dev/null || true
    rm -f "${PROJ_DIR}/data/raw/vlibras_db_v2_normalizado.csv" 2>/dev/null || true
    echo "[$(ts)] Estado limpo. Pipeline recomecara do zero."
fi

# ── Monitor.py: gestao de background ─────────────────────────────────────────
MONITOR_PID=""

iniciar_monitor() {
    local tag="$1"
    echo "[$(ts)] monitor.py START  tag=${tag}  interval=${MON_INTERVAL}s"
    "${PYTHON}" tools/monitor.py \
        --tag      "${tag}" \
        --interval "${MON_INTERVAL}" &
    MONITOR_PID=$!
    echo "[$(ts)] monitor PID=${MONITOR_PID}"
}

parar_monitor() {
    [[ -z "${MONITOR_PID}" ]] && return 0
    [[ -d "/proc/${MONITOR_PID}" ]] || { MONITOR_PID=""; return 0; }
    echo "[$(ts)] monitor STOP PID=${MONITOR_PID}..."
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
    echo "[$(ts)] Monitor encerrado. JSON em ${LOG_DIR}/"
}

# ── Trap: limpa em qualquer saida (erro, Ctrl+C, sucesso) ─────────────────────
cleanup() {
    local code=$?
    parar_monitor || true
    if [[ $code -ne 0 ]]; then
        sep
        echo "  PIPELINE ABORTOU  (exit code=${code})"
        echo "  Checkpoints de etapas concluidas foram preservados."
        echo "  Re-execute o script para retomar da etapa que falhou."
        sep
        echo "  Log: ${MASTER_LOG}"
        notificar "critical" \
            "❌ Pipeline v7 FALHOU (code=${code})" \
            "Re-execute para retomar. Log: artifacts/logs/"
        echo ""
        echo "Ultimas 15 linhas do log:"
        tail -15 "${MASTER_LOG}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Cabecalho ─────────────────────────────────────────────────────────────────
sep
echo "  Pipeline Libras v7  —  $(ts)"
sep
echo "  Projeto       : ${PROJ_DIR}"
echo "  Python        : ${PYTHON}"
echo "  venv          : ${VENV_DIR}"
echo "  Steps HPO     : ${STEPS_HPO} por trial"
echo "  Timeout HPO   : ${HPO_TIMEOUT}s  ($(fmt_elapsed ${HPO_TIMEOUT}))"
echo "  Skip HPO      : ${SKIP_HPO}"
echo "  Skip mesclar  : ${SKIP_MESCLAR}"
echo "  Reiniciar     : ${REINICIAR}"
echo "  Log           : ${MASTER_LOG}"
echo "  Checkpoints   : ${CKPT_DIR}/"
sep

# ── Ativar venv e inspecionar ambiente ────────────────────────────────────────
cd "${PROJ_DIR}"
# shellcheck source=/dev/null
source "${VENV_ACT}"

echo "[$(ts)] Ambiente:"
"${PYTHON}" - <<'PYEOF'
import sys, torch
print(f"  Python  : {sys.version.split()[0]}  ({sys.executable})")
print(f"  PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory // 1024**3
    print(f"  GPU     : {p.name}  {gb} GB VRAM  cc={p.major}.{p.minor}")
    print(f"  BF16    : {torch.cuda.is_bf16_supported()}")
else:
    print("  GPU     : AUSENTE — treino em CPU sera muito lento!")
    print("  Use --permitir-cpu nos scripts se for intencional.")
PYEOF

# ── Validacao de versao v7 (defesa contra aplicacao parcial) ──────────────────
# Sinais marcantes da v7 em cada arquivo. Se faltar qualquer um, abortar AGORA
# com mensagem explicita, em vez de falhar com erro confuso depois.
echo "[$(ts)] Validando versao dos scripts (sinais v7)..."
ERRO_VERSAO=0

check_sinal() {
    local arq="$1" sinal="$2" desc="$3"
    if [[ ! -f "${arq}" ]]; then
        echo "  [X] ${arq}: ARQUIVO AUSENTE"
        ERRO_VERSAO=1
        return
    fi
    if ! grep -q "${sinal}" "${arq}" 2>/dev/null; then
        echo "  [X] ${arq}: falta '${sinal}'"
        echo "      Sinal de ${desc}"
        ERRO_VERSAO=1
    else
        echo "  [OK] ${arq}: ${desc}"
    fi
}

check_sinal "${PROJ_DIR}/src/libras/common.py" \
            "_carregar_dims_iniciais" \
            "dims data-driven (v6.2+)"
check_sinal "${PROJ_DIR}/scripts/01_prepare_data.py" \
            "balancear_por_tema" \
            "balanceamento por tema (v7)"
check_sinal "${PROJ_DIR}/scripts/01_prepare_data.py" \
            "analise_e_derivar_dims" \
            "dims data-driven (v6.2+)"
check_sinal "${PROJ_DIR}/scripts/02_hpo.py" \
            "libras_hpo_v7" \
            "STUDY_NAME v7 (search space refinado)"
check_sinal "${PROJ_DIR}/scripts/03_train.py" \
            "num_proc=n_proc" \
            "pre-tokenizacao paralela (v7)"
check_sinal "${PROJ_DIR}/scripts/04_evaluate.py" \
            "inserir-historico-retroativo" \
            "GPS multi-versao (v7)"
check_sinal "${PROJ_DIR}/tools/mesclar_datasets.py" \
            "MARCADORES_INGLES_FUNCIONAIS" \
            "filtros cross-lingual (v7)"
check_sinal "${PROJ_DIR}/tools/preparar_vlibras_db.py" \
            "ryj88ckjww" \
            "preparador VLibrasBD v2 (v7)"

if [[ ${ERRO_VERSAO} -ne 0 ]]; then
    echo ""
    sep
    echo "  VERSAO INCOMPLETA — pipeline abortado antes de comecar"
    sep
    echo "  Os scripts v7 nao estao todos aplicados."
    echo "  Copie de /mnt/user-data/outputs/ para:"
    echo "    cp .../preparar_vlibras_db.py    tools/"
    echo "    cp .../mesclar_datasets.py       tools/"
    echo "    cp .../01_prepare_data.py        scripts/"
    echo "    cp .../02_hpo.py                 scripts/"
    echo "    cp .../03_train.py               scripts/"
    echo "    cp .../04_evaluate.py            scripts/"
    echo "  Depois: bash tools/pipeline_v7.sh --reiniciar"
    sep
    exit 2
fi
echo ""

if [[ "${SO_VALIDAR}" == "true" ]]; then
    sep
    echo "  --so-validar: AMBIENTE VALIDADO. Encerrando sem rodar."
    sep
    exit 0
fi

CONFIG="artifacts/configs/config_campeao.json"
MERGED_V7="${PROJ_DIR}/data/raw/dataset_merged_v7.csv"
VLIBRAS_RAW="${PROJ_DIR}/data/raw/vlibras_db_v2.csv"
VLIBRAS_NORM="${PROJ_DIR}/data/raw/vlibras_db_v2_normalizado.csv"
T_PIPELINE=$(date +%s)

# ==============================================================================
# Etapa 00: preparar VLibrasBD v2  (OPCIONAL — fallback graceful)
# ==============================================================================
if [[ "${SKIP_MESCLAR}" == "true" ]]; then
    log_etapa "00 — PULADO (--skip-mesclar)"
elif done_existe "step00_vlibras" && [[ "${REINICIAR}" != "true" ]] \
        && [[ -f "${VLIBRAS_NORM}" ]]; then
    log_etapa "00 — PULADO (vlibras_db_v2_normalizado.csv presente)"
else
    if [[ ! -f "${VLIBRAS_RAW}" ]]; then
        log_etapa "00 — VLibrasBD v2 nao baixado (FALLBACK)"
        echo ""
        echo "  VLibrasBD v2 nao encontrado em: ${VLIBRAS_RAW}"
        echo ""
        echo "  Esse dataset (127k pares, UFPB, CC BY-SA 4.0) tem 87% mais"
        echo "  pares que o libras_bilingual_dataset.csv v1 (68k) e separa"
        echo "  conteudo governamental de geral via flag is_government_source."
        echo ""
        echo "  PARA OBTER (uma vez):"
        echo "    1. Abra https://data.mendeley.com/datasets/ryj88ckjww/2"
        echo "    2. Clique 'Download All' (.zip ~30MB)"
        echo "    3. Extraia e mova o CSV para: ${VLIBRAS_RAW}"
        echo "    4. Rode: bash tools/pipeline_v7.sh"
        echo ""
        echo "  Sem o v2, o mesclar_datasets cai automaticamente no fallback"
        echo "  bilingual_v1. O pipeline CONTINUA, mas com menos dados."
        echo ""
        # NAO marca como done — para que rode de novo na proxima execucao
        # caso o usuario baixe.
    else
        log_etapa "00 — preparar_vlibras_db.py"
        T0=$(date +%s)
        "${PYTHON}" tools/preparar_vlibras_db.py
        E=$(( $(date +%s) - T0 ))
        if [[ ! -f "${VLIBRAS_NORM}" ]]; then
            echo "ERRO: preparar_vlibras_db terminou sem escrever:"
            echo "      ${VLIBRAS_NORM}"
            exit 1
        fi
        marcar_done "step00_vlibras"
        echo "[$(ts)] 00 OK em $(fmt_elapsed $E)"
        notificar "normal" "Etapa 00 OK" "preparar_vlibras_db em $(fmt_elapsed $E)"
    fi
fi

# ==============================================================================
# Etapa 00b: mesclar datasets  (sempre obrigatoria; idempotente)
# ==============================================================================
if [[ "${SKIP_MESCLAR}" == "true" ]]; then
    log_etapa "00b — PULADO (--skip-mesclar)"
    if [[ ! -f "${MERGED_V7}" ]]; then
        echo "ERRO: --skip-mesclar exige ${MERGED_V7} existente."
        echo "Rode sem --skip-mesclar pelo menos uma vez."
        exit 1
    fi
    echo "  Usando merged existente: ${MERGED_V7}"

elif done_existe "step00b_mesclar" && [[ "${REINICIAR}" != "true" ]] \
        && [[ -f "${MERGED_V7}" ]]; then
    log_etapa "00b — PULADO (dataset_merged_v7.csv presente)"
    # Imprime estatisticas do merged existente para auditoria
    "${PYTHON}" - <<PYEOF
import pandas as pd
from pathlib import Path
df = pd.read_csv("${MERGED_V7}")
n_pares = len(df) // 2
print(f"  merged_v7: {len(df)} linhas / {n_pares} pares unicos")
if "tema" in df.columns:
    dist = df.drop_duplicates(subset=["portugues_original","libras_glosa"])["tema"].value_counts().head(10)
    print("  Top 10 temas (pares unicos):")
    for t, n in dist.items():
        print(f"    {t:<40} {n:>6} ({100*n/n_pares:.1f}%)")
PYEOF

else
    log_etapa "00b — mesclar_datasets.py"
    T0=$(date +%s)
    "${PYTHON}" tools/mesclar_datasets.py
    E=$(( $(date +%s) - T0 ))
    if [[ ! -f "${MERGED_V7}" ]]; then
        echo "ERRO: mesclar_datasets terminou sem escrever:"
        echo "      ${MERGED_V7}"
        exit 1
    fi
    # Auditoria humana: mostrar amostra de descartados
    DESCARTADOS="${PROJ_DIR}/data/reports/mesclar_descartados_amostra.csv"
    if [[ -f "${DESCARTADOS}" ]]; then
        echo ""
        echo "  Amostra de pares DESCARTADOS pelos filtros (cabecalho + 10 linhas):"
        head -11 "${DESCARTADOS}" | sed 's/^/    /'
        echo "  Arquivo completo: ${DESCARTADOS}"
    fi
    marcar_done "step00b_mesclar"
    echo "[$(ts)] 00b OK em $(fmt_elapsed $E)"
    notificar "normal" "Etapa 00b OK" "mesclar_datasets em $(fmt_elapsed $E)"
fi

# ==============================================================================
# Etapa 01: prepare_data
# ==============================================================================
DIMS_JSON="${PROJ_DIR}/data/processed/dataset_dims.json"
if done_existe "step01_prepare" && [[ "${REINICIAR}" != "true" ]] \
        && [[ -f "${DIMS_JSON}" ]]; then
    log_etapa "01 — PULADO (checkpoint + dataset_dims.json presentes)"
    echo "  Para refazer: bash tools/pipeline_v7.sh --reiniciar"
elif done_existe "step01_prepare" && [[ ! -f "${DIMS_JSON}" ]]; then
    # Estado inconsistente: checkpoint existe mas dims.json ausente
    log_etapa "01 — RE-EXECUTAR (checkpoint existe mas dataset_dims.json AUSENTE)"
    T0=$(date +%s)
    "${PYTHON}" scripts/01_prepare_data.py
    E=$(( $(date +%s) - T0 ))
    if [[ ! -f "${DIMS_JSON}" ]]; then
        echo "ERRO: 01 terminou sem escrever ${DIMS_JSON}."
        exit 1
    fi
    marcar_done "step01_prepare"
    echo "[$(ts)] 01 OK em $(fmt_elapsed $E)"
    notificar "normal" "Etapa 01 OK" "prepare_data em $(fmt_elapsed $E) (re-execucao)"
else
    log_etapa "01 — 01_prepare_data.py"
    T0=$(date +%s)
    "${PYTHON}" scripts/01_prepare_data.py
    E=$(( $(date +%s) - T0 ))
    if [[ ! -f "${DIMS_JSON}" ]]; then
        echo "ERRO: 01 terminou sem escrever ${DIMS_JSON}."
        echo "       A versao em scripts/01_prepare_data.py nao e a v7."
        exit 1
    fi
    marcar_done "step01_prepare"
    echo "[$(ts)] 01 OK em $(fmt_elapsed $E)"
    notificar "normal" "Etapa 01 OK" "prepare_data em $(fmt_elapsed $E)"
fi

# ==============================================================================
# Etapa 02: HPO
# ==============================================================================
if [[ "${SKIP_HPO}" == "true" ]]; then
    log_etapa "02 — HPO PULADO (--skip-hpo)"
    if [[ ! -f "${CONFIG}" ]]; then
        echo "ERRO: --skip-hpo exige ${CONFIG} existente."
        echo "Rode sem --skip-hpo para gerar o config."
        exit 1
    fi
    echo "  Usando config: ${CONFIG}"

elif done_existe "step02_hpo" && [[ "${REINICIAR}" != "true" ]]; then
    log_etapa "02 — PULADO (checkpoint existe)"
    [[ -f "${CONFIG}" ]] || {
        echo "ERRO: checkpoint de 02 existe mas ${CONFIG} esta ausente."
        echo "Checkpoint invalido. Rode: bash tools/pipeline_v7.sh --reiniciar"
        exit 1
    }

else
    log_etapa "02 — 02_hpo.py  (steps=${STEPS_HPO}, timeout=$(fmt_elapsed ${HPO_TIMEOUT}))"
    iniciar_monitor "hpo"
    T0=$(date +%s)
    "${PYTHON}" scripts/02_hpo.py \
        --steps-hpo "${STEPS_HPO}" \
        --timeout   "${HPO_TIMEOUT}"
    E=$(( $(date +%s) - T0 ))
    parar_monitor
    marcar_done "step02_hpo"
    echo "[$(ts)] 02 OK em $(fmt_elapsed $E)"
    notificar "normal" \
        "HPO concluido ($(fmt_elapsed $E))" \
        "Iniciando treino definitivo..."
fi

# ==============================================================================
# Etapa 03: treino definitivo
# ==============================================================================
if done_existe "step03_train" && [[ "${REINICIAR}" != "true" ]]; then
    log_etapa "03 — PULADO (checkpoint existe)"
else
    log_etapa "03 — 03_train.py"
    iniciar_monitor "train"
    T0=$(date +%s)
    "${PYTHON}" scripts/03_train.py --config "${CONFIG}"
    E=$(( $(date +%s) - T0 ))
    parar_monitor
    marcar_done "step03_train"
    echo "[$(ts)] 03 OK em $(fmt_elapsed $E)"
    notificar "normal" \
        "Treino concluido ($(fmt_elapsed $E))" \
        "Iniciando avaliacao..."
fi

# ==============================================================================
# Etapa 04: avaliacao + GPS multi-versao
# ==============================================================================
if done_existe "step04_evaluate" && [[ "${REINICIAR}" != "true" ]]; then
    log_etapa "04 — PULADO (checkpoint existe)"
else
    log_etapa "04 — 04_evaluate.py  (GPS multi-versao + erros + retroativo v5)"
    T0=$(date +%s)
    # --inserir-historico-retroativo: idempotente (so insere v5 se nao existir).
    # Garante que a 1a run do v7 tenha baseline para comparar automaticamente.
    "${PYTHON}" scripts/04_evaluate.py --save-errors --inserir-historico-retroativo
    E=$(( $(date +%s) - T0 ))
    marcar_done "step04_evaluate"
    echo "[$(ts)] 04 OK em $(fmt_elapsed $E)"
fi

# ==============================================================================
# Etapa 05: analise de erros (rapida, sem checkpoint — corre sempre)
# ==============================================================================
log_etapa "analise_erros.py  (categoriza evaluate_erros.csv)"
"${PYTHON}" tools/analise_erros.py

# ==============================================================================
# Resumo final v7 com TRAJETORIA MULTI-VERSAO
# ==============================================================================
log_etapa "Gerando RESULTADO_v7.txt"
"${PYTHON}" - <<'PYEOF'
import json
from pathlib import Path

P = Path.cwd()

def J(rel):
    f = P / rel
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None

ev    = J("data/reports/evaluate_report.json")
gps   = J("data/reports/gps_recomendacao.json")
hpo   = J("data/reports/hpo_report.json")
prep  = J("data/reports/prepare_report.json")
mesc  = J("data/reports/mesclar_report.json")
vlib  = J("data/reports/preparar_vlibras_db_report.json")
ae    = J("data/reports/analise_erros.json")
hist  = J("data/reports/historico_avaliacoes.json")

S1 = "=" * 70
S2 = "─" * 70
L = [S1, "  RESULTADO PIPELINE LIBRAS v7", S1]

# Fontes
if vlib:
    L.append("VLibrasBD: {} pares  (governo={}, geral={})  ganho_vs_v1={}x".format(
        vlib.get("n_pares", 0),
        vlib.get("distribuicao_tema", {}).get("vlibrasdb_governo", 0),
        vlib.get("distribuicao_tema", {}).get("vlibrasdb_geral", 0),
        vlib.get("ganho_vs_v1", "?")))

if mesc:
    L.append("Mesclar  : {} pares finais  (descartados: {}/{}={:.1f}%)".format(
        mesc.get("pares_finais", 0),
        mesc.get("contagem_filtros", {}).get("longo_pt", 0) +
        mesc.get("contagem_filtros", {}).get("longo_glosa", 0) +
        mesc.get("contagem_filtros", {}).get("pontuacao_excessiva_pt", 0) +
        mesc.get("contagem_filtros", {}).get("ingles_em_pt", 0) +
        mesc.get("contagem_filtros", {}).get("enciclopedico", 0),
        sum(mesc.get("contagem_filtros", {}).values()),
        100 * (1 - mesc.get("taxa_retencao", 1))))

if prep:
    s = prep.get("saidas", {})
    L.append("Dataset  : {:>7,} pares treino | {:,} val | {:,} teste".format(
        s.get("treino", {}).get("pares", 0),
        s.get("val",    {}).get("pares", 0),
        s.get("teste",  {}).get("pares", 0)))
    bal = prep.get("balanceamento", {})
    if bal.get("aplicado"):
        L.append("Balanceam: max_pct_tema={}  removidos={}  cap_pares={}".format(
            bal.get("max_pct_tema", "?"),
            bal.get("removidos", "?"),
            bal.get("cap_pares", "?")))

# HPO
if hpo:
    m = hpo.get("melhor", {})
    p = m.get("params", {})
    L.append("HPO v7   : trial #{} | chrF={:.4f} | {}/{} trials OK | fim={}".format(
        m.get("trial"), m.get("chrf", 0),
        hpo.get("trials_ok", 0), hpo.get("trials_totais", 0),
        hpo.get("encerramento", "?")))
    if p:
        L.append("         : lr={:.2e}  batch={}  lora_r={}  alpha={}  drop={:.3f}  sched={}".format(
            p.get("learning_rate", 0), p.get("batch_size"),
            p.get("lora_r"), p.get("lora_alpha"),
            p.get("lora_dropout", 0), p.get("lr_scheduler_type")))

# Eval atual
if ev:
    for d, r in ev.get("por_direcao", {}).items():
        L.append("Eval     : [{:<12}]  chrF={:.2f}  BLEU4={:.2f}  EM={:.2f}%  n={}".format(
            d, r.get("chrf", 0), r.get("bleu4", 0),
            r.get("exact_match", 0), r.get("n", 0)))

# TRAJETORIA MULTI-VERSAO (mudanca chave da v7)
if hist and len(hist) >= 1:
    L.append(S2)
    L.append("TRAJETORIA MULTI-VERSAO:")
    L.append("  {:<2}  {:<28}  {:>9}  {:>9}  {:>9}".format(
        "#", "modelo", "chrf_med", "chrf_p2l", "chrf_l2p"))
    for i, h in enumerate(hist):
        modelo = Path(h.get("model_dir", "?")).name[:28]
        L.append("  {:<2}  {:<28}  {:>9.2f}  {:>9.2f}  {:>9.2f}".format(
            i, modelo,
            h.get("chrf_medio", 0),
            h.get("chrf_pt2libras", 0),
            h.get("chrf_libras2pt", 0)))
    # Melhor da historia
    melhor = max(hist, key=lambda h: h.get("chrf_medio", 0))
    L.append("  MELHOR HISTORICO: {} chrF_medio={:.2f}".format(
        Path(melhor.get("model_dir", "?")).name,
        melhor.get("chrf_medio", 0)))

# Categorias de erro
if ae:
    L.append(S2)
    for d, info in ae.get("por_direcao", {}).items():
        cats = info.get("categorias", {})
        partes = " | ".join("{:}={:.0f}%".format(k, v["pct"]) for k, v in cats.items())
        L.append("Erros    : [{:<12}]  {}".format(d, partes))

# GPS (com deltas multi-versao)
if gps:
    L.append(S2)
    L.append("GPS      : {}".format(gps.get("diagnostico", "")))
    L.append("Acao     : {}".format(gps.get("acao_prioritaria", "")))
    L.append("Razao    : {}".format(gps.get("razao", "")))
    # Deltas v7
    deltas = []
    if gps.get("delta_vs_anterior") is not None:
        deltas.append("vs_anterior={:+.2f}".format(gps["delta_vs_anterior"]))
    if gps.get("delta_vs_melhor_historico") is not None:
        deltas.append("vs_melhor={:+.2f}".format(gps["delta_vs_melhor_historico"]))
    if gps.get("delta_vs_baseline") is not None:
        deltas.append("vs_baseline={:+.2f}".format(gps["delta_vs_baseline"]))
    if deltas:
        L.append("Deltas   : {}".format("  ".join(deltas)))
    alvo = gps.get("vocabulario_alvo", [])
    if alvo:
        top5 = ["{} (tr={}, s={:.2f})".format(
                    v["palavra"], v["n_treino"], v["score_raridade"])
                for v in alvo[:5]]
        L.append("Vocab    : {}".format(", ".join(top5)))

L.append(S1)
txt = "\n".join(L) + "\n"
print(txt)

out = Path("RESULTADO_v7.txt")
out.write_text(txt, encoding="utf-8")
print("[Salvo em {}]".format(out.resolve()))
print("[Cole o conteudo de RESULTADO_v7.txt na conversa para continuar]")
PYEOF

# ── Notificacao e encerramento ────────────────────────────────────────────────
TEMPO_TOTAL=$(( $(date +%s) - T_PIPELINE ))

GPS_DIAG=$("${PYTHON}" -c "
import json
from pathlib import Path
p = Path('data/reports/gps_recomendacao.json')
if p.exists():
    print(json.loads(p.read_text()).get('diagnostico', '?'))
else:
    print('ver_log')
" 2>/dev/null || echo "ver_log")

notificar "normal" \
    "✅ Pipeline v7 CONCLUIDO  ($(fmt_elapsed ${TEMPO_TOTAL}))" \
    "GPS: ${GPS_DIAG} — Veja RESULTADO_v7.txt"

sep
echo "  PIPELINE v7 CONCLUIDO  —  $(ts)"
echo "  Tempo total  : $(fmt_elapsed ${TEMPO_TOTAL})"
sep
echo "  Cole na conversa  : ${PROJ_DIR}/RESULTADO_v7.txt"
echo "  GPS completo      : data/reports/gps_recomendacao.json"
echo "  Trajetoria        : data/reports/historico_avaliacoes.json"
echo "  Filtros descarte  : data/reports/mesclar_descartados_amostra.csv"
echo "  Log mestre        : ${MASTER_LOG}"
echo "  Monitor HPO       : ${LOG_DIR}/monitor_hpo_*.json"
echo "  Monitor treino    : ${LOG_DIR}/monitor_train_*.json"
echo "  Checkpoints       : ${CKPT_DIR}/"
sep
