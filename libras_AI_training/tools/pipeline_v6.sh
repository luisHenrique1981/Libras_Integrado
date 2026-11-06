#!/usr/bin/env bash
# ==============================================================================
# pipeline_v6.sh — Pipeline Libras v6. Deixe rodando e volte depois.
# ==============================================================================
#
# Sequencia:  01_prepare -> 02_hpo -> 03_train -> 04_evaluate -> analise_erros
# Paralelo:   monitor.py em background (HPO e treino com tags distintos)
# Logs:       artifacts/logs/pipeline_<TS>.log  (terminal + arquivo simultaneamente)
# Resumo:     RESULTADO_v6.txt na raiz (cole na conversa ao voltar)
#
# ── TMUX VS QUEDA DE ENERGIA ──────────────────────────────────────────────────
#  tmux protege contra DESCONEXAO DE TERMINAL, nao contra queda de energia.
#  Se o PC desligar, o processo morre. A solucao aqui sao os CHECKPOINTS:
#  cada etapa grava um arquivo .done ao concluir. Se o script for re-executado
#  (apos ligar o PC de volta), etapas ja concluidas sao puladas automaticamente.
#  Para recomecar do zero: bash tools/pipeline_v6.sh --reiniciar
#
# ── COMO EXECUTAR ─────────────────────────────────────────────────────────────
#
#  Forma 1 — sessao atual (terminal deve ficar aberto):
#    cd ~/Projetos/treinamento_modelo_IA/libras_AI_training
#    bash tools/pipeline_v6.sh
#
#  Forma 2 — tmux (recomendado; sobrevive fechamento de terminal):
#    tmux new -s libras
#    bash tools/pipeline_v6.sh        # dentro do tmux
#    Ctrl+B  D                        # destaca (terminal pode fechar)
#    tmux attach -t libras            # reconecta quando voltar
#
#  Forma 3 — nohup (sobrevive fechamento; nao permite reconectar ao vivo):
#    nohup bash tools/pipeline_v6.sh > nohup_pipeline.out 2>&1 &
#    echo "PID=$!"
#    tail -f nohup_pipeline.out
#
# ── OPCOES ────────────────────────────────────────────────────────────────────
#
#  --reiniciar          Limpa todos os checkpoints e recomeça do zero
#  --skip-hpo           Pula 02_hpo (usa config_campeao.json existente)
#  --steps-hpo N        Steps por trial no HPO (default 3000)
#  --timeout N          Budget total do HPO em segundos (default 28800 = 8h)
#  --monitor-interval N Segundos entre amostras do monitor (default 2)
#  -h | --help          Esta mensagem
#
#  Exemplos:
#    bash tools/pipeline_v6.sh --steps-hpo 300 --timeout 1200   # smoke de fluxo
#    bash tools/pipeline_v6.sh --skip-hpo                        # so re-treina
#    bash tools/pipeline_v6.sh --reiniciar                        # tudo do zero
# ==============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROJ_DIR="${HOME}/Projetos/treinamento_modelo_IA/libras_AI_training"
REINICIAR="false"
SKIP_HPO="false"
STEPS_HPO="3000"
HPO_TIMEOUT="28800"   # 8h
MON_INTERVAL="2"      # segundos entre amostras

# ── Parse CLI ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reiniciar)        REINICIAR="true" ;;
        --skip-hpo)         SKIP_HPO="true" ;;
        --steps-hpo)        STEPS_HPO="$2";    shift ;;
        --timeout)          HPO_TIMEOUT="$2";  shift ;;
        --monitor-interval) MON_INTERVAL="$2"; shift ;;
        -h|--help)
            grep '^#' "$0" | head -55 | sed 's/^# \?//'
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
# Procura .venv_pytorch (original) e .venv (symlink para ele) nesta ordem.
# Usar caminho ABSOLUTO para o interpretador Python em todos os comandos;
# elimina qualquer shadow do sistema ou de outro venv no PATH.
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
# O mkdir aqui e antes do exec para que o arquivo de log exista ao redirecionar.
LOG_DIR="${PROJ_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"
TS_INICIO="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/pipeline_${TS_INICIO}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

# ── Helpers ───────────────────────────────────────────────────────────────────
ts()  { date '+%Y-%m-%dT%H:%M:%S'; }
sep() { printf '%0.s─' $(seq 1 70); echo; }

fmt_elapsed() {
    # Recebe segundos, retorna HHhMMmSSs. Sem dependencias externas.
    local s=$1
    printf '%02dh%02dm%02ds' $((s / 3600)) $(( (s % 3600) / 60 )) $((s % 60))
}

log_etapa() { sep; echo "[$(ts)] ▶  $*"; sep; }

# notify-send funciona no KDE Plasma/Wayland via D-Bus da sessao do usuario.
# || return 0 garante que uma falha aqui nunca aborta o pipeline.
notificar() {
    local urgency="$1" title="$2" msg="$3"
    command -v notify-send &>/dev/null || return 0
    notify-send -u "${urgency}" -t 15000 -a "Libras Pipeline" \
                "${title}" "${msg}" 2>/dev/null || true
}

# ── Sistema de checkpoints ────────────────────────────────────────────────────
# Cada etapa grava <nome>.done apos concluir com exit code 0.
# Na re-execucao (apos queda de energia, por exemplo), etapas com .done
# sao puladas automaticamente. --reiniciar apaga todos os .done.
CKPT_DIR="${PROJ_DIR}/artifacts/pipeline_state"
mkdir -p "${CKPT_DIR}"

done_existe() { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

marcar_done() {
    local nome="$1"
    # touch e atomico no nivel de syscall; o arquivo ou existe ou nao.
    touch "${CKPT_DIR}/${nome}.done"
    echo "[$(ts)] ✓ Checkpoint gravado: ${nome}"
}

if [[ "${REINICIAR}" == "true" ]]; then
    echo "[$(ts)] --reiniciar: apagando checkpoints anteriores..."
    rm -f "${CKPT_DIR}"/*.done 2>/dev/null || true
    echo "[$(ts)] Checkpoints limpos. Pipeline recomeçara do zero."
fi

# ── Monitor.py: gestao de background ─────────────────────────────────────────
MONITOR_PID=""

iniciar_monitor() {
    local tag="$1"
    echo "[$(ts)] monitor.py START  tag=${tag}  interval=${MON_INTERVAL}s"
    # Usa o Python do venv para o monitor, garantindo deps corretas.
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
    # SIGTERM -> handler do monitor.py seta _stop=True -> loop drena -> grava JSON.
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    # wait bloqueia ate o processo terminar; || true pois exit code sera 143 (SIGTERM).
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
            "❌ Pipeline FALHOU (code=${code})" \
            "Re-execute para retomar. Log: artifacts/logs/"
        echo ""
        echo "Ultimas 15 linhas do log:"
        tail -15 "${MASTER_LOG}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Cabecalho ─────────────────────────────────────────────────────────────────
sep
echo "  Pipeline Libras v6  —  $(ts)"
sep
echo "  Projeto    : ${PROJ_DIR}"
echo "  Python     : ${PYTHON}"
echo "  venv       : ${VENV_DIR}"
echo "  Steps HPO  : ${STEPS_HPO} por trial"
echo "  Timeout    : ${HPO_TIMEOUT}s  ($(fmt_elapsed ${HPO_TIMEOUT}))"
echo "  Skip HPO   : ${SKIP_HPO}"
echo "  Reiniciar  : ${REINICIAR}"
echo "  Log        : ${MASTER_LOG}"
echo "  Checkpoints: ${CKPT_DIR}/"
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

# ── Validacao de versao (defesa contra aplicacao parcial) ─────────────────────
# Sintomas marcantes da v6.2 (data-driven dims) em arquivos-chave.
# Se faltar qualquer um, abortar AGORA com mensagem explicita, em vez de
# falhar em 02 com erro confuso de truncamento depois de 100s de tokenizacao.
echo "[$(ts)] Validando versao dos scripts..."
ERRO_VERSAO=0
ARQ_COMMON="${PROJ_DIR}/src/libras/common.py"
ARQ_PREP="${PROJ_DIR}/scripts/01_prepare_data.py"
if ! grep -q '_carregar_dims_iniciais' "${ARQ_COMMON}" 2>/dev/null; then
    echo "  [X] ${ARQ_COMMON}"
    echo "      Falta a funcao _carregar_dims_iniciais (sinal da v6.2)."
    echo "      common.py atual ainda tem MAX_INPUT/MAX_TARGET hardcoded."
    ERRO_VERSAO=1
fi
if ! grep -q 'analise_e_derivar_dims' "${ARQ_PREP}" 2>/dev/null; then
    echo "  [X] ${ARQ_PREP}"
    echo "      Falta a etapa analise_e_derivar_dims (sinal da v6.2)."
    echo "      01_prepare_data atual nao deriva nem persiste dataset_dims.json."
    ERRO_VERSAO=1
fi
if [[ ${ERRO_VERSAO} -ne 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  VERSAO INCOMPLETA — pipeline abortado antes de comecar"
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Os fixes da v6.2 nao estao aplicados nos caminhos certos."
    echo "  Copie os dois arquivos de /mnt/user-data/outputs/ para:"
    echo "    cp .../common.py         ${ARQ_COMMON}"
    echo "    cp .../01_prepare_data.py ${ARQ_PREP}"
    echo "  Depois: bash tools/pipeline_v6.sh --reiniciar"
    echo "──────────────────────────────────────────────────────────────────────"
    exit 2
fi
echo "  [OK] common.py tem _carregar_dims_iniciais"
echo "  [OK] 01_prepare_data.py tem analise_e_derivar_dims"
echo ""

CONFIG="artifacts/configs/config_campeao.json"
T_PIPELINE=$(date +%s)

# ==============================================================================
# Etapa 01: prepare_data
# ==============================================================================
if done_existe "step01_prepare" && [[ "${REINICIAR}" != "true" ]]; then
    # Defesa contra estado inconsistente: .done existe (rodou a v6.1 ou anterior),
    # mas dataset_dims.json nao (a v6.1 nao escrevia esse arquivo). Sem o JSON,
    # common.py cai no fallback 128/64 e 02 vai abortar com truncamento.
    DIMS_JSON="${PROJ_DIR}/data/processed/dataset_dims.json"
    if [[ ! -f "${DIMS_JSON}" ]]; then
        log_etapa "01 — RE-EXECUTAR (checkpoint existe mas dataset_dims.json AUSENTE)"
        echo "  Estado inconsistente detectado:"
        echo "    artifacts/pipeline_state/step01_prepare.done  : presente"
        echo "    data/processed/dataset_dims.json              : AUSENTE"
        echo "  Sintoma de aplicacao parcial do fix v6.2. Re-executando 01..."
        T0=$(date +%s)
        "${PYTHON}" scripts/01_prepare_data.py
        E=$(( $(date +%s) - T0 ))
        marcar_done "step01_prepare"
        echo "[$(ts)] 01 OK em $(fmt_elapsed $E)"
        notificar "normal" "Etapa 01 OK" "prepare_data em $(fmt_elapsed $E) (re-execucao)"
    else
        log_etapa "01 — PULADO (checkpoint + dataset_dims.json presentes)"
        echo "  Para refazer: bash tools/pipeline_v6.sh --reiniciar"
    fi
else
    log_etapa "01 — 01_prepare_data.py"
    T0=$(date +%s)
    "${PYTHON}" scripts/01_prepare_data.py
    E=$(( $(date +%s) - T0 ))
    # Validacao pos-execucao: 01 v6.2 DEVE ter escrito dataset_dims.json.
    # Se nao escreveu, a versao do 01 ainda nao e a v6.2 (validacao inicial
    # ja teria pego, mas defesa em profundidade nunca faz mal).
    if [[ ! -f "${PROJ_DIR}/data/processed/dataset_dims.json" ]]; then
        echo "ERRO: 01 terminou sem escrever data/processed/dataset_dims.json."
        echo "       A versao em scripts/01_prepare_data.py nao e a v6.2."
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
        echo "Checkpoint invalido. Rode: bash tools/pipeline_v6.sh --reiniciar"
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
# Etapa 04: avaliacao + GPS
# ==============================================================================
if done_existe "step04_evaluate" && [[ "${REINICIAR}" != "true" ]]; then
    log_etapa "04 — PULADO (checkpoint existe)"
else
    log_etapa "04 — 04_evaluate.py  (GPS + erros)"
    T0=$(date +%s)
    "${PYTHON}" scripts/04_evaluate.py --save-errors
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
# Resumo final
# ==============================================================================
log_etapa "Gerando RESULTADO_v6.txt"
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

ev   = J("data/reports/evaluate_report.json")
gps  = J("data/reports/gps_recomendacao.json")
hpo  = J("data/reports/hpo_report.json")
prep = J("data/reports/prepare_report.json")
ae   = J("data/reports/analise_erros.json")
hist = J("data/reports/historico_avaliacoes.json")

S1 = "=" * 62
S2 = "─" * 62
L = [S1, "  RESULTADO PIPELINE LIBRAS v6", S1]

if prep:
    s = prep.get("saidas", {})
    L.append("Dataset  : {:>7,} pares treino | {:,} val | {:,} teste  |  {}".format(
        s.get("treino", {}).get("pares", 0),
        s.get("val",    {}).get("pares", 0),
        s.get("teste",  {}).get("pares", 0),
        Path(prep.get("fonte", "")).name))

if hpo:
    m = hpo.get("melhor", {})
    p = m.get("params", {})
    L.append("HPO      : trial #{} | chrF={:.4f} | {}/{} trials OK | fim={}".format(
        m.get("trial"), m.get("chrf", 0),
        hpo.get("trials_ok", 0), hpo.get("trials_totais", 0),
        hpo.get("encerramento", "?")))
    if p:
        L.append("         : lr={:.2e}  batch={}  lora_r={}  alpha={}  sched={}".format(
            p.get("learning_rate", 0), p.get("batch_size"),
            p.get("lora_r"), p.get("lora_alpha"),
            p.get("lr_scheduler_type")))

if ev:
    for d, r in ev.get("por_direcao", {}).items():
        L.append("Eval     : [{:<12}]  chrF={:.2f}  BLEU4={:.2f}  EM={:.2f}%  n={}".format(
            d, r.get("chrf", 0), r.get("bleu4", 0),
            r.get("exact_match", 0), r.get("n", 0)))

if hist and len(hist) >= 2:
    prev, curr = hist[-2], hist[-1]
    delta = curr.get("chrf_medio", 0) - prev.get("chrf_medio", 0)
    L.append("Delta    : {:+.2f} chrF vs run anterior  ({})".format(
        delta, prev.get("timestamp", "?")[:10]))

if ae:
    L.append(S2)
    for d, info in ae.get("por_direcao", {}).items():
        cats = info.get("categorias", {})
        partes = " | ".join("{:}={:.0f}%".format(k, v["pct"]) for k, v in cats.items())
        L.append("Erros    : [{:<12}]  {}".format(d, partes))

if gps:
    L.append(S2)
    L.append("GPS      : {}".format(gps.get("diagnostico", "")))
    L.append("Acao     : {}".format(gps.get("acao_prioritaria", "")))
    L.append("Razao    : {}".format(gps.get("razao", "")))
    alvo = gps.get("vocabulario_alvo", [])
    if alvo:
        top5 = ["{} (tr={}, s={:.2f})".format(
                    v["palavra"], v["n_treino"], v["score_raridade"])
                for v in alvo[:5]]
        L.append("Vocab    : {}".format(", ".join(top5)))
    d = gps.get("delta_chrf")
    if d is not None:
        L.append("DeltaGPS : {:+.2f} chrF vs run anterior".format(d))

L.append(S1)
txt = "\n".join(L) + "\n"
print(txt)

out = Path("RESULTADO_v6.txt")
out.write_text(txt, encoding="utf-8")
print("[Salvo em {}]".format(out.resolve()))
print("[Cole o conteudo de RESULTADO_v6.txt na conversa para continuar]")
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
    "✅ Pipeline v6 CONCLUIDO  ($(fmt_elapsed ${TEMPO_TOTAL}))" \
    "GPS: ${GPS_DIAG} — Veja RESULTADO_v6.txt"

sep
echo "  PIPELINE CONCLUIDO  —  $(ts)"
echo "  Tempo total  : $(fmt_elapsed ${TEMPO_TOTAL})"
sep
echo "  Cole na conversa  : ${PROJ_DIR}/RESULTADO_v6.txt"
echo "  GPS completo      : data/reports/gps_recomendacao.json"
echo "  Log mestre        : ${MASTER_LOG}"
echo "  Monitor HPO       : ${LOG_DIR}/monitor_hpo_*.json"
echo "  Monitor treino    : ${LOG_DIR}/monitor_train_*.json"
echo "  Checkpoints       : ${CKPT_DIR}/"
sep
