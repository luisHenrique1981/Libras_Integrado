#!/usr/bin/env python3
"""
monitor.py — Amostrador leve de GPU/CPU/RAM para diagnostico de gargalo.

Roda EM PARALELO a 02_hpo.py ou 03_train.py (outro terminal). Amostra metricas
a cada N segundos e grava CSV (uma linha por amostra) + um resumo JSON com
estatisticas e um diagnostico heuristico de gargalo.

Objetivo: descobrir se o treino esta GPU-bound (bom), CPU/dataloader-bound,
ou subutilizando a placa por batch pequeno / geracao no eval.

Dependencias: apenas o driver NVIDIA (nvidia-smi). psutil e usado SE presente;
caso contrario cai para /proc (Linux). Nenhuma dependencia obrigatoria extra.

Pontos de verificacao / auditabilidade:
  - Avisa explicitamente se nvidia-smi ou psutil faltarem (nao falha em silencio).
  - Degrada a query do nvidia-smi para um subconjunto garantido se a completa
    falhar (campos de clock/power variam por driver) -> nunca perde as metricas
    nucleares (util, memoria, temp, power.draw).
  - flush() a cada amostra: o CSV sobrevive a um kill -9 do processo monitorado.
  - Captura SIGINT/SIGTERM: ao parar (Ctrl+C), escreve o resumo JSON antes de sair.

Uso:
  python tools/monitor.py --tag train --interval 2
  python tools/monitor.py --tag hpo --interval 2 --duration 600
  # pare com Ctrl+C; o resumo e escrito no encerramento.

Saidas (em artifacts/logs/):
  monitor_<tag>_<ts>.csv    uma linha por amostra
  monitor_<tag>_<ts>.json   resumo estatistico (min/mean/p50/p95/max) + diagnostico

Depois e so subir o .json (e, se quiser, o .csv) nesta conversa para analise.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "logs"

# Campos do nvidia-smi. A ordem casa com NVSMI_FIELDS.
# FULL inclui clocks/power.limit (podem nao existir em todo driver);
# MIN tem apenas os nucleares, universalmente suportados.
NVSMI_QUERY_FULL = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
                    "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem")
NVSMI_QUERY_MIN = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
                   "temperature.gpu,power.draw")
NVSMI_FIELDS = ["gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mb",
                "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w",
                "gpu_power_limit_w", "gpu_sm_clock_mhz", "gpu_mem_clock_mhz"]

CSV_FIELDS = (["t_iso", "t_rel_s"] + NVSMI_FIELDS +
              ["cpu_pct", "cpu_pct_max_core", "ram_used_gb", "ram_total_gb",
               "ram_pct", "swap_used_gb"])

_stop = False
_nvsmi_query = NVSMI_QUERY_FULL  # rebaixado para MIN se a query completa falhar


def _handle_signal(signum: int, frame: Any) -> None:
    global _stop
    _stop = True


def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    if x in ("", "[N/A]", "[Not Supported]", "[Unknown Error]", "N/A"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def ler_gpu() -> list[Optional[float]]:
    """Uma amostra do nvidia-smi alinhada a NVSMI_FIELDS (None onde indisponivel)."""
    global _nvsmi_query
    for _ in range(2):  # 1 tentativa + 1 apos rebaixar a query
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={_nvsmi_query}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True).stdout
            linha = out.strip().splitlines()[0] if out.strip() else ""
            vals: list[Optional[float]] = [_to_float(v) for v in linha.split(",")]
            vals += [None] * (len(NVSMI_FIELDS) - len(vals))  # MIN preenche o resto
            return vals[:len(NVSMI_FIELDS)]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            if _nvsmi_query == NVSMI_QUERY_FULL:
                _nvsmi_query = NVSMI_QUERY_MIN  # rebaixa uma vez e tenta de novo
                continue
            return [None] * len(NVSMI_FIELDS)
    return [None] * len(NVSMI_FIELDS)


# ── CPU/RAM: psutil se possivel, senao /proc ────────────────────────────────
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_prev_cpu: dict[str, float] = {}


def ler_cpu_ram_psutil() -> list[Optional[float]]:
    per = psutil.cpu_percent(percpu=True)  # desde a chamada anterior
    overall = round(sum(per) / len(per), 1) if per else None
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return [overall, round(max(per), 1) if per else None,
            round(vm.used / 1e9, 2), round(vm.total / 1e9, 2),
            round(vm.percent, 1), round(sw.used / 1e9, 2)]


def ler_cpu_ram_proc() -> list[Optional[float]]:
    # RAM via /proc/meminfo
    mem: dict[str, float] = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = ln.partition(":")
        parts = v.strip().split()
        if parts:
            mem[k.strip()] = float(parts[0]) * 1024.0  # kB -> bytes
    total = mem.get("MemTotal", 0.0)
    avail = mem.get("MemAvailable", 0.0)
    used = total - avail
    swap_used = mem.get("SwapTotal", 0.0) - mem.get("SwapFree", 0.0)
    # CPU via delta de /proc/stat entre chamadas
    cpu_pct: Optional[float] = None
    nums = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0.0)
    total_t = sum(nums)
    if _prev_cpu:
        d_idle = idle - _prev_cpu["idle"]
        d_total = total_t - _prev_cpu["total"]
        if d_total > 0:
            cpu_pct = round(100.0 * (1.0 - d_idle / d_total), 1)
    _prev_cpu["idle"], _prev_cpu["total"] = idle, total_t
    return [cpu_pct, None, round(used / 1e9, 2), round(total / 1e9, 2),
            round(100.0 * used / total, 1) if total else None,
            round(swap_used / 1e9, 2)]


def ler_cpu_ram() -> list[Optional[float]]:
    if _HAS_PSUTIL:
        return ler_cpu_ram_psutil()
    try:
        return ler_cpu_ram_proc()
    except Exception:
        return [None] * 6


def resumir(rows: list[dict]) -> dict:
    def stats(col: str) -> Optional[dict]:
        vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
        if not vals:
            return None
        vs = sorted(vals)
        p95 = vs[min(len(vs) - 1, int(0.95 * len(vs)))]
        return {"min": round(min(vals), 1), "mean": round(statistics.mean(vals), 1),
                "p50": round(statistics.median(vals), 1), "p95": round(p95, 1),
                "max": round(max(vals), 1)}

    cols = NVSMI_FIELDS + ["cpu_pct", "cpu_pct_max_core", "ram_used_gb",
                           "ram_pct", "swap_used_gb"]
    res = {c: stats(c) for c in cols}

    diag: list[str] = []
    gu = res.get("gpu_util_pct") or {}
    cpu_core = res.get("cpu_pct_max_core") or {}
    cpu_all = res.get("cpu_pct") or {}
    gmean = gu.get("mean")
    cmax = cpu_core.get("max") if cpu_core else cpu_all.get("max")

    if gmean is None:
        diag.append("Sem dados de GPU (nvidia-smi ausente ou query falhou).")
    elif gmean >= 85:
        diag.append(f"GPU-bound: util media {gmean}% (>=85%). Placa bem usada.")
    elif gmean < 60:
        if cmax is not None and cmax >= 90:
            diag.append(f"Provavel CPU/dataloader-bound: GPU media {gmean}% (<60%) "
                        f"e core de CPU saturado ({cmax}%). Aumente "
                        "dataloader_num_workers / reduza custo de pre-processamento.")
        else:
            diag.append(f"GPU subutilizada ({gmean}% <60%) sem CPU saturado: "
                        "tipico de batch pequeno, geracao autoregressiva no eval "
                        "ou overhead de carga de modelo. Considere padding dinamico, "
                        "group_by_length, eval batch maior e (com re-tuning de LR) "
                        "train batch maior.")
    else:
        diag.append(f"GPU parcialmente usada ({gmean}%, 60-85%). Margem para "
                    "padding dinamico, group_by_length e eval batch maior.")

    pwr = res.get("gpu_power_w") or {}
    plim = res.get("gpu_power_limit_w") or {}
    if pwr.get("mean") and plim.get("max"):
        diag.append(f"Potencia media {pwr['mean']}W de limite {plim['max']}W "
                    f"({round(100 * pwr['mean'] / plim['max'])}%).")
    vram = res.get("gpu_mem_used_mb") or {}
    vramt = res.get("gpu_mem_total_mb") or {}
    if vram.get("max") and vramt.get("max"):
        diag.append(f"VRAM pico {round(vram['max'])}MiB de {round(vramt['max'])}MiB "
                    f"({round(100 * vram['max'] / vramt['max'])}%).")
    return {"estatisticas": res, "diagnostico": diag}


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor leve de GPU/CPU/RAM.")
    p.add_argument("--tag", default="run", help="rotulo do run (ex.: hpo, train)")
    p.add_argument("--interval", type=float, default=2.0, help="segundos entre amostras")
    p.add_argument("--duration", type=float, default=0.0, help="0 = ate Ctrl+C")
    p.add_argument("--out-dir", default=str(LOGS))
    a = p.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if shutil.which("nvidia-smi") is None:
        print("[monitor] AVISO: nvidia-smi nao encontrado; GPU ficara vazia.",
              file=sys.stderr)
    if not _HAS_PSUTIL:
        print("[monitor] AVISO: psutil ausente; usando /proc "
              "(cpu_pct_max_core indisponivel).", file=sys.stderr)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"monitor_{a.tag}_{ts}.csv"
    json_path = out_dir / f"monitor_{a.tag}_{ts}.json"

    print(f"[monitor] tag={a.tag} interval={a.interval}s "
          f"duration={'inf' if a.duration <= 0 else a.duration}")
    print(f"[monitor] CSV : {csv_path}")
    print(f"[monitor] JSON: {json_path}  (escrito ao parar)")

    # priming: a 1a chamada de cpu_percent retorna 0 (sem janela de referencia)
    if _HAS_PSUTIL:
        psutil.cpu_percent(percpu=True)
    else:
        try:
            ler_cpu_ram_proc()
        except Exception:
            pass
    time.sleep(0.2)

    rows: list[dict] = []
    t0 = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        n = 0
        while not _stop:
            t = time.time()
            row_vals = ([datetime.now().isoformat(), round(t - t0, 1)]
                        + ler_gpu() + ler_cpu_ram())
            w.writerow(row_vals)
            f.flush()  # durabilidade: dado salvo mesmo se o processo for morto
            rows.append(dict(zip(CSV_FIELDS, row_vals)))
            n += 1
            if n % 30 == 0:
                print(f"[monitor] {n} amostras ({round(t - t0)}s decorridos)")
            if a.duration > 0 and (t - t0) >= a.duration:
                break
            # sleep que respeita o sinal de parada
            slept = 0.0
            while slept < a.interval and not _stop:
                time.sleep(min(0.25, a.interval - slept))
                slept += 0.25

    resumo = {
        "tag": a.tag, "timestamp": ts, "n_amostras": len(rows),
        "interval_s": a.interval, "duracao_s": round(time.time() - t0, 1),
        "tem_psutil": _HAS_PSUTIL, "nvsmi_query_usada": _nvsmi_query,
        **resumir(rows),
    }
    json_path.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[monitor] {len(rows)} amostras gravadas em {csv_path}")
    print(f"[monitor] resumo em {json_path}")
    for d in resumo.get("diagnostico", []):
        print(f"[monitor] >> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
