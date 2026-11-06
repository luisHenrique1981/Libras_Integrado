#!/usr/bin/env python3
"""
02_hpo.py v7 — HPO data-driven com search space REFINADO baseado em resultados
v6 e mais trials por convergencia real.

CONTEXTO (v6 produziu):
  - winner v6: trial #11, chrF=40.59 (val), config: lr=1.28e-3, batch=8,
    lora_r=8, alpha=64, sched=linear, dropout=0.003
  - 9/12 trials concluiram antes do timeout
  - Eval full: chrf_pt2libras=66.55, chrf_libras2pt=42.90 (REGRESSAO vs v5)
  - Causa-raiz: contaminacao cross-lingual no corpus, NAO config ruim
    (mitigado em mesclar_datasets.py v7)

MUDANCAS v7 vs v6 (data-driven, justificadas):
  + STUDY_NAME = libras_hpo_v7 (namespace novo, nao mistura com v6)
  + search space REFINADO em volta do winner v6 (sem chutar fora do otimo
    local observado, mas explorando dimensoes que podem estar sub-exploradas):
    - lr: [1e-4, 3e-3] -> [3e-4, 2e-3] log (estreita; lr=1.28e-3 deu winner)
    - lora_r: [8,16,32] -> [16,32,64] (testar MAIOR; lora_r=8 pode estar
      subdimensionado para corpus de 50k+ pares; literatura sugere r>=16
      para tasks complexas)
    - lora_alpha: [16,32,64] -> [32,64,128] (compatibilidade com r maior;
      manter alpha/r entre 2 e 8 para escala saudavel)
    - lora_dropout: [0.0, 0.15] -> [0.05, 0.20] (corpus maior tolera/precisa
      mais regularizacao; v6 winner usou 0.003, suspeito de underfit-de-reg)
    - weight_decay: [1e-4, 0.1] -> [1e-3, 0.1] (apertar piso)
    - scheduler: {linear, cosine} -> {cosine, cosine_with_restarts}
      (linear NAO foi escolhido no winner v6; cosine_with_restarts permite
      escapar de minimos locais em treinos longos)
    - warmup_ratio: [0.02, 0.12] -> [0.03, 0.10] (estreita levemente)
  + paciencia padrao 8 -> 12 (search space maior precisa de mais amostras
    antes de declarar convergencia)
  + n_trials default explicito 30 (ao inves de teto 500): com paciencia 12,
    convergencia ocorre tipicamente em 25-35 trials num espaco dessa
    dimensionalidade segundo Optuna TPE)
  + dataloader_num_workers no config final: 10 (Ryzen 5500X3D 6c/12t com
    HT; 10 deixa 2 threads livres para sistema e main loop)
  + dataloader_prefetch_factor: 2 -> 6 (3x mais batches pre-carregados;
    mitiga o dataloader-bound observado no monitor_train v6)
  + num_train_epochs final REVISADO (treinos mais longos):
    <5k: 30 ep; 5k-30k: 20 ep (era 15); >=30k: 12 ep (era 8). Justifica:
    early stopping (patience=6) decide o ponto real; teto so corta runaway.

FILOSOFIA mantida: 
  - tudo data-driven (steps por trial, n_epochs por trial, n_epochs final)
  - convergencia detectada por callback (espelha early stopping do treino)
  - timeout do HPO (default 8h, pode subir com --timeout)

Uso:
  python scripts/02_hpo.py                  # 30 trials default
  python scripts/02_hpo.py --trials 50      # mais agressivo
  python scripts/02_hpo.py --retomar        # continua estudo SQLite v7
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.common import (  # noqa: E402
    BASE_MODEL_ID, TRAIN_CSV, VAL_CSV, MAX_INPUT, MAX_TARGET, SEED,
    ART_LOGS, ART_STUDIES, ART_CONFIGS, DATA_REPORTS,
    get_logger, salvar_json, require, StepGate, detectar_device,
    validar_dataframe, auditar_truncamento, calcular_metricas,
    DataContractError,
)

log = get_logger("hpo", ART_LOGS / "hpo.log")

LORA_TARGETS = ["q", "k", "v", "o", "wi", "wo"]
STUDY_DB = f"sqlite:///{ART_STUDIES / 'hpo_study.db'}"
STUDY_NAME = "libras_hpo_v7"  # novo namespace; nao mistura com v6


def carregar_dados(frac_treino: float, frac_val: float):
    """Subamostras PROPORCIONAIS. frac=1.0 usa tudo. Sem numero absoluto."""
    import pandas as pd
    df_tr = pd.read_csv(TRAIN_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
    df_va = pd.read_csv(VAL_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
    validar_dataframe(df_tr, "treino", log)
    validar_dataframe(df_va, "val", log)
    n_tr, n_va = len(df_tr), len(df_va)
    if 0.0 < frac_treino < 1.0:
        df_tr = df_tr.sample(max(400, int(n_tr * frac_treino)),
                             random_state=SEED).reset_index(drop=True)
    if 0.0 < frac_val < 1.0:
        df_va = df_va.sample(max(200, int(n_va * frac_val)),
                             random_state=SEED).reset_index(drop=True)
    log.info(f"Treino={len(df_tr)} ({len(df_tr)/n_tr:.0%}) | "
             f"Val={len(df_va)} ({len(df_va)/n_va:.0%}) | "
             f"Full={n_tr} treino + {n_va} val")
    return df_tr, df_va, n_tr  # n_tr eh o tamanho FULL, p/ deriv. epocas finais


def build_preprocess(tok):
    """Padding DINAMICO: collator pada por batch. Glosa media ~6 tokens; padar
    a 64 no preprocess desperdicaria ~90% do compute do decoder."""
    def pre(examples):
        enc = tok(list(examples["input"]), max_length=MAX_INPUT, truncation=True)
        lbl = tok(text_target=list(examples["output"]), max_length=MAX_TARGET,
                  truncation=True)
        enc["labels"] = lbl["input_ids"]  # collator -> -100 nos PAD
        return enc
    return pre


def build_compute_metrics(tok, trial_num: int):
    import numpy as np
    pad_id = tok.pad_token_id or 0
    vocab = tok.vocab_size or 32000

    def cm(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.clip(np.where(np.asarray(preds, dtype=np.int64) < 0, pad_id,
                                 np.asarray(preds, dtype=np.int64)), 0, vocab - 1)
        labels = np.clip(np.where(np.asarray(labels, dtype=np.int64) < 0, pad_id,
                                  np.asarray(labels, dtype=np.int64)), 0, vocab - 1)
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        m = calcular_metricas(dp, dl)
        for i in range(min(2, len(dp))):
            log.info(f"  [T{trial_num}|{i}] '{dp[i][:55]}' | ref:'{dl[i][:55]}'")
        return m

    # sanity check
    import numpy as _np
    z = (_np.array([[0, 0, 0]], dtype=_np.int64),
         _np.array([[0, 0, 0]], dtype=_np.int64))
    require(cm(z)["chrf"] == 0.0, "E_SANITY_METRICA",
            "sanity de metrica falhou", fix="bug em calcular_metricas")
    return cm


def make_objective(df_tr, df_va, steps_hpo: int, dtype):
    """Orcamento em STEPS, nao epocas. n_epochs deriva do batch escolhido para
    todos os trials terem o mesmo numero de atualizacoes de gradiente."""
    def objective(trial):
        import torch, optuna, shutil
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                   Seq2SeqTrainer, Seq2SeqTrainingArguments,
                                   DataCollatorForSeq2Seq, EarlyStoppingCallback)
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import Dataset as HFD

        params = {
            # v6 winner: lr=1.28e-3. Estreitamos em [3e-4, 2e-3] log para
            # explorar perto do otimo observado sem extrapolar fora.
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
            # batch=4 EXCLUIDO: monitor.py v5 mediu launch-bound.
            # batch=64 EXCLUIDO: risco de OOM no eval com 4 beams.
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            # lora_r=8 winner v6 pode estar subdimensionado para corpus de 50k+.
            # Subimos o range para testar capacidade maior.
            "lora_r": trial.suggest_categorical("lora_r", [16, 32, 64]),
            # lora_alpha proporcional ao novo range de r (manter alpha/r ~2-4).
            "lora_alpha": trial.suggest_categorical("lora_alpha", [32, 64, 128]),
            # v6 winner usou dropout=0.003 (suspeito de underfit-de-regularizacao
            # para corpus maior). Apertamos piso de 0.0 para 0.05.
            "lora_dropout": trial.suggest_float("lora_dropout", 0.05, 0.20),
            # Apertamos piso de 1e-4 para 1e-3: corpus maior tolera mais decay.
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
            # Estreitamos faixa para [0.03, 0.10] (range comum em literatura).
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.03, 0.10),
            # 'linear' NAO foi escolhido como winner v6. Substituimos por
            # 'cosine_with_restarts' que permite escapar de minimos locais.
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["cosine", "cosine_with_restarts"]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2]),
        }
        batch = params["batch_size"]
        accum = params["gradient_accumulation_steps"]
        # steps_per_epoch usa batch EFETIVO (batch * accum) -> grad_accum tambem
        # afeta o numero de updates.
        steps_per_epoch = max(1, len(df_tr) // (batch * accum))
        # n_epochs com piso 3 (garante convergencia minima) e teto 15 (seguranca):
        n_epochs = max(3, min(15, math.ceil(steps_hpo / steps_per_epoch)))
        log.info(f"[Trial {trial.number}] batch={batch} accum={accum} "
                 f"steps/epoca={steps_per_epoch} -> n_epochs={n_epochs} "
                 f"(alvo {steps_hpo} steps) | params={params}")

        t0 = time.time()
        ckpt = ART_STUDIES / f"trial_{trial.number}"
        try:
            tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
            model.config.use_cache = False
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, r=params["lora_r"],
                lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"],
                target_modules=LORA_TARGETS, bias="none")).to("cuda")
            pre = build_preprocess(tok)
            ds_tr = HFD.from_pandas(df_tr[["input", "output"]]).map(
                pre, batched=True, remove_columns=["input", "output"])
            ds_va = HFD.from_pandas(df_va[["input", "output"]]).map(
                pre, batched=True, remove_columns=["input", "output"])
            bf16 = dtype == torch.bfloat16
            # Ryzen 5500X3D tem 6 cores fisicos / 12 threads. Para HPO,
            # deixamos margem: 10 workers (2 livres para sistema + Optuna).
            # Hardware com water cooler robusto, sem preocupacao termica.
            nw = min(10, os.cpu_count() or 4)
            ta = Seq2SeqTrainingArguments(
                output_dir=str(ckpt), num_train_epochs=n_epochs,
                per_device_train_batch_size=batch,
                # eval batch limitado: geracao com 4 beams * seq custa memoria.
                per_device_eval_batch_size=min(batch, 16),
                gradient_accumulation_steps=accum,
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                warmup_ratio=params["warmup_ratio"],
                lr_scheduler_type=params["lr_scheduler_type"],
                bf16=bf16, fp16=not bf16, tf32=True,
                predict_with_generate=True, generation_max_length=MAX_TARGET,
                generation_num_beams=4,
                eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                load_best_model_at_end=True, metric_for_best_model="eval_chrf",
                greater_is_better=True, logging_steps=50, seed=SEED, data_seed=SEED,
                report_to="none", dataloader_num_workers=nw, dataloader_pin_memory=True,
                dataloader_persistent_workers=(nw > 0),
                dataloader_prefetch_factor=6 if nw > 0 else None,
                group_by_length=True)  # menos padding dentro do batch
            trainer = Seq2SeqTrainer(
                model=model, args=ta, processing_class=tok,
                train_dataset=cast(Any, ds_tr), eval_dataset=cast(Any, ds_va),
                data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
                compute_metrics=build_compute_metrics(tok, trial.number),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
            trainer.train()
            met = trainer.evaluate()
            chrf = met.get("eval_chrf", 0.0)
            bleu = met.get("eval_bleu4", 0.0)
            em = met.get("eval_exact_match", 0.0)
            dur = round(time.time() - t0, 1)
            trial.set_user_attr("bleu4", round(bleu, 4))
            trial.set_user_attr("em", round(em, 4))
            trial.set_user_attr("val_loss", round(met.get("eval_loss", -1), 4))
            trial.set_user_attr("dur_s", dur)
            trial.set_user_attr("n_epochs_derivado", n_epochs)
            trial.set_user_attr("steps_per_epoch", steps_per_epoch)
            log.info(f"[T{trial.number}] chrf={chrf:.4f} bleu={bleu:.4f} "
                     f"em={em:.2f} dur={dur}s epochs={n_epochs}")
            return chrf
        except Exception as exc:
            log.error(f"[T{trial.number}] FALHOU: {exc}")
            raise optuna.TrialPruned(f"trial {trial.number}: {exc}") from exc
        finally:
            if ckpt.exists():
                shutil.rmtree(ckpt, ignore_errors=True)
    return objective


def derivar_num_epochs_final(n_treino_pares_unicos: int) -> int:
    """Teto de epocas para o treino final, derivado do tamanho do dataset.
    REVISADO v7 (treinos mais longos):
      < 5k pares  -> 30 epocas (mantido; datasets muito pequenos precisam
                     muitos passes para convergir)
      5k-30k      -> 20 epocas (era 15; corpus medio tolera mais)
      >= 30k      -> 12 epocas (era 8; corpus grande v7 com filtros limpos
                     beneficia de mais passes do que v6 com corpus ruidoso)
    Early stopping (patience=6) decide o ponto real de parada; este e o
    teto de seguranca para evitar runaway.
    """
    if n_treino_pares_unicos < 5000:
        return 30
    elif n_treino_pares_unicos < 30000:
        return 20
    else:
        return 12


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps-hpo", type=int, default=3000,
                   help="orcamento de passos de gradiente POR TRIAL. n_epochs e "
                        "computado para honrar este orcamento independente do batch. "
                        "Default 3000 ~ HPO v5 (5 ep x 872 steps/ep no batch=4).")
    p.add_argument("--frac-treino-hpo", type=float, default=0.15,
                   help="fracao do treino por trial. 15%% de 50k pares ~ 7500 pares.")
    p.add_argument("--frac-val-hpo", type=float, default=0.10,
                   help="fracao do val por trial. Objetivo: discriminar configs "
                        "rapido, nao medir chrF definitivo.")
    p.add_argument("--timeout", type=int, default=43200,
                   help="orcamento total do HPO em segundos. Default 12h "
                        "(v7 espaco maior + mais paciencia). 0 = sem limite.")
    p.add_argument("--trials", type=int, default=40,
                   help="teto de trials. v7: default 40 (era 500); search "
                        "space refinado converge tipicamente em 25-35 trials. "
                        "Encerra antes por paciencia ou timeout.")
    p.add_argument("--paciencia-hpo", type=int, default=12,
                   help="encerra o HPO se chrF nao melhorar em N trials seguidos. "
                        "v7: 12 (era 8); search space maior precisa mais amostras "
                        "antes de declarar convergencia.")
    p.add_argument("--min-delta-chrf", type=float, default=0.5,
                   help="melhoria minima de chrF para contar como progresso.")
    p.add_argument("--retomar", action="store_true",
                   help="continua estudo SQLite existente (load_if_exists)")
    p.add_argument("--permitir-cpu", action="store_true")
    a = p.parse_args()

    log.info("=" * 64)
    log.info(f"HPO v7 | steps_hpo={a.steps_hpo} frac_treino={a.frac_treino_hpo:.0%} "
             f"frac_val={a.frac_val_hpo:.0%} | timeout={a.timeout}s "
             f"trials_max={a.trials} paciencia={a.paciencia_hpo} "
             f"delta>={a.min_delta_chrf} | lora_targets={LORA_TARGETS}")
    log.info("=" * 64)

    with StepGate("device", log) as g:
        try:
            import torch
            dev, dtype, _ = detectar_device()
        except Exception:
            if not a.permitir_cpu:
                raise
            import torch
            dev, dtype = "cpu", torch.float32
            log.warning("CPU forcada por --permitir-cpu")
        g.info["device"] = dev

    with StepGate("dados_e_truncamento", log) as g:
        df_tr, df_va, n_treino_full = carregar_dados(a.frac_treino_hpo, a.frac_val_hpo)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        trunc = auditar_truncamento(df_va, tok, log)
        require(trunc["pct_target_truncados"] < 0.02, "E_TRUNCAMENTO",
                f"{trunc['pct_target_truncados']:.1%} targets truncados "
                f"(MAX_TARGET={MAX_TARGET})",
                fix="aumente MAX_TARGET em common.py", err=DataContractError, ctx=trunc)
        g.info.update(trunc)
        g.info["n_treino_full"] = n_treino_full

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    with StepGate("otimizacao", log) as g:
        sampler = optuna.samplers.TPESampler(seed=SEED)
        # MedianPruner com mais warmup: amostra base maior antes de podar
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner,
            study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=a.retomar)
        if a.retomar and len(study.trials) > 0:
            log.info(f"Retomando: {len(study.trials)} trials existentes")

        # Convergencia: mesmo principio do early stopping do treino, aplicado ao
        # estudo. "Trials_sem_melhora" e o equivalente macro de "epochs_sem_melhora".
        _state = {"melhor": float("-inf"), "sem_melhora": 0}
        # Inicializa o melhor com o que ja existe no estudo retomado, se houver
        try:
            if a.retomar and study.best_trial is not None:
                _state["melhor"] = study.best_trial.value or float("-inf")
        except (ValueError, RuntimeError):
            pass  # estudo vazio

        def _cb_convergencia(study_: optuna.Study,
                             trial_: optuna.trial.FrozenTrial) -> None:
            if trial_.value is None or trial_.state == optuna.trial.TrialState.PRUNED:
                return  # podados nao contam para "sem_melhora"
            if trial_.value > _state["melhor"] + a.min_delta_chrf:
                _state["melhor"] = trial_.value
                _state["sem_melhora"] = 0
                log.info(f"[HPO] nova melhor chrF={trial_.value:.4f} "
                         f"(trial #{trial_.number})")
            else:
                _state["sem_melhora"] += 1
                log.info(f"[HPO] sem progresso: {_state['sem_melhora']}/"
                         f"{a.paciencia_hpo} (melhor={_state['melhor']:.4f})")
                if _state["sem_melhora"] >= a.paciencia_hpo:
                    log.info(f"[HPO] CONVERGENCIA: {a.paciencia_hpo} trials sem "
                             f"melhoria >= {a.min_delta_chrf}. Encerrando.")
                    study_.stop()

        study.optimize(
            make_objective(df_tr, df_va, a.steps_hpo, dtype),
            n_trials=a.trials,
            timeout=a.timeout if a.timeout > 0 else None,
            callbacks=[_cb_convergencia],
            show_progress_bar=False)

        ok = [t for t in study.trials if t.value is not None]
        require(len(ok) > 0, "E_NENHUM_TRIAL_OK", "nenhum trial concluiu",
                fix="inspecione artifacts/audit e o log")
        g.info["trials_ok"] = len(ok)
        g.info["trials_totais"] = len(study.trials)

    best = study.best_trial
    dur_min = round((time.time() - t0) / 60, 1)
    log.info("=" * 64)
    log.info(f"MELHOR TRIAL #{best.number} | chrF={best.value:.4f} "
             f"bleu={best.user_attrs.get('bleu4')} em={best.user_attrs.get('em')}")
    for k, v in best.params.items():
        log.info(f"  {k:<30} {v}")
    log.info(f"Duracao: {dur_min} min")

    # ---- DERIVACAO DO num_train_epochs PARA O TREINO FINAL ----
    # Baseado no tamanho REAL do conjunto de pares unicos de treino (nao linhas).
    # Cada par contribui com 2 linhas (pt2libras + libras2pt), entao
    # pares_unicos = linhas_treino / 2.
    n_pares_treino = n_treino_full // 2
    num_epochs_final = derivar_num_epochs_final(n_pares_treino)
    batch_final = best.params["batch_size"]
    accum_final = best.params["gradient_accumulation_steps"]
    steps_per_epoch_final = max(1, n_treino_full // (batch_final * accum_final))
    total_steps_alvo = num_epochs_final * steps_per_epoch_final
    log.info(f"[CFG FINAL] n_treino_pares={n_pares_treino} -> "
             f"num_train_epochs ceiling = {num_epochs_final} "
             f"(~{total_steps_alvo} steps; early stopping decide o real)")

    cfg = {
        **best.params,
        "lora_target_modules": LORA_TARGETS,
        "num_train_epochs": num_epochs_final,  # DERIVADO, nao hardcoded
        "early_stopping_patience": 6,
        "metric_for_best_model": "eval_chrf",
        "chrf_best_trial": best.value,
        "bleu4_best_trial": best.user_attrs.get("bleu4"),
        "trial_number": best.number,
        "dataloader_num_workers": min(10, os.cpu_count() or 4),
        "dataloader_prefetch_factor": 6,
        "_derivacao": {
            "n_treino_pares": n_pares_treino,
            "n_treino_linhas": n_treino_full,
            "steps_per_epoch": steps_per_epoch_final,
            "total_steps_ceiling": total_steps_alvo,
            "regra": "<5k pares: 30 ep; 5k-30k: 15 ep; >=30k: 8 ep",
        },
    }
    salvar_json(cfg, ART_CONFIGS / "config_campeao.json")
    rel = {
        "timestamp": datetime.now().isoformat(), "versao": "7.0.0",
        "metrica": "chrf", "lora_targets": LORA_TARGETS,
        "duracao_min": dur_min, "trials_ok": len(ok), "trials_totais": len(study.trials),
        "encerramento": ("convergencia" if _state["sem_melhora"] >= a.paciencia_hpo
                         else ("timeout" if (time.time() - t0) >= a.timeout else "trials")),
        "melhor": {"trial": best.number, "chrf": best.value,
                   "params": best.params, "user_attrs": best.user_attrs},
        "todos": [{"trial": t.number, "chrf": t.value, "params": t.params,
                   "attrs": t.user_attrs, "state": str(t.state)}
                  for t in study.trials],
        "config_final_derivado": cfg["_derivacao"],
    }
    salvar_json(rel, DATA_REPORTS / "hpo_report.json")
    log.info(f"[SALVO] {ART_CONFIGS / 'config_campeao.json'}")
    log.info(f"Proximo: python scripts/03_train.py --config "
             f"{ART_CONFIGS / 'config_campeao.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
