#!/usr/bin/env python3
"""
03_train.py v7 — Treino definitivo com otimizacao de dataloader.

FILOSOFIA: este script e o CONSUMIDOR do que o 01 e o 02 derivaram. Todos os
hiperparametros vem do config_campeao.json (escrito pelo 02_hpo v7). Aqui nao
se chuta nada.

MUDANCAS v7 vs v6 (focadas em throughput; correcao ja vinha bem):
  + PRE-TOKENIZACAO PARALELA: dataset HuggingFace .map() com num_proc=N para
    paralelizar a tokenizacao ANTES do treino. Mitiga o dataloader-bound que
    o monitor v6 detectou (1 core saturado 100%, GPU 19.6%). Tokenizacao agora
    e CPU-bound em N processos, nao serial em 1 thread.
  + prefetch_factor: 2 -> 6 (padrao HF e 2; v7 sobe para 6 = mais batches
    prontos na fila quando GPU pede). Custa um pouco mais de RAM (folga em
    32GB) em troca de menos starvation de GPU.
  + dataloader_num_workers consumido do config (escrito pelo 02_hpo v7 = 10
    em Ryzen 5500X3D 6c/12t). Antes era min(8, cpu_count) = 8.
  + Caminho de --override-epochs mantido como escape hatch.

MEDIDAS QUE PRECISO COMPARAR pos-run:
  - GPU util medio deveria SUBIR de 19.6% para 40%+ (alvo: 60%+)
  - cpu_pct_max_core: continuar alto e ESPERADO agora (estamos pagando
    custo de mais workers; e o que queremos)
  - Tempo total: idealmente cai 25-40% para o mesmo numero de epocas

VS v5/v6:
  + --out-dir default agora libras_v7
  + eval_batch_size DERIVADO de batch_size (min(batch, 16))
"""
from __future__ import annotations
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libras.common import (  # noqa: E402
    BASE_MODEL_ID, TRAIN_CSV, VAL_CSV, TEST_CSV, MAX_INPUT, MAX_TARGET, SEED,
    ART_LOGS, ART_MODELS, DATA_REPORTS, DIR_PT2LIBRAS, DIR_LIBRAS2PT,
    get_logger, salvar_json, carregar_json, require,
    StepGate, StepError, detectar_device,
    validar_dataframe, auditar_truncamento, calcular_metricas, montar_input,
    DataContractError, EnvironmentError_,
)

log = get_logger("train", ART_LOGS / "train.log")

# Smoke: sanity-check de prefixo + tokens especiais. Com o corpus v6 (glosa mais
# rica, ex. [NAO_VER], GRANDE(+)) algumas dessas frases podem nao bater 100% —
# nao bloqueia. O sinal real de qualidade e o eval do hold-out logo abaixo.
SMOKE = [
    ("Eu preciso de ajuda.", DIR_PT2LIBRAS, "EU PRECISAR AJUDA [PONTO]"),
    ("Onde fica o banheiro?", DIR_PT2LIBRAS, "BANHEIRO ONDE [INTERROGACAO]"),
    ("EU PRECISAR AJUDA [PONTO]", DIR_LIBRAS2PT, "Eu preciso de ajuda."),
    ("BANHEIRO ONDE [INTERROGACAO]", DIR_LIBRAS2PT, "Onde fica o banheiro?"),
]


def build_preprocess(tok):
    """Padding DINAMICO — collator pada por batch. Glosa media ~6 tokens; padar
    a 64 desperdicaria ~90% do compute do decoder."""
    def pre(examples):
        enc = tok(list(examples["input"]), max_length=MAX_INPUT, truncation=True)
        lbl = tok(text_target=list(examples["output"]), max_length=MAX_TARGET,
                  truncation=True)
        enc["labels"] = lbl["input_ids"]  # collator converte PAD -> -100
        return enc
    return pre


def build_compute_metrics(tok):
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
        for i in range(min(3, len(dp))):
            log.info(f"  [pred {i}] '{dp[i][:60]}' | ref:'{dl[i][:60]}'")
        return m
    return cm


def make_audit_callback(tok, ds_va, max_samples=20):
    from transformers import TrainerCallback

    class AuditCB(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            model = kwargs.get("model")
            if model is None:
                return
            try:
                import torch
                epoch = round(state.epoch or 0.0, 2)
                sub = ds_va.select(range(min(max_samples, len(ds_va))))
                batch = tok.pad({"input_ids": sub["input_ids"]},
                                return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**batch, max_new_tokens=MAX_TARGET,
                                          num_beams=4)
                preds = tok.batch_decode(out, skip_special_tokens=True)
                pad_id = tok.pad_token_id or 0
                clean = [[t if t != -100 else pad_id for t in row]
                         for row in sub["labels"]]
                refs = tok.batch_decode(clean, skip_special_tokens=True)
                salvar_json(
                    {"epoch": epoch, "metrics": metrics or {},
                     "amostras": [{"pred": p, "ref": r}
                                  for p, r in zip(preds, refs)]},
                    ART_LOGS / f"audit_epoch_{str(epoch).replace('.', '_')}.json")
            except Exception as e:
                log.warning(f"AuditCB falhou: {e}")
    return AuditCB()


def smoke_test(model, tok, device) -> list[dict]:
    import torch
    log.info("-" * 60)
    log.info("SMOKE TEST (sanity-check de prefixo; nao bloqueia)")
    res, acertos = [], 0
    model.eval()
    for fonte, direcao, esperado in SMOKE:
        entrada = montar_input(fonte, direcao)
        ids = tok(entrada, return_tensors="pt", max_length=MAX_INPUT,
                  truncation=True).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_TARGET, num_beams=4)
        obtido = tok.decode(out[0], skip_special_tokens=True).strip()
        ok = obtido.upper().strip() == esperado.upper().strip()
        acertos += int(ok)
        log.info(f"  {'OK' if ok else 'X '} [{direcao}] {fonte!r}")
        log.info(f"     esperado: {esperado}")
        log.info(f"     obtido  : {obtido}")
        res.append({"entrada": entrada, "esperado": esperado,
                    "obtido": obtido, "ok": ok})
    log.info(f"  Smoke: {acertos}/{len(SMOKE)} (informativo)")
    log.info("-" * 60)
    return res


def avaliar_por_direcao_batched(model, tok, df, device, batch_size: int = 32) -> dict:
    """Eval em batches, nao 1-a-1. Reduz tempo do hold-out de ~3h para ~3min."""
    import torch
    out = {}
    for direcao in sorted(df["direcao"].unique()):
        sub = df[df["direcao"] == direcao].reset_index(drop=True)
        inputs = [str(x) for x in sub["input"]]
        refs = [str(x).strip() for x in sub["output"]]
        preds: list[str] = []
        for i in range(0, len(inputs), batch_size):
            lote = inputs[i:i + batch_size]
            try:
                enc = tok(lote, return_tensors="pt", padding=True,
                          max_length=MAX_INPUT, truncation=True).to(device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=MAX_TARGET,
                                        num_beams=4)
                preds.extend(tok.batch_decode(g, skip_special_tokens=True))
            except torch.cuda.OutOfMemoryError:
                # fallback 1-a-1: nao perde o lote num OOM esporadico
                torch.cuda.empty_cache()
                for txt in lote:
                    try:
                        e1 = tok(txt, return_tensors="pt", max_length=MAX_INPUT,
                                 truncation=True).to(device)
                        with torch.no_grad():
                            o1 = model.generate(**e1, max_new_tokens=MAX_TARGET,
                                                 num_beams=4)
                        preds.append(tok.decode(o1[0], skip_special_tokens=True))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        preds.append("[OOM]")
        preds = [p.strip() for p in preds]
        m = calcular_metricas(preds, refs)
        m["n"] = len(sub)
        out[direcao] = m
        log.info(f"  [{direcao}] chrF={m['chrf']:.2f} BLEU4={m['bleu4']:.2f} "
                 f"EM={m['exact_match']:.2f}% (n={len(sub)})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="config_campeao.json escrito pelo 02_hpo")
    p.add_argument("--override-epochs", type=int, default=None,
                   help="ESCAPE HATCH: forca num_train_epochs ignorando o config. "
                        "Uso normal: deixe o config decidir.")
    p.add_argument("--out-dir", default=str(ART_MODELS / "libras_v7"),
                   help="default: artifacts/models/libras_v7")
    p.add_argument("--permitir-cpu", action="store_true")
    a = p.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                               Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               DataCollatorForSeq2Seq, EarlyStoppingCallback)
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from datasets import Dataset as HFD
    import pandas as pd

    out_dir = Path(a.out_dir)

    with StepGate("device", log) as g:
        try:
            device, dtype, grad_ckpt = detectar_device()
        except EnvironmentError_:
            if not a.permitir_cpu:
                raise
            device, dtype, grad_ckpt = "cpu", torch.float32, False
            log.warning("CPU forcada por --permitir-cpu")
        g.info.update({"device": device, "bf16": dtype == torch.bfloat16,
                       "grad_ckpt": grad_ckpt})

    with StepGate("config", log) as g:
        cfg = carregar_json(Path(a.config))
        if a.override_epochs is not None:
            log.warning(f"[OVERRIDE] num_train_epochs: {cfg.get('num_train_epochs')} "
                        f"-> {a.override_epochs} (manual via CLI)")
            cfg["num_train_epochs"] = a.override_epochs
        targets = cfg.get("lora_target_modules", ["q", "k", "v", "o", "wi", "wo"])
        require("num_train_epochs" in cfg, "E_CFG_INCOMPLETO",
                "config sem num_train_epochs",
                fix="re-rode 02_hpo.py para gerar config valido",
                err=DataContractError)
        for k, v in cfg.items():
            if not k.startswith("_"):
                log.info(f"  {k:<30} {v}")
        # Mostra o bloco de derivacao do 02_hpo se presente
        if "_derivacao" in cfg:
            log.info(f"  [DERIVACAO] {cfg['_derivacao']}")
        g.info["lora_targets"] = targets

    with StepGate("dados", log) as g:
        df_tr = pd.read_csv(TRAIN_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
        df_va = pd.read_csv(VAL_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
        validar_dataframe(df_tr, "treino", log)
        validar_dataframe(df_va, "val", log)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        trunc = auditar_truncamento(df_tr, tok, log)
        require(trunc["pct_target_truncados"] < 0.02, "E_TRUNCAMENTO",
                f"{trunc['pct_target_truncados']:.1%} targets truncados",
                fix="aumente MAX_TARGET em common.py ou limpe outliers",
                err=DataContractError, ctx=trunc)
        g.info.update({"n_treino": len(df_tr), "n_val": len(df_va), **trunc})

    with StepGate("modelo_lora", log) as g:
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.1)),
            target_modules=targets, bias="none")).to(device)
        treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"  Treinaveis: {treinaveis:,} / {total:,} "
                 f"({100*treinaveis/total:.3f}%)")
        g.info["pct_treinaveis"] = round(100 * treinaveis / total, 4)

    ds_tr = ds_va = None
    with StepGate("tokenizar", log) as g:
        pre = build_preprocess(tok)
        # v7: PRE-TOKENIZACAO PARALELA. v6 fazia .map(batched=True) com num_proc=1
        # (default), e o monitor mostrou 1 core saturado 100% durante o treino.
        # Aqui paralelizamos tokenizacao em N processos. Ryzen 5500X3D: 6c/12t.
        # Deixamos 4 threads para o trainer (gradients, GPU launch, eval).
        n_proc = min(8, max(2, (os.cpu_count() or 4) - 4))
        log.info(f"  pre-tokenizando em {n_proc} processos paralelos")
        ds_tr = HFD.from_pandas(df_tr[["input", "output"]]).map(
            pre, batched=True, remove_columns=["input", "output"],
            num_proc=n_proc, desc="tokenizar_treino")
        ds_va = HFD.from_pandas(df_va[["input", "output"]]).map(
            pre, batched=True, remove_columns=["input", "output"],
            num_proc=n_proc, desc="tokenizar_val")
        g.info.update({"n_proc_tokenizar": n_proc,
                       "ds_tr_size": len(ds_tr), "ds_va_size": len(ds_va)})
    if ds_tr is None or ds_va is None:
        raise StepError(code="E_TOKENIZE",
                        msg="tokenizacao nao produziu datasets",
                        fix="inspecione a etapa 'tokenizar' e artifacts/audit")

    with StepGate("treinar", log) as g:
        bf16 = dtype == torch.bfloat16
        # DERIVACAO: num_workers do config (escrito pelo 02_hpo v7 = 10).
        # Fallback para 10 se config legacy. Ryzen 5500X3D 6c/12t: 10 workers
        # deixa 2 threads para sistema e main loop.
        nw = int(cfg.get("dataloader_num_workers", min(10, os.cpu_count() or 4)))
        # v7: prefetch maior. Default HF e 2; subimos para 6 = 3x mais batches
        # prontos na fila quando GPU pede dados. Custa RAM (folga em 32GB).
        prefetch = int(cfg.get("dataloader_prefetch_factor", 6))
        batch = int(cfg.get("batch_size", 8))
        accum = int(cfg.get("gradient_accumulation_steps", 1))
        epochs = int(cfg["num_train_epochs"])
        eval_batch = min(batch, 16)  # geracao com 4 beams custa memoria
        log.info(f"  [DERIVADO] eval_batch={eval_batch} (= min(batch={batch}, 16)) "
                 f"| num_workers={nw} | prefetch={prefetch} "
                 f"| epochs(teto)={epochs}")
        ta = Seq2SeqTrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=eval_batch,
            gradient_accumulation_steps=accum,
            group_by_length=True,
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg.get("weight_decay", 0.02)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.08)),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            bf16=bf16, fp16=not bf16, tf32=True,
            predict_with_generate=True, generation_max_length=MAX_TARGET,
            generation_num_beams=4,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_chrf",
            greater_is_better=True, logging_steps=50, seed=SEED, data_seed=SEED,
            report_to="none", dataloader_num_workers=nw,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=(nw > 0),
            dataloader_prefetch_factor=prefetch if nw > 0 else None,
            gradient_checkpointing=grad_ckpt)
        patience = int(cfg.get("early_stopping_patience", 6))
        trainer = Seq2SeqTrainer(
            model=model, args=ta, processing_class=tok,
            train_dataset=cast(Any, ds_tr), eval_dataset=cast(Any, ds_va),
            data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
            compute_metrics=build_compute_metrics(tok),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience),
                       make_audit_callback(tok, ds_va)])
        if device == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        trainer.train()
        g.info["duracao_h"] = round((time.time() - t0) / 3600, 3)
        # Epoca final REAL (pode ser menor que num_train_epochs por early stopping)
        g.info["epoca_final_efetiva"] = round(trainer.state.epoch or 0.0, 2)
        met_val = trainer.evaluate()
        g.info["chrf_val"] = met_val.get("eval_chrf")

    with StepGate("salvar_e_merge", log) as g:
        ldir = out_dir / "lora_only"
        model.save_pretrained(str(ldir))
        tok.save_pretrained(str(ldir))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
        peft_model = PeftModel.from_pretrained(base, str(ldir))
        merged = cast(Any, peft_model).merge_and_unload()
        merged.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        merged = merged.to(device)
        g.info["out_dir"] = str(out_dir)

    with StepGate("smoke", log) as g:
        smoke_res = smoke_test(merged, tok, device)
        g.info["smoke_acertos"] = sum(r["ok"] for r in smoke_res)

    teste_metrics = {}
    if TEST_CSV.exists():
        with StepGate("avaliacao_teste_holdout", log) as g:
            df_te = pd.read_csv(TEST_CSV).dropna(subset=["input", "output"]).reset_index(drop=True)
            validar_dataframe(df_te, "teste", log)
            log.info(f"Hold-out teste: {len(df_te)} linhas (avaliacao batched)")
            teste_metrics = avaliar_por_direcao_batched(merged, tok, df_te, device,
                                                         batch_size=eval_batch * 2)
            g.info["teste"] = teste_metrics
    else:
        log.warning("dataset de teste ausente -- avaliacao hold-out ignorada")

    rel = {
        "versao": "7.0.0",
        "metricas_val": met_val, "smoke_test": smoke_res,
        "teste_por_direcao": teste_metrics, "config": cfg,
        "out_dir": str(out_dir),
    }
    salvar_json(rel, DATA_REPORTS / "train_report.json")
    log.info("=" * 64)
    log.info(f"TREINO CONCLUIDO | chrF(val)={met_val.get('eval_chrf'):.4f} | "
             f"modelo: {out_dir}")
    log.info(f"Proximo: python scripts/04_evaluate.py --save-errors")
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
