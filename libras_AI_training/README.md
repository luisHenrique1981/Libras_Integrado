# Tradutor Libras ↔ Português (T5 + LoRA) — Pipeline v5

Pipeline de fine-tuning do `unicamp-dl/ptt5-base-portuguese-vocab` com LoRA para
tradução bidirecional Português ↔ glosa Libras. Reorganizado para execução
auditável, com gates de etapa e erros explícitos.

## Por que a v4 não atingia o resultado esperado

1. **Prefixo de tarefa inconsistente (causa do smoke 0/18).** O smoke test usava
   `"traduzir Portugues para Libras: ..."`, mas o pré-processamento tokenizava
   `input` verbatim, sem prefixo. Treino e inferência em formatos diferentes →
   o modelo recebia entrada fora-de-distribuição e degenerava (copiava a entrada).
   **Correção:** `common.montar_input()` é o ponto único de verdade; a etapa 01
   injeta o prefixo e `validar_dataframe()` recusa qualquer input sem prefixo.

2. **Métrica errada.** BLEU-4 sobre glosa de ~6 tokens é patológico (um erro de
   palavra destrói vários 4-gramas; brevity penalty pune o curto). A seleção de
   modelo otimizava ruído. **Correção:** chrF++ é a métrica primária em todas as
   etapas; BLEU-4 vira secundária.

3. **LoRA estreito.** `target_modules=["q","v"]` limita o remapeamento lexical,
   que é o gargalo de glosa↔PT. **Correção:** `["q","k","v","o","wi","wo"]`.

4. **HPO de 3 épocas em subconjunto.** Modelos longe de convergir; o "campeão"
   (T13) estava dentro do ruído (Z=1.36 vs grupo canônico). **Correção:** HPO com
   mais épocas, pruner por mediana e chrF como objetivo.

5. **Teto de dados.** ~1.742 pares únicos de treino é pouco para MT. Nenhum
   hiperparâmetro corrige isso. O maior ganho disponível é **mais dados**.

## Estrutura

```
src/libras/common.py      backbone: paths, logging, StepGate, erros, contratos, métricas, prefixos
scripts/00_check_env.py   gate de ambiente (deps, GPU, disco, dados)
scripts/01_prepare_data.py prep: prefixo canônico + split sem leakage + contrato
scripts/02_hpo.py         busca TPE (chrF, LoRA largo, pruner)
scripts/03_train.py       treino final + smoke correto + merge + avaliação de teste
scripts/04_evaluate.py    generalização no hold-out, por direção
tools/cleanup.py          limpeza reversível (move p/ lixeira, nunca deleta)
data/raw/                 datasets-fonte (você copia aqui)
data/processed/           treino/val/teste gerados pela etapa 01
data/reports/             relatórios JSON de cada etapa
artifacts/                modelos, configs, logs, studies, auditoria
```

## Ordem de execução

```bash
source .venv_pytorch/bin/activate     # sua venv existente
cd libras_AI_training

python scripts/00_check_env.py                       # gate de ambiente
# copie a fonte para data/raw/ (ver migração abaixo)
python scripts/01_prepare_data.py                    # gera data/processed/*.csv
python scripts/02_hpo.py --trials 30 --epochs 5      # config_campeao.json
python scripts/03_train.py --config artifacts/configs/config_campeao.json
python scripts/04_evaluate.py --save-errors          # hold-out de teste
```

Ou via Makefile: `make check && make prepare && make hpo && make train && make eval`.

## Migração da pasta antiga

```bash
# 1. coloque libras_AI_training/ dentro de ~/Projetos/treinameno_modelo_IA/
# 2. copie a fonte de dados:
cp ../dataset_construido_v4.csv      data/raw/dataset_construido.csv
cp ../libras_bilingual_dataset.csv  data/raw/        # opcional (fallback)
# 3. limpeza (dry-run primeiro!):
python tools/cleanup.py --raiz ..                    # revise a lista
python tools/cleanup.py --raiz .. --aplicar          # move p/ _arquivo_morto_<data>/
```

`construir_dataset_v4.py` continua sendo o gerador da fonte; mantenha-o. A etapa
01 RE-DERIVA `input`/`output` a partir de `portugues_original`/`libras_glosa`/
`direcao`, então o formato do `input` antigo não importa mais.

## Erros auditáveis

Toda etapa roda dentro de um `StepGate`. Em falha, escreve
`artifacts/audit/audit_<etapa>_<ts>.json` com código, mensagem, correção e
traceback, e para o pipeline com `StepError`. Não há falha silenciosa.
