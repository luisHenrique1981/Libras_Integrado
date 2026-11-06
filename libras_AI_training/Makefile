# Atalhos. Ative a venv antes:  source .venv_pytorch/bin/activate
PY=python

check:        ; $(PY) scripts/00_check_env.py
prepare:      ; $(PY) scripts/01_prepare_data.py
hpo:          ; $(PY) scripts/02_hpo.py --trials 30 --epochs 5
train:        ; $(PY) scripts/03_train.py --config artifacts/configs/config_campeao.json
eval:         ; $(PY) scripts/04_evaluate.py --save-errors
pipeline: check prepare hpo train eval

cleanup-dry:  ; $(PY) tools/cleanup.py --raiz ..
cleanup:      ; $(PY) tools/cleanup.py --raiz .. --aplicar
