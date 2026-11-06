"""Audita dados_traducao produzido pelo modulo traducao, sem treinar nem editar arquivos.
Uso: python integracao_professor/auditar_dataset.py --dados dados_traducao
"""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def auditar(pasta):
    raiz = Path(pasta)
    indice = raiz / 'indice.jsonl'
    if not indice.is_file():
        raise FileNotFoundError(f'Indice nao encontrado: {indice}')
    registros = [json.loads(s) for s in indice.read_text(encoding='utf-8').splitlines() if s.strip()]
    grupos = defaultdict(list)
    avisos = []
    formas = Counter()
    invalidos = 0
    tokens = Counter()
    for i, r in enumerate(registros, 1):
        grupo = str(r.get('grupo') or r.get('video_id') or 'SEM_GRUPO')
        grupos[grupo].append(r)
        nome = r.get('arquivo')
        if not nome or not (raiz / nome).is_file():
            invalidos += 1
            avisos.append(f'[{i}] arquivo ausente: {nome}')
            continue
        try:
            with np.load(raiz / nome, allow_pickle=False) as d:
                x = d['features']
                formas[(tuple(x.shape[1:]))] += 1
                if x.ndim != 2 or x.shape[0] < 3 or not np.isfinite(x).all():
                    invalidos += 1
                    avisos.append(f'[{i}] features invalidas: {nome} {x.shape}')
        except Exception as exc:
            invalidos += 1
            avisos.append(f'[{i}] erro lendo {nome}: {exc}')
        texto = str(r.get('texto', '')).strip()
        if not texto:
            avisos.append(f'[{i}] legenda vazia: {nome}')
        tokens.update(texto.casefold().split())
    if len(grupos) == 1:
        avisos.append('Somente um grupo/video. Validacao por trecho do mesmo video nao mede generalizacao.')
    if len(grupos) < 3:
        avisos.append('Menos de tres videos distintos reconhecidos no indice.')
    if len(formas) > 1:
        avisos.append(f'Dimensoes inconsistentes nas features: {dict(formas)}')
    if any('SEM_GRUPO' == g for g in grupos):
        avisos.append('Registros sem identificador de grupo/video, risco de vazamento treino/validacao.')
    rel = {
        'dataset': str(raiz), 'exemplos': len(registros),
        'videos_grupos': {g: len(rs) for g, rs in sorted(grupos.items())},
        'formatos_features': {str(k): v for k, v in formas.items()},
        'segmentos_invalidos': invalidos, 'vocabulario_aproximado': len(tokens),
        'avisos': avisos[:100],
        'nota': 'Conferencia de integridade, nao mede fidelidade das legendas nem qualidade da traducao.',
    }
    print(json.dumps(rel, indent=2, ensure_ascii=False))
    return rel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', default='dados_traducao')
    parser.add_argument('--relatorio', default='')
    args = parser.parse_args()
    rel = auditar(args.dados)
    if args.relatorio:
        p = Path(args.relatorio)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rel, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
