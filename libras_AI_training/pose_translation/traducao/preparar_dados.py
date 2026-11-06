"""Converte pares .npz+.vtt em segmentos com características padronizadas.

Uso: python -m traducao.preparar_dados --manifesto pares_libras.json --saida dados_traducao
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from .representacao import dwpose_para_features

# Reutiliza o extrator para o formato REAL dos uploads, sem retrabalhar vídeos.
from .legendas_poses import carregar_legendas, extrair_frame


def preparar(manifesto, saida, fps_saida=12):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    pares = json.loads(Path(manifesto).read_text(encoding='utf-8'))
    if not isinstance(pares, list) or not pares:
        raise ValueError('Manifesto deve ser uma lista de pares {poses, vtt}')
    escritos = 0
    ausentes = []
    with (saida / 'indice.jsonl').open('w', encoding='utf-8') as out:
        for n, par in enumerate(pares):
            arquivo_poses, arquivo_vtt = Path(par['poses']), Path(par['vtt'])
            if not arquivo_poses.exists() or not arquivo_vtt.exists():
                print(f'AVISO: par {n+1} ausente: {arquivo_poses}, {arquivo_vtt}')
                ausentes.append(n+1)
                continue
            legendas = carregar_legendas(arquivo_vtt)
            with np.load(arquivo_poses, allow_pickle=True) as dados:
                ids = dados['frame_indices'].astype(np.int64)
                payloads = dados['frame_payloads']
                fps = float(dados['fps'].item())
                video_id = str(dados['video_id'].item())
                if len(ids) != len(payloads) or fps <= 0:
                    raise ValueError(f'NPZ inconsistente: {arquivo_poses}')
                tempos = (ids - ids[0]) / fps
                passo = max(1, round(fps / fps_saida))
                for i, (inicio, fim, frase) in enumerate(legendas):
                    escolhidos = np.flatnonzero((tempos >= inicio) & (tempos < fim))[::passo]
                    if len(escolhidos) < 3:
                        continue
                    frames, detectados = [], []
                    for indice in escolhidos:
                        frame, presente = extrair_frame(payloads[indice])
                        frames.append(frame)
                        detectados.append(presente)
                    if np.mean(detectados) < 0.5:
                        continue
                    features = dwpose_para_features(np.stack(frames))
                    arquivo = f'par{n+1:02d}_{video_id}_{i:05d}.npz'
                    np.savez_compressed(saida / arquivo, features=features)
                    registro = dict(arquivo=arquivo, texto=frase, video_id=video_id,
                                    grupo=f'par{n+1:02d}_{video_id}', inicio=inicio, fim=fim,
                                    frames=len(features))
                    out.write(json.dumps(registro, ensure_ascii=False)+'\n')
                    escritos += 1
            print(f'Processado: {video_id}, {len(legendas)} trechos de legenda')
    print(f'Exemplos salvos: {escritos} em {saida}')
    if ausentes:
        raise FileNotFoundError(f'Pares ausentes: {ausentes}. Corrija o manifesto antes de treinar.')
    if escritos == 0:
        raise RuntimeError('Nenhum segmento valido foi gerado.')
    return escritos


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifesto', default='pares_libras.json')
    p.add_argument('--saida', default='dados_traducao')
    p.add_argument('--fps-saida', type=float, default=12)
    args = p.parse_args()
    preparar(args.manifesto, args.saida, args.fps_saida)
