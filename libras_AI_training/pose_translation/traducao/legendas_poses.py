"""Converte um par de legendas WebVTT e poses SignVerse-2M em exemplos de treino.

Uso:
 python preparar_dataset_libras.py --vtt caminho/arquivo.pt-BR.vtt --poses caminho/poses.npz --saida dataset_pronto

Saida: um arquivo .npz por segmento e indice.jsonl.
O codigo NAO executa MediaPipe nem baixa videos.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

PARTES = (
    ('body', 18), ('left_hand', 21), ('right_hand', 21), ('face', 68)
)
TOTAL_PONTOS = sum(n for _, n in PARTES)
RE_TEMPO = re.compile(r'(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})')


def segundos(tempo):
    m = RE_TEMPO.fullmatch(tempo.strip())
    if not m:
        raise ValueError(f'Horario VTT desconhecido: {tempo}')
    h, mm, ss, ms = m.groups()
    return int(h or 0)*3600 + int(mm)*60 + int(ss) + int(ms)/1000


def carregar_legendas(caminho):
    texto = Path(caminho).read_text(encoding='utf-8-sig')
    blocos = re.split(r'\n\s*\n', texto.replace('\r\n', '\n'))
    segmentos = []
    for bloco in blocos:
        linhas = [x.strip() for x in bloco.splitlines() if x.strip()]
        if not linhas or linhas[0].startswith(('WEBVTT', 'NOTE', 'STYLE', 'REGION')):
            continue
        linha_tempo = next((i for i, l in enumerate(linhas) if '-->' in l), None)
        if linha_tempo is None:
            continue
        esq, dir = linhas[linha_tempo].split('-->', 1)
        inicio = segundos(esq.strip())
        fim = segundos(dir.strip().split()[0])
        fala = ' '.join(linhas[linha_tempo+1:])
        fala = re.sub(r'<[^>]*>', '', fala).strip()
        if fim > inicio and fala:
            segmentos.append((inicio, fim, fala))
    return segmentos


def escolher_pessoa(frame):
    """Escolhe maior confianca media das maos entre pessoas presentes.
    Heuristica simples: em videos com multiplas pessoas, conferir manualmente.
    """
    qtd = int(frame.get('num_persons', 0))
    if qtd <= 0:
        return None
    candidatos = []
    for numero in range(qtd):
        prefixo = f'person_{numero:03d}_'
        confiancas = []
        for parte in ('left_hand', 'right_hand'):
            valor = frame.get(prefixo + parte + '_scores')
            if valor is not None:
                confiancas.extend(np.asarray(valor).ravel().tolist())
        score = float(np.mean(confiancas)) if confiancas else -1.0
        candidatos.append((score, prefixo))
    return max(candidatos)[1]


def extrair_frame(frame):
    """(128, 3): x, y, score. Coordenadas originais ja sao normalizadas."""
    prefixo = escolher_pessoa(frame)
    matriz = np.zeros((TOTAL_PONTOS, 3), dtype=np.float32)
    if prefixo is None:
        return matriz, False
    pos = 0
    for parte, tamanho in PARTES:
        coords = frame.get(prefixo + parte + '_keypoints')
        score = frame.get(prefixo + parte + '_scores')
        if coords is not None and score is not None:
            c = np.asarray(coords, dtype=np.float32)
            s = np.asarray(score, dtype=np.float32)
            if c.shape == (tamanho, 2) and s.shape == (tamanho,):
                matriz[pos:pos+tamanho, :2] = c
                matriz[pos:pos+tamanho, 2] = s
        pos += tamanho
    return np.nan_to_num(matriz, nan=0.0, posinf=0.0, neginf=0.0), True


def executar(vtt, poses, saida, fps_saida=12):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    legendas = carregar_legendas(vtt)
    with np.load(poses, allow_pickle=True) as dados:
        video_id = str(dados['video_id'].item())
        fps = float(dados['fps'].item())
        ids = dados['frame_indices'].astype(np.int64)
        payloads = dados['frame_payloads']
        if len(ids) != len(payloads):
            raise ValueError('frame_indices e frame_payloads com tamanhos diferentes')
        if fps <= 0:
            raise ValueError('FPS invalido')

        # frame_indices inicia em 1 neste dataset. Tempo do primeiro quadro = 0s.
        tempos = (ids - 1) / fps
        amostragem = max(1, round(fps / fps_saida))
        resultados = []
        with (saida / 'indice.jsonl').open('w', encoding='utf-8') as indice:
            for num, (inicio, fim, texto) in enumerate(legendas):
                selecionados = np.flatnonzero((tempos >= inicio) & (tempos < fim))
                selecionados = selecionados[::amostragem]
                if len(selecionados) == 0:
                    continue
                sequencia = []
                valido = []
                for j in selecionados:
                    pontos, encontrou = extrair_frame(payloads[j])
                    sequencia.append(pontos)
                    valido.append(encontrou)
                sequencia = np.stack(sequencia)
                mascara = np.asarray(valido, dtype=bool)
                arquivo = f'{video_id}_{num:04d}.npz'
                np.savez_compressed(
                    saida / arquivo,
                    landmarks=sequencia,
                    mask=mascara,
                    frame_indices=ids[selecionados],
                    texto=np.array(texto),
                )
                reg = {
                    'arquivo': arquivo, 'video_id': video_id,
                    'texto': texto, 'inicio': inicio, 'fim': fim,
                    'shape': list(sequencia.shape),
                    'frames_sem_pessoa': int((~mascara).sum()),
                    'fps_original': fps,
                    'fps_aproximado': fps / amostragem,
                }
                indice.write(json.dumps(reg, ensure_ascii=False) + '\n')
                resultados.append(reg)
    print(f'Video: {video_id}')
    print(f'Legendas: {len(legendas)} | Segmentos exportados: {len(resultados)}')
    print(f'Saida: {saida.resolve()}')
    if resultados:
        print('Primeiro exemplo:', resultados[0])
    return resultados


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vtt', required=True)
    parser.add_argument('--poses', required=True)
    parser.add_argument('--saida', default='dataset_pronto')
    parser.add_argument('--fps-saida', type=float, default=12)
    args = parser.parse_args()
    executar(args.vtt, args.poses, args.saida, args.fps_saida)
