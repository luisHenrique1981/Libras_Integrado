"""Execute na raiz de pose_translation: python diagnosticar_traducao.py --segmento dados_traducao/arquivo.npz"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from traducao.avaliar import abrir_modelo
from traducao.dataset import reduzir_tempo, ids_para_texto, tokens_texto


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', default='modelos/tradutor.pt')
    p.add_argument('--dados', default='dados_traducao')
    p.add_argument('--segmento', required=True)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()

    arquivo = Path(a.segmento)
    dados = Path(a.dados)
    indice = dados / 'indice.jsonl'
    if indice.exists():
        registro = next((json.loads(l) for l in indice.read_text(encoding='utf-8').splitlines()
                         if l.strip() and json.loads(l).get('arquivo') == arquivo.name), None)
        if registro:
            print('LEGENDA DE REFERENCIA:', registro.get('texto'))
            print('GRUPO DO VIDEO:', registro.get('grupo'))
        else:
            print('AVISO: segmento nao encontrado no indice.jsonl')
    else:
        print('AVISO: indice.jsonl nao encontrado')
        registro = None

    modelo, meta, device = abrir_modelo(a.modelo, a.cpu)
    print('GRUPOS TREINO:', meta.get('train_groups'))
    print('GRUPOS VALIDACAO:', meta.get('val_groups'))
    print('MELHOR CHECKPOINT: selecionado por menor loss de validacao (se treino original)')
    vocab = meta['vocab']
    inv = {i: s for s, i in vocab.items()}

    with np.load(arquivo, allow_pickle=False) as f:
        x = reduzir_tempo(f['features'], meta['max_frames'])
    print('FORMATO DE ENTRADA:', x.shape)
    print('VALORES FINITOS:', bool(np.isfinite(x).all()))
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    bos, eos = vocab['<bos>'], vocab['<eos>']
    tokens = torch.tensor([[bos]], device=device)
    with torch.no_grad():
        logits = modelo(xt, tokens)[0, -1]
        probabilidades = logits.softmax(dim=-1)
        probs, ids = torch.topk(probabilidades, min(8, len(vocab)))
        print('\nOITO PRIMEIRAS OPCOES PARA O PRIMEIRO TOKEN:')
        for tid, prob in zip(ids.tolist(), probs.tolist()):
            print(f'  {inv.get(tid, "?")!r:<24} {prob:.3%}')
        gerado = modelo.gerar(xt, bos, eos, meta['max_tokens'])[0].tolist()
    print('\nTOKENS GERADOS:', [inv.get(i, '<id-desconhecido>') for i in gerado])
    print('TRADUCAO:', repr(ids_para_texto(gerado, vocab)))
    if len(gerado) > 1 and gerado[1] == eos:
        print('DIAGNOSTICO: modelo previu <eos> como primeiro token de saida.')
    elif not ids_para_texto(gerado, vocab):
        print('DIAGNOSTICO: saida sem tokens de palavras; revise tokens especiais.')
    else:
        print('DIAGNOSTICO: modelo produziu texto; a qualidade precisa ser avaliada.')


if __name__ == '__main__':
    main()
