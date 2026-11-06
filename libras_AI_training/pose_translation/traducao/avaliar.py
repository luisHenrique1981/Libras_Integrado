"""Gera traduções em um segmento já preparado. Não usa legendas como saída."""
import argparse
import numpy as np
import torch
from .dataset import ids_para_texto, reduzir_tempo
from .modelo_transformer import TradutorTransformer


def abrir_modelo(caminho, cpu=False):
    device=torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    checkpoint=torch.load(caminho, map_location=device, weights_only=True)
    modelo=TradutorTransformer(checkpoint['dim_entrada'],len(checkpoint['vocab']),
          checkpoint['d_model'],checkpoint['cabecas'],checkpoint['camadas']).to(device)
    modelo.load_state_dict(checkpoint['state_dict']); modelo.eval()
    return modelo, checkpoint, device


def traduzir_features(features, modelo, meta, device):
    x=reduzir_tempo(features,meta['max_frames'])
    x=torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        y=modelo.gerar(x,meta['vocab']['<bos>'],meta['vocab']['<eos>'],meta['max_tokens'])
    return ids_para_texto(y[0].tolist(),meta['vocab'])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--modelo',default='modelos/tradutor.pt')
    p.add_argument('--segmento',required=True,help='arquivo .npz contendo features')
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    modelo,meta,device=abrir_modelo(a.modelo,a.cpu)
    with np.load(a.segmento,allow_pickle=False) as f:features=f['features']
    print('TRADUÇÃO GERADA:',traduzir_features(features,modelo,meta,device))
