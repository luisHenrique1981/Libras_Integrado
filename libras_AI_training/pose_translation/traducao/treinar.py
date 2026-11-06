"""Treinamento supervisionado por frase (gloss-free) com teacher forcing.
Uso: python -m traducao.treinar --dados dados_traducao --epocas 15
"""
import argparse
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (montar_vocab, DatasetTraducoes, collate, carregar_indice,
                      ids_para_texto)
from .modelo_transformer import TradutorTransformer


def dividir(registros, seed=42):
    grupos = sorted({r['grupo'] for r in registros})
    rng = random.Random(seed)
    rng.shuffle(grupos)
    if len(grupos) >= 2:
        treino_grupos = set(grupos[:-1]); val_grupos = {grupos[-1]}
        return ([r for r in registros if r['grupo'] in treino_grupos],
                [r for r in registros if r['grupo'] in val_grupos], 'por vídeo')
    # Atenção: divide por trechos de MESMO vídeo => vazamento / avaliação otimista.
    print('AVISO: somente um vídeo: divisão por trecho NÃO mede generalização.')
    dados = registros[:]
    rng.shuffle(dados)
    n = max(1, int(len(dados)*0.15))
    return dados[n:], dados[:n], 'por trecho (exploratória)'


def executar(args):
    random.seed(args.seed);torch.manual_seed(args.seed)
    registros = carregar_indice(args.dados)
    if len(registros) < 2:
        raise ValueError('É necessário ter pelo menos dois trechos válidos')
    treino, validacao, tipo_split = dividir(registros, args.seed)
    vocab = montar_vocab(treino)
    print(f'Treino={len(treino)} | Validação={len(validacao)} ({tipo_split}) | Vocabulário={len(vocab)}')
    if len(vocab) < 30:
        print('AVISO: vocabulário muito pequeno, os resultados não serão confiáveis.')
    ds_train = DatasetTraducoes(args.dados, treino, vocab, args.max_frames, args.max_tokens)
    ds_val = DatasetTraducoes(args.dados, validacao, vocab, args.max_frames, args.max_tokens)
    loader_t = DataLoader(ds_train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    loader_v = DataLoader(ds_val, batch_size=args.batch, shuffle=False, collate_fn=collate)
    dim = ds_train[0][0].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    modelo = TradutorTransformer(dim, len(vocab), args.d_model, args.cabecas, args.camadas).to(device)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr)
    criterio = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    destino = Path(args.saida); destino.parent.mkdir(parents=True, exist_ok=True)
    melhor = float('inf')

    for epoca in range(1, args.epocas+1):
        perdas=[]
        modelo.train()
        for x, y, src_pad in loader_t:
            x,y,src_pad = x.to(device), y.to(device), src_pad.to(device)
            entrada, alvo = y[:, :-1], y[:, 1:]
            pred = modelo(x, entrada, src_pad, entrada==vocab['<pad>'])
            loss = criterio(pred.reshape(-1, len(vocab)), alvo.reshape(-1))
            otimizador.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step(); perdas.append(loss.item())
        modelo.eval(); val_loss=[]
        with torch.no_grad():
            for x,y,src_pad in loader_v:
                x,y,src_pad = x.to(device),y.to(device),src_pad.to(device)
                pred = modelo(x,y[:,:-1],src_pad, y[:,:-1]==vocab['<pad>'])
                l = criterio(pred.reshape(-1,len(vocab)),y[:,1:].reshape(-1))
                val_loss.append(l.item())
        v = sum(val_loss)/len(val_loss)
        print(f'Época {epoca:03d}: treino={sum(perdas)/len(perdas):.4f} | validação={v:.4f}')
        if v < melhor:
            melhor=v
            torch.save({
                'state_dict':modelo.state_dict(), 'vocab':vocab, 'dim_entrada':dim,
                'd_model':args.d_model, 'cabecas':args.cabecas, 'camadas':args.camadas,
                'max_frames':args.max_frames, 'max_tokens':args.max_tokens,
                'tipo_split':tipo_split, 'train_groups':sorted({r['grupo'] for r in treino}),
                'val_groups':sorted({r['grupo'] for r in validacao}),
            }, destino)
    print('Modelo salvo em', destino)
    print('Atenção: loss não comprova qualidade de tradução; teste em vídeo NÃO visto.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dados',default='dados_traducao')
    p.add_argument('--saida',default='modelos/tradutor.pt')
    p.add_argument('--epocas',type=int,default=15)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--d-model',type=int,default=128)
    p.add_argument('--cabecas',type=int,default=4)
    p.add_argument('--camadas',type=int,default=2)
    p.add_argument('--max-frames',type=int,default=128)
    p.add_argument('--max-tokens',type=int,default=45)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu',action='store_true')
    executar(p.parse_args())
