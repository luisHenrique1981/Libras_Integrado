"""Dataset de sequências de landmarks e legendas em PT-BR."""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

ESPECIAIS = ['<pad>', '<bos>', '<eos>', '<unk>']


def tokens_texto(s):
    return re.findall(r"\w+|[^\w\s]", s.lower(), flags=re.UNICODE)


def montar_vocab(registros, vocab_max=4000):
    freq = Counter(tok for reg in registros for tok in tokens_texto(reg['texto']))
    return {palavra: i for i, palavra in enumerate(ESPECIAIS + [
        t for t, _ in freq.most_common(vocab_max - len(ESPECIAIS))
        if t not in ESPECIAIS
    ])}


def texto_para_ids(texto, vocab, max_tokens=45):
    return [vocab['<bos>']] + [vocab.get(t, vocab['<unk>']) for t in tokens_texto(texto)[:max_tokens-2]] + [vocab['<eos>']]


def ids_para_texto(ids, vocab):
    inverso = {v: k for k, v in vocab.items()}
    palavras = []
    for tid in ids:
        tok = inverso.get(int(tid), '<unk>')
        if tok == '<eos>':
            break
        if tok in ESPECIAIS:
            continue
        palavras.append(tok)
    frase = ' '.join(palavras)
    return re.sub(r'\s+([.,!?;:])', r'\1', frase)


def reduzir_tempo(x, max_frames=128):
    if len(x) > max_frames:
        indices = np.linspace(0, len(x)-1, max_frames).round().astype(int)
        x = x[indices]
    return x.astype('float32')


class DatasetTraducoes(Dataset):
    def __init__(self, pasta, registros, vocab, max_frames=128, max_tokens=45):
        self.pasta = Path(pasta)
        self.registros = registros
        self.vocab = vocab
        self.max_frames = max_frames
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, i):
        reg = self.registros[i]
        with np.load(self.pasta / reg['arquivo'], allow_pickle=False) as f:
            x = reduzir_tempo(f['features'], self.max_frames)
        y = texto_para_ids(reg['texto'], self.vocab, self.max_tokens)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def collate(amostras):
    b = len(amostras)
    maior_x = max(x.shape[0] for x, _ in amostras)
    maior_y = max(y.shape[0] for _, y in amostras)
    dim = amostras[0][0].shape[1]
    X = torch.zeros(b, maior_x, dim)
    Y = torch.zeros(b, maior_y, dtype=torch.long)
    src_pad = torch.ones(b, maior_x, dtype=torch.bool)
    for i, (x, y) in enumerate(amostras):
        X[i, :len(x)] = x
        Y[i, :len(y)] = y
        src_pad[i, :len(x)] = False
    return X, Y, src_pad


def carregar_indice(pasta):
    return [json.loads(l) for l in (Path(pasta) / 'indice.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
