import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        p = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(p * div)
        pe[:, 1::2] = torch.cos(p * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TradutorTransformer(nn.Module):
    def __init__(self, dim_entrada, vocab_size, d_model=128, cabecas=4, camadas=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(dim_entrada, d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=cabecas, num_encoder_layers=camadas,
            num_decoder_layers=camadas, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.saida = nn.Linear(d_model, vocab_size)

    def forward(self, x, tokens_entrada, src_mask=None, tgt_mask=None):
        src = self.pos(self.input_proj(x))
        tgt = self.pos(self.token_emb(tokens_entrada) * math.sqrt(self.token_emb.embedding_dim))
        mask_causal = torch.triu(torch.ones(
            tokens_entrada.shape[1], tokens_entrada.shape[1],
            dtype=torch.bool, device=tokens_entrada.device
        ), diagonal=1)
        saida = self.transformer(
            src, tgt, tgt_mask=mask_causal,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        return self.saida(saida)

    @torch.no_grad()
    def gerar(self, x, bos_id, eos_id, max_tokens=45):
        self.eval()
        tokens = torch.full((x.size(0), 1), bos_id, device=x.device, dtype=torch.long)
        acabados = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
        for _ in range(max_tokens):
            logits = self(x, tokens)[:, -1, :]
            proximos = logits.argmax(-1)
            tokens = torch.cat([tokens, proximos[:, None]], dim=1)
            acabados |= proximos == eos_id
            if acabados.all():
                break
        return tokens
