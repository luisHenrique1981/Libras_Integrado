#!/usr/bin/env python3
"""Seleciona ate N pares de uma pasta PUBLICA do Google Drive.
Faz listagem sem baixar blobs; baixa ate M legendas pequenas para examinar texto;
baixa apenas N arquivos .npz pareados (mais suas legendas).

O pareamento por pasta eh conservador, mas deve ser revisado pelo usuario.
Nao comprova idioma de sinais, alinhamento ou ocorrencia do sinal isolado.
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath
from collections import defaultdict

DRIVE = 'https://drive.google.com/drive/folders/1YFao8jmTT5Y6IGJEz8aZjThrJcNjplLr'
TIMES = re.compile(r'(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}\s*-->\s*(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}')
TAGS = re.compile(r'<[^>]+>')

def normalizar(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')

def analisar_vtt(path):
    raw = path.read_text(encoding='utf-8-sig', errors='replace')
    blocks = re.split(r'\n\s*\n', raw.replace('\r\n', '\n'))
    hits = []
    count = 0
    for block in blocks:
        lines = block.splitlines()
        idx = next((i for i, x in enumerate(lines) if TIMES.search(x)), -1)
        if idx < 0:
            continue
        body = ' '.join(TAGS.sub('', x).strip() for x in lines[idx+1:] if x.strip())
        if not body:
            continue
        count += 1
        low = normalizar(body)
        for word in ('casa', 'oi'):
            if re.search(r'(?<!\w)' + word + r'(?!\w)', low):
                hits.append({'palavra':word,'tempo':lines[idx].strip(),'texto':body})
    return count, hits

def main():
    p = argparse.ArgumentParser(description='Selecao LIMITADA de pares Drive (nao baixa a pasta inteira).')
    p.add_argument('--url',default=DRIVE)
    p.add_argument('--quantidade',type=int,default=10)
    p.add_argument('--max-legendas',type=int,default=100,help='Maximo de VTT baixados para inspecao (padrao 100).')
    p.add_argument('--saida',default='drive_libras_10')
    p.add_argument('--somente-relatorio',action='store_true',help='Nao baixar arquivos NPZ ainda.')
    a = p.parse_args()
    if not (1 <= a.quantidade <= 50 and 1 <= a.max_legendas <= 1000):
        p.error('--quantidade deve estar entre 1 e 50 e --max-legendas entre 1 e 1000')
    try:
        import gdown
    except ImportError:
        p.error('Instale: python -m pip install --upgrade gdown')
    out = Path(a.saida)
    out.mkdir(parents=True,exist_ok=True)
    print('Listando arquivos do Drive (sem download de conteudo)...',flush=True)
    try:
        entries = gdown.download_folder(url=a.url,skip_download=True,quiet=True,remaining_ok=True)
    except Exception as exc:
        print('Nao foi possivel listar a pasta publica:',exc)
        return 2
    if not entries:
        print('Nenhum arquivo listado. Verifique se a pasta pode ser acessada por quem tem o link.')
        return 2
    rows = []
    for x in entries:
        name = str(x.path).replace('\\','/')
        suf = PurePosixPath(name).suffix.lower()
        if suf not in ('.npz','.vtt'):
            continue
        rows.append({'id':x.id,'path':name,'ext':suf,'folder':str(PurePosixPath(name).parent)})
    (out/'catalogo.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['folder']].append(r)
    pairs=[]
    for folder, files in grouped.items():
        npzs=[x for x in files if x['ext']=='.npz']
        vtts=[x for x in files if x['ext']=='.vtt']
        if len(npzs)==1 and len(vtts)==1:
            pairs.append({'folder':folder,'npz':npzs[0],'vtt':vtts[0]})
    pairs.sort(key=lambda e:e['folder'].lower())
    print(f'Arquivos listados: {len(rows)} | pares candidatos de pasta unica: {len(pairs)}')
    print(f'Analisando no maximo {a.max_legendas} legendas, sem baixar NPZ ainda.')
    evaluated=[]
    for i, pair in enumerate(pairs[:a.max_legendas],1):
        dest=out/'legendas_inspecao'/f'{i:04d}.vtt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = gdown.download(id=pair['vtt']['id'],output=str(dest),quiet=True)
            if not result or not dest.is_file():
                raise RuntimeError('download nao concluido')
            n,hits=analisar_vtt(dest)
            evaluated.append({**pair,'segmentos':n,'hits':hits,'legenda_local':str(dest)})
        except Exception as exc:
            print(f'Falha na legenda {pair["vtt"]["path"]}: {exc}')
        if i%10==0:
            print(f'Legendas verificadas: {i}/{min(len(pairs),a.max_legendas)}',flush=True)
    # Prioriza presença da expressão, depois diversidade textual (n. segmentos), nunca tamanho NPZ desconhecido.
    evaluated.sort(key=lambda p:(bool(p['hits']),len(p['hits']),p['segmentos']),reverse=True)
    selected=evaluated[:a.quantidade]
    with (out/'ocorrencias_casa_oi.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['pasta','palavra','tempo','texto'])
        writer.writeheader()
        for e in evaluated:
            for h in e['hits']:
                writer.writerow({'pasta':e['folder'],**h})
    summary=[{'pasta':e['folder'],'npz':e['npz']['path'],'vtt':e['vtt']['path'],
              'segmentos':e['segmentos'],'ocorrencias':len(e['hits'])} for e in selected]
    (out/'selecionados.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Selecionados: {len(selected)} pares. Relatorios em {out}')
    if a.somente_relatorio:
        print('Modo relatorio: NENHUM NPZ foi baixado.')
        return 0
    manifest=[]
    for i, e in enumerate(selected,1):
        dest=out/'selecionados'/f'video{i:02d}'
        dest.mkdir(parents=True,exist_ok=True)
        npz_path=dest/'poses.npz'
        vtt_path=dest/'legenda.vtt'
        try:
            import shutil
            shutil.copyfile(e['legenda_local'],vtt_path)
            print(f'Baixando NPZ {i}/{len(selected)}: {e["npz"]["path"]}',flush=True)
            result=gdown.download(id=e['npz']['id'],output=str(npz_path),quiet=False)
            if not result or not npz_path.is_file() or npz_path.stat().st_size==0:
                raise RuntimeError('arquivo nao foi baixado')
            manifest.append({'poses':npz_path.as_posix(),'vtt':vtt_path.as_posix()})
        except Exception as exc:
            print('ERRO ao baixar:',exc)
    (out/'pares_selecionados.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Baixados {len(manifest)} pares completos. Verifique idioma e conteudo antes de treinar.')
    return 0 if manifest else 2

if __name__=='__main__':
    raise SystemExit(main())
