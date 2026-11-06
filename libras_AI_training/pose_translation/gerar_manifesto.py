
import json
from pathlib import Path

PASTA = Path("dados_exemplo")
SAIDA = Path("pares_libras.json")

pares = []
erros = []

for pasta in sorted(
    PASTA.glob("video*"),
    key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 9999
):
    if not pasta.is_dir():
        continue

    poses = list(pasta.glob("*.npz"))
    legendas = list(pasta.glob("*.vtt"))

    if len(poses) != 1 or len(legendas) != 1:
        erros.append(
            f"{pasta}: encontrados "
            f"{len(poses)} NPZ e {len(legendas)} VTT"
        )
        continue

    pares.append({
        "poses": poses[0].as_posix(),
        "vtt": legendas[0].as_posix()
    })

if erros:
    print("Corrija estas pastas:")
    for erro in erros:
        print(" -", erro)
    raise SystemExit(1)

if not pares:
    raise SystemExit("Nenhum par encontrado.")

# Evita sobrescrever o manifesto existente
# sem guardar uma copia.
if SAIDA.exists():
    backup = SAIDA.with_suffix(".json.bak")
    backup.write_bytes(SAIDA.read_bytes())

SAIDA.write_text(
    json.dumps(pares, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Manifesto atualizado: {len(pares)} pares")
for par in pares:
    print(par["poses"], "+", par["vtt"])
