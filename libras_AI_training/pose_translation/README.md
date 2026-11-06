# Tradução por landmarks + legendas (integração experimental)

Este é um **módulo adicional dentro do projeto do professor** (`libras_AI_training/pose_translation`). O treinamento original T5/LoRA em `libras_AI_training/scripts/` e o aplicativo Android `libras_bridge/` permanecem intactos.

## Objetivo e limites

Treinar um Transformer de poses para texto usando apenas **SignVerse DWPose `.npz` + legendas `.vtt`**, sem baixar ou processar vídeo durante o treinamento. Não usa glosas, e não exige que os sinais isolados sejam rotulados manualmente. **Ainda é protótipo de pesquisa:** com três vídeos pode memorizar legendas e falhar com pessoas novas. A webcam usa MediaPipe com representação aproximada; isso não equivale a DWPose e não garante tradução correta.

A implementação de tradução por landmarks **não substitui** o `libras_AI_training` textual do professor. As duas vertentes coexistem e ainda não estão conectadas ao app Android.

## Instalação (Windows PowerShell)

Abra a pasta `libras_AI_training/pose_translation` no terminal do VS Code (ou navegue até ela). Use um ambiente dedicado para não conflitar com TensorFlow do projeto original:

```powershell
py -3.11 -m venv .venv_transformer
.\.venv_transformer\Scripts\Activate.ps1
python -m pip install -r requirements_pose.txt
python -m pip check
```

### Organizar dados

Copie `pares_libras.exemplo.json` para `pares_libras.json` e ajuste **os nomes reais** dos arquivos nas pastas `dados_exemplo/video1`, `video2` e `video3`. Cada par precisa ter as poses **do mesmo vídeo** que as legendas. O manifesto de exemplo usa nomes observados no seu VS Code, mas valide a grafia. O arquivo de video3 termina em `.pt.vtt`, que é diferente de `.pt-BR.vtt`; isso não indica por si só que os sinais sejam Libras. Confirme o idioma de sinais nos metadados. Não inclua os dados grandes no commit.

### Preparar, treinar, avaliar

Execute estes comandos **dentro de `libras_AI_training/pose_translation`**:

```powershell
python -m traducao.preparar_dados --manifesto pares_libras.json --saida dados_traducao
python -m traducao.treinar --dados dados_traducao --epocas 5 --batch 4 --d-model 64 --cabecas 4 --camadas 1 --saida modelos/tradutor.pt
python -m traducao.avaliar --modelo modelos/tradutor.pt --segmento "dados_traducao/NOME_REAL.npz"
python -m traducao.traduzir_webcam --modelo modelos/tradutor.pt --janela 64 --a-cada 24
```

A avaliação no `.npz` é **mais importante que a câmera** inicialmente: remove diferenças entre MediaPipe e DWPose. O script de webcam só executa inferência após capturar uma janela completa, com mãos visíveis em pelo menos 25% dos frames. Ele não faz classificação confiável de glosas ou detecção de fronteiras de frases.

### Auditoria (opcional)

```powershell
python auditoria/auditar_dataset.py --dados dados_traducao --relatorio relatorio_integracao.json
```

## O que foi preservado do professor

- `libras_AI_training/scripts/` e `src/`: pipeline textual T5/LoRA e avaliação originais.
- `teachablemachine/`: extrator/classificador por uma mão.
- `libras_bridge/`: aplicativo Android, sem mudanças no APK/Gradle/Kotlin.
- `ProjetoVisaoComputacional/`: outras ferramentas.

## Para enviar ao GitHub do professor

Faça fork ou crie branch com autorização de escrita, copie esta integração para uma cópia **git clone** do repositório e revise `git status`. Este ZIP é um snapshot do repositório, não contém a pasta `.git`. Use pull request. Não faça commit de `.npz`, `.vtt`, checkpoints `.pt`, modelos ou dados pessoais sem verificar licença e autorização.

## Próximas etapas técnicas

1. Testar aprendizado e generalização em vídeos/sinalizantes não vistos.
2. Confirmar idioma de sinais de cada par e refinar alinhamento legenda-poses.
3. Padronizar detector offline/online (mesmo DWPose em ambos ou adaptação de domínio validada).
4. Só então integrar o checkpoint no app Android por backend/API ou exportação compatível. Não existe ainda integração direta com `libras_bridge`.
