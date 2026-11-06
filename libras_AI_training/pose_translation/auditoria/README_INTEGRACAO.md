# Integração avaliada: ProjetoVisaoComputacional + libras_projeto_completo

## Diagnóstico da base recebida

- `libras_AI_training/`: fine-tuning de `unicamp-dl/ptt5-base-portuguese-vocab` com LoRA, **entrada textual** português/glosa e saída textual; NÃO aceita landmarks diretamente.
- `teachablemachine/extrair_landmarks.py`: landmarks MediaPipe de 21 pontos da primeira mão por frame, 63 valores; requer vídeos e rótulos; não usa `.vtt` como tradução de sequência.
- `teachablemachine/treinar_classificador.py`: MLP Dense(128,64,32) com saída de classes, não decoder de tradução.
- `teachablemachine/inferir_sinal.py`: média das probabilidades por frame, descarta a ordem temporal.
- `libras_bridge/`: aplicativo Android Kotlin com câmera e avatar, porém requer `models/libras_gesture.task`, ausente no ZIP; não implementa inferência do Transformer PyTorch.

## Integração recomendada

1. Preserve `traducao/preparar_dados.py`, `representacao.py`, `modelo_transformer.py`, `treinar.py` do projeto atual; entrada: DWPose `.npz` + `.vtt`.
2. Use `auditar_dataset.py` neste pacote para conferir o índice gerado, arquivo por arquivo, e verificar quantos vídeos distintos entraram.
3. Adote do professor **conceitos** de contratos de dados, auditoria, divisão sem vazamento e métricas de tradução. **Não cole** `01_prepare_data.py` sobre o atual: espera textos/glosas em CSV e um tokenizador T5.
4. Avalie no vídeo não visto por treinamento. Se houver somente três vídeos, resultados serão altamente incertos e dependentes do conteúdo.
5. Para webcam, usar uma representação temporal consistente. MediaPipe e DWPose não são equivalentes: comparar projeções na mesma amostra antes de avaliar traduções.
6. O pipeline T5 do professor só seria útil como **segundo estágio glosa textual -> português** se houvesse um modelo validado que previsse glosas a partir das poses. Como o projeto atual é gloss-free, essa etapa não é necessária.
7. Android pode ser uma interface futura: exportar um modelo compatível (ONNX/TFLite validado) ou servir a tradução num backend. O ZIP não fornece uma API de integração pronta.

## Uso do auditor (sem extrair vídeos, sem TensorFlow)

Copie a pasta `integracao_professor` para a raiz do projeto e execute no PowerShell:

```powershell
python integracao_professor/auditar_dataset.py --dados dados_traducao --relatorio relatorio_integracao.json
```

A verificação é estrutural, não um teste de acurácia. Não modifica datasets ou modelos.

## Observações específicas

O README do professor cita uma etapa `scripts/00_check_env.py` que **não consta no ZIP recebido**. Resultados v7 relatam regressão de chrF médio frente a v6, portanto não devem ser interpretados como prova de desempenho superior. O `libras_gesture.task` Android também não consta no ZIP.
