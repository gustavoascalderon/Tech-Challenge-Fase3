# Relatorio de validacao do modelo

**Resultado geral: APROVADO**

Teste limpo (sem vazamento): **1788** laudos.

| Metrica | Valor |
|---|---|
| Acuracia sklearn | 74.4% |
| Acuracia ONNX | 74.7% |
| F1 macro ONNX | 72.2% |
| Baseline (classe majoritaria) | 49.8% |
| Concordancia ONNX x sklearn | 99.3% |

## Matriz de confusao (ONNX) - linhas: real, colunas: previsto

| real \ previsto | atencao | normal | urgente |
|---|---|---|---|
| atencao | 274 | 91 | 36 |
| normal | 67 | 321 | 109 |
| urgente | 35 | 114 | 741 |

## Verificacoes

| Verificacao | Status |
|---|---|
| acuracia_onnx_maior_que_baseline | OK |
| concordancia_onnx_sklearn_>=_0.97 | OK |
| probabilidades_somam_1 | OK |
| contrato_da_api_ok | OK |
| api_rejeita_texto_curto_422 | OK |
