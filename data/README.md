# Dados

Arquivos em `raw/` (Medical Abstracts TC Corpus, licença CC BY-SA 3.0):
`medical_tc_train.csv`, `medical_tc_test.csv`, `medical_tc_labels.csv`.

Fonte: https://github.com/sebischair/Medical-Abstracts-TC-Corpus

Para baixar novamente:
```
curl -L -o data/raw/medical_tc_train.csv https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv
curl -L -o data/raw/medical_tc_test.csv  https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv
```

## Observações de qualidade

O corpus original tem textos repetidos entre treino e teste e textos com rótulos conflitantes.
O projeto trata isso em `src/triagem/dados.py`; o diagnóstico completo está em `results/auditoria_dados.md`
(gerado por `python scripts/auditar_dados.py`).
