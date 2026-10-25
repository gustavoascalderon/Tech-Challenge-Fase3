# Relatorio de auditoria dos dados

**Resultado geral: APROVADO**

## Verificacoes

| Verificacao | Status | Detalhe |
|---|---|---|
| Sem texto repetido entre treino e teste (vazamento) | OK | 988 textos repetidos no original, 0 apos limpeza |
| Treino com no minimo 2000 amostras (exigencia do enunciado) | OK | 7565 amostras |
| As 3 classes de urgencia estao presentes | OK | {'urgente': 0.4846, 'normal': 0.2998, 'atencao': 0.2156} |
| Nenhuma classe com menos de 5% do treino | OK | menor classe: 21.6% |
| Sem textos nulos ou vazios | OK | menor texto: 169 caracteres |
| Sem texto com urgencia conflitante apos limpeza | OK | duplicatas e conflitos removidos |

## Dados originais (antes da limpeza)

| Item | Treino | Teste |
|---|---|---|
| Linhas | 11550 | 2888 |
| Textos duplicados | 2105 | 118 |
| Textos com condicao conflitante | 1956 | 113 |
| Textos com urgencia conflitante | 1880 | 107 |
| Palavras (min / mediana / max) | 24 / 175 / 596 | 29 / 179 / 528 |

Textos identicos presentes em treino **e** teste: **988** (removidos do teste para evitar vazamento).

## Dados limpos (usados no treino e na avaliacao)

- Treino: **7565** laudos | Teste: **1788** laudos
- Distribuicao de urgencia (treino): {'urgente': 0.4846, 'normal': 0.2998, 'atencao': 0.2156}
- Distribuicao de urgencia (teste): {'urgente': 0.4978, 'normal': 0.278, 'atencao': 0.2243}
- Baseline (classe majoritaria no teste): 49.8%

## Regras de limpeza

1. Texto normalizado (minusculas, espacos unificados).
2. Condicao convertida para urgencia (mapeamento proxy documentado no README).
3. Texto com urgencias diferentes no mesmo conjunto: removido (rotulo ambiguo).
4. Texto repetido com a mesma urgencia: mantido uma vez.
5. Textos do teste que tambem aparecem no treino: removidos do teste.
