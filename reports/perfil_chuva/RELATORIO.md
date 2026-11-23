# Perfil da chuva prevista: prever o pico de 10 min (onda 16, Fase 4.2)

Pré-registro: `projeto-mapi-api/docs/PRE-REGISTRO-PERFIL-CHUVA.md` (escrito antes). Rodado por `python -m mapi_ai.rain.perfil`
em 2026-11-23 (`reports/perfil_chuva/perfil_chuva.json`). Só chuva: nenhum rótulo de alagamento.

## Resultado no teste do rain_v1 (depois de 31/07/2025), só linhas com chuva medida no horizonte (≥ 1 mm)

| Horizonte | Linhas (dias) | Pico medido: mediana / p95 (mm) | MAE uniforme | MAE (b) bloco alternado | MAE (a) prever o pico | (a) − (b), IC95 | Cobertura do q95 do pico (a) |
|---|---|---|---|---|---|---|---|
| 1 h | 26.075 (323) | 1,57 / 6,00 | 1,933 | 1,752 | **1,618** | −0,133 [−0,143; −0,124] | 0,708 |
| 3 h | 53.414 (337) | 1,60 / 6,49 | 2,161 | 1,872 | **1,690** | −0,183 [−0,195; −0,169] | 0,831 |
| 6 h | 80.188 (349) | 1,79 / 6,90 | 2,364 | 1,986 | **1,761** | −0,225 [−0,244; −0,208] | 0,887 |

Fração do maior bloco de 10 min no bloco alternado da IDF: 0,348 (1 h), 0,235 (3 h), 0,193 (6 h).

**Escolha pelo critério escrito antes: (a).** MAE médio 1,69 mm contra 1,87 mm de (b); a diferença média tem IC95 de −0,189 a
−0,172 mm, longe de zero, então não há empate.

## Leitura

- Espalhar por igual erra o pico por 1,9 a 2,4 mm e sempre para baixo (viés igual ao MAE): é o "apaga o pico" do prompt.
- As três opções subestimam o pico (viés de −1,5 a −2,4 mm). O q50 do acumulado já é baixo em chuva forte, e as opções (a)
  e (b) limitam o pico ao acumulado q50.
- O q95 do pico de (a) cobre 71% a 89%, abaixo dos 95% nominais. O cenário pessimista do alerta fica menos pessimista do que
  o nome diz: limitação para o relatório.

## O que muda

- Modelo novo `rain_v1p` (`models/v4/rain/rain_v1p_pico10.joblib`): regressor de quantis 0,5 e 0,95 do pico de 10 min por
  horizonte, mesmas entradas do modelo servido no horizonte (gb_rede em 1 h, gb_rede_om em 3 e 6 h), mesmo treino e parada
  antecipada do rain_v1. Os regressores do acumulado não mudam.
- No balde: pico no passo do meio do horizonte, limitado ao acumulado; o resto do acumulado por igual nos outros passos
  (`mapi_ai.rain.perfil.peak_profile`).
