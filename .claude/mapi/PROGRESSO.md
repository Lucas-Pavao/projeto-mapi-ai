# Progresso — projeto-mapi-ai (IN1165)

Estado geral da revisão: `projeto-mapi-api/.claude/mapi/PROGRESSO.md`. Aqui, só as etapas do projeto final IN1165.
Contrato de entrada: `projeto-mapi-api/docs/handoff/etapa0.json`.

| Etapa | Status | Commit | Notas |
|---|---|---|---|
| 1 Dataset | concluida | (este) | flood_scenario_labels 0 linhas; CKAN 225 ocorrências / 18 dias / 0 com hora; 0 pontos-dia a 500 m (1 km: 6; 2 km: 45) → Plano B. Sintético: 5 000 horas reais (estratos da tabela de 4 408 eventos Cemaden), 7,7% positivos, fórmula em `reports/synthetic_ground_truth.md`. `load_dataset()` exige `--synthetic`. Nada gravado no banco. |

## Decisões
| Data | Decisão | Por quê |
|---|---|---|
| 2026-11-15 | Plano B | 0 positivos reais a 500 m, nenhum com hora (piso 30) |
| 2026-11-15 | Features e chuva reais; só o rótulo é sintético | calibração pelas distribuições reais sem inventar números |
| 2026-11-15 | Coeficientes do gerador (6; 10; 2; 6) em vez de (1,5; 2,5; 0,8; 3,0) | com os primeiros a hora seca tinha 4,3% de positivos (rótulo ≈ ruído); escolhido antes de treinar, só pela taxa por estrato: seco 0,5%, FORTE 28% |
| 2026-11-15 | Dados extraídos versionados em `data/in1165/` com SHA-256 | o notebook precisa rodar em ambiente limpo, sem o banco |
