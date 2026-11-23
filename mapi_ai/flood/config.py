"""
Alerta de alagamento por regressão + física (proposta de 20/11). Configuração única, fixada ANTES de qualquer teste.

Caminho: chuva medida (rede Cemaden/PED) -> rain_v1 (quantis de chuva em 1, 3 e 6 h, validado) -> balde M1a a 10 min
(físico, sem treino, não calibrado) -> regra fixa sobre a lâmina. Nenhum artefato aqui é treinado com rótulo de
alagamento, sintético ou real.
"""
VERSION = "flood_reg_v1"

# Lâmina que define alagamento no MAPi: 15 cm (docs/PARAMETROS-MODELO-FISICO.md do projeto-mapi-api, "h_crit = 15 cm",
# faixa 10 a 30 só como sensibilidade). Definição física, congelada; não é ajustada para acertar frequência.
HCRIT_CM = 15.0

HORIZONS = (1, 3, 6)          # horas, os mesmos do rain_v1
SCENARIOS = ("q50", "q95")    # provável e pessimista (para não perder alagamento)
DT_MIN = 10.0

LEVELS = ("ALERTA", "ATENCAO", "SEM_ALERTA", "SEM_DADO")
RULE = {
    "ALERTA": "a lâmina máxima do cenário q50 (provável) chega a 15 cm no horizonte",
    "ATENCAO": "só a lâmina máxima do cenário q95 (pessimista) chega a 15 cm no horizonte",
    "SEM_ALERTA": "nenhum dos dois cenários chega a 15 cm no horizonte (não quer dizer sem risco)",
    "SEM_DADO": "falta entrada para calcular; nunca vira SEM_ALERTA",
}
RULE_TEXT = ("ALERTA se a lâmina do cenário q50 chega a 15 cm; ATENÇÃO se só a do cenário q95 chega; SEM ALERTA caso "
             "contrário. Limiar e regra fixados antes do teste (20/11).")

# Como a chuva prevista entra no balde: o rain_v1 prevê o ACUMULADO em (t, t+h]; sem previsão da distribuição no
# tempo, o acumulado é espalhado por igual nos passos de 10 min do horizonte. Declarado: isso dilui o pico nos
# horizontes de 3 e 6 h (limitação para o relatório).
DISTRIBUTION = "chuva prevista em (t, t+h] distribuída por igual nos passos de 10 min do horizonte"
# A lâmina máxima do horizonte inclui a lâmina de agora (o alerta "já agora" não some porque a galeria escoa).
SHEET_MAX = "máximo entre a lâmina de agora e a lâmina prevista a cada 10 min no horizonte"

DECOMP_HISTORY_STEPS = 18     # 3 h medidas antes de agora na decomposição

# Motivos de SEM_DADO (texto curto para a tela). Nenhum vira SEM_ALERTA.
REASONS = {
    "CHUVA_AUSENTE": "sem leitura de chuva válida perto do ponto",
    "PREVISAO_INDISPONIVEL": "previsão de chuva (rain_v1) fora do ar",
    "SEM_TERRENO": "sem terreno do ponto",
    "TERRENO_EM_REVISAO": "terreno do ponto em revisão",
    "SERIE_INCOMPLETA": "série de chuva de 10 min com lacuna",
    "MARE_AUSENTE": "sem maré prevista para algum passo (refluxo ligado no nível)",
}

# Fatores por ponto (ADR 0030): refluxo de maré e limpeza (Central 156), cada um com um modo.
# OFF: não calcula. INFO: calcula e devolve a contribuição, sem mudar o nível. NIVEL: muda o nível.
FACTOR_MODES = ("OFF", "INFO", "NIVEL")
HORIZON_STEPS = int(max(HORIZONS) * 60 / DT_MIN)   # 36 passos de 10 min: a maré vai até o fim do maior horizonte
# Cenário de viés do MDT (ADR 0030, item 2): MDT - RN = -0,369 m, mediana em 17 RRNN em terra
# (projeto-mapi-api/docs/extratos/referencia-vertical.md:101). A soleira no NR sobe 0,369 m. Só sensibilidade.
MDT_BIAS_SCENARIO_M = -0.369
FACTOR_NOTE = ("refluxo: capacidade x clamp((soleira_NR + D - maré_NR)/D, 0, 1), soleira_NR = soleira_MDT - viés_MDT + Δ "
               "(ADR 0009 emenda 3); limpeza: P(lâmina >= 15 cm) = (1 - p_E)·1[livre] + p_E·1[entupida] "
               "(docs/PRE-REGISTRO-ENTUPIMENTO-156.md)")

# Contrafactual: busca por bissecção da chuva extra em 1 h (somada ao cenário q50 de 1 h) que leva a lâmina a 15 cm.
CF_MAX_MM = 300.0
CF_TOL_MM = 0.01

# Sensibilidade (Etapa 3.4): faixas registradas antes, em docs/PARAMETROS-MODELO-FISICO.md e no balde a 10 min.
# f: fração da capacidade da galeria, "sem fonte", 0 (obstruída) a 1 (projeto) em passos de 20%.
# C: 0,70 a 0,95 (central 0,86). i_T: IDF de Silva e Araújo com TR 2 a 10 anos e Te 5 a 15 min.
SENSITIVITY = {"f": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "C": (0.70, 0.86, 0.95),
               "tr_anos": (2.0, 5.0, 10.0), "te_min": (5.0, 10.0, 15.0)}
CENTRAL = {"f": 1.0, "C": 0.86, "tr_anos": 5.0, "te_min": 10.0}
SENSITIVITY_SOURCES = {
    "f": "sem fonte (cadastro da galeria não é público; faixa de 0 a projeto)",
    "C": "URB Recife (2018) e faixa do documento de parâmetros",
    "tr_anos": "Silva e Araújo (2013), Eq. 16; TR é escolha de projeto",
    "te_min": "tempo de entrada típico de microdrenagem; central 10 min",
}

EXPERIMENTAL_NOTE = ("experimental: a validação externa (Etapa 2.3, dias da Defesa Civil) ainda não foi aprovada; o "
                     "balde não está calibrado")
