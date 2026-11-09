# Previsão da Produção de Arroz na América do Sul com AutoML

Reanálise crítica de um estudo publicado (Grion et al., SBSI 2026) que prevê a produção anual de arroz em 12 países sul-americanos. O projeto reproduz o experimento original, **identifica um vazamento de dados (target leakage) não discutido pelos autores** e compara a configuração manual de XGBoost com uma busca automatizada de modelos via **FLAML**.

> Projeto da disciplina de Machine Learning — ICMC/USP.
> Notebook completo: [`previsão_produção_arroz.ipynb`](previsão_produção_arroz.ipynb) · [versão no Colab](https://colab.research.google.com/drive/10nXDPiV8XljL7Xt5gw06amHY0oB6eodT?usp=sharing)

## Destaques

- **Reprodução validada:** MAPE de 9,14% na réplica, dentro da faixa reportada no artigo (6,12%–12,22%).
- **Target leakage encontrado:** a variável-alvo é uma identidade aritmética dos preditores — `Produção = Área × Rendimento / 1000` (erro médio de 0,0005%, correlação 1,000000). Além disso, área e rendimento do ano previsto não estão disponíveis no momento real da previsão.
- **AutoML reduziu o erro em quase 2/3** no modelo global (MAPE de 31,29% → 11,50%), escolhendo um Extra Trees com apenas 4 árvores em vez do XGBoost de 200 árvores do artigo.
- **Resultado honesto sobre limitações:** sem o vazamento, o AutoML **não** superou a configuração manual — e piorou quando havia apenas 9 registros de treino por modelo.

## Problema e dados

- **Fonte:** FAOSTAT (produção, área colhida, rendimento) + NASA POWER (temperatura média, precipitação).
- **Tamanho:** 156 registros — 12 países × 13 anos (2011–2023), sem valores faltantes.
- **Divisão cronológica:** treino até 2020, teste em 2021–2023 (36 observações). Divisão aleatória foi evitada por permitir uso de informação futura.

## Metodologia

Foram avaliados **2 cenários × 2 estratégias**, sempre comparando o XGBoost com os hiperparâmetros do artigo contra o modelo escolhido pelo AutoML:

| Cenário | Atributos | Treino |
|---|---|---|
| **A — Réplica fiel** | Mesmos do artigo, referentes ao próprio ano previsto (com vazamento) | 120 registros |
| **B — Sem vazamento** | Apenas informação do ano anterior (lag de 1 ano) — uso operacional real | 108 registros |

- **Estratégias:** um modelo por país (desenho original) vs. um modelo global com o país como atributo categórico.
- **AutoML:** FLAML otimizando MAPE diretamente; orçamento de 90 s (global) e 15 s por país.
- **Métricas:** MAPE (principal), MAE e RMSE.

## Resultados

MAPE (%) no conjunto de teste (2021–2023):

| Cenário | Estratégia | XGBoost manual | AutoML | Variação |
|---|---|---:|---:|---:|
| A (réplica) | Por país | 9,14 | **8,29** | −0,85 |
| A (réplica) | Global | 31,29 | **11,50** | **−19,79** |
| B (lag 1) | Por país | **13,79** | 19,86 | +6,07 |
| B (lag 1) | Global | **14,11** | 14,88 | +0,77 |

**Principais conclusões:**

1. **O bom desempenho original depende do vazamento + modelagem por país.** Dentro de um país, `área × rendimento` é quase linear, o que explica até os erros < 1% da Regressão Linear no artigo. No modelo global (produções de 9,5×10⁴ a 1,3×10⁷ t), essa aproximação deixa de valer.
2. **AutoML corrige má calibração de hiperparâmetros.** 200 árvores com learning rate 0,08 é uma receita para bases grandes, não para 120 registros. O FLAML encontrou que 4 estimadores bastavam.
3. **AutoML não cria informação.** No cenário sem vazamento, o desempenho é limitado pelos dados. Com 9 registros por país e validação cruzada de 3 folds, a busca sobreajustou — a piora veio quase toda de um único país (Venezuela: ~8% → ~66%).
4. **Adaptação por série:** nos modelos por país, o AutoML escolheu algoritmos diferentes conforme o regime (Extra Trees predominou no Cenário A; XGBoost no B).

> **Em síntese:** a automação corrige configuração, não substitui dados.

## Stack

Python 3.12 · FLAML 2.6.0 · XGBoost · scikit-learn · pandas · NumPy · Matplotlib · Google Colab (seed 42)

## Como executar

Abra o notebook no [Google Colab](https://colab.research.google.com/drive/10nXDPiV8XljL7Xt5gw06amHY0oB6eodT?usp=sharing) ou localmente:

```bash
pip install flaml xgboost scikit-learn pandas numpy matplotlib
jupyter notebook "previsão_produção_arroz.ipynb"
```

O notebook requer o arquivo `dataset_NASA_FAOSTAT.csv`, disponibilizado pelos autores do estudo original.

## Referências

- Grion, R. et al. *Predictive Modeling of Rice Production in South America Using Machine Learning.* SBSI 2026, p. 577–596.
- Wang, C. et al. *FLAML: A Fast and Lightweight AutoML Library.* MLSys 2021.
- Kaufman, S. et al. *Leakage in Data Mining: Formulation, Detection, and Avoidance.* KDD 2011.

---

**Victor Hugo Oliveira da Silva** — ICMC/USP
