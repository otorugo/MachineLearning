# Projetos de Machine Learning — ICMC/USP

Projetos da disciplina de Aplicações de Aprendizado de Máquina (ICMC/USP). Cada um cobre uma área diferente: dados tabulares e séries temporais, áudio e visão computacional. Todos seguem o mesmo cuidado metodológico: baseline de referência, conjunto de teste usado uma única vez, sementes fixas e uma discussão honesta das limitações.

| Projeto | Área | Técnicas | Resultado principal |
|---|---|---|---|
| [Previsão da produção de arroz com AutoML](previsao_colheita/) | Tabular / séries temporais | FLAML, XGBoost, Extra Trees | Identificação de *target leakage* em um artigo publicado. O AutoML reduziu o MAPE de 31,3% para 11,5% |
| [Identificação de idioma em áudio telefônico](classificação_idioma_audio/) | Áudio / sinais | Filtro passa-banda, MFCC, SVM RBF | 82,1% de acurácia entre português e espanhol, contra 55,5% do baseline |
| [Classificação de imagens no Fashion-MNIST](classificacao_imagens/) | Visão computacional | ResNet-50, ConvNeXt-Tiny, ViT-Small, transfer learning | 95,4% de acurácia (ConvNeXt), com 71% menos erros que o baseline |

## Destaques

- **Pensamento crítico sobre a literatura:** a [reanálise do estudo de produção de arroz](previsao_colheita/) mostra que a variável-alvo era uma identidade aritmética dos preditores. Por isso, o projeto propõe um cenário sem vazamento, que corresponde ao uso real do modelo.
- **Engenharia de atributos guiada pelo domínio:** no [projeto de áudio](classificação_idioma_audio/), o filtro e os parâmetros do MFCC foram escolhidos a partir da faixa de frequência da telefonia e da literatura de identificação de idioma.
- **Comparação justa de arquiteturas:** no [projeto de imagens](classificacao_imagens/), as três redes foram treinadas com o mesmo protocolo e comparadas em acurácia e em custo de treino e de inferência.

## Stack

Python · scikit-learn · PyTorch · timm · FLAML · XGBoost · librosa · SciPy · pandas · NumPy · Matplotlib · Google Colab / Kaggle

## Estrutura

```
.
├── previsao_colheita/            # AutoML para previsão de produção agrícola
├── classificação_idioma_audio/   # Identificação de idioma com MFCC + SVM
└── classificacao_imagens/        # Transfer learning com CNNs e ViT
```

Cada pasta tem o notebook executado e um README com a metodologia, os resultados e as instruções para rodar o projeto.

---

**Victor Hugo Oliveira da Silva** — ICMC/USP · victorsilva698@usp.br
