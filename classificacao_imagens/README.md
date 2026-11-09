# Classificação de Imagens no Fashion-MNIST: ResNet × ConvNeXt × ViT

Comparação de três gerações de arquiteturas de visão computacional na classificação de roupas do **Fashion-MNIST**: uma CNN clássica (**ResNet-50**), uma CNN moderna (**ConvNeXt-Tiny**) e um Vision Transformer (**ViT-Small/16**). Os três modelos partem de pesos pré-treinados no ImageNet e passam por **fine-tuning completo com o mesmo protocolo**. A comparação mede acurácia e custo computacional, e uma regressão logística serve de baseline.

> Projeto da disciplina de Machine Learning — ICMC/USP.
> Notebook completo: [`classificacao_imagens.ipynb`](classificacao_imagens.ipynb) · [versão no Kaggle](https://www.kaggle.com/code/otorugo/notebook956ceb4a49)

## Destaques

- **A ConvNeXt-Tiny foi a mais precisa:** 95,43% de acurácia no teste, com 71% menos erros que o baseline e 43% menos que a ResNet-50.
- **O ViT-Small/16 teve o melhor custo-benefício:** 94,72% de acurácia, sendo o mais rápido no treino (3,8 min) e na inferência (0,25 ms/img, 3,7× mais rápido que a ConvNeXt).
- **Comparação justa:** mesmos hiperparâmetros, mesmos dados, mesma semente e pesos pré-treinados só no ImageNet-1k para os três modelos.
- **O ganho do transfer learning ficou nas classes difíceis:** o recall de *Shirt* subiu de 56,7% para 84,9%, e o de *Coat*, de 75,5% para 95,0%.
- **Baseline validado com a literatura:** a regressão logística teve 84,25% de acurácia, contra 84,2% no artigo original do Fashion-MNIST.

## Dados

- **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist):** 70.000 imagens 28×28 em tons de cinza, em 10 classes balanceadas.
- **Divisão:** 54.000 imagens de treino e 6.000 de validação (estratificada, seed 42), mais as 10.000 de teste originais, usadas uma única vez.
- **O que torna a tarefa difícil:** classes visualmente parecidas, como *T-shirt*, *Shirt*, *Pullover* e *Coat*, ou *Sneaker* e *Ankle Boot*.

## Metodologia

**Pré-processamento (feito em lote, na GPU):** normalização para [0, 1] → flip horizontal (só no treino) → resize para 112×112 → 1 canal replicado em 3 → normalização com as estatísticas de cada modelo.

| Modelo | Pesos (timm) | Parâmetros |
|---|---|---:|
| ResNet-50 | `resnet50.a1_in1k` | 23,5 M |
| ConvNeXt-Tiny | `convnext_tiny.fb_in1k` | 27,8 M |
| ViT-Small/16 | `vit_small_patch16_224.augreg_in1k` | 21,6 M |

**Treinamento (idêntico para os três):** AdamW (weight decay 0,05), learning rate 10⁻⁴ com OneCycle, 5 épocas, batch 128, precisão mista (fp16). No teste, foi usado o checkpoint com a melhor acurácia de validação.

## Resultados

Conjunto de teste (n = 10.000):

| Modelo | Acc. val. | Acc. teste | F1 macro | Treino (min) | Inferência (ms/img) |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,861 | 0,843 | 0,842 | 1,6 | 0,004* |
| ResNet-50 | 0,932 | 0,920 | 0,920 | 5,6 | 0,43 |
| **ConvNeXt-Tiny** | **0,957** | **0,954** | **0,954** | 6,4 | 0,94 |
| ViT-Small/16 | 0,950 | 0,947 | 0,947 | **3,8** | **0,25** |

<sub>*Medido na CPU, por isso não é diretamente comparável com as redes. Todas as redes rodaram em uma GPU NVIDIA T4.</sub>

**O que os resultados mostram:**

- **Não há vencedor absoluto entre CNNs e Transformers.** O ViT superou com folga a ResNet, mas ficou um pouco atrás da ConvNeXt, que é uma CNN com escolhas de projeto modernas. Na prática, a ConvNeXt compensa quando cada ponto de acurácia importa, e o ViT, quando tempo e custo pesam mais.
- **O ViT foi rápido porque a sequência é curta:** com entrada de 112×112 e patches de 16×16, cada imagem vira só 49 tokens, e a atenção fica barata. Já as convoluções depthwise 7×7 da ConvNeXt aproveitam pior a GPU.
- **Os erros se concentram nas roupas de parte de cima.** Todos os modelos confundem *Shirt* com *T-shirt* e *Coat*. Vários erros são ambíguos até para uma pessoa olhando a imagem em 28×28.
- **Sobreajuste sob controle:** o ViT chegou a 99,4% de acurácia no treino contra 94,95% na validação, mas a escolha do melhor checkpoint evitou que isso afetasse o teste. A ResNet ainda não tinha convergido em 5 épocas.

**Limitações e próximos passos:** cada modelo foi treinado com uma única semente e sem ajuste de learning rate por arquitetura. Os caminhos naturais são rodar várias sementes, treinar por mais épocas com mais data augmentation e usar a resolução de 224×224.

## Stack

Python 3.13 · PyTorch · timm · scikit-learn · NumPy · Matplotlib · Kaggle (GPU NVIDIA T4)

## Como executar

O jeito mais simples é abrir o notebook no [Kaggle](https://www.kaggle.com/code/otorugo/notebook956ceb4a49), com a GPU ativada. Para rodar localmente:

```bash
pip install torch timm scikit-learn numpy matplotlib
```

Baixe os arquivos `idx3-ubyte` e `idx1-ubyte` do [repositório oficial do Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) e ajuste no notebook o caminho dos dados (hoje ele aponta para `/kaggle/input/...`).

## Referências

- Xiao, H. et al. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747, 2017.
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Liu, Z. et al. *A ConvNet for the 2020s.* CVPR 2022.
- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.

---

**Victor Hugo Oliveira da Silva** — ICMC/USP
