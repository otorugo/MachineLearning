# Identificação de Idioma em Áudio Telefônico (Português × Espanhol)

Classificador que identifica se uma gravação telefônica está em **português europeu (pt-PT)** ou **espanhol (es-ES)**, usando processamento de sinais clássico (filtro passa-banda + MFCC) e um **SVM com kernel RBF**. Os dados vêm do conjunto público **MINDS-14**, com ligações reais para um sistema de e-banking.

> Projeto da disciplina de Machine Learning — ICMC/USP.
> Notebook completo: [`classificação_idioma_audio.ipynb`](classificação_idioma_audio.ipynb) · [versão no Kaggle](https://www.kaggle.com/code/otorugo/t3-de-aplica-es-de-aprendizado-de-m-quina)

## Destaques

- **Acurácia de 82,1% e F1 macro de 0,819** no teste (218 gravações). Um classificador que sempre responde a classe majoritária fica em 55,5% e 0,357.
- **Pipeline sem vazamento de dados:** padronização e SVM no mesmo `Pipeline` do scikit-learn, Grid Search só no treino e conjunto de teste usado uma única vez.
- **Atributos guiados pelo domínio:** filtro na faixa da telefonia (300–3400 Hz) e parâmetros de MFCC baseados na literatura de identificação de idioma em fala telefônica.
- **A otimização valeu a pena:** o F1 de validação subiu de 0,790 (valores padrão) para 0,870.
- **Resultado comparável à literatura:** erro de 17,9%, na mesma faixa dos modelos acústicos (GMM) de Zissman (1996), que erraram cerca de 20%.

## Dados

- **Fonte:** [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) (Hugging Face), subconjuntos pt-PT e es-ES.
- **Tamanho:** 1.090 gravações, sendo 604 em pt-PT (55,4%) e 486 em es-ES (44,6%).
- **Formato:** 8 kHz, mono, com duração de 2 s a 68 s (mediana de 10,6 s).
- **Divisão:** 80% treino (872) e 20% teste (218), estratificada por idioma.

## Pipeline

```
Áudio 8 kHz → Filtro passa-banda 300–3400 Hz → MFCC 1–12 → média + desvio padrão → StandardScaler → SVM (RBF)
```

1. **Filtro passa-banda:** Butterworth de 4ª ordem com fase zero (`sosfiltfilt`). Ele coloca todas as gravações na mesma faixa de frequências, independentemente do aparelho ou do canal. O filtro atenua cerca de 25 dB em 200 Hz e 50 dB em 4 kHz.
2. **MFCC:** janela de 20 ms, passo de 10 ms, FFT de 256 pontos, 40 filtros mel e 13 coeficientes. O MFCC 0 foi descartado porque mede o volume da gravação, não o idioma.
3. **Vetor fixo de 24 atributos:** média e desvio padrão de cada coeficiente ao longo do tempo, o que permite comparar gravações de qualquer duração.
4. **SVM RBF:** `C` e `γ` escolhidos por Grid Search (20 combinações) com validação cruzada estratificada de 5 partições, otimizando o F1 macro para compensar o desbalanceamento entre os idiomas.

## Resultados

| Modelo | Acurácia | F1 macro |
|---|---:|---:|
| Classe majoritária (pt-PT) | 0,555 | 0,357 |
| **SVM otimizado** (C = 100, γ = 0,1) | **0,821** | **0,819** |

| Idioma | Precision | Recall | F1 | Gravações |
|---|---:|---:|---:|---:|
| es-ES | 0,796 | 0,804 | 0,800 | 97 |
| pt-PT | 0,842 | 0,835 | 0,838 | 121 |

**O que os resultados mostram:**

- **Os erros estão equilibrados:** 19 gravações em espanhol foram classificadas como português e 20 em português como espanhol. Mesmo com mais exemplos de pt-PT, o modelo não passou a favorecer esse idioma.
- **A queda da validação (0,870) para o teste (0,819)** é esperada. Escolher a melhor entre 20 combinações superestima o resultado da validação, e o erro padrão da acurácia com 218 gravações é de cerca de 2,6 p.p.
- **O limite desta abordagem:** média e desvio do MFCC ignoram a ordem dos sons. Na literatura, sistemas que modelam sequências de fonemas erram entre 6% e 10%. Esse é o próximo passo natural para melhorar o modelo.

## Stack

Python 3.12 · librosa 0.11 · scikit-learn 1.6 · SciPy · NumPy · pandas · Hugging Face `datasets` · Kaggle (seed 42)

## Como executar

Abra o notebook no [Kaggle](https://www.kaggle.com/code/otorugo/t3-de-aplica-es-de-aprendizado-de-m-quina) ou rode localmente:

```bash
pip install librosa scikit-learn scipy numpy pandas matplotlib soundfile "datasets<4"
```

O dataset é baixado automaticamente com `load_dataset("PolyAI/minds14", "all", split="train")`.

## Referências

- Gerz, D. et al. *Multilingual and Cross-Lingual Intent Detection from Spoken Data.* EMNLP 2021.
- Zissman, M. A. *Comparison of Four Approaches to Automatic Language Identification of Telephone Speech.* IEEE TSAP, 1996.
- Davis, S.; Mermelstein, P. *Comparison of parametric representations for monosyllabic word recognition.* IEEE TASSP, 1980.

---

**Victor Hugo Oliveira da Silva** — ICMC/USP
