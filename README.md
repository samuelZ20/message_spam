# 📩 Detector de SPAM em SMS

> 👋 **Sobre este projeto**
>
> Este é um projeto de estudos: estou começando minha jornada em **Machine Learning** e quis colocar em prática, de ponta a ponta, o que venho aprendendo. A ideia foi passar por todas as etapas de um projeto real, desde explorar e limpar os dados, testar vários modelos e escolher o melhor, até disponibilizar o resultado numa API e numa interface simples.
>
> Ainda há muito a melhorar, e sugestões e feedbacks são muito bem-vindos! 🚀

Projeto de Machine Learning que classifica mensagens SMS (em inglês) como **spam** ou **ham** (mensagem legítima). Inclui análise exploratória, pipeline de treino, uma API REST com FastAPI e uma interface web com Streamlit.

## Modelos testados

No notebook `notebooks/exploracao.ipynb`, os textos foram transformados em features **TF-IDF** (3000 termos) e 10 classificadores do scikit-learn foram comparados, com 80% dos dados para treino e 20% para teste:

| Sigla    | Modelo                                   | Configuração                           |
|----------|------------------------------------------|----------------------------------------|
| SVC      | Support Vector Classifier                | `kernel="sigmoid"`, `gamma=1.0`        |
| KNN      | K-Nearest Neighbors                      | padrão                                 |
| NB       | Naive Bayes Multinomial                  | padrão                                 |
| DT       | Árvore de Decisão                        | `max_depth=5`                          |
| LR       | Regressão Logística                      | `solver="liblinear"`, `penalty="l1"`   |
| RF       | Random Forest                            | `n_estimators=50`                      |
| AdaBoost | AdaBoost                                 | `n_estimators=50`                      |
| Bagging  | Bagging                                  | `n_estimators=50`                      |
| ETC      | Extra Trees                              | `n_estimators=50`                      |
| GBDT     | Gradient Boosting                        | `n_estimators=50`                      |

As métricas usadas foram a **acurácia** (quanto o modelo acerta no geral) e a **precisão** (das mensagens marcadas como spam, quantas eram realmente spam). A precisão importa bastante aqui, porque marcar uma mensagem legítima como spam (falso positivo) é pior do que deixar passar um spam.

## Conclusões

- **SVC** e **Random Forest** tiveram as maiores acurácias (~97,58%).
- **Naive Bayes** teve **precisão de 100%**: nenhuma mensagem legítima foi marcada como spam.
- **Gradient Boosting**, **AdaBoost**, **Regressão Logística** e **Bagging** também foram bem, com acurácias entre 94,68% e 96,03%.
- Na análise exploratória, as mensagens de spam são, em média, **bem mais longas** que as legítimas (~138 contra ~70 caracteres), e o dataset é **desbalanceado** (~87% ham e ~13% spam). Por isso só a acurácia não basta para avaliar os modelos.
- O **SVC** foi escolhido para produção porque teve a melhor acurácia com uma precisão alta. Ele é treinado em `src/train.py`.

### Resultado do modelo final

Modelo: **SVC (kernel sigmoid)** sobre features **TF-IDF** (3000 termos), avaliado em 20% dos dados reservados para teste.

| Métrica   | Valor  |
|-----------|--------|
| Acurácia  | 97,58% |
| Precisão  | 96,69% |

As métricas são salvas em `models/metrics.json` a cada treino.

> **Um aprendizado no caminho:** no notebook, o TF-IDF foi ajustado com o dataset inteiro antes de separar treino e teste, então informação do teste "vazava" para o treino (*data leakage*). No `src/train.py` isso foi corrigido: a separação acontece **antes**, e o vetorizador aprende o vocabulário só com os dados de treino.

## Estrutura do projeto

```
msg_spam/
├── api/
│   └── main.py              # API FastAPI (/predict, /metrics, /status)
├── data/
│   └── raw/spam.csv         # Dataset SMS Spam Collection
├── frontend/
│   └── app.py               # Interface Streamlit
├── models/                  # Modelo, vetorizador e métricas gerados pelo treino
├── notebooks/
│   └── exploracao.ipynb     # Análise exploratória (EDA)
├── src/
│   ├── data.py              # Carregamento e limpeza do dataset
│   ├── preprocessing.py     # Limpeza de texto (NLTK)
│   └── train.py             # Treino, avaliação e salvamento do modelo
├── Dockerfile               # Imagem única usada pela API e pelo frontend
├── docker-compose.yml       # Sobe API + interface juntas
├── .dockerignore
└── requirements.txt
```

## Pipeline

1. **Dados** (`src/data.py`): lê o CSV, remove colunas vazias, codifica o rótulo (`ham=0`, `spam=1`) e remove duplicatas.
2. **Pré-processamento** (`src/preprocessing.py`): minúsculas → tokenização → mantém apenas tokens alfanuméricos → remove stopwords e pontuação → stemming (Porter).
3. **Treino** (`src/train.py`): divide treino/teste (80/20) **antes** de vetorizar, ajusta o TF-IDF só no treino, treina o SVC e salva `model.joblib`, `vectorizer.joblib` e `metrics.json` em `models/`.

## Como rodar

Há duas formas: com **Docker** (mais simples) ou **localmente** com Python.

### Com Docker

Pré-requisito: [Docker](https://docs.docker.com/get-docker/) com Docker Compose.

Na raiz do projeto:

```bash
docker compose up --build
```

Isso constrói uma única imagem (`msg-spam`) e sobe dois containers:

| Serviço    | Endereço                                        | Descrição                  |
|------------|-------------------------------------------------|----------------------------|
| `api`      | `http://localhost:8000` (docs em `/docs`)       | API FastAPI                |
| `frontend` | `http://localhost:8501`                         | Interface Streamlit        |

Detalhes da imagem:

- Baseada em `python:3.13-slim`.
- Os pacotes do NLTK (`stopwords`, `punkt`, `punkt_tab`) são baixados **durante o build**, então o container não precisa de internet ao subir.
- O modelo é **treinado durante o build** (`python -m src.train`), por isso a pasta `models/` local é ignorada pelo `.dockerignore`, assim como `venv/` e `notebooks/`.
- No Docker, o frontend acessa a API pela variável de ambiente `URL_API=http://api:8000` (definida no `docker-compose.yml`). Fora do Docker, o padrão é `http://127.0.0.1:8000`.

Comandos úteis:

```bash
docker compose up -d --build   # sobe em segundo plano
docker compose logs -f         # acompanha os logs
docker compose down            # para e remove os containers
```

> Para retreinar o modelo (por exemplo, após mudar `src/` ou o dataset), basta reconstruir a imagem com `docker compose up --build`.

### Localmente (sem Docker)

#### 1. Instalar dependências

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

> Na primeira execução, o NLTK baixa automaticamente os pacotes `stopwords` e `punkt`.

#### 2. Treinar o modelo

Execute a partir da raiz do projeto:

```bash
python -m src.train
```

#### 3. Subir a API

```bash
uvicorn api.main:app --reload
```

A API fica em `http://127.0.0.1:8000` e a documentação interativa em `http://127.0.0.1:8000/docs`.

#### 4. Abrir a interface

Em outro terminal (com a API rodando):

```bash
streamlit run frontend/app.py
```

## Endpoints da API

| Método | Rota       | Descrição                                           |
|--------|------------|-----------------------------------------------------|
| GET    | `/`        | Mensagem de boas-vindas                             |
| GET    | `/status`  | Status da API e se o modelo está carregado          |
| POST   | `/predict` | Classifica uma mensagem                             |
| GET    | `/metrics` | Retorna as métricas do último treino                |

Exemplo de requisição:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free ticket. Call now!"}'
```

Resposta:

```json
{
  "texto": "Congratulations! You won a free ticket. Call now!",
  "spam": true,
  "classe": "spam"
}
```

Se o modelo ainda não foi treinado, `/predict` retorna `503` e `/metrics` retorna `404`.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI / Kaggle): cerca de 5,5 mil mensagens SMS em inglês rotuladas como `ham` ou `spam`.

## Tecnologias

Python · pandas · scikit-learn · NLTK · FastAPI · Uvicorn · Streamlit · Docker · Jupyter · matplotlib · seaborn · wordcloud
