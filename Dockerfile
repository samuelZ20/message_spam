FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pacotes do NLTK baixados no build (evita download ao subir o container)
RUN python -m nltk.downloader -d $NLTK_DATA stopwords punkt punkt_tab

# Código e dados
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY data/ data/
RUN mkdir -p models

# Treina o modelo durante o build (gera models/*.joblib e metrics.json)
RUN python -m src.train

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
