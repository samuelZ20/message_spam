import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import transformar_texto

#1) Carregar o modelo e vetorizador

CAMINHO_MODELO = "models/model.joblib"
CAMINHO_VETORIZADOR = "models/vectorizer.joblib"
CAMINHO_METRICAS = "models/metrics.json"

if os.path.exists(CAMINHO_MODELO) and os.path.exists(CAMINHO_VETORIZADOR):
    modelo = joblib.load(CAMINHO_MODELO)
    tfid = joblib.load(CAMINHO_VETORIZADOR)
    modelo_carregado = True
else:
    modelo = None
    tfid = None
    modelo_carregado = False


#2) Formato do que a API recebe
class Mensagem(BaseModel):
    text: str

app = FastAPI(title="SPAM Message API", description="API para classificar mensagens como SPAM ou HAM")

#3) Endpoints

@app.get("/")
def home():
    return {"mensagem": "Bem vindo ao SPAM Message API!"}

@app.get("/status")
def status():
    return {"status": "API is running!", "modelo_carregado": modelo_carregado}


@app.post("/predict")
def predict(mensagem: Mensagem):
    if not modelo_carregado:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Treine o modelo primeiro.")

    texto_limpo = transformar_texto(mensagem.text)
    vetor = tfid.transform([texto_limpo])
    previsao = int(modelo.predict(vetor)[0])

    return {
        "texto" : mensagem.text,
        "spam" : previsao ==1,
        "classe" : "spam" if previsao == 1 else "ham",
    }

@app.get("/metrics")
def metrics():
    try:
        with open(CAMINHO_METRICAS, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Métricas não encontradas. Treine o modelo primeiro.")
    