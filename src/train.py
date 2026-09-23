import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

from src.data import carregar_dados
from src.preprocessing import transformar_texto

#1) Carregar os dados
df = carregar_dados()

# 2) Pré-processar o texto
df['transformed_text'] = df['text'].apply(transformar_texto)

# 3) Separar treino e teste ANTES de vetorizar
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed_text'], df['target'], test_size=0.20, random_state=2
)

# 4) Vetorizar: o TF-IDF aprende o vocabulário só com o treino
tfid = TfidfVectorizer(max_features=3000)
X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)

# 5) Treinar o modelo
modelo = SVC(kernel="sigmoid", gamma=1.0)
modelo.fit(X_train, y_train)

# 6) Avaliar
y_pred = modelo.predict(X_test)
metricas = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
}
print("Accuracy: ", metricas["accuracy"])
print("Precision:", metricas["precision"])

# 7) Salvar o vetorizador e o modelo
joblib.dump(tfid, "models/vectorizer.joblib")
joblib.dump(modelo, "models/model.joblib")
with open("models/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2)
print("Vetorizador, modelo e métricas salvos na pasta models/")