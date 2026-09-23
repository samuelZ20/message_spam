import os
import requests
import streamlit as st

# No Docker a URL vem da variável de ambiente (http://api:8000)
URL_API = os.getenv("URL_API", "http://127.0.0.1:8000")

st.title("📩 Detector de SPAM")
st.write("Digite uma mensagem (em inglês) e descubra se ela é spam.")

# Caixa de texto
texto = st.text_area("Mensagem:", height=150)

# Botão
if st.button("Analisar"):
    if texto.strip() == "":
        st.warning("Digite uma mensagem antes de analisar.")
    else:
        try:
            resposta = requests.post(f"{URL_API}/predict", json={"text": texto})
            resultado = resposta.json()

            # Acurácia do modelo (medida no conjunto de teste)
            acuracia = None
            resp_metricas = requests.get(f"{URL_API}/metrics")
            if resp_metricas.status_code == 200:
                acuracia = resp_metricas.json().get("accuracy")
            texto_acuracia = f" (acurácia do modelo: {acuracia:.2%})" if acuracia is not None else ""

            if resposta.status_code != 200:
                st.error(f"Erro da API ({resposta.status_code}): {resultado.get('detail')}")
            elif resultado["spam"]:
                st.error(f"🚨 Essa mensagem é SPAM!{texto_acuracia}")
            else:
                st.success(f"✅ Essa mensagem parece legítima (HAM).{texto_acuracia}")

        except requests.exceptions.ConnectionError:
            st.error("Não consegui conectar na API. Ela está rodando?")