import streamlit as st
import joblib
import os

st.title("Classificador de sentimentos")

texto = st.text_input("Digite um Tweet:")

modelo_path = "model.joblib"
vetor_path = "vectorizer.joblib"

if os.path.exists(modelo_path) and os.path.exists(vetor_path):
    model = joblib.load(modelo_path)
    vectorizer = joblib.load(vetor_path)

    if st.button("Analisar"):
        if texto.strip():
            texto_vetor = vectorizer.transform([texto])
            pred = model.predict(texto_vetor)[0]
            st.write(f"Sentimento: {pred}")
        else:
            st.warning("Por favor, digite um texto para análise.")
else:
    st.error(
        "Modelo ou vetor não foram encontrados. Certifique-se que os arquivos mode e vectorizer estão na raiz do projeto."
    )
