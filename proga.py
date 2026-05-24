# Создайте новый файл
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Определение языка текста")

tab1, tab2, tab3 = st.tabs(["Предсказание", "Статистика", "Справка"])

with tab1:
    user_input = st.text_area("Введите текст:")
    if st.button("Определить"):
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
        data = response.json()
        st.success(f"Язык: {data['language']} (Уверенность: {data['confidence']:.2f})")

with tab2:
    st.header("Статистика датасета")
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Всего записей: {len(df)}")
        st.dataframe(df.head())

with tab3:
    st.write("Команды API:")
    st.code("POST /predict - JSON: {'text': 'your text'}")
