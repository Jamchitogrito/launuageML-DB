import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Определение языка текста")

tab1, tab2, tab3 = st.tabs(["Предсказание", "Статистика", "Справка"])

with tab1:
    user_input = st.text_area("Введите текст:")
    if st.button("Определить"):
        if not user_input.strip():
            st.warning("Введите текст")
        else:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": user_input}
                )
                data = response.json()
                st.success(f"Язык: **{data['language']}**")
                st.metric("Уверенность модели", f"{data['probability'] * 100:.1f}%")
                if data['probability'] < 0.8:
                    st.warning("Модель не уверена в результате (< 80%)")
                st.subheader("Топ-5 языков:")
                top5 = list(data['probabilities'].items())[:5]
                for lang, prob in top5:
                    st.progress(prob, text=f"{lang}: {prob*100:.1f}%")
            except Exception as e:
                st.error(f"Ошибка: {e}")

with tab2:
    st.header("Статистика датасета")

    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Общая информация")
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего текстов", len(df))
        col2.metric("Языков", df['Language'].nunique())
        col3.metric("Средняя длина", f"{df['Text'].astype(str).apply(len).mean():.0f} симв.")
        lang_counts = df['Language'].value_counts()
        df['text_len'] = df['Text'].astype(str).apply(len)

        st.subheader("Первые записи")
        st.dataframe(df.head())

        st.subheader("Распределение по языкам")
        lang_counts = df['language'].value_counts()
        st.bar_chart(lang_counts)

        st.subheader("Длина текстов")
        df['text_len'] = df['text'].astype(str).apply(len)
        fig, ax = plt.subplots()
        ax.hist(df['text_len'], bins=30, color='steelblue')
        ax.set_xlabel("Длина текста (символы)")
        ax.set_ylabel("Количество")
        st.pyplot(fig)

with tab3:
    st.header("Справка")

    st.subheader("Как пользоваться")
    st.write("""
    1. Перейди на вкладку **Предсказание**
    2. Введи текст в поле
    3. Нажми кнопку **Определить**
    4. Получи результат с уверенностью модели
    """)

    st.subheader("Команды API")
    st.code("POST /predict\nBody: {\"text\": \"your text\"}", language="bash")
    st.code("POST /predict/batch\nBody: {\"texts\": [\"text1\", \"text2\"]}", language="bash")
    st.code("GET /health", language="bash")
    st.code("GET /languages", language="bash")

    st.subheader("Пример ответа API")
    st.json({
        "language": "en",
        "language_full": "English",
        "probability": 0.88,
        "top_languages": {
            "en": 0.88,
            "de": 0.07,
            "fr": 0.05
        },
        "is_confident": True
    })