import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_css():
    st.markdown("""
    <style>
    div[data-testid="stAppViewContainer"] {
        background-color: #f0f8ff; 
        background-image: url('https://img.freepik.com/premium-vector/abstract-dark-background-triangles-shades-black-gray-colors_444390-4195.jpg');  /* Картинка фона */
        h1 {
            color: #FF6347 !important;
            font-size: 50px!important;
        }
        h3 {
            color: orange !important;
        }  
        button {
        background-color: orange !important;  
        color: white;
        font-size: 50px;     
    }
        
    </style>
    """, unsafe_allow_html=True)

load_css()

if 'area' not in st.session_state:
    st.session_state.area = 0

st.title('Предсказать стоимость квартиры в зависимости от площади')

with st.expander("Описание проекта"):
    st.write('''Прогнозирование стоимости квартиры на основе регрессионной модели. Введите площадь квартиры, и мы предскажем её стоимость.''')

with st.container():
    st.subheader("Введите площадь квартиры")
    st.slider("Площадь квартиры (м²)", min_value=1, max_value=400, key='area', step=1, help="Выберите площадь квартиры для прогнозирования стоимости.")
    st.write(f"Вы выбрали площадь: {st.session_state.area} м²")

model_file_path = "project_house (1).sav"
reg = pickle.load(open(model_file_path, 'rb'))

def predict_price():
    input_dataframe = pd.DataFrame({
        'area': [st.session_state.area],
    })
    prediction = reg.predict(input_dataframe)
    return prediction[0]

def reset_input():
    st.session_state.area = 0

def exit_app():
    st.session_state.clear()

st.button("Сбросить", on_click=reset_input)

if st.button('Предсказать'):
    prediction = predict_price()
    st.write("Предсказанная стоимость квартиры:")
    st.metric(label="Цена", value=f"{prediction:.2f} ₽", delta="")
else:
    st.write("Ожидание данных для прогнозирования...")

import streamlit as st

def exit_app():
    st.session_state.exit_flag = True

if "exit_flag" not in st.session_state:
    st.session_state.exit_flag = False

if st.session_state.exit_flag:
    st.write("Вы вышли из приложения. Пожалуйста, перезагрузите страницу, чтобы начать заново.")
else:
    if st.button("Выйти", key="exit"):
        exit_app()
        st.write("Вы вышли из приложения. Пожалуйста, перезагрузите страницу, чтобы начать заново.")



