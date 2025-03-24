import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_data
def load_historical_data():
    return pd.read_csv('./vladislav_work/last_chance_to_made_app/data/processed_data.csv', parse_dates=['dt'])

def prepare_input(data, date):
    latest = data[data['dt'] <= date].tail(4)
    features = {
        'lag_1': latest.iloc[-1]['Цена на арматуру'],
        'lag_2': latest.iloc[-2]['Цена на арматуру'],
        'lag_3': latest.iloc[-3]['Цена на арматуру'],
        'lag_4': latest.iloc[-4]['Цена на арматуру'],
        'rolling_mean_4': latest['Цена на арматуру'].mean(),
        'year': date.year,
        'month': date.month,
        'week_of_year': date.isocalendar().week
    }
    return pd.DataFrame([features])

st.title('Оптимизация закупок арматуры')
st.write("Рекомендация системы по количеству недель для тендера")

date = st.date_input(
    'Выберите дату проведения тендера',
    min_value=datetime(2015,1,1),
    max_value=datetime(2023,12,31)
)

if st.button('Получить рекомендацию'):
    model = load_model()
    data = load_historical_data()
    input_df = prepare_input(data, date)
    
    prediction = model.predict(input_df)
    n_weeks = int(np.clip(round(prediction[0]), 1, 6))
    
    st.success(f"Рекомендуется заключить тендер на {n_weeks} недель")
    
    st.subheader('История цен')
    chart_data = data[['dt', 'Цена на арматуру']].set_index('dt')
    st.line_chart(chart_data)