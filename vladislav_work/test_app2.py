import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostRegressor
import sys

# Добавляем текущую директорию в путь
sys.path.append('.')

# Импортируем класс модели из существующего файла
from model_script import ReinforcementPriceModel

# Настройка страницы
st.set_page_config(
    page_title="Система прогнозирования цен на арматуру",
    page_icon="📊",
    layout="wide",
)

# Функция для загрузки данных
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

# Функция для построения графика исторических цен
def plot_historical_prices(data, date_col, price_col):
    fig = px.line(data, x=date_col, y=price_col, title='Историческая динамика цен на арматуру')
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Цена',
        height=500,
    )
    return fig

# Функция для визуализации прогноза
def plot_forecast(historical_data, forecast_data, date_col, price_col):
    # Создаем объединенный DataFrame для визуализации
    hist_df = historical_data.copy()
    hist_df['Тип'] = 'Исторические данные'
    
    # Преобразуем прогнозы в DataFrame
    forecast_df = pd.DataFrame(forecast_data['weekly_predictions'])
    forecast_df.rename(columns={'dt': date_col, 'price': price_col}, inplace=True)
    forecast_df['Тип'] = 'Прогноз'
    
    # Объединяем данные
    combined_df = pd.concat([
        hist_df[[date_col, price_col, 'Тип']].tail(52),  # Берем последний год для визуализации
        forecast_df[[date_col, price_col, 'Тип']]
    ])
    combined_df[date_col] = pd.to_datetime(combined_df[date_col])
    
    # Строим график
    fig = px.line(combined_df, x=date_col, y=price_col, color='Тип',
                 title='Исторические данные и прогноз цен на арматуру')
    fig.update_layout(
        xaxis_title='Дата',
        yaxis_title='Цена',
        height=500,
        legend_title='Данные'
    )
    
    # Добавляем вертикальную линию для текущей даты
    current_date = hist_df[date_col].max()
    fig.add_vline(x=current_date, line_dash="dash", line_color="gray",
                 annotation_text="Текущая дата", annotation_position="top right")
    
    return fig

def main():
    # Заголовок приложения
    st.title("Система прогнозирования цен на арматуру")
    
    # Боковая панель для загрузки данных и настройки модели
    with st.sidebar:
        st.header("Настройки")
        
        # Загрузка данных
        data_option = st.radio(
            "Источник данных:",
            ["Использовать демонстрационные данные", "Загрузить свои данные"]
        )
        
        if data_option == "Использовать демонстрационные данные":
            train_path = './vladislav_work/processed_data.csv'
            additional_path = None
            
            if not os.path.exists(train_path):
                st.error("Демонстрационные данные не найдены. Пожалуйста, загрузите свои данные.")
                data_option = "Загрузить свои данные"
        
        if data_option == "Загрузить свои данные":
            train_file = st.file_uploader("Загрузите файл с историческими ценами на арматуру", type=['csv'])
            additional_file = st.file_uploader("Загрузите файл с дополнительными данными (необязательно)", type=['csv'])
            
            if train_file:
                train_path = train_file
            else:
                st.warning("Пожалуйста, загрузите файл с историческими данными")
                return
                
            additional_path = additional_file if additional_file else None
        
        # Настройки прогноза
        st.header("Параметры прогноза")
        weeks_ahead = st.slider("Количество недель для прогноза", 1, 24, 4)
        
        # Модель: обучить новую или загрузить существующую
        model_option = st.radio(
            "Модель:",
            ["Обучить новую модель", "Использовать сохраненную модель"]
        )
        
        if model_option == "Обучить новую модель":
            test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random seed", 0, 1000, 42)
            
            train_button = st.button("Обучить модель")
        else:
            model_path = './model/reinforcement_price_model.pkl'
            if not os.path.exists(model_path):
                st.error("Сохраненная модель не найдена. Необходимо обучить новую модель.")
                model_option = "Обучить новую модель"
            else:
                predict_button = st.button("Сделать прогноз")
    
    # Основная панель
    # Загрузка и отображение данных
    try:
        if 'train_path' in locals():
            with st.spinner('Загрузка данных...'):
                data = load_data(train_path)
                
                if data is not None:
                    # Определяем колонки с датой и ценой
                    date_column = 'dt'
                    price_column = 'Цена на арматуру'
                    
                    # Проверка наличия нужных колонок
                    if date_column not in data.columns:
                        st.error(f"Колонка с датой '{date_column}' не найдена в данных")
                        return
                    if price_column not in data.columns:
                        st.error(f"Колонка с ценой '{price_column}' не найдена в данных")
                        return
                    
                    # Преобразование даты
                    data[date_column] = pd.to_datetime(data[date_column])
                    
                    # Отображаем данные
                    st.header("Исторические данные о ценах на арматуру")
                    
                    # Отображаем график
                    fig = plot_historical_prices(data, date_column, price_column)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Показываем таблицу с данными
                    with st.expander("Показать таблицу с данными"):
                        st.dataframe(data.sort_values(by=date_column, ascending=False))
                    
                    # Загружаем дополнительные данные
                    additional_data = None
                    if additional_path:
                        with st.spinner('Загрузка дополнительных данных...'):
                            additional_data = load_data(additional_path)
                            
                            if additional_data is not None:
                                with st.expander("Показать дополнительные данные"):
                                    st.dataframe(additional_data)
                    
                    # Обучение модели
                    if model_option == "Обучить новую модель" and train_button:
                        with st.spinner('Обучение модели...'):
                            try:
                                # Инициализируем модель
                                model = ReinforcementPriceModel()
                                
                                # Перенаправляем вывод print в streamlit
                                with st.expander("Детали обучения модели"):
                                    # Обучаем модель
                                    results = model.train(
                                        data, 
                                        additional_data=additional_data,
                                        test_size=test_size,
                                        random_state=random_state
                                    )
                                    
                                    # Отображаем метрики
                                    st.subheader("Метрики модели")
                                    metrics = results['metrics']
                                    cols = st.columns(3)
                                    cols[0].metric("MAE (валидация)", f"{metrics['val_mae']:.2f}")
                                    cols[1].metric("RMSE (валидация)", f"{metrics['val_rmse']:.2f}")
                                    cols[2].metric("R² (валидация)", f"{metrics['val_r2']:.4f}")
                                    
                                    # Отображаем важность признаков
                                    st.subheader("Важность признаков")
                                    feat_imp = results['feature_importance']
                                    st.dataframe(feat_imp.head(10))
                                
                                # Сохраняем модель
                                model_path = model.save_model()
                                
                                st.success(f"Модель успешно обучена и сохранена: {model_path}")
                                
                                # Рекомендуем пользователю сделать прогноз
                                st.info("Теперь вы можете использовать сохраненную модель для прогноза. Выберите 'Использовать сохраненную модель' в настройках.")
                                
                            except Exception as e:
                                st.error(f"Ошибка при обучении модели: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                    
                    # Прогнозирование
                    if model_option == "Использовать сохраненную модель" and predict_button:
                        with st.spinner('Выполнение прогноза...'):
                            try:
                                # Инициализируем модель и загружаем сохраненную
                                model = ReinforcementPriceModel()
                                model.load_model()
                                
                                # Получаем текущую дату (последнюю дату в данных)
                                current_date = data[date_column].max()
                                
                                # Делаем прогноз
                                forecast = model.predict_future(
                                    data,
                                    additional_data=additional_data,
                                    weeks_ahead=weeks_ahead,
                                    current_date=current_date
                                )
                                
                                if forecast['success']:
                                    # Создаем раздел с прогнозом
                                    st.header("Результаты прогнозирования")
                                    
                                    # Отображаем рекомендацию
                                    st.subheader("Рекомендация по тендеру")
                                    recommendation_col, confidence_col = st.columns(2)
                                    
                                    recommendation_col.info(forecast['recommendation'])
                                    confidence_col.metric("Уверенность в прогнозе", forecast['confidence'])
                                    
                                    # Отображаем прогноз цен
                                    st.subheader("Прогноз цен на арматуру")
                                    
                                    metrics_cols = st.columns(3)
                                    metrics_cols[0].metric(
                                        "Последняя известная цена", 
                                        f"{forecast['last_price']:.2f} руб.",
                                    )
                                    metrics_cols[1].metric(
                                        "Средний прогноз", 
                                        f"{forecast['average_prediction']:.2f} руб.",
                                        f"{forecast['price_change_pct']:.2f}%"
                                    )
                                    metrics_cols[2].metric(
                                        "Недель в прогнозе", 
                                        forecast['weeks_ahead']
                                    )
                                    
                                    # Визуализируем прогноз
                                    forecast_fig = plot_forecast(data, forecast, date_column, price_column)
                                    st.plotly_chart(forecast_fig, use_container_width=True)
                                    
                                    # Детальный прогноз по неделям
                                    st.subheader("Детальный прогноз по неделям")
                                    weekly_df = pd.DataFrame(forecast['weekly_predictions'])
                                    weekly_df['price'] = weekly_df['price'].round(2)
                                    st.dataframe(weekly_df)
                                    
                                    # Создаем график с недельным прогнозом
                                    weekly_fig = px.line(
                                        weekly_df, 
                                        x='dt', 
                                        y='price', 
                                        title='Прогноз цен на арматуру по неделям'
                                    )
                                    weekly_fig.update_layout(
                                        xaxis_title='Дата',
                                        yaxis_title='Цена',
                                        height=400
                                    )
                                    st.plotly_chart(weekly_fig, use_container_width=True)
                                    
                                    # Рекомендации по принятию решений
                                    st.subheader("Что это означает для вашего тендера")
                                    
                                    if forecast['price_change_pct'] > 5:
                                        st.warning("""
                                        **Прогнозируется повышение цен!**
                                        
                                        Ожидается рост цен на арматуру в ближайшие недели. Рекомендуется:
                                        - Провести небольшой тендер на ближайшие недели
                                        - Планировать крупный тендер после ожидаемого снижения цен
                                        - Рассмотреть альтернативных поставщиков
                                        """)
                                    elif forecast['price_change_pct'] < -5:
                                        st.success("""
                                        **Прогнозируется снижение цен!**
                                        
                                        Ожидается падение цен на арматуру в ближайшие недели. Рекомендуется:
                                        - Провести крупный тендер на весь прогнозируемый период
                                        - Зафиксировать более низкие цены на длительный срок
                                        - Увеличить объем закупки, пока цены снижаются
                                        """)
                                    else:
                                        st.info("""
                                        **Прогнозируется стабильность цен**
                                        
                                        Существенных изменений цен на арматуру в ближайшие недели не ожидается. Рекомендуется:
                                        - Проводить стандартные тендеры с обычным объемом
                                        - Следовать принятой закупочной политике
                                        - Контролировать цены каждую неделю для своевременной корректировки стратегии
                                        """)
                                
                            except Exception as e:
                                st.error(f"Ошибка при выполнении прогноза: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                    
        else:
            st.warning("Пожалуйста, загрузите данные для начала работы")
    
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
        import traceback
        st.error(traceback.format_exc())
    
    # Информация о проекте
    with st.expander("О системе"):
        st.write("""
        ## Система прогнозирования цен на арматуру
        
        Это приложение помогает категорийному менеджеру, закупающему арматуру, экономить, 
        предоставляя прогнозы будущих цен и рекомендации по проведению тендеров.
        
        ### Возможности:
        - Загрузка и визуализация исторических данных о ценах на арматуру
        - Обучение модели CatBoost для прогнозирования цен
        - Прогнозирование цен на выбранное количество недель вперед
        - Рекомендации по объему и времени проведения тендера
        
        ### Как использовать:
        1. Загрузите файл с историческими данными о ценах
        2. Загрузите дополнительные данные (если есть)
        3. Обучите модель или используйте сохраненную
        4. Выберите количество недель для прогноза
        5. Получите прогноз и рекомендации
        """)

if __name__ == "__main__":
    main()