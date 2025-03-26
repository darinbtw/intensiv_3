import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox, StringVar, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from matplotlib.dates import DateFormatter
import xgboost as xgb
import os
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline

# Улучшенная цветовая схема
PRIMARY_COLOR = '#1E3A8A'    # Темно-синий цвет
SECONDARY_COLOR = '#3B82F6'  # Голубой
BACKGROUND_COLOR = '#F3F4F6' # Светло-серый фон
TEXT_COLOR = '#1F2937'       # Темно-серый цвет текста
ACCENT_COLOR = '#2563EB'     # Яркий синий для акцентов
SUCCESS_COLOR = '#10B981'    # Зеленый для положительных изменений
WARNING_COLOR = '#F59E0B'    # Оранжевый для предупреждений
DANGER_COLOR = '#EF4444'     # Красный для отрицательных изменений

class RebarPricePredictor:
    def __init__(self, data_path=None):
        self.model_path = 'rebar_price_model.pkl'
        self.scaler_path = 'rebar_price_scaler.pkl'
        self.data_path = data_path or './vladislav_work/last_chance_to_made_app/processed_data.csv'
        
        # Загрузка и подготовка данных
        self.load_data()
        
        # Если есть сохраненная модель, загрузим ее, иначе создадим новую
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
        else:
            self.prepare_data()
            self.train_model()
            self.save_model()
    
    def load_data(self):
        try:
            # Сначала пробуем загрузить обработанные данные
            if os.path.exists(self.data_path):
                self.train_data = pd.read_csv(self.data_path)
                
                # Проверяем формат даты в данных
                if 'dt' in self.train_data.columns:
                    self.train_data['dt'] = pd.to_datetime(self.train_data['dt'], errors='coerce')
                else:
                    # Если столбец с датой называется иначе, ищем его
                    date_columns = [col for col in self.train_data.columns if 'dt' in col.lower() or 'дата' in col.lower() or 'date' in col.lower()]
                    if date_columns:
                        self.train_data.rename(columns={date_columns[0]: 'dt'}, inplace=True)
                        self.train_data['dt'] = pd.to_datetime(self.train_data['dt'], errors='coerce')
                    else:
                        # Если столбца с датой нет, создаем искусственную дату
                        self.train_data['dt'] = pd.date_range(start='2015-01-01', periods=len(self.train_data), freq='W-MON')
                
                # Проверяем наличие столбца с ценой
                price_columns = [col for col in self.train_data.columns if 'цена' in col.lower() or 'price' in col.lower()]
                if price_columns:
                    self.train_data.rename(columns={price_columns[0]: 'price'}, inplace=True)
                else:
                    # Если явно не указан столбец с ценой, используем второй числовой столбец
                    numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:
                        self.train_data.rename(columns={numeric_cols[0]: 'price'}, inplace=True)
                    else:
                        raise ValueError("Не найден столбец с ценами в данных")
            else:
                # Если обработанных данных нет, пробуем загрузить сырые данные
                train_path = './vladislav_work/last_chance_to_made_app/train.csv'
                test_path = './vladislav_work/last_chance_to_made_app/test.csv'
                
                if os.path.exists(train_path):
                    train_df = pd.read_csv(train_path)
                    
                    # Если есть тестовые данные, объединяем их
                    if os.path.exists(test_path):
                        test_df = pd.read_csv(test_path)
                        self.train_data = pd.concat([train_df, test_df], ignore_index=True)
                    else:
                        self.train_data = train_df
                    
                    # Проверяем наличие столбца с датой
                    date_columns = [col for col in self.train_data.columns if 'дата' in col.lower() or 'date' in col.lower()]
                    if date_columns:
                        self.train_data.rename(columns={date_columns[0]: 'dt'}, inplace=True)
                        self.train_data['dt'] = pd.to_datetime(self.train_data['dt'], errors='coerce')
                    else:
                        # Если столбца с датой нет, создаем искусственную дату
                        self.train_data['dt'] = pd.date_range(start='2015-01-01', periods=len(self.train_data), freq='W-MON')
                    
                    # Проверяем наличие столбца с ценой
                    price_columns = [col for col in self.train_data.columns if 'цена' in col.lower() or 'price' in col.lower()]
                    if price_columns:
                        self.train_data.rename(columns={price_columns[0]: 'price'}, inplace=True)
                    else:
                        # Если явно не указан столбец с ценой, используем первый числовой столбец
                        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) > 0:
                            self.train_data.rename(columns={numeric_cols[0]: 'price'}, inplace=True)
                        else:
                            raise ValueError("Не найден столбец с ценами в данных")
                else:
                    raise FileNotFoundError("Файлы с данными не найдены")
                
                # Сохраняем обработанные данные
                self.train_data.to_csv(self.data_path, index=False)
                
            # Сортируем данные по дате
            self.train_data = self.train_data.sort_values('dt')
            
            # Проверяем наличие пропущенных значений и заполняем их
            if self.train_data['price'].isnull().any():
                self.train_data['price'] = self.train_data['price'].interpolate(method='linear')
            
            print(f"Данные успешно загружены: {len(self.train_data)} записей")
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def prepare_data(self):
        """Подготовка данных для обучения модели"""
        # Создаем признаки
        # Временные признаки
        self.train_data['day_of_week'] = self.train_data['dt'].dt.dayofweek
        self.train_data['month'] = self.train_data['dt'].dt.month
        self.train_data['year'] = self.train_data['dt'].dt.year
        self.train_data['quarter'] = self.train_data['dt'].dt.quarter
        self.train_data['week_of_year'] = self.train_data['dt'].dt.isocalendar().week
        
        # Лаговые признаки
        for i in range(1, 13):
            self.train_data[f'price_lag_{i}'] = self.train_data['price'].shift(i)
        
        # Скользящие средние и статистики
        for window in [2, 4, 8, 12, 26]:
            self.train_data[f'price_ma_{window}'] = self.train_data['price'].rolling(window=window).mean()
            self.train_data[f'price_std_{window}'] = self.train_data['price'].rolling(window=window).std()
            self.train_data[f'price_min_{window}'] = self.train_data['price'].rolling(window=window).min()
            self.train_data[f'price_max_{window}'] = self.train_data['price'].rolling(window=window).max()
        
        # Скорость изменения цены
        self.train_data['price_diff_1'] = self.train_data['price'].diff(1)
        self.train_data['price_diff_4'] = self.train_data['price'].diff(4)
        self.train_data['price_pct_change_1'] = self.train_data['price'].pct_change(1)
        self.train_data['price_pct_change_4'] = self.train_data['price'].pct_change(4)
        
        # Экспоненциальные скользящие средние
        self.train_data['price_ewm_4'] = self.train_data['price'].ewm(span=4).mean()
        self.train_data['price_ewm_12'] = self.train_data['price'].ewm(span=12).mean()
        
        # Относительные показатели
        self.train_data['price_rel_ma_4'] = self.train_data['price'] / self.train_data['price_ma_4']
        self.train_data['price_rel_ma_12'] = self.train_data['price'] / self.train_data['price_ma_12']
        
        # Тренд и сезонность (упрощенно)
        self.train_data['trend'] = np.arange(len(self.train_data))
        self.train_data['trend_squared'] = self.train_data['trend'] ** 2
        
        # Удаляем строки с NaN значениями
        self.train_data = self.train_data.dropna()
        
        # Определяем признаки для модели
        self.features = [
            'day_of_week', 'month', 'year', 'quarter', 'week_of_year',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lag_8', 'price_lag_12',
            'price_ma_4', 'price_ma_8', 'price_ma_12', 'price_ma_26',
            'price_std_4', 'price_std_12',
            'price_min_12', 'price_max_12',
            'price_diff_1', 'price_diff_4',
            'price_pct_change_1', 'price_pct_change_4',
            'price_ewm_4', 'price_ewm_12',
            'price_rel_ma_4', 'price_rel_ma_12',
            'trend', 'trend_squared'
        ]
        
        # Подготовка признаков и целевой переменной
        self.X = self.train_data[self.features]
        self.y = self.train_data['price']
        
        print(f"Данные подготовлены: {len(self.features)} признаков")
    
    def train_model(self):
        """Обучение модели прогнозирования цен"""
        # Разделение данных на обучающую и тестовую выборки с учетом временной структуры
        train_size = int(len(self.X) * 0.8)
        X_train, X_test = self.X[:train_size], self.X[train_size:]
        y_train, y_test = self.y[:train_size], self.y[train_size:]
        
        print(f"Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")
        
        # Создаем пайплайн с масштабированием и моделью
        self.scaler = StandardScaler()
        
        # Используем XGBoost для лучшей точности
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        
        # Обучаем масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучаем модель
        print("Обучение модели...")
        self.model = xgb_model.fit(X_train_scaled, y_train)
        
        # Оцениваем модель
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Метрики модели: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
        
        # Визуализируем важность признаков
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 12))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(self.features)[sorted_idx])
        plt.title('Важность признаков XGBoost')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Сохраним фактические и предсказанные значения для визуализации
        self.test_actual = y_test
        self.test_predicted = y_pred
    
    def save_model(self):
        """Сохранение обученной модели и скейлера"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Модель сохранена в {self.model_path}")
    
    def load_model(self):
        """Загрузка сохраненной модели и скейлера"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Определяем признаки для модели
            self.features = [
                'day_of_week', 'month', 'year', 'quarter', 'week_of_year',
                'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lag_8', 'price_lag_12',
                'price_ma_4', 'price_ma_8', 'price_ma_12', 'price_ma_26',
                'price_std_4', 'price_std_12',
                'price_min_12', 'price_max_12',
                'price_diff_1', 'price_diff_4',
                'price_pct_change_1', 'price_pct_change_4',
                'price_ewm_4', 'price_ewm_12',
                'price_rel_ma_4', 'price_rel_ma_12',
                'trend', 'trend_squared'
            ]
            
            # Подготовка данных, если не подготовлены
            if 'price_lag_1' not in self.train_data.columns:
                self.prepare_data()
            
            print(f"Модель загружена из {self.model_path}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {str(e)}")
            self.prepare_data()
            self.train_model()
            self.save_model()
    
    # В функции predict_next_n_weeks исправьте работу с Series:
    def predict_next_n_weeks(self, current_data, n_weeks=6):
        """Прогнозирование цен на следующие N недель"""
        predictions = []
        prediction_dates = []
        temp_data = current_data.copy()
        
        for week in range(1, n_weeks + 1):
            # Подготовка признаков для прогноза
            latest_row = temp_data.iloc[-1:].copy()
            
            # Расчет следующей даты - извлекаем скалярное значение
            next_date = latest_row['dt'].iloc[0] + pd.Timedelta(days=7)
            prediction_dates.append(next_date)
            
            # Создаем новую строку данных
            new_row = latest_row.copy()
            new_row['dt'] = next_date
            new_row['day_of_week'] = next_date.dayofweek
            new_row['month'] = next_date.month
            new_row['year'] = next_date.year
            new_row['quarter'] = next_date.quarter
            new_row['week_of_year'] = int(next_date.isocalendar()[1])  # Используйте int() для безопасного получения номера недели
            
            # Обновляем зависящие от времени признаки - используем .iloc[0] для получения скалярных значений
            new_row['trend'] = temp_data['trend'].iloc[-1] + 1
            new_row['trend_squared'] = new_row['trend'].iloc[0] ** 2  # Используем .iloc[0]
            
            # Масштабирование признаков
            X_pred = new_row[self.features]
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Прогнозирование
            price_pred = self.model.predict(X_pred_scaled)[0]
            
            # Сохранение прогноза
            new_row['price'] = price_pred
            predictions.append(price_pred)
            
            # Обновление лаговых признаков - используем .iloc[0] для скалярных значений
            for i in range(12, 1, -1):
                if f'price_lag_{i}' in new_row.columns and f'price_lag_{i-1}' in temp_data.columns:
                    new_row[f'price_lag_{i}'] = temp_data[f'price_lag_{i-1}'].iloc[-1]
            
            new_row['price_lag_1'] = temp_data['price'].iloc[-1]
            
            # Обновление скользящих средних - извлекаем скалярные значения для вычислений
            for window in [2, 4, 8, 12, 26]:
                if len(temp_data) >= window:
                    prices = list(temp_data.iloc[-(window-1):]['price'].values) + [price_pred]
                    new_row[f'price_ma_{window}'] = np.mean(prices)
                    new_row[f'price_std_{window}'] = np.std(prices) if len(prices) > 1 else 0
                    new_row[f'price_min_{window}'] = np.min(prices)
                    new_row[f'price_max_{window}'] = np.max(prices)
            
            # Обновление показателей изменения - извлекаем скалярные значения
            last_price = temp_data['price'].iloc[-1]
            new_row['price_diff_1'] = price_pred - last_price
            
            if len(temp_data) >= 4:
                price_4_weeks_ago = temp_data['price'].iloc[-4]
                new_row['price_diff_4'] = price_pred - price_4_weeks_ago
                new_row['price_pct_change_4'] = (price_pred / price_4_weeks_ago - 1) if price_4_weeks_ago != 0 else 0
            
            new_row['price_pct_change_1'] = (price_pred / last_price - 1) if last_price != 0 else 0
            
            # Обновление экспоненциальных средних - извлекаем скалярные значения
            ewm_4_last = temp_data['price_ewm_4'].iloc[-1]
            ewm_12_last = temp_data['price_ewm_12'].iloc[-1]
            new_row['price_ewm_4'] = (ewm_4_last * 0.6) + (price_pred * 0.4)
            new_row['price_ewm_12'] = (ewm_12_last * 0.85) + (price_pred * 0.15)
            
            # Обновление относительных показателей - используем безопасное получение значений
            ma_4 = new_row['price_ma_4'].iloc[0] if 'price_ma_4' in new_row.columns else 1
            ma_12 = new_row['price_ma_12'].iloc[0] if 'price_ma_12' in new_row.columns else 1
            
            new_row['price_rel_ma_4'] = price_pred / ma_4 if ma_4 != 0 else 1
            new_row['price_rel_ma_12'] = price_pred / ma_12 if ma_12 != 0 else 1
            
            # Добавление прогноза к данным для следующей итерации
            temp_data = pd.concat([temp_data, new_row])
        
        return predictions, prediction_dates
    
    def calculate_strategy_costs(self, current_price, predictions, weekly_volume):
        """Расчет стоимости для каждой стратегии закупки (1-6 недель)"""
        strategy_costs = {}
        total_weeks = 6
        
        for weeks_ahead in range(1, 7):
            total_cost = 0
            weeks_covered = 0
            
            while weeks_covered < total_weeks:
                # Количество недель для текущей закупки
                purchase_weeks = min(weeks_ahead, total_weeks - weeks_covered)
                
                if weeks_covered == 0:
                    # Первая закупка по текущей цене
                    price = current_price
                else:
                    # Последующие закупки по прогнозным ценам
                    price = predictions[weeks_covered - 1]
                
                # Стоимость закупки = цена × объем × кол-во недель
                cost = price * weekly_volume * purchase_weeks
                total_cost += cost
                
                weeks_covered += purchase_weeks
            
            strategy_costs[weeks_ahead] = total_cost
        
        return strategy_costs
    
    def recommend_tender_period(self, current_data, weekly_volume=1):
        """Рекомендация оптимального периода тендера (1-6 недель)"""
        # Прогнозируем цены на следующие 6 недель
        predictions, prediction_dates = self.predict_next_n_weeks(current_data)
        current_price = current_data.iloc[-1]['price']
        
        # Рассчитываем стоимость для каждой стратегии закупки
        strategy_costs = self.calculate_strategy_costs(current_price, predictions, weekly_volume)
        
        # Находим стратегию с минимальной стоимостью
        recommended_period = min(strategy_costs, key=strategy_costs.get)
        
        # Анализируем тренд цен для дополнительной проверки
        price_trend = []
        for i in range(len(predictions) - 1):
            if predictions[i+1] > predictions[i]:
                price_trend.append(1)  # Рост
            elif predictions[i+1] < predictions[i]:
                price_trend.append(-1)  # Снижение
            else:
                price_trend.append(0)  # Стабильность
        
        # Рассчитываем общий тренд
        if sum(price_trend) > 0 and all(p >= 0 for p in price_trend):
            # Стабильный рост - рекомендуем большую закупку
            trend_recommendation = 6
        elif sum(price_trend) < 0 and all(p <= 0 for p in price_trend):
            # Стабильное снижение - рекомендуем малую закупку
            trend_recommendation = 1
        else:
            # Смешанный тренд - доверяем расчетам по стоимости
            trend_recommendation = recommended_period
        
        # Финальная рекомендация - выбираем между расчетом по стоимости и трендовым анализом
        if trend_recommendation != recommended_period:
            # Проверяем, насколько отличаются стоимости
            cost_diff = abs(strategy_costs[trend_recommendation] - strategy_costs[recommended_period])
            cost_ratio = cost_diff / strategy_costs[recommended_period]
            
            # Если разница менее 2%, предпочитаем трендовый анализ
            if cost_ratio < 0.02:
                recommended_period = trend_recommendation
        
        return recommended_period, predictions, prediction_dates, strategy_costs

class RebarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("АРМАПРОГНОЗ: система рекомендаций по закупке арматуры")
        self.root.geometry("1200x900")
        self.root.configure(background=BACKGROUND_COLOR)
        
        # Создаем стиль с корпоративными цветами
        self.style = ttk.Style()
        self.setup_styles()
        
        # Определяем переменные
        self.weekly_volume = StringVar(value="1")
        self.data_path = StringVar(value="./processed_data.csv")
        
        # Создаем фрейм с полосой прокрутки
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.main_container, background=BACKGROUND_COLOR)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Настраиваем интерфейс
        self.setup_ui()
        
        # Инициализируем предиктор
        self.initialize_predictor()
        
        # Привязываем прокрутку колесиком мыши
        self.root.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.root.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.root.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down
    
    def setup_styles(self):
        """Настройка стилей для виджетов"""
        self.style.configure("TFrame", background=BACKGROUND_COLOR)
        self.style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=("Arial", 10))
        self.style.configure("TButton", background=PRIMARY_COLOR, foreground="white", font=("Arial", 10, "bold"))
        self.style.configure("TLabelframe", background=BACKGROUND_COLOR)
        self.style.configure("TLabelframe.Label", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=("Arial", 11, "bold"))
        self.style.configure("TEntry", fieldbackground="white")
        
        # Дополнительные стили
        self.style.configure("Title.TLabel", font=("Arial", 18, "bold"), foreground=PRIMARY_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Subtitle.TLabel", font=("Arial", 14, "bold"), foreground=PRIMARY_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Info.TLabel", font=("Arial", 10), foreground=TEXT_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Success.TLabel", font=("Arial", 12, "bold"), foreground=SUCCESS_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Warning.TLabel", font=("Arial", 12, "bold"), foreground=WARNING_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Danger.TLabel", font=("Arial", 12, "bold"), foreground=DANGER_COLOR, background=BACKGROUND_COLOR)
        
        # Стили для кнопок
        self.style.configure("Primary.TButton", background=PRIMARY_COLOR, foreground="white")
        self.style.configure("Secondary.TButton", background=SECONDARY_COLOR, foreground="white")
        self.style.map("Primary.TButton", background=[("active", ACCENT_COLOR)])
        self.style.map("Secondary.TButton", background=[("active", SECONDARY_COLOR)])
    
    def _on_mousewheel(self, event):
        """Обработка прокрутки колесиком мыши"""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной контейнер
        main_frame = ttk.Frame(self.scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Заголовок и описание
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame, 
            text="АРМАПРОГНОЗ", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(10, 5))
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Интеллектуальная система рекомендаций для оптимизации закупок арматуры",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 10))
        
        description_label = ttk.Label(
            header_frame,
            text="Система анализирует исторические цены и прогнозирует оптимальный период закупки арматуры для минимизации затрат.",
            style="Info.TLabel",
            wraplength=1100
        )
        description_label.pack(pady=(0, 10))
        
        # Секция настроек
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding=15)
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Создаем сетку для лучшего выравнивания
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.X, pady=5)
        settings_grid.columnconfigure(0, weight=1)
        settings_grid.columnconfigure(1, weight=1)
        settings_grid.columnconfigure(2, weight=1)
        settings_grid.columnconfigure(3, weight=1)
        
        # Объем еженедельной потребности
        volume_label = ttk.Label(settings_grid, text="Еженедельная потребность (тонн):")
        volume_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        volume_entry = ttk.Entry(settings_grid, textvariable=self.weekly_volume, width=15)
        volume_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Кнопка выбора файла данных
        file_label = ttk.Label(settings_grid, text="Файл данных:")
        file_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        file_path_frame = ttk.Frame(settings_grid)
        file_path_frame.grid(row=0, column=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        file_path_entry = ttk.Entry(file_path_frame, textvariable=self.data_path, width=25)
        file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(file_path_frame, text="Обзор", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Секция текущих данных
        data_frame = ttk.LabelFrame(main_frame, text="Текущие данные", padding=15)
        data_frame.pack(fill=tk.X, pady=(0, 15))
        
        current_data_grid = ttk.Frame(data_frame)
        current_data_grid.pack(fill=tk.X, pady=5)
        current_data_grid.columnconfigure(0, weight=1)
        current_data_grid.columnconfigure(1, weight=2)
        current_data_grid.columnconfigure(2, weight=1)
        current_data_grid.columnconfigure(3, weight=2)
        
        # Текущая дата и цена
        date_label = ttk.Label(current_data_grid, text="Последняя дата:")
        date_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.current_date_label = ttk.Label(current_data_grid, text="", font=("Arial", 10, "bold"))
        self.current_date_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        price_label = ttk.Label(current_data_grid, text="Текущая цена:")
        price_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        self.current_price_label = ttk.Label(current_data_grid, text="", font=("Arial", 10, "bold"))
        self.current_price_label.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Тренд цены
        trend_label = ttk.Label(current_data_grid, text="Тренд (4 недели):")
        trend_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.trend_label = ttk.Label(current_data_grid, text="", font=("Arial", 10, "bold"))
        self.trend_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        volatility_label = ttk.Label(current_data_grid, text="Волатильность:")
        volatility_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        self.volatility_label = ttk.Label(current_data_grid, text="", font=("Arial", 10, "bold"))
        self.volatility_label.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Секция рекомендаций
        rec_frame = ttk.LabelFrame(main_frame, text="Рекомендация по закупке", padding=15)
        rec_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Основная рекомендация
        self.recommendation_label = ttk.Label(rec_frame, text="", font=("Arial", 16, "bold"), foreground=PRIMARY_COLOR)
        self.recommendation_label.pack(pady=(5, 10))
        
        # Пояснение к рекомендации
        self.explanation_label = ttk.Label(rec_frame, text="", wraplength=1100)
        self.explanation_label.pack(pady=(0, 10))
        
        # Прогнозные цены
        self.forecast_frame = ttk.Frame(rec_frame)
        self.forecast_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Экономия от рекомендации
        self.savings_label = ttk.Label(rec_frame, text="", style="Success.TLabel")
        self.savings_label.pack(pady=(0, 5))
        
        # График прогноза цен
        chart_frame = ttk.LabelFrame(main_frame, text="Прогноз цен на арматуру", padding=15)
        chart_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        self.canvas_fig = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.X)
        
        # График сравнения стратегий
        strategy_frame = ttk.LabelFrame(main_frame, text="Сравнение стратегий закупок", padding=15)
        strategy_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.strategy_figure = Figure(figsize=(10, 4), dpi=100)
        self.strategy_plot = self.strategy_figure.add_subplot(111)
        
        self.strategy_canvas = FigureCanvasTkAgg(self.strategy_figure, strategy_frame)
        self.strategy_canvas.get_tk_widget().pack(fill=tk.X)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Обновление прогноза
        refresh_button = ttk.Button(
            button_frame, 
            text="Обновить прогноз", 
            command=self.update_recommendation,
            style="Primary.TButton"
        )
        refresh_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Загрузка данных
        reload_button = ttk.Button(
            button_frame, 
            text="Перезагрузить данные", 
            command=self.reload_data,
            style="Secondary.TButton"
        )
        reload_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Справка
        help_button = ttk.Button(
            button_frame, 
            text="Справка", 
            command=self.show_help
        )
        help_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def initialize_predictor(self):
        """Инициализация предиктора цен"""
        try:
            # Проверяем указанный путь к данным
            data_path = self.data_path.get() if os.path.exists(self.data_path.get()) else None
            self.predictor = RebarPricePredictor(data_path)
            self.load_current_data()
        except Exception as e:
            messagebox.showerror(
                "Ошибка инициализации", 
                f"Не удалось инициализировать предиктор: {str(e)}\n\n"
                "Убедитесь, что файлы с данными находятся в указанной директории."
            )
    
    def browse_file(self):
        """Диалог выбора файла данных"""
        filename = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*"))
        )
        if filename:
            self.data_path.set(filename)
    
    def reload_data(self):
        """Перезагрузка данных и пересоздание модели"""
        try:
            # Проверяем указанный путь к данным
            data_path = self.data_path.get() if os.path.exists(self.data_path.get()) else None
            
            # Удаляем старые файлы модели, чтобы пересоздать их
            if os.path.exists(self.predictor.model_path):
                os.remove(self.predictor.model_path)
            if os.path.exists(self.predictor.scaler_path):
                os.remove(self.predictor.scaler_path)
                
            # Инициализируем предиктор заново
            self.predictor = RebarPricePredictor(data_path)
            self.load_current_data()
            
            messagebox.showinfo("Успешно", "Данные успешно перезагружены и модель переобучена.")
        except Exception as e:
            messagebox.showerror(
                "Ошибка перезагрузки", 
                f"Не удалось перезагрузить данные: {str(e)}"
            )
    
    def load_current_data(self):
        """Загрузка актуальных данных для прогноза"""
        # Получаем последние данные
        self.current_data = self.predictor.train_data.copy()
        
        # Отображаем текущую дату и цену
        if not self.current_data.empty:
            last_row = self.current_data.iloc[-1]
            self.current_date = last_row['dt']
            self.current_price = last_row['price']
            
            # Обновляем метки
            self.current_date_label.config(text=self.current_date.strftime("%d.%m.%Y"))
            self.current_price_label.config(text=f"{self.current_price:.2f} руб/т")
            
            # Рассчитываем и отображаем тренд
            if len(self.current_data) >= 5:
                last_5_prices = self.current_data.iloc[-5:]['price'].values
                price_change = (last_5_prices[-1] - last_5_prices[0]) / last_5_prices[0] * 100
                
                if price_change > 2:
                    trend_text = f"↗ Рост ({price_change:.1f}%)"
                    self.trend_label.config(text=trend_text, foreground=DANGER_COLOR)
                elif price_change < -2:
                    trend_text = f"↘ Падение ({price_change:.1f}%)"
                    self.trend_label.config(text=trend_text, foreground=SUCCESS_COLOR)
                else:
                    trend_text = f"→ Стабильный ({price_change:.1f}%)"
                    self.trend_label.config(text=trend_text, foreground=TEXT_COLOR)
                
                # Рассчитываем волатильность
                volatility = np.std(last_5_prices) / np.mean(last_5_prices) * 100
                
                if volatility < 2:
                    vol_text = f"Низкая ({volatility:.1f}%)"
                    self.volatility_label.config(text=vol_text, foreground=SUCCESS_COLOR)
                elif volatility < 5:
                    vol_text = f"Средняя ({volatility:.1f}%)"
                    self.volatility_label.config(text=vol_text, foreground=WARNING_COLOR)
                else:
                    vol_text = f"Высокая ({volatility:.1f}%)"
                    self.volatility_label.config(text=vol_text, foreground=DANGER_COLOR)
            
            # Обновляем рекомендацию
            self.update_recommendation()
        else:
            messagebox.showerror("Ошибка данных", "Не удалось загрузить актуальные данные.")
    
    def update_recommendation(self):
        """Обновление рекомендации на основе текущих данных"""
        try:
            # Получаем еженедельный объем из поля ввода
            try:
                weekly_volume = float(self.weekly_volume.get())
                if weekly_volume <= 0:
                    raise ValueError("Объем должен быть положительным числом")
            except ValueError as e:
                messagebox.showerror(
                    "Ошибка ввода", 
                    f"Некорректное значение еженедельной потребности: {str(e)}"
                )
                return
            
            # Получаем рекомендацию
            recommended_period, predictions, prediction_dates, strategy_costs = self.predictor.recommend_tender_period(
                self.current_data, 
                weekly_volume=weekly_volume
            )
            
            # Формируем текст рекомендации
            if recommended_period == 1:
                period_text = "1 неделю"
            elif recommended_period < 5:
                period_text = f"{recommended_period} недели"
            else:
                period_text = f"{recommended_period} недель"
                
            self.recommendation_label.config(
                text=f"Рекомендуемый период закупки: {period_text}"
            )
            
            # Формируем объяснение
            if recommended_period == 1:
                explanation = (
                    "Рекомендуется краткосрочная закупка (1 неделя), поскольку прогнозируется снижение "
                    "цен в ближайшем будущем. Такая стратегия позволит воспользоваться более выгодными "
                    "ценами в последующие периоды."
                )
                self.explanation_label.config(text=explanation)
            elif recommended_period >= 5:
                explanation = (
                    "Рекомендуется долгосрочная закупка, поскольку прогнозируется рост цен в ближайшем "
                    "будущем. Фиксация текущей цены на длительный период позволит избежать повышенных "
                    "затрат при закупке по будущим, более высоким ценам."
                )
                self.explanation_label.config(text=explanation)
            else:
                explanation = (
                    f"Рекомендуется среднесрочная закупка ({period_text}). Данный период обеспечивает "
                    f"оптимальный баланс между текущими ценами и прогнозируемыми изменениями "
                    f"в ближайшем будущем, что позволит минимизировать общие затраты."
                )
                self.explanation_label.config(text=explanation)
            
            # Обновляем прогнозные цены
            self.update_forecast_prices(prediction_dates, predictions)
            
            # Рассчитываем экономию
            worst_strategy = max(strategy_costs.items(), key=lambda x: x[1])
            best_strategy = min(strategy_costs.items(), key=lambda x: x[1])
            
            savings = worst_strategy[1] - best_strategy[1]
            savings_pct = (savings / worst_strategy[1]) * 100
            
            self.savings_label.config(
                text=f"Потенциальная экономия: {savings:.2f} руб. ({savings_pct:.2f}%) "
                     f"по сравнению с наименее выгодной стратегией ({worst_strategy[0]} {'нед.' if worst_strategy[0] > 1 else 'нед.'})"
            )
            
            # Обновляем графики
            self.plot_predictions(prediction_dates, predictions)
            self.plot_strategy_comparison(strategy_costs)
            
            # Обновляем область прокрутки
            self.canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обновить рекомендацию: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_forecast_prices(self, prediction_dates, predictions):
        """Обновление отображения прогнозных цен"""
        # Очищаем предыдущие виджеты
        for widget in self.forecast_frame.winfo_children():
            widget.destroy()
        
        # Создаем таблицу для прогнозных цен
        headers = ["Период", "Дата", "Прогноз цены", "Изменение", "Тренд"]
        
        # Создаем заголовки таблицы
        for i, header in enumerate(headers):
            label = ttk.Label(
                self.forecast_frame, 
                text=header, 
                font=("Arial", 10, "bold"),
                padding=5
            )
            label.grid(row=0, column=i, sticky=tk.W, padx=5, pady=(0, 5))
        
        # Добавляем горизонтальную линию под заголовками
        separator = ttk.Separator(self.forecast_frame, orient="horizontal")
        separator.grid(row=1, column=0, columnspan=len(headers), sticky=tk.EW, pady=(0, 5))
        
        # Отображаем прогнозные цены для каждой недели
        for i, (date, price) in enumerate(zip(prediction_dates, predictions)):
            week_num = i + 1
            date_str = date.strftime("%d.%m.%Y")
            
            # Определяем цвет в зависимости от изменения цены
            if i == 0:
                price_change = price - self.current_price
                pct_change = (price_change / self.current_price) * 100
                color = SUCCESS_COLOR if price_change < 0 else DANGER_COLOR if price_change > 0 else TEXT_COLOR
            else:
                price_change = price - predictions[i-1]
                pct_change = (price_change / predictions[i-1]) * 100
                color = SUCCESS_COLOR if price_change < 0 else DANGER_COLOR if price_change > 0 else TEXT_COLOR
            
            # Отображаем информацию
            ttk.Label(self.forecast_frame, text=f"Неделя {week_num}").grid(
                row=i+2, column=0, padx=5, pady=2, sticky=tk.W
            )
            
            ttk.Label(self.forecast_frame, text=date_str).grid(
                row=i+2, column=1, padx=5, pady=2, sticky=tk.W
            )
            
            ttk.Label(
                self.forecast_frame, 
                text=f"{price:.2f} руб/т",
                foreground=color
            ).grid(row=i+2, column=2, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(
                self.forecast_frame, 
                text=f"{price_change:+.2f} руб ({pct_change:+.2f}%)",
                foreground=color
            ).grid(row=i+2, column=3, padx=5, pady=2, sticky=tk.W)
            
            # Отображаем тренд
            trend_symbol = "↗" if price_change > 0 else "↘" if price_change < 0 else "→"
            ttk.Label(
                self.forecast_frame, 
                text=trend_symbol,
                font=("Arial", 12, "bold"),
                foreground=color
            ).grid(row=i+2, column=4, padx=5, pady=2, sticky=tk.W)
    
    def plot_predictions(self, prediction_dates, predictions):
        """Построение графика прогноза цен"""
        self.plot.clear()
        
        # Получаем исторические данные (последние 26 недель)
        historical_data = self.current_data.iloc[-26:]
        dates = historical_data['dt'].tolist()
        prices = historical_data['price'].tolist()
        
        # Создаем непрерывную линию
        all_dates = dates + [dates[-1]] + prediction_dates
        all_prices = prices + [prices[-1]] + predictions
        
        # Стилизация графика
        self.plot.figure.set_facecolor(BACKGROUND_COLOR)
        self.plot.set_facecolor(BACKGROUND_COLOR)
        
        # Построение исторических данных
        self.plot.plot(dates, prices, marker='o', linestyle='-', color=PRIMARY_COLOR, linewidth=2, markersize=5, label='Исторические цены')
        
        # Построение прогнозных данных
        self.plot.plot(prediction_dates, predictions, marker='s', linestyle='--', color=ACCENT_COLOR, linewidth=2, markersize=6, label='Прогноз')
        
        # Выделяем текущую цену
        self.plot.scatter([dates[-1]], [prices[-1]], color=SUCCESS_COLOR, s=120, zorder=5, label='Текущая цена')
        
        # Добавляем аннотации с ценами к прогнозным точкам
        for date, price in zip(prediction_dates, predictions):
            self.plot.annotate(
                f"{price:.0f}", 
                (date, price),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=SECONDARY_COLOR, alpha=0.8)
            )
        
        # Определяем диапазон для осей
        min_price = min(min(prices), min(predictions)) * 0.98
        max_price = max(max(prices), max(predictions)) * 1.02
        
        # Настройка графика
        self.plot.set_title('Исторические и прогнозируемые цены на арматуру', fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
        self.plot.set_xlabel('Дата', fontsize=11, fontweight='bold')
        self.plot.set_ylabel('Цена (руб/т)', fontsize=11, fontweight='bold')
        self.plot.tick_params(axis='both', colors=TEXT_COLOR)
        self.plot.legend(loc='upper left', fancybox=True, shadow=True)
        self.plot.grid(True, alpha=0.3, linestyle='--')
        
        # Форматирование даты на оси X
        self.plot.xaxis.set_major_formatter(DateFormatter('%d.%m.%y'))
        self.figure.autofmt_xdate(rotation=45)
        
        # Настраиваем границы осей
        self.plot.set_ylim(min_price, max_price)
        
        # Добавляем вертикальную линию для текущей даты
        self.plot.axvline(x=dates[-1], color=PRIMARY_COLOR, linestyle=':', alpha=0.7)
        
        # Добавляем область для прогноза
        self.plot.axvspan(dates[-1], prediction_dates[-1], alpha=0.1, color=ACCENT_COLOR)
        
        # Обновляем график
        self.figure.tight_layout()
        self.canvas_fig.draw()
    
    def plot_strategy_comparison(self, strategy_costs):
        """Построение графика сравнения стратегий закупки"""
        self.strategy_plot.clear()
        
        # Получаем данные для графика
        weeks = list(strategy_costs.keys())
        costs = list(strategy_costs.values())
        
        # Определяем лучшую и худшую стратегии
        best_strategy_idx = costs.index(min(costs))
        worst_strategy_idx = costs.index(max(costs))
        
        # Создаем цвета, выделяем лучшую и худшую стратегии
        colors = ['#c6c6c6'] * len(weeks)
        colors[best_strategy_idx] = SUCCESS_COLOR
        colors[worst_strategy_idx] = DANGER_COLOR
        
        # Настройка внешнего вида
        self.strategy_plot.figure.set_facecolor(BACKGROUND_COLOR)
        self.strategy_plot.set_facecolor(BACKGROUND_COLOR)
        
        # Создаем столбчатую диаграмму
        bars = self.strategy_plot.bar(weeks, costs, color=colors, width=0.6, alpha=0.9)
        
        # Добавляем подписи стоимости над каждым столбцом
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            self.strategy_plot.text(
                bar.get_x() + bar.get_width()/2.,
                height + (max(costs) - min(costs)) * 0.01,
                f'{cost:.0f}',
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold'
            )
        
        # Добавляем подписи к лучшей и худшей стратегиям
        bars[best_strategy_idx].set_label('Рекомендуемая стратегия')
        bars[worst_strategy_idx].set_label('Наименее выгодная стратегия')
        
        # Настройка графика
        self.strategy_plot.set_xlabel('Период закупки (недель)', fontsize=11, fontweight='bold')
        self.strategy_plot.set_ylabel('Общие затраты (руб)', fontsize=11, fontweight='bold')
        self.strategy_plot.set_title('Сравнение стратегий закупки арматуры', fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
        self.strategy_plot.tick_params(axis='both', colors=TEXT_COLOR)
        self.strategy_plot.legend(loc='upper right', fancybox=True, shadow=True)
        self.strategy_plot.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Настраиваем границы осей
        min_cost = min(costs) * 0.98
        max_cost = max(costs) * 1.02
        self.strategy_plot.set_ylim(min_cost, max_cost)
        
        # Добавляем описательный текст
        min_cost_value = min(costs)
        max_cost_value = max(costs)
        savings = max_cost_value - min_cost_value
        savings_pct = (savings / max_cost_value) * 100
        
        if weeks[best_strategy_idx] == 1:
            strategy_text = "Краткосрочная закупка (1 неделя)"
        elif weeks[best_strategy_idx] >= 5:
            strategy_text = f"Долгосрочная закупка ({weeks[best_strategy_idx]} недель)"
        else:
            strategy_text = f"Среднесрочная закупка ({weeks[best_strategy_idx]} недели)"
        
        self.strategy_plot.text(
            0.5, 0.05,
            f"{strategy_text} позволяет сэкономить {savings:.0f} руб. ({savings_pct:.1f}%)",
            ha='center', va='bottom',
            transform=self.strategy_plot.transAxes,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
        
        # Обновляем график
        self.strategy_figure.tight_layout()
        self.strategy_canvas.draw()
    
    def show_help(self):
        """Отображение справочной информации"""
        help_text = """Система рекомендаций для закупки арматуры "АРМАПРОГНОЗ"

ФУНКЦИОНАЛЬНОСТЬ:
- Анализ исторических цен на арматуру
- Прогнозирование цен на ближайшие 6 недель
- Определение оптимального периода тендера
- Расчёт потенциальной экономии от оптимизации закупок
- Визуализация прогнозных цен и сравнение стратегий

КАК ПОЛЬЗОВАТЬСЯ:
1. Введите еженедельную потребность в арматуре (в тоннах)
2. При необходимости укажите путь к файлу с данными
3. Нажмите "Обновить прогноз" для получения рекомендации
4. Изучите прогнозные цены и сравнение стратегий закупки
5. Следуйте рекомендации системы для оптимизации затрат

О МОДЕЛИ ПРОГНОЗИРОВАНИЯ:
- Используется алгоритм XGBoost для анализа временных рядов
- Учитываются сезонные колебания, тренды и другие факторы
- Точность прогноза зависит от качества исторических данных
- Регулярное обновление данных повышает точность прогнозов

ТЕРМИНЫ:
- Период закупки — количество недель, на которое заключается тендер
- Волатильность — мера изменчивости цен на рынке
- Тренд — направление движения цен (рост, падение, стабильность)
- Стратегия закупки — план проведения тендеров на определенный период

КОНТАКТЫ:
При возникновении вопросов обращайтесь в отдел аналитики.
"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Справка о системе")
        help_window.geometry("650x600")
        help_window.configure(background=BACKGROUND_COLOR)
        
        # Заголовок
        ttk.Label(
            help_window,
            text="СПРАВКА ПО СИСТЕМЕ",
            style="Title.TLabel",
        ).pack(pady=(15, 5))
        
        # Основной текст
        help_frame = ttk.Frame(help_window)
        help_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        help_text_widget = tk.Text(
            help_frame,
            wrap=tk.WORD,
            background=BACKGROUND_COLOR,
            foreground=TEXT_COLOR,
            font=("Arial", 11),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.config(state=tk.DISABLED)  # Запрещаем редактирование
        
        # Кнопка закрытия
        close_button = ttk.Button(
            help_window,
            text="Закрыть",
            command=help_window.destroy,
            style="Primary.TButton"
        )
        close_button.pack(pady=(0, 15))

# Запуск приложения
if __name__ == "__main__":
    try:
        # Устанавливаем стиль для Tkinter
        root = tk.Tk()
        
        # Устанавливаем иконку приложения (если доступна)
        try:
            root.iconbitmap("app_icon.ico")
        except:
            pass
            
        # Запускаем приложение
        app = RebarApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Ошибка запуска приложения: {str(e)}")
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение: {str(e)}")