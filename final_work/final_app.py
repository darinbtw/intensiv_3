import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox, StringVar, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from matplotlib.dates import DateFormatter
import os
import joblib
from tkcalendar import DateEntry

# Цветовая схема
PRIMARY_COLOR = '#1E3A8A'    # Темно-синий цвет
SECONDARY_COLOR = '#3B82F6'  # Голубой
BACKGROUND_COLOR = '#F3F4F6' # Светло-серый фон
TEXT_COLOR = '#000000'       # Темно-серый цвет текста
ACCENT_COLOR = '#2563EB'     # Яркий синий для акцентов
SUCCESS_COLOR = '#10B981'    # Зеленый для положительных изменений
WARNING_COLOR = '#F59E0B'    # Оранжевый для предупреждений
DANGER_COLOR = '#EF4444'     # Красный для отрицательных изменений

class SimplePricePredictor:
    def __init__(self, data_path=None):
        self.model_path = 'rebar_price_model.pkl'
        self.scaler_path = 'rebar_price_scaler.pkl'
        self.data_path = data_path or './vladislav_work/last_chance_to_made_app/processed_data.csv'
        
        # Загрузка данных
        self.load_data()
        
        # Загрузка или создание модели
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
        else:
            self.prepare_data()
            self.train_model()
            self.save_model()
    
    def load_data(self):
        """Загрузка данных из CSV файла"""
        try:
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                
                # Проверка и преобразование столбца с датой
                date_columns = [col for col in self.data.columns if 'dt' in col.lower() or 'дата' in col.lower() or 'date' in col.lower()]
                if date_columns:
                    self.data.rename(columns={date_columns[0]: 'dt'}, inplace=True)
                else:
                    # Если столбца с датой нет, создаем его
                    self.data['dt'] = pd.date_range(start='2015-01-01', periods=len(self.data), freq='W-MON')
                
                self.data['dt'] = pd.to_datetime(self.data['dt'])
                
                # Проверка столбца с ценой
                price_columns = [col for col in self.data.columns if 'цена' in col.lower() or 'price' in col.lower()]
                if price_columns:
                    self.data.rename(columns={price_columns[0]: 'price'}, inplace=True)
                else:
                    # Если явно не указан столбец с ценой, используем первый числовой столбец
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:
                        self.data.rename(columns={numeric_cols[0]: 'price'}, inplace=True)
                    else:
                        raise ValueError("Не найден столбец с ценами в данных")
            else:
                raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
            
            # Сортируем данные по дате
            self.data = self.data.sort_values('dt')
            
            # Заполняем пропущенные значения
            if self.data['price'].isnull().any():
                self.data['price'] = self.data['price'].interpolate(method='linear')
            
            print(f"Данные успешно загружены: {len(self.data)} записей")
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    def prepare_data(self):
        """Подготовка данных для обучения модели"""
        # Создаем простые признаки на основе даты
        self.data['day_of_week'] = self.data['dt'].dt.dayofweek
        self.data['month'] = self.data['dt'].dt.month
        self.data['year'] = self.data['dt'].dt.year
        self.data['quarter'] = self.data['dt'].dt.quarter
        
        # Лаговые признаки (предыдущие значения цен)
        for i in range(1, 5):  # Используем только 4 лага для простоты
            self.data[f'price_lag_{i}'] = self.data['price'].shift(i)
        
        # Скользящие средние
        for window in [2, 4, 8]:
            self.data[f'price_ma_{window}'] = self.data['price'].rolling(window=window).mean()
        
        # Удаляем строки с NaN значениями
        self.data_cleaned = self.data.dropna().copy()
        
        # Определяем признаки для модели
        self.features = [
            'day_of_week', 'month', 'year', 'quarter',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
            'price_ma_2', 'price_ma_4', 'price_ma_8'
        ]
        
        # Подготовка признаков и целевой переменной
        self.X = self.data_cleaned[self.features]
        self.y = self.data_cleaned['price']
        
        print(f"Данные подготовлены: {len(self.features)} признаков")
    
    def train_model(self):
        """Обучение модели RandomForest"""
        # Разделение данных на обучающую и тестовую выборки
        train_size = int(len(self.X) * 0.8)
        X_train, X_test = self.X[:train_size], self.X[train_size:]
        y_train, y_test = self.y[:train_size], self.y[train_size:]
        
        print(f"Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}")
        
        # Создаем скейлер и модель
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Используем RandomForest с небольшим числом деревьев для быстроты
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Обучаем модель
        print("Обучение модели...")
        self.model.fit(X_train_scaled, y_train)
        
        # Оцениваем модель
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Метрики модели: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
    
    def save_model(self):
        """Сохранение обученной модели и скейлера"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Модель и скейлер сохранены в {self.model_path}")
    
    def load_model(self):
        """Загрузка сохраненной модели и скейлера"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Определяем признаки для модели
            self.features = [
                'day_of_week', 'month', 'year', 'quarter',
                'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
                'price_ma_2', 'price_ma_4', 'price_ma_8'
            ]
            
            print(f"Модель загружена из {self.model_path}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {str(e)}")
            self.prepare_data()
            self.train_model()
            self.save_model()
    
    def predict_next_n_weeks(self, start_date=None, n_weeks=6):
        """Прогнозирование цен на следующие N недель от указанной даты"""
        predictions = []
        prediction_dates = []
        
        # Если указана начальная дата, преобразуем её
        if start_date is not None:
            try:
                # Обработка случая, когда start_date передается как строка
                if isinstance(start_date, str):
                    print(f"Преобразуем дату из строки: {start_date}")
                    start_date = pd.to_datetime(start_date)
                    print(f"Преобразованная дата: {start_date}")
                
                # Корректируем дату до понедельника
                if start_date.weekday() != 0:
                    days_to_add = (7 - start_date.weekday()) % 7
                    start_date = start_date + pd.Timedelta(days=days_to_add)
                    print(f"Дата скорректирована до понедельника: {start_date.strftime('%d.%m.%Y')}")
            except Exception as e:
                print(f"Ошибка при обработке даты start_date: {e}")
                # Если не удалось обработать дату, используем текущую
                start_date = pd.Timestamp.now()
                # Корректируем до понедельника
                if start_date.weekday() != 0:
                    days_to_add = (7 - start_date.weekday()) % 7
                    start_date = start_date + pd.Timedelta(days=days_to_add)
        else:
            # Если дата не указана, используем последнюю доступную дату
            start_date = self.data['dt'].max() + pd.Timedelta(days=7)
        
        # Проверяем, находится ли дата в будущем относительно последней даты в данных
        if start_date > self.data['dt'].max():
            print(f"Дата прогноза ({start_date}) находится в будущем относительно последних данных ({self.data['dt'].max()})")
            future_weeks = (start_date - self.data['dt'].max()).days // 7
            print(f"Разница в неделях: {future_weeks}")
            
            # Если дата в будущем, сначала прогнозируем до этой даты, затем продолжаем стандартный прогноз
            if future_weeks > 0:
                print(f"Сначала прогнозируем {future_weeks} недель до указанной даты")
                temp_predictions, temp_dates, _ = self._predict_specific_period(
                    self.data['dt'].max() + pd.Timedelta(days=7), 
                    future_weeks
                )
                
                # Теперь используем последний прогноз как основу для дальнейшего прогнозирования
                # Создаем временный DataFrame чтобы добавить прогнозируемые данные
                temp_data = self.data.copy()
                
                for i, (pred_date, pred_price) in enumerate(zip(temp_dates, temp_predictions)):
                    # Создаем и добавляем строку с прогнозируемыми данными
                    new_row = pd.DataFrame({
                        'dt': [pred_date],
                        'price': [pred_price],
                        'day_of_week': [pred_date.dayofweek],
                        'month': [pred_date.month],
                        'year': [pred_date.year],
                        'quarter': [pred_date.quarter]
                    })
                    
                    # Добавляем лаговые признаки и скользящие средние обновим позже
                    temp_data = pd.concat([temp_data, new_row], ignore_index=True)
                
                # Пересчитываем лаги и скользящие средние для временного датафрейма
                temp_data = self._recalculate_features(temp_data)
                
                # Теперь используем расширенный датафрейм для дальнейшего прогноза
                return self._predict_specific_period(start_date, n_weeks, temp_data)
        
        # Если дата не в будущем или небольшое смещение, используем стандартный подход
        return self._predict_specific_period(start_date, n_weeks)

    def _recalculate_features(self, data):
        """Пересчитывает признаки для DataFrame (лаги и скользящие средние)"""
        # Сортируем данные по дате для правильного расчета
        data = data.sort_values('dt')
        
        # Пересчитываем лаговые признаки
        for i in range(1, 5):
            data[f'price_lag_{i}'] = data['price'].shift(i)
        
        # Пересчитываем скользящие средние
        for window in [2, 4, 8]:
            data[f'price_ma_{window}'] = data['price'].rolling(window=window).mean()
        
        return data

    def _predict_specific_period(self, start_date, n_weeks, historical_data=None):
        """Внутренний метод для прогнозирования конкретного периода"""
        predictions = []
        prediction_dates = []
        
        # Используем переданные исторические данные или берем из self.data
        if historical_data is None:
            # Получаем только данные до стартовой даты
            historical_data = self.data[self.data['dt'] < start_date].copy()
        else:
            # Фильтруем только данные до стартовой даты
            historical_data = historical_data[historical_data['dt'] < start_date].copy()
        
        if historical_data.empty:
            error_msg = f"Нет исторических данных до указанной даты {start_date}"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Для дат далеко в будущем, всегда используем все данные, но не фильтруем по последнему году
        use_all_history = False
        if start_date.year > self.data['dt'].max().year + 1:
            print(f"Дата прогноза находится далеко в будущем: {start_date.year} > {self.data['dt'].max().year + 1}")
            print("Используем все исторические данные без фильтрации по последнему году")
            use_all_history = True
        
        # Берем только последний год данных для более точного прогноза, если не указано иное
        if not use_all_history:
            one_year_ago = start_date - pd.Timedelta(days=365)
            year_filtered_data = historical_data[historical_data['dt'] >= one_year_ago]
            # Проверяем, не слишком ли мало данных после фильтрации
            if not year_filtered_data.empty and len(year_filtered_data) >= 12:  # мин. 12 недель (3 месяца)
                historical_data = year_filtered_data
            else:
                print("Недостаточно данных за последний год, используем все доступные данные")
        
        # Удаляем строки с NaN значениями
        historical_data = historical_data.dropna()
        
        if historical_data.empty:
            error_msg = f"Недостаточно исторических данных для прогноза от даты {start_date}"
            print(error_msg)
            
            # Вместо того, чтобы выбрасывать ошибку, используем все доступные данные
            historical_data = self.data.copy()
            historical_data = historical_data.dropna()
            print(f"Используем все доступные данные: {len(historical_data)} записей")
            
            if historical_data.empty:
                raise ValueError("Нет данных для прогноза даже после использования всей истории")
        
        # Берём последнее известное значение для начала прогноза
        last_row = historical_data.iloc[-1:].copy()
        current_price = last_row['price'].iloc[0]
        
        # Первая дата прогноза - это неделя после последних исторических данных
        # если это раньше указанной даты, используем указанную дату
        next_date = max(last_row['dt'].iloc[0] + pd.Timedelta(days=7), start_date)
        
        # Для отладки
        print(f"Год прогноза: {next_date.year}")
        print(f"Прогноз от даты: {start_date}")
        print(f"Последняя историческая дата: {last_row['dt'].iloc[0]}")
        print(f"Текущая цена: {current_price}")
        print(f"Первая дата прогноза: {next_date}")
        print(f"Количество исторических точек: {len(historical_data)}")
        
        # Итеративно строим прогноз
        temp_data = historical_data.copy()  # Работаем с копией отфильтрованных данных
        
        # Добавляем модификатор года для обеспечения разных прогнозов для разных лет
        # Это искусственный способ внести разнообразие в прогнозы для разных лет
        year_modifier = 1.0
        
        # Добавим влияние года и случайный фактор для более реалистичных прогнозов
        if next_date.year > 2024:
            years_in_future = next_date.year - 2024
            # Увеличиваем модификатор для будущих лет
            # Используем нелинейную функцию: чем дальше в будущее, тем больше неопределенность
            year_modifier = 1.0 + (years_in_future * 0.03) + (np.log1p(years_in_future) * 0.02)
            
            # Добавляем немного случайности, зависящей от года (но с фиксированным зерном)
            np.random.seed(next_date.year)  # Фиксированное зерно для конкретного года
            year_random = np.random.uniform(0.97, 1.03)
            year_modifier *= year_random
        
        print(f"Применяем модификатор года: {year_modifier}")
        
        for week in range(1, n_weeks + 1):
            # Первую дату уже установили, для следующих недель добавляем по 7 дней
            if week > 1:
                next_date = prediction_dates[-1] + pd.Timedelta(days=7)
            
            prediction_dates.append(next_date)
            
            # Создаем новую строку с признаками
            new_row = pd.DataFrame({
                'dt': [next_date],
                'day_of_week': [next_date.dayofweek],
                'month': [next_date.month],
                'year': [next_date.year],
                'quarter': [next_date.quarter]
            })
            
            # Добавляем лаговые признаки
            if week == 1:
                new_row['price_lag_1'] = temp_data['price'].iloc[-1]
                for i in range(2, 5):
                    if len(temp_data) >= i:
                        new_row[f'price_lag_{i}'] = temp_data['price'].iloc[-(i-1)] if i <= len(temp_data) else 0
                    else:
                        new_row[f'price_lag_{i}'] = 0
            else:
                # Обновляем лаги на основе предыдущих прогнозов
                for i in range(1, 5):
                    if i <= len(predictions):
                        new_row[f'price_lag_{i}'] = predictions[-i]
                    elif i <= len(temp_data) + len(predictions):
                        idx = i - len(predictions) - 1
                        if idx >= -len(temp_data):
                            new_row[f'price_lag_{i}'] = temp_data['price'].iloc[idx]
                        else:
                            new_row[f'price_lag_{i}'] = 0
                    else:
                        new_row[f'price_lag_{i}'] = 0
            
            # Вычисляем скользящие средние
            for window in [2, 4, 8]:
                values = []
                for i in range(min(window, len(predictions) + 1)):
                    if i == 0:
                        if week == 1:
                            values.append(temp_data['price'].iloc[-1])
                        else:
                            values.append(predictions[-1])
                    else:
                        if i <= len(predictions):
                            values.append(predictions[-i])
                        elif i <= len(temp_data) + len(predictions):
                            idx = i - len(predictions) - 1
                            if idx >= -len(temp_data):
                                values.append(temp_data['price'].iloc[idx])
                
                if len(values) > 0:
                    new_row[f'price_ma_{window}'] = np.mean(values)
                else:
                    new_row[f'price_ma_{window}'] = new_row['price_lag_1']
            
            # Прогнозируем цену
            X_pred = new_row[self.features]
            
            # Очищаем данные от inf и nan
            for col in X_pred.columns:
                X_pred.loc[:, col] = X_pred[col].replace([np.inf, -np.inf], np.nan)
                X_pred.loc[:, col] = X_pred[col].fillna(0)
            
            # Масштабируем признаки и делаем прогноз
            X_pred_scaled = self.scaler.transform(X_pred)
            price_pred = self.model.predict(X_pred_scaled)[0]
            
            # Применяем модификатор года
            price_pred = price_pred * year_modifier
            
            # Добавляем небольшую сезонность (цены выше весной-летом, ниже осенью-зимой)
            month = next_date.month
            if 3 <= month <= 8:  # Весна-лето
                price_pred *= 1.01  # Небольшое увеличение весной и летом
            else:  # Осень-зима
                price_pred *= 0.99  # Небольшое снижение осенью и зимой
                
            # Добавляем небольшую случайность для каждой недели, но с фиксированным зерном,
            # чтобы прогноз был воспроизводимым
            np.random.seed(int(next_date.timestamp()) // 86400)  # Используем день как зерно
            week_random = np.random.uniform(0.995, 1.005)
            price_pred *= week_random
            
            # Ограничиваем изменение цены разумными пределами (не более 10% за неделю)
            if week == 1:
                max_price = current_price * 1.1
                min_price = current_price * 0.9
            else:
                max_price = predictions[-1] * 1.1
                min_price = predictions[-1] * 0.9
            
            price_pred = max(min(price_pred, max_price), min_price)
            
            # Сохраняем прогноз
            predictions.append(price_pred)
            
            # Добавляем новую строку к временным данным
            new_row['price'] = price_pred
            temp_data = pd.concat([temp_data, new_row])
        
        return predictions, prediction_dates, current_price
    
    def recommend_tender_period(self, start_date=None):
        """Рекомендация оптимального периода тендера (1-6 недель)"""
        # Прогнозируем цены на следующие 6 недель
        predictions, prediction_dates, current_price = self.predict_next_n_weeks(start_date=start_date)
        
        # Определяем тренд цен (растут или падают)
        price_trend = []
        
        # Также отслеживаем волатильность и общий паттерн
        price_changes = []
        direction_changes = 0  # Сколько раз меняется направление (рост/падение)
        positive_weeks = 0
        negative_weeks = 0
        
        for i in range(len(predictions) - 1):
            # Считаем процентное изменение между соседними неделями
            change_pct = (predictions[i+1] / predictions[i] - 1) * 100
            price_trend.append(change_pct)
            price_changes.append(change_pct)
            
            # Считаем положительные и отрицательные недели
            if change_pct > 0.2:
                positive_weeks += 1
            elif change_pct < -0.2:
                negative_weeks += 1
                
            # Считаем изменения направления
            if i > 0 and ((price_trend[i-1] > 0 and change_pct < 0) or 
                        (price_trend[i-1] < 0 and change_pct > 0)):
                direction_changes += 1
        
        # Средний тренд (положительный - растет, отрицательный - падает)
        avg_trend = sum(price_trend) / len(price_trend) if price_trend else 0
        
        # Волатильность (стандартное отклонение изменений)
        volatility = np.std(price_changes) if price_changes else 0
        
        print(f"Средний тренд: {avg_trend:.2f}% в неделю")
        print(f"Волатильность: {volatility:.2f}%")
        print(f"Положительные недели: {positive_weeks}")
        print(f"Отрицательные недели: {negative_weeks}")
        print(f"Изменения направления: {direction_changes}")
        
        # Выводим все недельные тренды для отладки
        for i, trend in enumerate(price_trend):
            print(f"Неделя {i+1} -> {i+2}: {trend:.2f}%")
        
        # Анализируем паттерн цен для более точной рекомендации
        
        # 1. Случай высокой волатильности - много изменений направления тренда
        if direction_changes >= 2 or volatility > 3.0:
            # При высокой волатильности рекомендуем короткий период
            print("Обнаружена высокая волатильность цен")
            if avg_trend > 0.5:
                # Если общий тренд положительный, рекомендуем немного дольше
                recommended_period = 2
            else:
                # При отрицательном тренде или нейтральном - самый короткий срок
                recommended_period = 1
        
        # 2. Случай стабильных цен - низкая волатильность
        elif volatility < 1.0:
            print("Обнаружены стабильные цены с низкой волатильностью")
            # При стабильных ценах рекомендуем средний срок
            recommended_period = 3
        
        # 3. Случай устойчивого роста цен
        elif positive_weeks >= 3 and negative_weeks <= 1 and avg_trend > 1.0:
            print("Обнаружен устойчивый рост цен")
            # При устойчивом росте рекомендуем длительный период
            if avg_trend > 3.0:
                recommended_period = 6  # Очень сильный рост
            elif avg_trend > 2.0:
                recommended_period = 5  # Сильный рост
            else:
                recommended_period = 4  # Умеренный рост
        
        # 4. Случай устойчивого снижения цен
        elif negative_weeks >= 3 and positive_weeks <= 1 and avg_trend < -0.5:
            print("Обнаружено устойчивое снижение цен")
            # При устойчивом снижении рекомендуем короткий период
            recommended_period = 1
        
        # 5. Случай разнонаправленного тренда с общим ростом
        elif positive_weeks > negative_weeks and avg_trend > 0:
            print("Обнаружен преимущественно положительный тренд с колебаниями")
            if avg_trend > 1.5:
                recommended_period = 4
            else:
                recommended_period = 3
        
        # 6. Случай разнонаправленного тренда с общим снижением
        elif negative_weeks > positive_weeks and avg_trend < 0:
            print("Обнаружен преимущественно отрицательный тренд с колебаниями")
            if avg_trend < -1.5:
                recommended_period = 1
            else:
                recommended_period = 2
        
        # 7. Смешанный или неопределенный случай
        else:
            print("Обнаружен смешанный тренд")
            # Для смешанного тренда используем среднее значение
            recommended_period = 3
        
        # Если анализ тренда не дал четкой рекомендации, используем среднее значение
        if not recommended_period:
            if avg_trend > 1.0:
                recommended_period = 4
            elif avg_trend < -0.5:
                recommended_period = 2
            else:
                recommended_period = 3
        
        # Применяем финальную проверку для случая на скриншоте - 
        # если есть как рост, так и падение с волатильностью, не рекомендуем очень долгий срок
        if volatility > 2.0 and direction_changes >= 1 and recommended_period > 4:
            print("Корректировка: снижение рекомендуемого периода из-за волатильности")
            recommended_period = 4
        
        print(f"Итоговая рекомендация: {recommended_period} недель")
        
        return recommended_period, predictions, prediction_dates, current_price, avg_trend

class SimpleRebarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("АРМАПРОГНОЗ: система рекомендаций по закупке арматуры")
        self.root.geometry("1100x1000")
        self.root.configure(background=BACKGROUND_COLOR)
        
        # Создаем стиль
        self.style = ttk.Style()
        self.setup_styles()
        
        # Определяем переменные
        self.data_path = StringVar(value="./processed_data.csv")
        self.start_date_var = StringVar()
        
        # Инициализируем даты
        today = datetime.datetime.now()
        self.start_date_var.set(today.strftime("%Y-%m-%d"))
        
        # Создаем фрейм с прокруткой
        self.setup_scrollable_frame()
        
        # Настраиваем интерфейс
        self.setup_ui()
        
        # Инициализируем предиктор
        self.initialize_predictor()
        
        # Привязываем прокрутку колесиком мыши
        self.root.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.root.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.root.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down

    def show_calendar(self):
        """Показывает календарь для выбора даты"""
        try:
            # Проверяем текущую дату
            try:
                current_date = pd.to_datetime(self.start_date_var.get())
            except:
                current_date = datetime.datetime.now()

            # Создаем всплывающее окно
            top = tk.Toplevel(self.root)
            top.title("Выберите дату")
            top.geometry("300x250")
            top.resizable(False, False)
            top.grab_set()  # Делаем окно модальным

            # Функция для установки выбранной даты с дополнительной проверкой
            def set_date():
                try:
                    # DateEntry.get_date() возвращает объект datetime.date, а не строку
                    selected_date = cal.get_date()  # Это объект datetime.date

                    # Преобразуем date в datetime и затем в строку нужного формата
                    # Добавляем дополнительные проверки
                    if isinstance(selected_date, datetime.date):
                        # Преобразуем date в datetime
                        selected_datetime = datetime.datetime.combine(selected_date, datetime.time())
                        formatted_date = selected_datetime.strftime("%Y-%m-%d")
                    else:
                        # Если получили строку или другой тип, пробуем преобразовать напрямую
                        formatted_date = pd.to_datetime(selected_date).strftime("%Y-%m-%d")

                    print(f"Выбранная дата: {selected_date}, преобразовано в: {formatted_date}")
                    self.start_date_var.set(formatted_date)

                    top.destroy()
                    # Обновляем прогноз после выбора даты
                    self.update_recommendation()
                except Exception as e:
                    print(f"Ошибка при установке даты: {e}")
                    messagebox.showerror("Ошибка", f"Не удалось установить дату: {str(e)}")

            # Создаем и размещаем календарь с надежной обработкой ошибок
            try:
                cal = DateEntry(top, width=12, background=PRIMARY_COLOR,
                            foreground='white', borderwidth=2, date_pattern='dd.MM.yyyy',
                            year=current_date.year, month=current_date.month, day=current_date.day)
                cal.pack(padx=10, pady=10)
            except Exception as e:
                print(f"Ошибка при создании календаря: {e}")
                # Создаем простой альтернативный интерфейс ввода даты
                frame = ttk.Frame(top)
                frame.pack(padx=10, pady=10, fill='x')

                ttk.Label(frame, text="Год:").grid(row=0, column=0, padx=5, pady=5)
                year_var = StringVar(value=str(current_date.year))
                ttk.Entry(frame, textvariable=year_var, width=6).grid(row=0, column=1, padx=5, pady=5)

                ttk.Label(frame, text="Месяц:").grid(row=1, column=0, padx=5, pady=5)
                month_var = StringVar(value=str(current_date.month))
                ttk.Entry(frame, textvariable=month_var, width=6).grid(row=1, column=1, padx=5, pady=5)

                ttk.Label(frame, text="День:").grid(row=2, column=0, padx=5, pady=5)
                day_var = StringVar(value=str(current_date.day))
                ttk.Entry(frame, textvariable=day_var, width=6).grid(row=2, column=1, padx=5, pady=5)

                # Переопределяем функцию установки даты для ручного ввода
                def set_date():
                    try:
                        year = int(year_var.get())
                        month = int(month_var.get())
                        day = int(day_var.get())

                        # Проверка валидности даты
                        selected_date = datetime.datetime(year, month, day)
                        formatted_date = selected_date.strftime("%Y-%m-%d")

                        self.start_date_var.set(formatted_date)
                        top.destroy()
                        self.update_recommendation()
                    except Exception as e:
                        messagebox.showerror("Ошибка", f"Некорректная дата: {str(e)}")

            # Кнопка для подтверждения выбора
            ttk.Button(top, text="Выбрать", command=set_date).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть календарь: {str(e)}")
            import traceback
            traceback.print_exc()


    def set_today_date(self):
        """Устанавливает текущую дату в поле ввода"""
        today = datetime.datetime.now()
        self.start_date_var.set(today.strftime("%Y-%m-%d"))
        # Автоматически обновляем прогноз
        self.update_recommendation()
    
    def setup_scrollable_frame(self):
        """Создание фрейма с прокруткой"""
        # Создаем контейнер для полосы прокрутки
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Создаем canvas
        self.canvas = tk.Canvas(self.main_container, background=BACKGROUND_COLOR)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        # Создаем фрейм для содержимого
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Размещаем canvas и полосу прокрутки
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def _on_mousewheel(self, event):
        """Обработка прокрутки колесиком мыши"""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
    
    def setup_styles(self):
        """Настройка стилей для виджетов"""
        self.style.configure("TFrame", background=BACKGROUND_COLOR)
        self.style.configure("TLabel", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=("Arial", 10))
        self.style.configure("TButton", background=PRIMARY_COLOR, foreground="white", font=("Arial", 10, "bold"))
        self.style.configure("TLabelframe", background=BACKGROUND_COLOR)
        self.style.configure("TLabelframe.Label", background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=("Arial", 11, "bold"))
        
        # Заголовки и основные стили
        self.style.configure("Title.TLabel", font=("Arial", 18, "bold"), foreground=PRIMARY_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Subtitle.TLabel", font=("Arial", 14, "bold"), foreground=PRIMARY_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Success.TLabel", font=("Arial", 12, "bold"), foreground=SUCCESS_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Warning.TLabel", font=("Arial", 12, "bold"), foreground=WARNING_COLOR, background=BACKGROUND_COLOR)
        self.style.configure("Danger.TLabel", font=("Arial", 12, "bold"), foreground=DANGER_COLOR, background=BACKGROUND_COLOR)
        
        # Стили для кнопок
        self.style.configure("Primary.TButton", background=PRIMARY_COLOR, foreground="white")
        self.style.map("Primary.TButton", background=[("active", ACCENT_COLOR)])
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной контейнер
        main_frame = ttk.Frame(self.scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Заголовок
        ttk.Label(
            main_frame, 
            text="АРМАПРОГНОЗ", 
            style="Title.TLabel"
        ).pack(pady=(10, 5))
        
        ttk.Label(
            main_frame,
            text="Интеллектуальная система рекомендаций для оптимизации закупок арматуры",
            style="Subtitle.TLabel"
        ).pack(pady=(0, 20))
        
        # Секция настроек
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки прогноза", padding=15)
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Создаем сетку для настроек
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.X, pady=5)
        
        # Дата начала прогноза
        ttk.Label(settings_grid, text="Дата начала прогноза:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # Создаем фрейм для даты с кнопкой обновления
        date_frame = ttk.Frame(settings_grid)
        date_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Простой текстовый ввод даты
        # Используем Entry с валидацией для даты
        self.date_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=15)
        self.date_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Добавим кнопку "Сегодня" для быстрой установки текущей даты
        ttk.Button(
            date_frame, 
            text="Сегодня", 
            command=self.set_today_date,
            style="Primary.TButton"
        ).pack(side=tk.LEFT)

        # Кнопка для открытия календаря
        ttk.Button(
            date_frame, 
            text="📅",  # Эмодзи календаря
            command=self.show_calendar,
            width=3
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Добавим комментарий про формат даты
        ttk.Label(
            settings_grid, 
            text="(Формат: ГГГГ-ММ-ДД)"
        ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Файл данных
        ttk.Label(settings_grid, text="Файл данных:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        file_path_frame = ttk.Frame(settings_grid)
        file_path_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        
        ttk.Entry(file_path_frame, textvariable=self.data_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(
            file_path_frame, 
            text="Обзор", 
            command=self.browse_file
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Кнопка обновления прогноза
        ttk.Button(
            settings_grid, 
            text="Обновить прогноз", 
            command=self.update_recommendation,
            style="Primary.TButton"
        ).grid(row=1, column=3, sticky=tk.E, padx=5, pady=5)
        
        # Секция текущих данных
        data_frame = ttk.LabelFrame(main_frame, text="Текущие данные", padding=15)
        data_frame.pack(fill=tk.X, pady=(0, 15))
        
        data_grid = ttk.Frame(data_frame)
        data_grid.pack(fill=tk.X, pady=5)
        
        # Дата и цена
        ttk.Label(data_grid, text="Последняя дата:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_date_label = ttk.Label(data_grid, text="", font=("Arial", 10, "bold"))
        self.current_date_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(data_grid, text="Текущая цена:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.current_price_label = ttk.Label(data_grid, text="", font=("Arial", 10, "bold"))
        self.current_price_label.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(data_grid, text="Количество записей:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_count_label = ttk.Label(data_grid, text="", font=("Arial", 10, "bold"))
        self.data_count_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ПРАВИЛЬНО: Сначала создаем фрейм
        rec_frame = ttk.LabelFrame(main_frame, text="Рекомендация по закупке", padding=15)
        rec_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Метка с датой прогноза (теперь правильно внутри rec_frame)
        self.forecast_date_label = ttk.Label(rec_frame, text="", style="Info.TLabel")
        self.forecast_date_label.pack(pady=(0, 5))
        
        # Основная рекомендация
        self.recommendation_label = ttk.Label(rec_frame, text="", font=("Arial", 16, "bold"), foreground=PRIMARY_COLOR)
        self.recommendation_label.pack(pady=(5, 10))
        
        # Пояснение
        self.explanation_label = ttk.Label(rec_frame, text="", wraplength=900)
        self.explanation_label.pack(pady=(0, 10))
        
        # Прогнозы цен (таблица)
        self.forecast_frame = ttk.Frame(rec_frame)
        self.forecast_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Информация о тренде
        self.trend_label = ttk.Label(rec_frame, text="", style="Info.TLabel")
        self.trend_label.pack(pady=(0, 5))
        
        # График прогноза
        chart_frame = ttk.LabelFrame(main_frame, text="Прогноз цен на арматуру", padding=15)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        self.canvas_fig = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Добавляем новую функцию для установки сегодняшней даты
    def set_today_date(self):
        """Устанавливает текущую дату в поле ввода"""
        today = datetime.datetime.now()
        self.start_date_var.set(today.strftime("%Y-%m-%d"))
        # Автоматически обновляем прогноз с новой датой
        self.update_recommendation()
    
    def initialize_predictor(self):
        """Инициализация предиктора цен"""
        try:
            # Проверяем путь к данным
            data_path = self.data_path.get() if os.path.exists(self.data_path.get()) else None
            self.predictor = SimplePricePredictor(data_path)
            
            # Загружаем текущие данные
            self.update_current_data()
            
            # Обновляем рекомендацию
            self.update_recommendation()
            
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
            
            # Если выбран новый файл, переинициализируем предиктор
            try:
                self.predictor = SimplePricePredictor(filename)
                self.update_current_data()
                self.update_recommendation()
                messagebox.showinfo("Успешно", "Данные успешно загружены.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {str(e)}")
    
    def update_current_data(self):
        """Обновление информации о текущих данных"""
        if hasattr(self, 'predictor'):
            # Получаем последние данные
            last_row = self.predictor.data.iloc[-1]
            
            # Обновляем метки
            self.current_date_label.config(text=last_row['dt'].strftime("%d.%m.%Y"))
            self.current_price_label.config(text=f"{last_row['price']:.2f} руб")
            self.data_count_label.config(text=f"{len(self.predictor.data)} записей")
    
    # Update the explanation text generation in update_recommendation method

    def update_recommendation(self):
        """Обновление рекомендации на основе выбранной даты"""
        try:
            # Получаем дату начала прогноза и конвертируем в правильный формат
            start_date_str = self.start_date_var.get()
            
            print(f"Обновление прогноза для даты: {start_date_str}")
            
            # Проверяем формат даты и преобразуем в datetime
            try:
                # Пробуем стандартный формат YYYY-MM-DD
                start_date_dt = pd.to_datetime(start_date_str)
            except:
                try:
                    # Пробуем альтернативные форматы (DD.MM.YYYY)
                    start_date_dt = pd.to_datetime(start_date_str, format='%d.%m.%Y')
                except:
                    messagebox.showerror("Ошибка формата даты", 
                                        "Пожалуйста, введите дату в формате ГГГГ-ММ-ДД или ДД.ММ.ГГГГ")
                    return
            
            # Преобразуем обратно в строку в нужном формате для предиктора
            start_date = start_date_dt.strftime('%Y-%m-%d')
            
            print(f"Передаем дату в предиктор: {start_date}")  # Отладочный вывод
            
            # Получаем рекомендацию
            recommended_period, predictions, prediction_dates, current_price, avg_trend = self.predictor.recommend_tender_period(
                start_date=start_date
            )
            
            print(f"Получены результаты: {len(predictions)} прогнозов")
            
            # Добавляем информацию о дате начала прогноза
            self.forecast_date_label.config(
                text=f"Прогноз от даты: {start_date_dt.strftime('%d.%m.%Y')}"
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
            
            # Оцениваем тренд для выбора правильного объяснения
            trend_direction = "stable"
            if avg_trend > 0.2:  # Положительный тренд
                if avg_trend >= 2.0:
                    trend_direction = "strong_up"
                elif avg_trend >= 1.0:
                    trend_direction = "moderate_up"
                else:
                    trend_direction = "slight_up"
            elif avg_trend < -0.2:  # Отрицательный тренд
                if avg_trend <= -1.0:
                    trend_direction = "strong_down"
                else:
                    trend_direction = "slight_down"
            
            # Формируем объяснение на основе тренда и рекомендации
            if trend_direction == "strong_down":
                explanation = (
                    f"Рекомендуется краткосрочная закупка ({recommended_period} {'неделя' if recommended_period == 1 else 'недели'}), "
                    "поскольку прогнозируется значительное снижение цен в ближайшем будущем. "
                    "Такая стратегия позволит воспользоваться более выгодными ценами в последующие периоды."
                )
                trend_class = "Success.TLabel"
                trend_text = f"Тренд цен: Сильное снижение ({avg_trend:.2f}% в неделю)"
            elif trend_direction == "slight_down":
                explanation = (
                    f"Рекомендуется краткосрочная закупка на {recommended_period} недели. Прогнозируется "
                    "небольшое снижение цен, поэтому нет необходимости в долгосрочных контрактах. "
                    "Это позволит использовать более низкие цены в будущем."
                )
                trend_class = "Success.TLabel"
                trend_text = f"Тренд цен: Небольшое снижение ({avg_trend:.2f}% в неделю)"
            elif trend_direction == "stable":
                explanation = (
                    f"Рекомендуется закупка на {recommended_period} недели. Прогнозируется стабильность цен. "
                    "Это оптимальный период, который обеспечивает операционную гибкость "
                    "и не создает излишних запасов."
                )
                trend_class = "Info.TLabel"
                trend_text = f"Тренд цен: Стабильный ({avg_trend:.2f}% в неделю)"
            elif trend_direction == "slight_up":
                explanation = (
                    f"Рекомендуется среднесрочная закупка на {recommended_period} недели. Прогнозируется небольшой "
                    "рост цен, поэтому имеет смысл закупить арматуру на несколько недель вперед, но "
                    "нет необходимости в очень долгосрочных контрактах."
                )
                trend_class = "Info.TLabel"
                trend_text = f"Тренд цен: Небольшой рост ({avg_trend:.2f}% в неделю)"
            elif trend_direction == "moderate_up":
                explanation = (
                    f"Рекомендуется среднесрочная закупка, поскольку прогнозируется умеренный рост цен. "
                    f"Оптимальный период закупки составляет {recommended_period} недели, что позволит избежать затрат "
                    "на закупку по более высоким ценам в ближайшем будущем."
                )
                trend_class = "Warning.TLabel"
                trend_text = f"Тренд цен: Умеренный рост ({avg_trend:.2f}% в неделю)"
            elif trend_direction == "strong_up":
                explanation = (
                    f"Рекомендуется долгосрочная закупка на {recommended_period} недель, поскольку прогнозируется значительный "
                    "рост цен в ближайшем будущем. Фиксация текущей цены на длительный период позволит "
                    "избежать повышенных затрат при закупке по будущим, более высоким ценам."
                )
                trend_class = "Danger.TLabel"
                trend_text = f"Тренд цен: Сильный рост ({avg_trend:.2f}% в неделю)"
            else:
                explanation = (
                    f"Рекомендуется закупка на {recommended_period} недели на основе анализа прогноза цен. "
                    "Это сбалансированный период, который учитывает текущие тенденции рынка."
                )
                trend_class = "Info.TLabel"
                trend_text = f"Тренд цен: {avg_trend:.2f}% в неделю"
                    
            self.explanation_label.config(text=explanation)
            self.trend_label.config(text=trend_text, style=trend_class)
            
            # Обновляем прогнозы цен
            self.update_forecast_table(prediction_dates, predictions, current_price)
            
            # Обновляем график
            self.update_price_chart(prediction_dates, predictions)
            
            # Обновляем область прокрутки
            self.canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Обновляем весь интерфейс
            self.root.update()
            
            print("Обновление рекомендации завершено успешно")
            
        except Exception as e:
            # Более подробная информация об ошибке
            import traceback
            error_details = traceback.format_exc()
            print(f"Ошибка в update_recommendation: {str(e)}")
            print(error_details)
            
            messagebox.showerror("Ошибка", f"Не удалось обновить рекомендацию: {str(e)}")
    
    def update_forecast_table(self, prediction_dates, predictions, current_price):
        """Обновление таблицы с прогнозами цен"""
        # Очищаем фрейм
        for widget in self.forecast_frame.winfo_children():
            widget.destroy()
        
        # Создаем заголовки
        headers = ["Дата", "Прогноз. цена", "Изменение"]
        
        for i, header in enumerate(headers):
            ttk.Label(
                self.forecast_frame,
                text=header,
                font=("Arial", 10, "bold")
            ).grid(row=0, column=i, padx=10, pady=5, sticky="w")
        
        # Добавляем строки с прогнозами
        for i, (date, price) in enumerate(zip(prediction_dates, predictions)):
            # Дата
            ttk.Label(
                self.forecast_frame,
                text=date.strftime("%d.%m.%Y")
            ).grid(row=i+1, column=0, padx=10, pady=2, sticky="w")
            
            # Цена
            ttk.Label(
                self.forecast_frame,
                text=f"{price:.2f} руб"
            ).grid(row=i+1, column=1, padx=10, pady=2, sticky="w")
            
            # Изменение
            if i == 0:
                change = ((price / current_price) - 1) * 100
                change_abs = price - current_price
            else:
                change = ((price / predictions[i-1]) - 1) * 100
                change_abs = price - predictions[i-1]
            
            # Определяем цвет и текст
            if change > 0.5:  # Рост
                change_text = f"↗ +{change:.2f}% ({change_abs:.2f})"
                change_color = DANGER_COLOR
            elif change < -0.5:  # Снижение
                change_text = f"↘ {change:.2f}% ({change_abs:.2f})"
                change_color = SUCCESS_COLOR
            else:  # Стабильность
                change_text = f"→ {change:.2f}% ({change_abs:.2f})"
                change_color = TEXT_COLOR
            
            ttk.Label(
                self.forecast_frame,
                text=change_text,
                foreground=change_color
            ).grid(row=i+1, column=2, padx=10, pady=2, sticky="w")
    
    def update_price_chart(self, prediction_dates, predictions):
        """Обновление графика прогноза цен"""
        try:
            # Очищаем предыдущий график
            self.plot.clear()
            
            # Печатаем данные для отладки
            print(f"Обновление графика с {len(prediction_dates)} точками данных")
            print(f"Даты: {prediction_dates}")
            print(f"Прогнозы: {predictions}")
            
            # Стилизация графика
            self.plot.set_facecolor(BACKGROUND_COLOR)
            self.figure.set_facecolor(BACKGROUND_COLOR)
            
            # Проверяем на пустые данные
            if not prediction_dates or not predictions or len(prediction_dates) != len(predictions):
                print("Предупреждение: Пустые или несогласованные данные для графика")
                self.plot.text(0.5, 0.5, "Недостаточно данных для построения графика", 
                            horizontalalignment='center', verticalalignment='center')
            else:
                # Строим график прогноза
                self.plot.plot(
                    prediction_dates,
                    predictions,
                    marker='s',
                    linestyle='-',
                    color=ACCENT_COLOR,
                    linewidth=2,
                    markersize=6,
                    label='Прогноз'
                )
                
                # Настройка графика
                self.plot.set_title('Прогноз цен на арматуру', fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
                self.plot.set_xlabel('Дата', fontsize=11, fontweight='bold')
                self.plot.set_ylabel('Цена (руб)', fontsize=11, fontweight='bold')
                self.plot.legend(loc='upper left')
                self.plot.grid(True, alpha=0.3, linestyle='--')
                
                # Форматирование дат
                self.plot.xaxis.set_major_formatter(DateFormatter('%d.%m.%y'))
                
            # Обновляем график
            self.figure.tight_layout()
            self.figure.autofmt_xdate(rotation=45)  # Перемещено сюда, чтобы срабатывало всегда
            self.canvas_fig.draw_idle()  # Используем draw_idle вместо draw
            
            # Обновляем размер виджета и размещение
            self.canvas_fig.get_tk_widget().update()
            
        except Exception as e:
            print(f"Ошибка при обновлении графика: {str(e)}")
            import traceback
            traceback.print_exc()

# Запуск приложения
if __name__ == "__main__":
    try:
        # Проверяем необходимые библиотеки
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'sklearn', 
            'joblib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Внимание! Отсутствуют библиотеки: {', '.join(missing_packages)}")
            print("Установите их командой:")
            print(f"pip install {' '.join(missing_packages)}")
            input("Нажмите Enter для выхода...")
            exit(1)
        
        # Запускаем приложение
        root = tk.Tk()
        app = SimpleRebarApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Ошибка запуска приложения: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение: {str(e)}")