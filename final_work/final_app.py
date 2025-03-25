import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from matplotlib.dates import DateFormatter

PRIMARY_BLUE = '#FFFFF'      # Темно-синий цвет
LIGHT_BLUE = '#0099FF'        # Светло-голубой фон
TEXT_COLOR = '#FFFFFF'        # Темно-серый цвет текста
ACCENT_COLOR = '#FFFFFF'      # Яркий синий для акцентов
WHITE_COLOR = '#FFFFFF'       # Белый цвет

class RebarPricePredictor:
    def __init__(self):
        # Load the data
        try:
            self.train_data = pd.read_csv('./vladislav_work/last_chance_to_made_app/processed_data.csv')
            # Проверяем формат даты в данных
            if 'dt' in self.train_data.columns:
                # Преобразуем столбец date в формат datetime
                self.train_data['dt'] = pd.to_datetime(self.train_data['dt'], errors='coerce')
            else:
                # Если столбец с датой называется иначе, ищем его
                date_columns = [col for col in self.train_data.columns if 'dt' in col.lower() or 'дата' in col.lower()]
                if date_columns:
                    self.train_data['dt'] = pd.to_datetime(self.train_data[date_columns[0]], errors='coerce')
                else:
                    # Если столбца с датой нет, создаем искусственную дату
                    self.train_data['dt'] = pd.date_range(start='2015-01-01', periods=len(self.train_data), freq='W-MON')
            
            # Проверяем наличие столбца с ценой
            price_columns = [col for col in self.train_data.columns if 'Цена на арматуру' in col.lower() or 'цена' in col.lower()]
            if price_columns:
                self.train_data.rename(columns={price_columns[0]: 'Цена на арматуру'}, inplace=True)
            else:
                # Если явно не указан столбец с ценой, используем второй числовой столбец (предполагая, что первый - это индекс или дата)
                numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    self.train_data.rename(columns={numeric_cols[0]: 'Цена на арматуру'}, inplace=True)
                else:
                    raise ValueError("Не найден столбец с ценами в данных")
            
            # Prepare the data
            self.prepare_data()
            
            # Train the model
            self.train_model()

            
            
        except Exception as e:
            print(f"Ошибка при инициализации предиктора: {str(e)}")
            raise
    
    def prepare_data(self):
        # Sort by date
        self.train_data = self.train_data.sort_values('dt')
        
        # Feature engineering
        # Add day of week, month, year features
        self.train_data['day_of_week'] = self.train_data['dt'].dt.dayofweek
        self.train_data['month'] = self.train_data['dt'].dt.month
        self.train_data['year'] = self.train_data['dt'].dt.year
        
        # Add lag features (previous weeks' prices)
        for i in range(1, 5):
            self.train_data[f'price_lag_{i}'] = self.train_data['Цена на арматуру'].shift(i)
        
        # Calculate moving averages
        self.train_data['ma_4'] = self.train_data['Цена на арматуру'].rolling(window=4).mean()
        self.train_data['ma_12'] = self.train_data['Цена на арматуру'].rolling(window=12).mean()
        
        # Calculate price momentum (percent change)
        self.train_data['price_pct_change'] = self.train_data['Цена на арматуру'].pct_change(periods=4)
        
        # Drop rows with NaN values from lag features
        self.train_data = self.train_data.dropna()
        
        # Prepare features and target
        self.features = [
            'day_of_week', 'month', 'year',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
            'ma_4', 'ma_12', 'price_pct_change'
        ]
        
        self.X = self.train_data[self.features]
        self.y = self.train_data['Цена на арматуру']
    
    def train_model(self):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train a Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Model MAE: {mae}")
    
    def predict_next_n_weeks(self, current_data, n_weeks=6):
        """Predict prices for the next N weeks"""
        predictions = []
        prediction_dates = []
        temp_data = current_data.copy()
        
        for week in range(1, n_weeks + 1):
            # Prepare features for prediction
            latest_row = temp_data.iloc[-1:].copy()
            
            # Безопасно вычисляем следующую дату
            next_date = latest_row['dt'].iloc[0] + pd.Timedelta(days=7)
            prediction_dates.append(next_date)
            
            # Создаем новую строку данных
            new_row = latest_row.copy()
            new_row['dt'] = next_date
            new_row['day_of_week'] = next_date.dayofweek
            new_row['month'] = next_date.month
            new_row['year'] = next_date.year
            
            # Scale the features
            X_pred = new_row[self.features]
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make prediction
            price_pred = self.model.predict(X_pred_scaled)[0]
            
            # Save the prediction
            new_row['Цена на арматуру'] = price_pred
            predictions.append(price_pred)
            
            # Update lag features for next prediction
            for i in range(4, 1, -1):
                new_row[f'price_lag_{i}'] = temp_data.iloc[-1][f'price_lag_{i-1}']
            
            new_row['price_lag_1'] = temp_data.iloc[-1]['Цена на арматуру']
            
            # Update moving averages and momentum
            prices = list(temp_data.iloc[-11:]['Цена на арматуру']) + [price_pred]
            new_row['ma_4'] = np.mean(prices[-4:])
            new_row['ma_12'] = np.mean(prices[-12:])
            new_row['price_pct_change'] = (price_pred / prices[-5] - 1) if prices[-5] != 0 else 0
            
            # Add the prediction to temp_data for the next iteration
            temp_data = pd.concat([temp_data, new_row])
            
        return predictions, prediction_dates
    
    def calculate_strategy_costs(self, current_price, predictions, weekly_volume):
        """Calculate the cost for each purchase strategy (1-6 weeks)"""
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
        """Recommend optimal tender period (1-6 weeks)"""
        predictions, prediction_dates = self.predict_next_n_weeks(current_data)
        current_price = current_data.iloc[-1]['Цена на арматуру']
        
        # Рассчитываем стоимость для каждой стратегии закупки
        strategy_costs = self.calculate_strategy_costs(current_price, predictions, weekly_volume)
        
        # Находим стратегию с минимальной стоимостью
        recommended_period = min(strategy_costs, key=strategy_costs.get)
        
        # Для наглядности, если цены стабильно падают - рекомендуем 1 неделю
        # Если цены стабильно растут - рекомендуем 6 недель
        if all(x > y for x, y in zip(predictions[:-1], predictions[1:])):
            # Цены стабильно падают
            recommended_period = 1
        elif all(x < y for x, y in zip(predictions[:-1], predictions[1:])):
            # Цены стабильно растут
            recommended_period = 6
        
        return recommended_period, predictions, prediction_dates, strategy_costs

class RebarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система рекомендаций для закупки арматуры")
        self.root.geometry("1000x800")
        self.root.configure(background=LIGHT_BLUE)  # Установка фона корневого окна
        
        # Создаем стиль с корпоративными цветами
        self.style = ttk.Style()

        # Определяем переменные
        self.weekly_volume = StringVar(value="1")
        
        # Create a canvas with scrollbar for the entire application
        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
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
        
        # Setup UI in the scrollable frame
        self.setup_ui()
        
        # Initialize the predictor
        try:
            self.predictor = RebarPricePredictor()
            self.load_current_data()
        except Exception as e:
            messagebox.showerror("Ошибка загрузки данных", 
                                f"Не удалось загрузить данные: {str(e)}\n"
                                "Убедитесь, что файлы train.csv и test.csv находятся в текущей директории.")
        
        # Bind mousewheel to scroll canvas
        self.root.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.root.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.root.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down
    
    def _on_mousewheel(self, event):
        # Handle scrolling with mouse wheel
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.scrollable_frame, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Система рекомендаций для закупки арматуры", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Current data frame
        data_frame = ttk.LabelFrame(main_frame, text="Текущие данные", padding="10")
        data_frame.pack(fill=tk.X, pady=10)
        
        # Current date and price
        input_frame = ttk.Frame(data_frame)
        input_frame.pack(fill=tk.X)
        
        # Добавляем сетку для лучшего выравнивания
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=2)
        input_frame.columnconfigure(2, weight=1)
        input_frame.columnconfigure(3, weight=2)
        
        # Текущая дата и цена
        ttk.Label(input_frame, text="Текущая дата:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.current_date_label = ttk.Label(input_frame, text="", font=("Arial", 10, "bold"))
        self.current_date_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(input_frame, text="Текущая цена:", style='TLabelframe').grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.current_price_label = ttk.Label(input_frame, text="", font=("Arial", 10, "bold"))
        self.current_price_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Объем еженедельной потребности
        ttk.Label(input_frame, text="Еженедельная потребность (тонн):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        volume_entry = ttk.Entry(input_frame, textvariable=self.weekly_volume, width=10)
        volume_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Recommendation frame
        rec_frame = ttk.LabelFrame(main_frame, text="Рекомендация по закупке", style='TLabelframe')
        rec_frame.pack(fill=tk.X, pady=10)
        
        self.recommendation_label = ttk.Label(rec_frame, text="", font=("Arial", 14, "bold"))
        self.recommendation_label.pack(pady=10)
        
        self.explanation_label = ttk.Label(rec_frame, text="", wraplength=950)
        self.explanation_label.pack(pady=5)
        
        # Прогнозные цены
        self.forecast_frame = ttk.Frame(rec_frame)
        self.forecast_frame.pack(fill=tk.X, pady=5)
        
        # Экономия от рекомендации
        self.savings_label = ttk.Label(rec_frame, text="", font=("Arial", 12, "bold"), foreground="green")
        self.savings_label.pack(pady=5)
        
        # Graph frame for price prediction
        graph_frame = ttk.LabelFrame(main_frame, text="Прогноз цен", style='TLabelframe')
        graph_frame.pack(fill=tk.X, pady=10)
        
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        self.canvas_fig = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.X)
        
        # График сравнения стратегий
        strategy_graph_frame = ttk.LabelFrame(main_frame, text="Сравнение стратегий закупок", style='TLabelframe')
        strategy_graph_frame.pack(fill=tk.X, pady=10)
        
        self.strategy_figure = Figure(figsize=(8, 3), dpi=100)
        self.strategy_plot = self.strategy_figure.add_subplot(111)
        
        self.strategy_canvas = FigureCanvasTkAgg(self.strategy_figure, strategy_graph_frame)
        self.strategy_canvas.get_tk_widget().pack(fill=tk.X)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # В методе setup_ui(), измените создание кнопок:
        refresh_button = ttk.Button(button_frame, text="Обновить прогноз", command=self.update_recommendation)
        refresh_button.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        help_button = ttk.Button(button_frame, text="Справка о приложении", command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

    def show_help(self):
        """Display a help dialog with application information"""
        help_text = """Система рекомендаций для закупки арматуры

Основные возможности:
1. Прогнозирование цен на арматуру на ближайшие 6 недель
2. Рекомендация оптимального периода закупки
3. Визуализация исторических и прогнозируемых цен
4. Сравнение стратегий закупок

Как пользоваться:
- Введите еженедельную потребность в тоннах
- Нажмите "Обновить прогноз" для получения рекомендаций
- Изучите графики прогноза цен и сравнения стратегий

Рекомендации основаны на:
- Исторических ценах
- Тенденциях рынка
- Прогнозировании с использованием машинного обучения

Точность прогноза зависит от качества и объема исторических данных."""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Справка")
        help_window.geometry("500x500")
        help_window.configure(background=LIGHT_BLUE)
        
        help_label = ttk.Label(help_window, 
                               text=help_text, 
                               wraplength=480, 
                               justify=tk.LEFT, 
                               style='TLabel')
        help_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        close_button = ttk.Button(help_window, text="Закрыть", style='TButton', command=help_window.destroy)
        close_button.pack(pady=10)
    
    def load_current_data(self):
        """Load the most recent data for prediction"""
        # Assuming the train data is sorted by date
        self.current_data = self.predictor.train_data.copy()
        
        # Display current date and price
        last_row = self.current_data.iloc[-1]
        self.current_date = last_row['dt']
        self.current_price = last_row['Цена на арматуру']
        
        self.current_date_label.config(text=self.current_date.strftime("%d.%m.%Y"))
        self.current_price_label.config(text=f"{self.current_price:.2f} руб./т")
        
        # Update recommendation
        self.update_recommendation()
    
    def update_recommendation(self):
        """Update the recommendation based on current data"""
        try:
            # Получаем еженедельный объем из поля ввода с проверкой
            try:
                weekly_volume = float(self.weekly_volume.get())
            except ValueError:
                messagebox.showerror("Ошибка ввода", "Введите корректное числовое значение для еженедельной потребности.")
                return
            
            # Get recommendation
            recommended_period, predictions, prediction_dates, strategy_costs = self.predictor.recommend_tender_period(
                self.current_data, 
                weekly_volume=weekly_volume
            )
            
            # Update recommendation label
            self.recommendation_label.config(
                text=f"Рекомендуемый период закупки: {recommended_period} {'неделя' if recommended_period == 1 else 'недели' if 1 < recommended_period < 5 else 'недель'}"
            )
            
            # Update explanation
            if recommended_period == 1:
                explanation = "Рекомендуется закупка на 1 неделю, так как прогнозируется снижение цен в ближайшем будущем."
            elif recommended_period >= 5:
                explanation = "Рекомендуется закупка на длительный период, так как прогнозируется рост цен в ближайшем будущем."
            else:
                explanation = f"Рекомендуется закупка на {recommended_period} {'недели' if recommended_period < 5 else 'недель'} на основе прогноза динамики цен."
            
            self.explanation_label.config(text=explanation)
            
            # Обновляем прогнозные цены
            self.update_forecast_prices(prediction_dates, predictions)
            
            # Рассчитываем и отображаем экономию
            worst_strategy = max(strategy_costs.items(), key=lambda x: x[1])
            best_strategy = min(strategy_costs.items(), key=lambda x: x[1])
            
            savings = worst_strategy[1] - best_strategy[1]
            savings_pct = (savings / worst_strategy[1]) * 100
            
            self.savings_label.config(
                text=f"Экономия от следования рекомендации: {savings:.2f} руб. ({savings_pct:.2f}%)"
            )
            
            # Update graphs
            self.plot_predictions(prediction_dates, predictions)
            self.plot_strategy_comparison(strategy_costs)
            
            # Update the canvas to show all content
            self.canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обновить рекомендацию: {str(e)}")
    
    def update_forecast_prices(self, prediction_dates, predictions):
        """Update the forecast prices display"""
        # Очищаем предыдущие виджеты
        for widget in self.forecast_frame.winfo_children():
            widget.destroy()
        
        # Создаем заголовки
        ttk.Label(self.forecast_frame, text="Прогнозные цены:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Отображаем прогнозные цены для каждой недели
        for i, (date, price) in enumerate(zip(prediction_dates, predictions)):
            week_num = i + 1
            date_str = date.strftime("%d.%m.%Y")
            
            # Определяем цвет в зависимости от изменения цены
            if i == 0:
                price_change = price - self.current_price
                color = "green" if price_change < 0 else "red" if price_change > 0 else "black"
            else:
                price_change = price - predictions[i-1]
                color = "green" if price_change < 0 else "red" if price_change > 0 else "black"
            
            # Отображаем информацию
            ttk.Label(self.forecast_frame, text=f"Неделя {week_num} ({date_str}):").grid(row=0, column=i+1, padx=10, pady=5)
            price_label = ttk.Label(self.forecast_frame, 
                                    text=f"{price:.2f} руб./т", 
                                    foreground=color,
                                    font=("Arial", 10, "bold"))
            price_label.grid(row=1, column=i+1, padx=10, pady=5)
            
            # Отображаем изменение цены
            if i == 0:
                change_text = f"{price_change:+.2f} ({price_change/self.current_price*100:+.2f}%)"
            else:
                change_text = f"{price_change:+.2f} ({price_change/predictions[i-1]*100:+.2f}%)"
            
            change_label = ttk.Label(self.forecast_frame, text=change_text, foreground=color)
            change_label.grid(row=2, column=i+1, padx=10, pady=5)
    
    def plot_predictions(self, prediction_dates, predictions):
        """Plot the price predictions"""
        self.plot.clear()
        
        # Get historical data for plotting (last 12 weeks)
        historical_data = self.current_data.iloc[-12:]
        dates = historical_data['dt'].tolist()
        prices = historical_data['Цена на арматуру'].tolist()
        
        # Добавляем текущую дату и цену в список будущих дат и цен для непрерывности линии
        all_dates = dates + [dates[-1]] + prediction_dates
        all_prices = prices + [prices[-1]] + predictions
        
        # Plot the continuous line
        self.plot.plot(all_dates, all_prices, marker='', linestyle='-', color='blue', alpha=0.3)
        
        # Plot historical data
        self.plot.plot(dates, prices, marker='o', linestyle='-', color='blue', label='Исторические цены')
        
        # Plot predicted data
        self.plot.plot(prediction_dates, predictions, marker='x', linestyle='--', color='red', label='Прогноз')
        
        # Highlight current price
        self.plot.scatter([dates[-1]], [prices[-1]], color='green', s=100, zorder=5, label='Текущая цена')
        
        # Добавляем аннотации с ценами к прогнозным точкам
        for date, price in zip(prediction_dates, predictions):
            self.plot.annotate(f"{price:.0f}", 
                             (date, price),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center')
        
        # Format the plot
        self.plot.set_title('Исторические и прогнозируемые цены на арматуру')
        self.plot.set_xlabel('Дата')
        self.plot.set_ylabel('Цена (руб./т)')
        self.plot.legend()
        self.plot.grid(True, alpha=0.3)
        
        # Format date axis
        self.plot.xaxis.set_major_formatter(DateFormatter('%d.%m.%y'))
        self.figure.autofmt_xdate(rotation=45)
        
        # Настраиваем диапазон осей
        min_price = min(min(prices), min(predictions)) * 0.98
        max_price = max(max(prices), max(predictions)) * 1.02
        self.plot.set_ylim(min_price, max_price)
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas_fig.draw()
    
    def plot_strategy_comparison(self, strategy_costs):
        """Plot comparison of different purchase strategies"""
        self.strategy_plot.clear()
        
        # Convert strategy costs to list
        weeks = list(strategy_costs.keys())
        costs = list(strategy_costs.values())
        
        # Determine the best strategy
        best_strategy_idx = costs.index(min(costs))
        
        # Create colors, highlight the best strategy
        colors = ['lightgray'] * len(weeks)
        colors[best_strategy_idx] = 'green'
        
        # Create the bar chart
        bars = self.strategy_plot.bar(weeks, costs, color=colors)
        
        # Add cost labels above each bar
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            self.strategy_plot.text(
                bar.get_x() + bar.get_width()/2.,
                height + costs[best_strategy_idx] * 0.01,
                f'{cost:.0f}',
                ha='center', va='bottom', rotation=0
            )
        
        # Label the recommended strategy
        bars[best_strategy_idx].set_label('Рекомендуемая стратегия')
        
        # Format the plot
        self.strategy_plot.set_xlabel('Период закупки (недель)')
        self.strategy_plot.set_ylabel('Общие затраты (руб.)')
        self.strategy_plot.set_title('Сравнение стратегий закупки')
        self.strategy_plot.legend()
        self.strategy_plot.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0
        min_cost = min(costs) * 0.95
        max_cost = max(costs) * 1.05
        self.strategy_plot.set_ylim(min_cost, max_cost)
        
        # Update canvas
        self.strategy_figure.tight_layout()
        self.strategy_canvas.draw()

# Run the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RebarApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Ошибка запуска приложения: {str(e)}")
        messagebox.showerror("Ошибка", f"Ошибка запуска приложения: {str(e)}")