import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

class EnhancedRebarPredictor:
    def __init__(self, data_path='./vladislav_work/last_chance_to_made_app/processed_data.csv'):
        """
        Enhanced Rebar Price Predictor with advanced feature engineering and model selection
        
        Parameters:
        - data_path: Path to the CSV file containing historical rebar price data
        """
        self.data_path = data_path
        self.load_and_prepare_data()
        self.create_advanced_features()
        self.split_data()
        self.train_models()
    
    def load_and_prepare_data(self):
        """Load and preprocess the raw data"""
        try:
            # Load data with flexible date and price column detection
            self.train_data = pd.read_csv(self.data_path)
            
            # Date handling
            if 'dt' not in self.train_data.columns:
                date_columns = [col for col in self.train_data.columns if 'dt' in col.lower() or 'дата' in col.lower()]
                if date_columns:
                    self.train_data.rename(columns={date_columns[0]: 'dt'}, inplace=True)
                else:
                    # Generate synthetic dates if no date column exists
                    self.train_data['dt'] = pd.date_range(start='2015-01-01', periods=len(self.train_data), freq='W-MON')
            
            # Convert to datetime
            self.train_data['dt'] = pd.to_datetime(self.train_data['dt'], errors='coerce')
            
            # Price column handling
            price_columns = [col for col in self.train_data.columns if 'Цена на арматуру' in col.lower() or 'цена' in col.lower()]
            if price_columns:
                self.train_data.rename(columns={price_columns[0]: 'price'}, inplace=True)
            else:
                # Use the second numeric column if no explicit price column
                numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    self.train_data.rename(columns={numeric_cols[1]: 'price'}, inplace=True)
                else:
                    raise ValueError("No price column found in the data")
            
            # Sort by date
            self.train_data.sort_values('dt', inplace=True)
            self.train_data.reset_index(drop=True, inplace=True)
        
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_advanced_features(self):
        """
        Create sophisticated time series and statistical features
        """
        # Temporal features
        self.train_data['year'] = self.train_data['dt'].dt.year
        self.train_data['month'] = self.train_data['dt'].dt.month
        self.train_data['week'] = self.train_data['dt'].dt.isocalendar().week
        self.train_data['day_of_year'] = self.train_data['dt'].dt.dayofyear
        
        # Season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Autumn
        
        self.train_data['season'] = self.train_data['month'].apply(get_season)
        
        # Lag features
        for lag in [1, 2, 3, 4, 12]:
            self.train_data[f'price_lag_{lag}'] = self.train_data['price'].shift(lag)
        
        # Rolling window features
        windows = [4, 12, 24]
        for window in windows:
            self.train_data[f'price_rolling_mean_{window}'] = self.train_data['price'].rolling(window=window).mean()
            self.train_data[f'price_rolling_std_{window}'] = self.train_data['price'].rolling(window=window).std()
        
        # Percentage change features
        for period in [1, 4, 12]:
            self.train_data[f'price_pct_change_{period}'] = self.train_data['price'].pct_change(periods=period)
        
        # Exponential weighted moving average
        self.train_data['ewma_price'] = self.train_data['price'].ewm(span=12).mean()
        
        # Time-based trend feature
        self.train_data['trend'] = np.arange(len(self.train_data))
        
        # Drop rows with NaN values
        self.train_data.dropna(inplace=True)
    
    def split_data(self, test_size=0.2):
        """Split data into features and target with time series split"""
        # Select features
        feature_columns = [
            'year', 'month', 'week', 'day_of_year', 'season', 'trend',
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lag_12',
            'price_rolling_mean_4', 'price_rolling_mean_12', 'price_rolling_mean_24',
            'price_rolling_std_4', 'price_rolling_std_12', 'price_rolling_std_24',
            'price_pct_change_1', 'price_pct_change_4', 'price_pct_change_12',
            'ewma_price'
        ]
        
        self.X = self.train_data[feature_columns]
        self.y = self.train_data['price']
        
        # Time series split for more reliable validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, shuffle=False
        )
    
    def train_models(self):
        """
        Train multiple models and select the best performer
        """
        # Preprocessing pipeline
        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_regression, k=10)),  # Select top 10 features
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        # Model definitions
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model evaluation and selection
        self.model_performances = {}
        
        for name, model in models.items():
            # Create full pipeline
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                full_pipeline, 
                self.X_train, 
                self.y_train, 
                cv=tscv, 
                scoring='neg_mean_absolute_error'
            )
            
            # Fit on full training data
            full_pipeline.fit(self.X_train, self.y_train)
            
            # Predictions and metrics
            y_pred = full_pipeline.predict(self.X_test)
            
            self.model_performances[name] = {
                'MAE': mean_absolute_error(self.y_test, y_pred),
                'MSE': mean_squared_error(self.y_test, y_pred),
                'R2': r2_score(self.y_test, y_pred),
                'CV_MAE_Mean': -cv_scores.mean(),
                'CV_MAE_Std': cv_scores.std()
            }
        
        # Select best model based on cross-validation MAE
        self.best_model_name = min(
            self.model_performances, 
            key=lambda x: self.model_performances[x]['CV_MAE_Mean']
        )
        
        # Store the best model
        self.best_model = list(models.values())[list(models.keys()).index(self.best_model_name)]
        
        # Final training of the best model
        self.preprocessor = preprocessor
        self.final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.best_model)
        ])
        self.final_model.fit(self.X_train, self.y_train)
    
    def predict(self, forecast_periods=6):
        """
        Forecast future prices
        """
        def get_season(month):
            if month in [12, 1, 2]:
                return 0
            elif month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8]:
                return 2
            else:
                return 3
        
        # Get the last data point
        last_data = self.train_data.iloc[-1:].copy()
        predictions = []
        prediction_dates = []
        
        for _ in range(forecast_periods):
            # Generate features for next period
            next_date = last_data['dt'].iloc[0] + pd.Timedelta(weeks=1)
            next_row = last_data.copy()
            
            # Update temporal features
            next_row['dt'] = next_date
            next_row['year'] = next_date.year
            next_row['month'] = next_date.month
            next_row['week'] = next_date.isocalendar().week
            next_row['day_of_year'] = next_date.dayofyear
            next_row['season'] = get_season(next_date.month)
            next_row['trend'] += 1
            
            # Update lag and rolling features
            next_row['price_lag_1'] = last_data['price'].iloc[0]
            for i in range(2, 5):
                next_row[f'price_lag_{i}'] = last_data[f'price_lag_{i-1}'].iloc[0]
            next_row['price_lag_12'] = last_data['price_lag_11'].iloc[0]
            
            # Compute rolling features
            recent_prices = self.train_data['price'].iloc[-24:].tolist() + [next_row['price_lag_1']]
            for window in [4, 12, 24]:
                next_row[f'price_rolling_mean_{window}'] = np.mean(recent_prices[-window:])
                next_row[f'price_rolling_std_{window}'] = np.std(recent_prices[-window:])
            
            # Percentage change and EWMA
            next_row['price_pct_change_1'] = (next_row['price_lag_1'] / last_data['price'].iloc[0]) - 1
            next_row['price_pct_change_4'] = (next_row['price_lag_1'] / last_data[f'price_lag_4'].iloc[0]) - 1
            next_row['price_pct_change_12'] = (next_row['price_lag_1'] / last_data[f'price_lag_12'].iloc[0]) - 1
            next_row['ewma_price'] = 0.8 * next_row['price_lag_1'] + 0.2 * last_data['ewma_price'].iloc[0]
            
            # Prepare features for prediction
            predict_features = next_row[self.X.columns]
            
            # Predict next price
            predicted_price = self.final_model.predict(predict_features.to_frame().T)[0]
            
            predictions.append(predicted_price)
            prediction_dates.append(next_date)
            
            # Update last_data for next iteration
            next_row['price'] = predicted_price
            last_data = next_row
        
        return predictions, prediction_dates

    def recommend_tender_period(self, weekly_volume=1):
        """Recommend optimal tender period"""
        predictions, prediction_dates = self.predict()
        current_price = self.train_data.iloc[-1]['price']
        
        # Calculate strategy costs
        strategy_costs = {}
        total_weeks = 6
        
        for weeks_ahead in range(1, 7):
            total_cost = 0
            weeks_covered = 0
            
            while weeks_covered < total_weeks:
                # Weeks for current purchase
                purchase_weeks = min(weeks_ahead, total_weeks - weeks_covered)
                
                if weeks_covered == 0:
                    # First purchase at current price
                    price = current_price
                else:
                    # Subsequent purchases at predicted prices
                    price = predictions[weeks_covered - 1]
                
                # Purchase cost
                cost = price * weekly_volume * purchase_weeks
                total_cost += cost
                
                weeks_covered += purchase_weeks
            
            strategy_costs[weeks_ahead] = total_cost
        
        # Find strategy with minimum cost
        recommended_period = min(strategy_costs, key=strategy_costs.get)
        
        # Add trend-based adjustments
        if all(x > y for x, y in zip(predictions[:-1], predictions[1:])):
            # Prices consistently decreasing
            recommended_period = 1
        elif all(x < y for x, y in zip(predictions[:-1], predictions[1:])):
            # Prices consistently increasing
            recommended_period = 6
        
        return recommended_period, predictions, prediction_dates, strategy_costs

class RebarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система рекомендаций для закупки арматуры")
        self.root.geometry("1200x1000")
        
        # Define variables
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
            self.predictor = EnhancedRebarPredictor()
            self.load_current_data()
        except Exception as e:
            messagebox.showerror("Ошибка загрузки данных", 
                                f"Не удалось загрузить данные: {str(e)}\n"
                                "Убедитесь, что файл processed_data.csv находится в правильной директории.")
        
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
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Расширенная система рекомендаций для закупки арматуры", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Current data frame
        data_frame = ttk.LabelFrame(main_frame, text="Текущие данные", padding="10")
        data_frame.pack(fill=tk.X, pady=10)
        
        # Current date and price
        input_frame = ttk.Frame(data_frame)
        input_frame.pack(fill=tk.X)
        
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=2)
        input_frame.columnconfigure(2, weight=1)
        input_frame.columnconfigure(3, weight=2)
        
        # Current date and price
        ttk.Label(input_frame, text="Текущая дата:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.current_date_label = ttk.Label(input_frame, text="", font=("Arial", 10, "bold"))
        self.current_date_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(input_frame, text="Текущая цена:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.current_price_label = ttk.Label(input_frame, text="", font=("Arial", 10, "bold"))
        self.current_price_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Weekly volume
        ttk.Label(input_frame, text="Еженедельная потребность (тонн):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        volume_entry = ttk.Entry(input_frame, textvariable=self.weekly_volume, width=10)
        volume_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Model Performance Frame
        perf_frame = ttk.LabelFrame(main_frame, text="Производительность модели", padding="10")
        perf_frame.pack(fill=tk.X, pady=10)
        
        self.model_performance_text = tk.Text(perf_frame, height=10, width=120, wrap=tk.WORD)
        self.model_performance_text.pack(padx=5, pady=5)
        
        # Recommendation frame
        rec_frame = ttk.LabelFrame(main_frame, text="Рекомендация по закупке", padding="10")
        rec_frame.pack(fill=tk.X, pady=10)
        
        self.recommendation_label = ttk.Label(rec_frame, text="", font=("Arial", 14, "bold"))
        self.recommendation_label.pack(pady=10)
        
        self.explanation_label = ttk.Label(rec_frame, text="", wraplength=950)
        self.explanation_label.pack(pady=5)
        
        # Forecast prices frame
        self.forecast_frame = ttk.Frame(rec_frame)
        self.forecast_frame.pack(fill=tk.X, pady=5)
        
        # Savings label
        self.savings_label = ttk.Label(rec_frame, text="", font=("Arial", 12, "bold"), foreground="green")
        self.savings_label.pack(pady=5)
        
        # Graphs frame
        graphs_frame = ttk.Frame(main_frame)
        graphs_frame.pack(fill=tk.X, pady=10)
        
        # Price prediction graph
        price_graph_frame = ttk.LabelFrame(graphs_frame, text="Прогноз цен", padding="10")
        price_graph_frame.pack(fill=tk.X, pady=5)
        
        self.price_figure = Figure(figsize=(10, 4), dpi=100)
        self.price_plot = self.price_figure.add_subplot(111)
        
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, price_graph_frame)
        self.price_canvas.get_tk_widget().pack(fill=tk.X)
        
        # Strategy comparison graph
        strategy_graph_frame = ttk.LabelFrame(graphs_frame, text="Сравнение стратегий закупок", padding="10")
        strategy_graph_frame.pack(fill=tk.X, pady=5)
        
        self.strategy_figure = Figure(figsize=(10, 3), dpi=100)
        self.strategy_plot = self.strategy_figure.add_subplot(111)
        
        self.strategy_canvas = FigureCanvasTkAgg(self.strategy_figure, strategy_graph_frame)
        self.strategy_canvas.get_tk_widget().pack(fill=tk.X)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        refresh_button = ttk.Button(button_frame, text="Обновить прогноз", command=self.update_recommendation)
        refresh_button.pack(side=tk.RIGHT, padx=5)

        help_button = ttk.Button(button_frame, text="О приложении", command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5)
    
    def show_help(self):
        """Display help dialog about the application"""
        help_text = """Расширенная система рекомендаций для закупки арматуры

Возможности:
1. Прогнозирование цен с использованием машинного обучения
2. Анализ нескольких моделей прогнозирования
3. Рекомендация оптимальной стратегии закупок
4. Визуализация прогнозов и сравнение стратегий

Как использовать:
- Введите еженедельную потребность в тоннах
- Нажмите "Обновить прогноз"
- Изучите графики и рекомендации

Технические особенности:
- Использованы модели: Random Forest, Gradient Boosting, Ridge Regression
- Учитываются сезонность, тренды и исторические данные
- Автоматический выбор лучшей модели"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("О приложении")
        help_window.geometry("500x500")
        
        help_label = ttk.Label(help_window, text=help_text, wraplength=480, justify=tk.LEFT)
        help_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        close_button = ttk.Button(help_window, text="Закрыть", command=help_window.destroy)
        close_button.pack(pady=10)
    
    def load_current_data(self):
        """Load the most recent data for prediction"""
        self.current_data = self.predictor.train_data.copy()
        
        # Display current date and price
        last_row = self.current_data.iloc[-1]
        self.current_date = last_row['dt']
        self.current_price = last_row['price']
        
        self.current_date_label.config(text=self.current_date.strftime("%d.%m.%Y"))
        self.current_price_label.config(text=f"{self.current_price:.2f} руб./т")
        
        # Display model performance
        self.display_model_performance()
        
        # Update recommendation
        self.update_recommendation()
    
    def display_model_performance(self):
        """Display model performance metrics"""
        self.model_performance_text.delete(1.0, tk.END)
        self.model_performance_text.insert(tk.END, "Производительность моделей:\n\n")
        
        for model, metrics in self.predictor.model_performances.items():
            self.model_performance_text.insert(tk.END, f"{model}:\n")
            for metric, value in metrics.items():
                self.model_performance_text.insert(tk.END, f"  {metric}: {value:.4f}\n")
            self.model_performance_text.insert(tk.END, "\n")
        
        self.model_performance_text.insert(tk.END, f"Выбрана модель: {self.predictor.best_model_name}")
    
    def update_recommendation(self):
        """Update the recommendation based on current data"""
        try:
            # Get weekly volume with validation
            try:
                weekly_volume = float(self.weekly_volume.get())
            except ValueError:
                messagebox.showerror("Ошибка ввода", "Введите корректное числовое значение для еженедельной потребности.")
                return
            
            # Get recommendation
            recommended_period, predictions, prediction_dates, strategy_costs = self.predictor.recommend_tender_period(
                weekly_volume=weekly_volume
            )
            
            # Update recommendation label
            recommendation_text = f"Рекомендуемый период закупки: {recommended_period} {'неделя' if recommended_period == 1 else 'недели' if 1 < recommended_period < 5 else 'недель'}"
            self.recommendation_label.config(text=recommendation_text)
            
            # Update explanation
            if recommended_period == 1:
                explanation = "Рекомендуется закупка на 1 неделю, так как прогнозируется снижение цен в ближайшем будущем."
            elif recommended_period >= 5:
                explanation = "Рекомендуется закупка на длительный период, так как прогнозируется рост цен в ближайшем будущем."
            else:
                explanation = f"Рекомендуется закупка на {recommended_period} {'недели' if recommended_period < 5 else 'недель'} на основе прогноза динамики цен."
            
            self.explanation_label.config(text=explanation)
            
            # Update forecast prices
            self.update_forecast_prices(prediction_dates, predictions)
            
            # Calculate and display savings
            worst_strategy = max(strategy_costs.items(), key=lambda x: x[1])
            best_strategy = min(strategy_costs.items(), key=lambda x: x[1])
            
            savings = worst_strategy[1] - best_strategy[1]
            savings_pct = (savings / worst_strategy[1]) * 100
            
            self.savings_label.config(
                text=f"Экономия от следования рекомендации: {savings:.2f} руб. ({savings_pct:.2f}%)"
            )
            
            # Update graphs
            self.plot_price_predictions(prediction_dates, predictions)
            self.plot_strategy_comparison(strategy_costs)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обновить рекомендацию: {str(e)}")
    
    def update_forecast_prices(self, prediction_dates, predictions):
        """Update the forecast prices display"""
        # Clear previous widgets
        for widget in self.forecast_frame.winfo_children():
            widget.destroy()
        
        # Create headers
        ttk.Label(self.forecast_frame, text="Прогнозные цены:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Display predicted prices for each week
        for i, (date, price) in enumerate(zip(prediction_dates, predictions)):
            week_num = i + 1
            date_str = date.strftime("%d.%m.%Y")
            
            # Determine color based on price change
            if i == 0:
                price_change = price - self.current_price
                color = "green" if price_change < 0 else "red" if price_change > 0 else "black"
            else:
                price_change = price - predictions[i-1]
                color = "green" if price_change < 0 else "red" if price_change > 0 else "black"
            
            # Display information
            ttk.Label(self.forecast_frame, text=f"Неделя {week_num} ({date_str}):").grid(row=0, column=i+1, padx=10, pady=5)
            price_label = ttk.Label(self.forecast_frame, 
                                    text=f"{price:.2f} руб./т", 
                                    foreground=color,
                                    font=("Arial", 10, "bold"))
            price_label.grid(row=1, column=i+1, padx=10, pady=5)
            
            # Display price change
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