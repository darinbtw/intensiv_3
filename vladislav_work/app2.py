from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, 
                           QComboBox, QHBoxLayout, QFileDialog, QMessageBox, QDateEdit,
                           QGroupBox, QGridLayout, QSpinBox, QTabWidget, QFrame)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QIcon
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import joblib
import os
from datetime import datetime, timedelta

class PredictionModel:
    """Class to handle prediction model operations"""
    
    def __init__(self):
        # Path to the model file (should be loaded from a trained model)
        self.model_path = 'model/reinforcement_price_model.pkl'
        self.model = None
        self.train_data = None
        self.additional_data = None
        
    def load_model(self):
        """Load the trained model if exists, otherwise return False"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_data(self, train_path, additional_path=None):
        """Load historical price data and additional features"""
        try:
            self.train_data = pd.read_csv(train_path)
            # Convert date to datetime format
            self.train_data['dt'] = pd.to_datetime(self.train_data['dt'])
            
            if additional_path and os.path.exists(additional_path):
                self.additional_data = pd.read_csv(additional_path)
                self.additional_data['dt'] = pd.to_datetime(self.additional_data['dt'])
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def predict(self, weeks_ahead=1, current_date=None):
        """
        Make price predictions for N weeks ahead
        Returns a dictionary with predictions and recommendations
        """
        if self.model is None:
            # For demonstration, we'll implement a simple prediction logic
            # In a real application, this would use the actual trained model
            
            # If no model is available, use a simple moving average approach
            if self.train_data is None:
                return {
                    "success": False,
                    "message": "No data available for prediction"
                }
            
            # For demonstration, we'll use the last 4 weeks average price as prediction
            # In a real model, we would use the trained model to make predictions
            recent_prices = self.train_data.sort_values('dt', ascending=False).head(4)['Цена на арматуру'].values
            avg_price = np.mean(recent_prices)
            last_price = self.train_data.sort_values('dt', ascending=False).iloc[0]['Цена на арматуру']
            
            # Generate predictions for each of the requested weeks
            predictions = []
            for i in range(1, weeks_ahead + 1):
                # Adding some random variation to simulate predictions
                pred_price = avg_price * (1 + np.random.normal(0, 0.02))
                predictions.append(pred_price)
            
            avg_prediction = np.mean(predictions)
            
            # Basic recommendation logic
            if avg_prediction > last_price * 1.05:
                recommendation = "HOLD: Prices are expected to rise significantly. Recommend a smaller tender now."
                confidence = "Medium"
            elif avg_prediction < last_price * 0.95:
                recommendation = "BUY: Prices are expected to drop. Recommend a larger tender for the entire period."
                confidence = "High"
            else:
                recommendation = "NEUTRAL: Prices are expected to remain stable. Proceed with standard tender volume."
                confidence = "Medium"
            
            # Generate weekly breakdown
            weekly_data = []
            start_date = pd.to_datetime(current_date) if current_date else pd.to_datetime('today')
            for i in range(weeks_ahead):
                pred_date = start_date + timedelta(days=7 * (i+1))
                weekly_data.append({
                    "week": i+1,
                    "dt": pred_date.strftime('%Y-%m-%d'),
                    "Цена на арматуру": predictions[i],
                })
            
            return {
                "success": True,
                "average_prediction": avg_prediction,
                "last_price": last_price,
                "price_change_pct": ((avg_prediction - last_price) / last_price) * 100,
                "recommendation": recommendation,
                "confidence": confidence,
                "weeks_ahead": weeks_ahead,
                "weekly_predictions": weekly_data
            }
        else:
            # Here would go the code to use the actual trained model
            # As we don't have it, we'll return a similar structure as above
            pass

class MatplotlibCanvas(FigureCanvas):
    """Canvas for Matplotlib plots"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Приложение для предсказания цен на арматуру")
        self.model = PredictionModel()
        self.init_ui()
        self.resize(900, 700)

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Title
        title_label = QLabel("Система прогнозирования цен на арматуру")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Помощник категорийного менеджера для оптимизации закупок")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.prediction_tab = QWidget()
        self.data_tab = QWidget()
        self.history_tab = QWidget()
        
        self.tabs.addTab(self.prediction_tab, "Прогнозирование")
        self.tabs.addTab(self.data_tab, "Данные")
        self.tabs.addTab(self.history_tab, "История")
        
        main_layout.addWidget(self.tabs)
        
        # Setup prediction tab
        self.setup_prediction_tab()
        
        # Setup data tab
        self.setup_data_tab()
        
        # Setup history tab
        self.setup_history_tab()
        
        # Status bar
        self.status_label = QLabel("Готово к работе")
        self.status_label.setAlignment(Qt.AlignRight)
        main_layout.addWidget(self.status_label)

    def setup_prediction_tab(self):
        layout = QVBoxLayout()
        self.prediction_tab.setLayout(layout)
        
        # Parameters group
        params_group = QGroupBox("Параметры прогноза")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        
        # Date selection
        date_label = QLabel("Дата прогноза:")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        params_layout.addWidget(date_label, 0, 0)
        params_layout.addWidget(self.date_edit, 0, 1)
        
        # Weeks ahead selection
        weeks_label = QLabel("Количество недель:")
        self.weeks_spinbox = QSpinBox()
        self.weeks_spinbox.setRange(1, 12)
        self.weeks_spinbox.setValue(4)
        params_layout.addWidget(weeks_label, 1, 0)
        params_layout.addWidget(self.weeks_spinbox, 1, 1)
        
        # Strategy selection
        strategy_label = QLabel("Стратегия закупки:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Консервативная", "Сбалансированная", "Агрессивная"])
        params_layout.addWidget(strategy_label, 2, 0)
        params_layout.addWidget(self.strategy_combo, 2, 1)
        
        layout.addWidget(params_group)
        
        # Prediction button
        self.predict_button = QPushButton("Выполнить прогноз")
        self.predict_button.clicked.connect(self.predict_price)
        self.predict_button.setMinimumHeight(40)
        layout.addWidget(self.predict_button)
        
        # Results group
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.result_label = QLabel("Для получения прогноза нажмите кнопку выше")
        self.result_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_label)
        
        # Add chart placeholder
        self.chart_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        results_layout.addWidget(self.chart_canvas)
        
        layout.addWidget(results_group)
        
        # Recommendation group
        rec_group = QGroupBox("Рекомендация")
        rec_layout = QVBoxLayout()
        rec_group.setLayout(rec_layout)
        
        self.recommendation_label = QLabel("Здесь будет отображена рекомендация после выполнения прогноза")
        self.recommendation_label.setWordWrap(True)
        rec_layout.addWidget(self.recommendation_label)
        
        layout.addWidget(rec_group)

    def setup_data_tab(self):
        layout = QVBoxLayout()
        self.data_tab.setLayout(layout)
        
        # Data loading group
        data_group = QGroupBox("Загрузка данных")
        data_layout = QGridLayout()
        data_group.setLayout(data_layout)
        
        # Training data
        train_label = QLabel("Исторические данные:")
        self.train_path_label = QLabel("Файл не выбран")
        self.train_path_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.train_path_label.setStyleSheet("background-color: white;")
        train_button = QPushButton("Выбрать файл")
        train_button.clicked.connect(lambda: self.select_file('train'))
        
        data_layout.addWidget(train_label, 0, 0)
        data_layout.addWidget(self.train_path_label, 0, 1)
        data_layout.addWidget(train_button, 0, 2)
        
        # Additional data
        add_label = QLabel("Дополнительные данные:")
        self.add_path_label = QLabel("Файл не выбран")
        self.add_path_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.add_path_label.setStyleSheet("background-color: white;")
        add_button = QPushButton("Выбрать файл")
        add_button.clicked.connect(lambda: self.select_file('additional'))
        
        data_layout.addWidget(add_label, 1, 0)
        data_layout.addWidget(self.add_path_label, 1, 1)
        data_layout.addWidget(add_button, 1, 2)
        
        # Load data button
        load_button = QPushButton("Загрузить данные")
        load_button.clicked.connect(self.load_data)
        load_button.setMinimumHeight(40)
        
        data_layout.addWidget(load_button, 2, 0, 1, 3)
        
        layout.addWidget(data_group)
        
        # Data preview group
        preview_group = QGroupBox("Предпросмотр данных")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        
        self.data_preview_label = QLabel("Загрузите данные для предпросмотра")
        self.data_preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.data_preview_label)
        
        # Add data visualization placeholder
        self.data_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        preview_layout.addWidget(self.data_canvas)
        
        layout.addWidget(preview_group)

    def setup_history_tab(self):
        layout = QVBoxLayout()
        self.history_tab.setLayout(layout)
        
        # History group
        history_group = QGroupBox("История прогнозов")
        history_layout = QVBoxLayout()
        history_group.setLayout(history_layout)
        
        self.history_label = QLabel("Здесь будет отображаться история прогнозов")
        self.history_label.setAlignment(Qt.AlignCenter)
        history_layout.addWidget(self.history_label)
        
        # Add history visualization placeholder
        self.history_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        history_layout.addWidget(self.history_canvas)
        
        layout.addWidget(history_group)
        
        # Clear history button
        clear_button = QPushButton("Очистить историю")
        clear_button.clicked.connect(self.clear_history)
        layout.addWidget(clear_button)

    def select_file(self, file_type):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Выберите файл данных", "", "CSV Files (*.csv);;All Files (*)")
        
        if file_path:
            if file_type == 'train':
                self.train_path_label.setText(file_path)
            elif file_type == 'additional':
                self.add_path_label.setText(file_path)

    def load_data(self):
        train_path = self.train_path_label.text()
        additional_path = self.add_path_label.text()
        
        if train_path == "Файл не выбран":
            QMessageBox.warning(self, "Ошибка", "Выберите файл с историческими данными")
            return
        
        success = self.model.load_data(train_path, additional_path if additional_path != "Файл не выбран" else None)
        
        if success:
            QMessageBox.information(self, "Успех", "Данные успешно загружены")
            self.status_label.setText("Данные загружены")
            
            # Update data preview
            self.update_data_preview()
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные")

    def update_data_preview(self):
        if self.model.train_data is not None:
            # Clear the previous plot
            self.data_canvas.axes.clear()
            
            # Plot the price history
            self.data_canvas.axes.plot(self.model.train_data['dt'], self.model.train_data['Цена на арматуру'], 'b-')
            self.data_canvas.axes.set_title('История цен на арматуру')
            self.data_canvas.axes.set_xlabel('Дата')
            self.data_canvas.axes.set_ylabel('Цена')
            self.data_canvas.axes.grid(True)
            
            # Format the date on x-axis
            self.data_canvas.fig.autofmt_xdate()
            
            # Redraw the canvas
            self.data_canvas.draw()
            
            # Update the preview label
            self.data_preview_label.setText(f"Загружено {len(self.model.train_data)} записей")

    def predict_price(self):
        if self.model.train_data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные")
            return
        
        # Get parameters
        selected_date = self.date_edit.date().toPyDate()
        weeks_ahead = self.weeks_spinbox.value()
        strategy = self.strategy_combo.currentText()
        
        # Make prediction
        result = self.model.predict(weeks_ahead=weeks_ahead, current_date=selected_date)
        
        if result["success"]:
            # Update result label
            price_change = result["price_change_pct"]
            change_direction = "повышение" if price_change > 0 else "снижение"
            
            self.result_label.setText(
                f"Прогноз на {weeks_ahead} недель вперед:\n"
                f"Средняя цена: {result['average_prediction']:.2f}\n"
                f"Текущая цена: {result['last_price']:.2f}\n"
                f"Изменение: {abs(price_change):.2f}% ({change_direction})\n"
                f"Уверенность: {result['confidence']}"
            )
            
            # Update recommendation
            self.recommendation_label.setText(result["recommendation"])
            
            # Update plot
            self.update_prediction_plot(result)
            
            # Add to history (in a real app, this would be stored persistently)
            self.update_history_plot(result)
            
            self.status_label.setText(f"Прогноз выполнен {datetime.now().strftime('%H:%M:%S')}")
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось выполнить прогноз")

    def update_prediction_plot(self, result):
        # Clear the previous plot
        self.chart_canvas.axes.clear()
        
        # Get data for plotting
        weeks = [pred["week"] for pred in result["weekly_predictions"]]
        prices = [pred["Цена на арматуру"] for pred in result["weekly_predictions"]]
        
        # Plot the predictions
        self.chart_canvas.axes.plot(weeks, prices, 'b-o')
        self.chart_canvas.axes.axhline(y=result["last_price"], color='r', linestyle='--', label='Текущая цена')
        
        # Add labels and title
        self.chart_canvas.axes.set_title('Прогноз цен на арматуру')
        self.chart_canvas.axes.set_xlabel('Неделя')
        self.chart_canvas.axes.set_ylabel('Цена')
        self.chart_canvas.axes.grid(True)
        self.chart_canvas.axes.legend()
        
        # Redraw the canvas
        self.chart_canvas.draw()

    def update_history_plot(self, result):
        # In a real application, this would retrieve history from storage
        # For this demo, we'll just show the current prediction
        
        # Clear the previous plot
        self.history_canvas.axes.clear()
        
        # Get data for plotting
        weeks = [pred["week"] for pred in result["weekly_predictions"]]
        prices = [pred["Цена на арматуру"] for pred in result["weekly_predictions"]]
        
        # Plot the predictions
        self.history_canvas.axes.plot(weeks, prices, 'g-o', label=datetime.now().strftime('%Y-%m-%d'))
        
        # Add labels and title
        self.history_canvas.axes.set_title('История прогнозов')
        self.history_canvas.axes.set_xlabel('Неделя')
        self.history_canvas.axes.set_ylabel('Цена')
        self.history_canvas.axes.grid(True)
        self.history_canvas.axes.legend()
        
        # Redraw the canvas
        self.history_canvas.draw()
        
        # Update history label
        self.history_label.setText(f"Последний прогноз: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def clear_history(self):
        # Clear the history plot
        self.history_canvas.axes.clear()
        self.history_canvas.draw()
        self.history_label.setText("История прогнозов очищена")
        QMessageBox.information(self, "Информация", "История прогнозов очищена")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())