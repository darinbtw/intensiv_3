#это вторая версия приложения на ARIMA

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

train_data = pd.read_excel('./vladislav_work/processed_data.xlsx')
train_data['dt'] = pd.to_datetime(train_data['dt'])
train_data['День'] = train_data['dt'].dt.day
train_data['Месяц'] = train_data['dt'].dt.month
train_data['Год'] = train_data['dt'].dt.year
train_data['ДеньНедели'] = train_data['dt'].dt.dayofweek
train_data['ЦенаL1'] = train_data['Цена на арматуру'].shift(1)
train_data['ЦенаL2'] = train_data['Цена на арматуру'].shift(2)
train_data.dropna(inplace=True)

future_df = pd.DataFrame()

def train_and_predict():
    global future_df
    try:
        weeks = int(week_choice.get())

        if weeks < 1 or weeks > 6:
            messagebox.showwarning("Warning", "Количество недель должно быть от 1 до 6!")
            return

        y_train = train_data['Цена на арматуру']
        model = ARIMA(y_train, order=(3, 1, 2))  # Параметры ARIMA, которые вы используете
        model_fit = model.fit()

        future_dates = pd.date_range(start=train_data['dt'].max() + pd.Timedelta(days=1), periods=weeks, freq='W-MON')

        future_predictions = model_fit.forecast(steps=weeks)

        # Добавляем первую точку вручную
        first_forecast_point = y_train.iloc[-1]  # Берём последнее значение как начало
        future_dates = [train_data['dt'].iloc[-1]] + list(future_dates)  # Добавляем последнюю известную дату к будущим датам
        full_predictions = [first_forecast_point] + list(future_predictions)

        future_df = pd.DataFrame({'dt': future_dates, 'Цена на арматуру': full_predictions})

        plot_graph()
        display_predictions()

    except Exception as e:
        messagebox.showerror("Error", f"Ошибка при прогнозировании: {e}")

def plot_graph():
    global future_df
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_data['dt'], train_data['Цена на арматуру'], label='Исторические данные', color='blue', linewidth=2)
    ax.plot(future_df['dt'], future_df['Цена на арматуру'], label='Предсказанные цены', color='red', linestyle='-', linewidth=2)
    ax.set_title('Прогноз цен на арматуру')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Цена на арматуру')
    ax.legend()
    ax.grid()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def display_predictions():
    global future_df
    for widget in table_frame.winfo_children():
        widget.destroy()

    table = ttk.Treeview(table_frame, columns=("Дата", "Цена на арматуру"), show="headings", height=10)
    table.heading("Дата", text="Дата")
    table.heading("Цена на арматуру", text="Цена на арматуру")
    table.pack(fill=tk.BOTH, expand=True)

    for _, row in future_df.iterrows():
        table.insert("", "end", values=(row["dt"].strftime('%Y-%m-%d'), row["Цена на арматуру"]))

    export_button = tk.Button(table_frame, text="Экспортировать в Excel", command=export_to_excel)
    export_button.pack(pady=5)

def export_to_excel():
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            future_df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Данные успешно экспортированы в {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Ошибка при экспорте данных: {e}")

root = tk.Tk()
root.title("Прогноз цен на арматуру")
root.state('zoomed')

control_frame = tk.Frame(root)
control_frame.pack(fill=tk.X, padx=10, pady=10)

tk.Label(control_frame, text="Выберите количество недель для прогноза:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

week_choice = ttk.Combobox(control_frame, values=[1, 2, 3, 4, 5, 6], state="readonly", font=("Arial", 12))
week_choice.pack(side=tk.LEFT, padx=5)
week_choice.set(1)

tk.Button(control_frame, text="Прогнозировать", font=("Arial", 12), command=train_and_predict).pack(side=tk.LEFT, padx=5)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

graph_frame = tk.Frame(main_frame)
graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

table_frame = tk.Frame(main_frame)
table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

root.mainloop()