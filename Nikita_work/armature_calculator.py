import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from datetime import datetime

# Загрузка модели
try:
    model = joblib.load('armature_price_predictor.pkl')
except Exception as e:
    messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
    exit()

def predict_price():
    try:
        # Получение данных от пользователя
        date = entry_date.get()
        demand = entry_demand.get()
        currency_rate = entry_currency_rate.get()

        # Проверка, что поля не пустые
        if not date or not demand or not currency_rate:
            messagebox.showerror("Ошибка", "Все поля должны быть заполнены!")
            return

        # Преобразование данных в числа
        demand = float(demand)
        currency_rate = float(currency_rate)

        # Преобразование даты в день года
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday

        # Создание DataFrame для предсказания
        input_data = pd.DataFrame({
            'day_of_year': [day_of_year],
            'demand': [demand],
            'currency_rate': [currency_rate]
        })

        # Предсказание
        predicted_price = model.predict(input_data)[0]
        print(f"Предсказанная цена: {predicted_price:.2f} руб.")  # Вывод в консоль для отладки
        label_result.config(text=f"Предсказанная цена: {predicted_price:.2f} руб.", fg='#00796b')
    except ValueError:
        messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числовые значения.")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

# Создание основного окна
root = tk.Tk()
root.title("Калькулятор цены на арматуру")
root.geometry("400x300")
root.configure(bg='#e0f7fa')  # Светло-голубой фон

# Установка иконки окна (убедитесь, что файл samolet_logo.ico существует)
try:
    root.iconbitmap('samolet_logo.ico')
except Exception as e:
    print(f"Не удалось загрузить иконку: {e}")

# Фрейм для ввода данных
input_frame = tk.Frame(root, bg='#e0f7fa')
input_frame.grid(row=0, column=0, padx=10, pady=10)

# Поля ввода
label_date = tk.Label(input_frame, text="Дата (ГГГГ-ММ-ДД):", bg='#e0f7fa', font=('Arial', 12))
label_date.grid(row=0, column=0, padx=5, pady=5, sticky="w")

entry_date = tk.Entry(input_frame, bg='#ffffff')  # Белый фон для поля ввода
entry_date.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

label_demand = tk.Label(input_frame, text="Спрос:", bg='#e0f7fa', font=('Arial', 12))
label_demand.grid(row=1, column=0, padx=5, pady=5, sticky="w")

entry_demand = tk.Entry(input_frame, bg='#ffffff')  # Белый фон для поля ввода
entry_demand.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

label_currency_rate = tk.Label(input_frame, text="Курс валют:", bg='#e0f7fa', font=('Arial', 12))
label_currency_rate.grid(row=2, column=0, padx=5, pady=5, sticky="w")

entry_currency_rate = tk.Entry(input_frame, bg='#ffffff')  # Белый фон для поля ввода
entry_currency_rate.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

# Кнопка предсказания
button_predict = tk.Button(root, text="Предсказать цену", bg='#80deea', fg='#000000', font=('Arial', 12), command=predict_price)
button_predict.grid(row=1, column=0, pady=10)

# Результат
label_result = tk.Label(root, text="Предсказанная цена: ", bg='#e0f7fa', font=('Arial', 12), fg='#00796b')
label_result.grid(row=2, column=0, pady=10)

# Запуск основного цикла
root.mainloop()