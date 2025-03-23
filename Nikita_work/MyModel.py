import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Пример данных (замените на свои данные)
df = pd.read_excel("train.xlsx")

# Проверка на стационарность
def check_stationarity(series):
    result = adfuller(series)
    return result[1] <= 0.05  # p-value <= 0.05 означает стационарность

for col in df.columns:
    if not check_stationarity(df[col]):
        df[col] = df[col].diff().dropna()  # Применяем дифференцирование

# Удаляем NaN значения после дифференцирования
df = df.dropna()

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Обучение модели VAR
model = VAR(train)
results = model.fit(maxlags=15, ic='aic')  # Автоматический выбор порядка p с помощью AIC

# Прогнозирование
lag_order = results.k_ar
forecast_input = train.values[-lag_order:]
forecast = results.forecast(y=forecast_input, steps=len(test))

# Преобразование прогноза в DataFrame
forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

# Оценка модели
rmse = np.sqrt(mean_squared_error(test, forecast_df))
print(f"RMSE: {rmse}")

# Визуализация результатов
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(test.index, test[col], label=f'Фактические значения {col}')
    plt.plot(forecast_df.index, forecast_df[col], label=f'Прогноз {col}', linestyle='--')
plt.legend()
plt.show()