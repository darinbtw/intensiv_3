import pandas as pd
import glob

# Список всех CSV и XLSX файлов в папке
all_files = glob.glob("*.xlsx") + glob.glob("*.csv")

# Создаем пустой DataFrame для итоговых данных
combined_data = pd.DataFrame()

for file in all_files:
    try:
        # Чтение файлов
        if file.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')

        # Добавление данных в общий DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    except Exception as e:
        print(f"Ошибка в файле {file}: {e}")

# Сохранение в один файл
combined_data.to_excel("Объединенный_файл.xlsx", index=False)  # или .to_csv()