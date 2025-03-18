import pandas as pd
import os
import glob

def merge_all_files(directory_path, output_file='combined_data.xlsx'):
    """
    Объединяет все CSV и XLSX файлы из указанной директории в один файл.
    Сохраняет все столбцы из каждого файла.
    
    Параметры:
    directory_path (str): Путь к директории с файлами
    output_file (str): Имя выходного файла (по умолчанию 'combined_data.xlsx')
    """
    # Получаем списки всех CSV и XLSX файлов в директории
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    xlsx_files = glob.glob(os.path.join(directory_path, '*.xlsx'))
    
    all_files = csv_files + xlsx_files
    
    if not all_files:
        print(f"В директории {directory_path} не найдено CSV или XLSX файлов.")
        return
    
    print(f"Найдено файлов: {len(all_files)}")
    
    # Создаем пустой DataFrame для результата
    combined_df = pd.DataFrame()
    
    # Обрабатываем каждый файл
    for file_path in all_files:
        print(f"Обработка файла: {os.path.basename(file_path)}")
        
        try:
            # Читаем файл в зависимости от его расширения
            if file_path.endswith('.csv'):
                # Пробуем разные кодировки и разделители для CSV
                try:
                    df = pd.read_csv(file_path)
                except:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                    except:
                        try:
                            df = pd.read_csv(file_path, delimiter=';')
                        except:
                            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8-sig')
            else:
                df = pd.read_excel(file_path)
            
            # Добавляем столбец с источником данных
            df['source_file'] = os.path.basename(file_path)
            
            # Объединяем с основным DataFrame
            if combined_df.empty:
                combined_df = df
            else:
                # Объединяем все столбцы, заполняя отсутствующие значения NaN
                combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)
        
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {str(e)}")
    
    # Сохраняем результат
    if not combined_df.empty:
        combined_df.to_excel(output_file, index=False)
        print(f"Данные успешно объединены в файл: {output_file}")
        print(f"Общее количество строк: {combined_df.shape[0]}")
        print(f"Общее количество столбцов: {combined_df.shape[1]}")
    else:
        print("Не удалось объединить данные. Проверьте файлы и их содержимое.")

# Пример использования
if __name__ == "__main__":
    # Замените на путь к вашей директории с файлами
    directory = input("Введите путь к директории с файлами: ")
    output = input("Введите имя выходного файла (по умолчанию 'combined_data.xlsx'): ")
    
    if not output:
        output = 'combined_data.xlsx'
    
    merge_all_files(directory, output)