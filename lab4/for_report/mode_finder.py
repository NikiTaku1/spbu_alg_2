import os
import pandas as pd
from collections import Counter
import datetime

def analyze_csv_files(folder_path):
    """
    Анализирует CSV файлы в указанной папке в порядке их даты создания,
    находит моду для категориальных столбцов 'Symptoms', 'Analysis' и 'Doctor',
    а также среднее, медиану и моду для столбца 'Price'.

    Args:
        folder_path (str): Путь к папке с CSV файлами.

    Returns:
        dict: Словарь, где ключи - имена файлов, а значения - словари с результатами анализа.
              Файлы в словаре представлены в порядке их даты создания.
              Если файл не найден, возвращает пустой словарь.
    """

    results = {}
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

    # Сортируем файлы по дате создания
    filepaths.sort(key=os.path.getctime)

    for filepath in filepaths:
        filename = os.path.basename(filepath)  # Получаем только имя файла из пути
        try:
            df = pd.read_csv(filepath)

            # Анализ категориальных столбцов
            try:
                symptoms_mode = Counter(df['Symptoms']).most_common(1)[0][0]
            except KeyError:
                symptoms_mode = None # Или другое значение по умолчанию, если столбец отсутствует

            try:
                analysis_mode = Counter(df['Analysis']).most_common(1)[0][0]
            except KeyError:
                analysis_mode = None

            try:
                doctor_mode = Counter(df['Doctor']).most_common(1)[0][0]
            except KeyError:
                doctor_mode = None

            # Анализ числового столбца 'Price'
            try:
                price_mean = df['Price'].mean()
                price_median = df['Price'].median()
                price_mode = Counter(df['Price']).most_common(1)[0][0]  # Используем Counter для поиска моды
                if price_mean is not None:
                    price_mean = round(price_mean, 2) # Округляем до сотых
            except KeyError:
                price_mean = None
                price_median = None
                price_mode = None
            except TypeError:
                print(f"Ошибка: Столбец 'Price' в {filename} содержит нечисловые данные. Пропуск анализа Price.")
                price_mean = None
                price_median = None
                price_mode = None
            except Exception as e:
                print(f"Непредвиденная ошибка при анализе 'Price' в {filename}: {e}")
                price_mean = None
                price_median = None
                price_mode = None


            results[filename] = {
                'Symptoms_mode': symptoms_mode,
                'Analysis_mode': analysis_mode,
                'Doctor_mode': doctor_mode,
                'Price_mean': price_mean,
                'Price_median': price_median,
                'Price_mode': price_mode
            }

        except FileNotFoundError:
            print(f"Файл не найден: {filepath}")
        except pd.errors.EmptyDataError:
            print(f"Файл пуст: {filepath}")
        except pd.errors.ParserError:
            print(f"Ошибка парсинга файла: {filepath}.  Возможно, файл поврежден или имеет неправильный формат CSV.")
        except Exception as e:
            print(f"Непредвиденная ошибка при обработке файла {filename}: {e}")
            # Можно добавить здесь более подробную обработку ошибок,
            # например, записать ошибку в лог-файл.

    return results


# Пример использования:
folder_path = "C://Users//Zver//Desktop//uni//ALG//lab4//for_report//data"  # Замените на фактический путь к вашей папке
analysis_results = analyze_csv_files(folder_path)

if analysis_results:
    for filename, data in analysis_results.items():
        print(f"Результаты анализа для файла: {filename}")
        for key, value in data.items():
            print(f"  {key}: {value}")
        print("-" * 30)
else:
    print("В указанной папке не найдено подходящих CSV файлов или произошла ошибка при их обработке.")