import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Для стиля кнопок

def compare_csv_files(file1_path, file2_path, categorical_cols, numerical_col, output_text):
    """
    Сравнивает два CSV файла и вычисляет общее среднее процентное значение ошибок
    по категориальным и числовому столбцу. Выводит результат в текстовое поле.
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        output_text.insert(tk.END, f"Ошибка: Файл не найден: {e}\n")
        return
    except pd.errors.ParserError as e:
        output_text.insert(tk.END, f"Ошибка: Ошибка при парсинге CSV файла: {e}\n")
        return

    if len(df1) != len(df2):
        output_text.insert(tk.END, "Ошибка: Файлы имеют разное количество строк.\n")
        return

    output_text.delete("1.0", tk.END)  # Очистка текстового поля

    error_percentages = []  # Сюда будем складывать все ошибки в процентах

    # Категориальные столбцы
    for col in categorical_cols:
        if col not in df1.columns or col not in df2.columns:
            output_text.insert(tk.END, f"Предупреждение: Столбец '{col}' отсутствует в одном из файлов.\n")
            continue
        errors = (df1[col] != df2[col]).sum()
        error_percentage = (errors / len(df1)) * 100 if len(df1) > 0 else 0
        error_percentages.append(error_percentage)

    # Числовой столбец
    if numerical_col not in df1.columns or numerical_col not in df2.columns:
        output_text.insert(tk.END, f"Предупреждение: Столбец '{numerical_col}' отсутствует в одном из файлов.\n")
    else:
        mse = np.mean((df1[numerical_col] - df2[numerical_col]) ** 2)
        mean_value = df1[numerical_col].mean()
        mse_percentage = (mse / mean_value**2) * 100 if mean_value != 0 else 0
        error_percentages.append(mse_percentage)

    # Подсчёт среднего по всем ошибкам
    if error_percentages:
        overall_average = sum(error_percentages) / len(error_percentages)
        output_text.insert(tk.END, f"Общий средний процент ошибок: {overall_average:.2f}%\n")
    else:
        output_text.insert(tk.END, "Нет данных для вычисления среднего процента ошибок.\n")



def browse_file(entry):
    """Открывает диалоговое окно выбора файла и вставляет выбранный путь в Entry."""
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def compare_files_button_click():
    """Обработчик нажатия кнопки сравнения."""
    file1_path = file1_entry.get()
    file2_path = file2_entry.get()

    if not file1_path or not file2_path:
        messagebox.showerror("Ошибка", "Пожалуйста, выберите оба файла.")
        return

    compare_csv_files(file1_path, file2_path, categorical_cols, numerical_col, output_text)


# Настройки столбцов
categorical_cols = ['Symptoms', 'Analysis', 'Doctor']
numerical_col = 'Price'

# Создание главного окна
root = tk.Tk()
root.title("Сравнение CSV файлов")

# Создание и стилизация элементов интерфейса с использованием ttk для современной темы
style = ttk.Style(root)
style.theme_use('clam')  # Или 'alt', 'default', 'classic', 'vista'

# Frame для организации элементов
frame = ttk.Frame(root, padding=(10, 10))
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Выбор первого файла
file1_label = ttk.Label(frame, text="Первый CSV файл:")
file1_label.grid(row=0, column=0, sticky=tk.W)
file1_entry = ttk.Entry(frame, width=50)
file1_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
file1_button = ttk.Button(frame, text="Обзор", command=lambda: browse_file(file1_entry))
file1_button.grid(row=0, column=2, sticky=tk.W)

# Выбор второго файла
file2_label = ttk.Label(frame, text="Второй CSV файл:")
file2_label.grid(row=1, column=0, sticky=tk.W)
file2_entry = ttk.Entry(frame, width=50)
file2_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
file2_button = ttk.Button(frame, text="Обзор", command=lambda: browse_file(file2_entry))
file2_button.grid(row=1, column=2, sticky=tk.W)

# Кнопка сравнения
compare_button = ttk.Button(frame, text="Сравнить файлы", command=compare_files_button_click)
compare_button.grid(row=2, column=0, columnspan=3, pady=10)

# Текстовое поле для вывода результатов
output_label = ttk.Label(frame, text="Результаты:")
output_label.grid(row=3, column=0, sticky=tk.W)
output_text = tk.Text(frame, height=10, width=70)
output_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))

# Конфигурация строк и столбцов для адаптивного размера
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)  # Чтобы текстовое поле расширялось

root.mainloop()