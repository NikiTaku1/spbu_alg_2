import pandas as pd
import numpy as np
import random

def generate_random_directed_distance_matrix(n, min_distance=1, max_distance=100):
    """
    Генерирует случайную матрицу расстояний для ориентированного графа.
    В формате DataFrame pandas.

    Args:
        n (int): Размерность матрицы (количество вершин).
        min_distance (int): Минимальное значение расстояния.
        max_distance (int): Максимальное значение расстояния.

    Returns:
        pandas.DataFrame: Сгенерированная матрица расстояний.
    """
    vertices = [chr(65 + i) for i in range(n)]
    df = pd.DataFrame(index=vertices, columns=vertices)
    for i in range(n):
        for j in range(n):
            if i == j:
                df.iloc[i, j] = 0
            else:
                distance = random.randint(min_distance, max_distance)
                df.iloc[i, j] = distance
    return df


def save_matrix_to_csv(df, n, filename="distance_matrix_{}.csv"):
    """
    Сохраняет матрицу расстояний в файл CSV.

    Args:
        df (pandas.DataFrame): Матрица расстояний.
        filename (str): Имя файла для сохранения.
    """
    try:
        df.to_csv(filename.format(n))
        print(f"Матрица расстояний сохранена в файл '{filename.format(n)}'")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    matrix_size = 30
    min_dist = 1
    max_dist = 50

    distance_matrix = generate_random_directed_distance_matrix(matrix_size, min_dist, max_dist)
    save_matrix_to_csv(distance_matrix, matrix_size)