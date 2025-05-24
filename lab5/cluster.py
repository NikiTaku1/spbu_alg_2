import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# ---------- Генерация датасета с числовыми и категориальными признаками ----------
def generate_dataset(n_samples=300, n_num_features=12, n_cat_features=5, n_clusters=5):
    num_feature_names = [
        'Length', 'Width', 'Height', 'Weight', 'Density',
        'Temperature', 'Pressure', 'Volume', 'Speed', 'Acceleration',
        'Force', 'Energy'
    ][:n_num_features]

    X_num, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_num_features,
                         cluster_std=1.5, random_state=42)

    df_num = pd.DataFrame(X_num, columns=num_feature_names)

    np.random.seed(42)
    cat_feature_values = {
        'Color': ['Red', 'Green', 'Blue', 'Yellow', 'Purple'],
        'Shape': ['Circle', 'Square', 'Triangle', 'Hexagon', 'Pentagon', 'Ellipse'],
        'Size': ['Small', 'Medium', 'Large', 'Extra Large'],
        'Material': ['Wood', 'Metal', 'Plastic', 'Glass'],
        'Texture': ['Smooth', 'Rough', 'Matte', 'Glossy']
    }

    cat_keys = list(cat_feature_values.keys())[:n_cat_features]
    df_cat = pd.DataFrame({
        key: np.random.choice(cat_feature_values[key], size=n_samples)
        for key in cat_keys
    })

    df = pd.concat([df_num, df_cat], axis=1)
    df['Target'] = y

    return df

# ---------- Предобработка данных ----------
def preprocess_data(df, anonymized=False):
    if not anonymized:
        num_cols = df.select_dtypes(include=[np.number]).columns.drop('Target')
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        scaler = StandardScaler()
        scaled_num = scaler.fit_transform(df[num_cols])

        try:
            encoder = OneHotEncoder(sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(sparse=False)

        encoded_cat = encoder.fit_transform(df[cat_cols])

        X = np.hstack([scaled_num, encoded_cat])
        return X, scaler, encoder, num_cols, cat_cols
    else:
        cat_cols = df.columns.drop('Target')

        try:
            encoder = OneHotEncoder(sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(sparse=False)

        X = encoder.fit_transform(df[cat_cols])
        return X, None, encoder, [], cat_cols

# ---------- Кластеризация maximin ----------
def maximin_clustering(X, k=None):
    centers_idx = [np.random.randint(0, len(X))]
    centers = [X[centers_idx[0]]]

    while True:
        dists = cdist(X, np.array(centers), metric='euclidean')
        min_dists = np.min(dists, axis=1)
        new_center_idx = np.argmax(min_dists)

        if k and len(centers) >= k:
            break
        if any(np.array_equal(X[new_center_idx], c) for c in centers):
            break

        centers.append(X[new_center_idx])
        centers_idx.append(new_center_idx)

    dists = cdist(X, np.array(centers), metric='euclidean')
    labels = np.argmin(dists, axis=1)
    return labels, centers

# ---------- Выбор информативных признаков ----------
def select_informative_features(df_num, top_n=10):
    center = df_num.mean(axis=0)
    spread = ((df_num - center) ** 2).mean(axis=0)
    return spread.sort_values(ascending=False).head(top_n).index.tolist()

# ---------- Анонимизация с округлением числовых и смысловым обобщением категориальных ----------
def anonymize(df, k=5, text_output=None):
    df_anon = df.copy()
    num_cols = df_anon.select_dtypes(include=[np.number]).columns.drop('Target')
    cat_cols = df_anon.select_dtypes(include=['object', 'category']).columns

    def transform_number(x):
        if pd.isnull(x):
            return 'Missing'
        sign = 1
        int_part = abs(int(x))
        digits_count = len(str(int_part)) if int_part != 0 else 1
        return str(sign * digits_count)

    for col in num_cols:
        df_anon[col] = df_anon[col].apply(transform_number)

    cat_threshold = max(k, 10)
    for col in cat_cols:
        counts = df_anon[col].value_counts()
        rare = counts[counts < cat_threshold].index
        df_anon.loc[df_anon[col].isin(rare), col] = 'Other'

    df_anon.fillna('Missing', inplace=True)
    df_anon.replace('', 'Missing', inplace=True)

    quasi_id = df_anon.drop(columns='Target').apply(lambda row: tuple(row), axis=1)
    freq = quasi_id.value_counts()
    k_anon = freq.min()

    #if text_output:
    #    text_output.insert(tk.END, f"Минимальная частота квази-идентификаторов после обобщения: {k_anon}\n")
    #    rare_ids = freq[freq < k].index.tolist()
    #    if rare_ids:
    #        text_output.insert(tk.END, f"Количество уникальных (меньше k) квази-идентификаторов: {len(rare_ids)}\n")
    #    else:
    #        text_output.insert(tk.END, "Все квази-идентификаторы имеют частоту >= k\n")

        # --- Новый вывод: частоты всех уникальных строк ---
    #    text_output.insert(tk.END, "\nЧастоты всех уникальных строк:\n")
    #    for row_val, count in freq.items():
    #        text_output.insert(tk.END, f"{row_val} : {count}\n")

    #else:
    #    print(f"Минимальная частота квази-идентификаторов после обобщения: {k_anon}")
    #    rare_ids = freq[freq < k].index.tolist()
    #    if rare_ids:
    #        print(f" Количество уникальных (меньше k) квази-идентификаторов: {len(rare_ids)}")
    #    else:
    #        print("Все квази-идентификаторы имеют частоту >= k")

    #    print("\nЧастоты всех уникальных строк:")
    #    for row_val, count in freq.items():
    #        print(f"{row_val} : {count}")

    return df_anon, k_anon



# ---------- Построение графика ----------
def cluster_and_plot(data, label, text_output, color='tab10'):
    labels, centers = maximin_clustering(data, k=current_k)

    n_features = data.shape[1]
    n_components = 2 if n_features >= 2 else n_features 

    reduced = PCA(n_components=n_components).fit_transform(data)

    fig, ax = plt.subplots()
    if n_components == 1:
        ax.scatter(reduced[:, 0], np.zeros_like(reduced[:, 0]), c=labels, cmap=color)
        ax.set_ylim(-1, 1)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap=color)
    ax.set_title(label)
    fig.canvas.manager.set_window_title(label)
    fig.show()

    return labels
# ---------- Основной обработчик ----------
def process_clustering():
    text_output.delete(1.0, tk.END)

    try:
        top_n = int(entry_top_n.get())
        if top_n < 1:
            raise ValueError
    except ValueError:
        text_output.insert(tk.END, "Некорректное число информативных признаков, используется значение по умолчанию 10\n")
        top_n = 10

    global current_k
    try:
        current_k = int(entry_clusters.get())
        if current_k < 1:
            raise ValueError
    except ValueError:
        text_output.insert(tk.END, "Некорректное число кластеров, используется значение по умолчанию 5\n")
        current_k = 5

    try:
        n_samples = int(entry_samples.get())
        if n_samples < 10:
            raise ValueError
    except ValueError:
        text_output.insert(tk.END, "Некорректный размер датасета, используется 300\n")
        n_samples = 300

    df = generate_dataset(n_samples=n_samples, n_cat_features=5, n_clusters=current_k)
    y_true = df['Target']

    df.to_csv('dataset_all_features.csv', index=False)
    text_output.insert(tk.END, "Сохранён датасет с всеми признаками: dataset_all_features.csv\n\n")

    X_full, scaler, encoder, num_cols, cat_cols = preprocess_data(df)

    text_output.insert(tk.END, "Кластеризация по всем признакам...\n")
    labels_all = cluster_and_plot(X_full, "Кластеры (все признаки)", text_output)
    rand_all = rand_score(y_true, labels_all)

    text_output.insert(tk.END, f"Индекс Ранда: {rand_all:.3f}\n\n")

    top_num_features = select_informative_features(df[num_cols], top_n=top_n)
    text_output.insert(tk.END, f"Информативные числовые признаки ({top_n}): {', '.join(top_num_features)}\n")

    scaled_top_num = scaler.fit_transform(df[top_num_features])
    X_top = scaled_top_num

    df_top = pd.concat([df[top_num_features], df['Target']], axis=1)
    df_top.to_csv('dataset_top_features.csv', index=False)
    text_output.insert(tk.END, "Сохранён датасет с информативными признаками: dataset_top_features.csv\n\n")

    labels_top = cluster_and_plot(X_top, "Кластеры (информативные признаки)", text_output)
    rand_top = rand_score(y_true, labels_top)
    text_output.insert(tk.END, f"Индекс Ранда: {rand_top:.3f}\n\n")

    df_anon1, k_anon_val1 = anonymize(df, k=5)
    df_anon1.to_csv('dataset_k_anonymized.csv', index=False)
    text_output.insert(tk.END, f"Сохранён обезличенный датасет: dataset_anonymized_orig.csv\n")
    text_output.insert(tk.END, f"Показатель k-анонимности: {k_anon_val1}\n\n")

    X_anon1, _, _, _, _ = preprocess_data(df_anon1, anonymized=True)

    text_output.insert(tk.END, "Кластеризация по обезличенным данным всего датасета...\n")
    labels_anon1 = cluster_and_plot(X_anon1, "Кластеры (анонимизация всего датасета)", text_output)
    rand_anon1 = rand_score(y_true, labels_anon1)
    text_output.insert(tk.END, f"Индекс Ранда: {rand_anon1:.3f}\n\n")

    df_anon2, k_anon_val2 = anonymize(df_top, k=5)
    df_anon2.to_csv('dataset_k_anonymized.csv', index=False)
    text_output.insert(tk.END, f"Сохранён обезличенный датасет: dataset_anonymized_mod.csv\n")
    text_output.insert(tk.END, f"Показатель k-анонимности: {k_anon_val2}\n\n")

    X_anon2, _, _, _, _ = preprocess_data(df_anon2, anonymized=True)

    text_output.insert(tk.END, "Кластеризация по обезличенным данным...\n")
    labels_anon2 = cluster_and_plot(X_anon2, "Кластеры (анонимизация для измененного датасета)", text_output)
    rand_anon2 = rand_score(y_true, labels_anon2)
    text_output.insert(tk.END, f"Индекс Ранда: {rand_anon2:.3f}\n\n")

    text_output.insert(tk.END, "Сравнение:\n")
    text_output.insert(tk.END, f"- Все признаки:     {rand_all:.3f}\n")
    text_output.insert(tk.END, f"- Информативные:    {rand_top:.3f}\n")
    text_output.insert(tk.END, f"- Обезличенный (весь):     {rand_anon1:.3f}\n")
    text_output.insert(tk.END, f"- Обезличенный (изм):     {rand_anon2:.3f}\n")

# ---------- GUI ----------
root = tk.Tk()
root.title("Кластеризация")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

tk.Label(frame, text="Генерация и кластеризация датасета", font=("Arial", 14)).pack(pady=5)

frame_top_n = tk.Frame(frame)
frame_top_n.pack(pady=5)
tk.Label(frame_top_n, text="Количество информативных признаков:").pack(side=tk.LEFT)
entry_top_n = tk.Entry(frame_top_n, width=5)
entry_top_n.insert(0, "10")
entry_top_n.pack(side=tk.LEFT)

frame_clusters = tk.Frame(frame)
frame_clusters.pack(pady=5)
tk.Label(frame_clusters, text="Количество кластеров:").pack(side=tk.LEFT)
entry_clusters = tk.Entry(frame_clusters, width=5)
entry_clusters.insert(0, "5")
entry_clusters.pack(side=tk.LEFT)

frame_samples = tk.Frame(frame)
frame_samples.pack(pady=5)
tk.Label(frame_samples, text="Размер датасета:").pack(side=tk.LEFT)
entry_samples = tk.Entry(frame_samples, width=7)
entry_samples.insert(0, "300")
entry_samples.pack(side=tk.LEFT)

btn = tk.Button(frame, text="Выполнить кластеризацию", command=process_clustering, width=30)
btn.pack(pady=10)

text_output = tk.Text(frame, height=25, width=85, font=("Courier", 10))
text_output.pack()

current_k = 5

root.mainloop()