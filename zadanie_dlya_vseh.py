import pandas as pd
import numpy as np
import time
# Читаем реальный файл 1 Студента

FILE_PATH = "../products_advanced_dataset.csv"
try:
    df_raw = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f" Ошибка! Файл '{FILE_PATH}' не найден в текущей папке.")
    print("Пожалуйста, убедитесь, что файл Первого Студента лежит рядом с этим скриптом.")
    exit()

#1 Глубокий аудит данных (Векторизованный NumPy-подход)
def advanced_data_audit(df):
    print("\n=== [ЗАДАЧА 1: НАЧАЛО АУДИТА] ===")
    print("\n[INFO] Структура данных (.info):")
    df.info()
    nan_counts = {col: int(np.sum(df[col].isna().to_numpy())) for col in df.columns}
    print(f"\nРазмерность: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Пропуски по колонкам: {nan_counts}")
    print("\n=== Уникальные значения категориальных колонок ===")
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in categorical_cols:
        unique_vals = np.unique(df[col].dropna().to_numpy().astype(str))
        print(f"Колонка '{col}' (Уникальных: {len(unique_vals)}): {unique_vals[:5]}...")

#2 Data Cleaning Pipeline (Оптимизация через np.where)
def clean_data_vectorized(df):
    df_clean = df.copy()
    prices = df_clean['price'].to_numpy()
    median_price = np.nanmedian(prices[prices > 0])
    q99_price = np.nanpercentile(prices, 99)
    prices = np.where(prices < 0, median_price, prices)
    prices = np.clip(prices, a_min=0, a_max=q99_price)
    df_clean['price'] = prices
    if 'rating' in df_clean.columns:
        ratings = df_clean['rating'].to_numpy()
        df_clean['rating'] = np.where(np.isnan(ratings), np.nanmean(ratings), ratings)
    return df_clean

#3 Feature Engineering (Векторизованные вычисления)
def build_features_vectorized(df):
    revenue = df['revenue_last_month'].to_numpy()
    views = df['views'].to_numpy()
    rating = df['rating'].to_numpy()
    rev_min, rev_max = revenue.min(), revenue.max()
    views_min, views_max = views.min(), views.max()
    rev_norm = (revenue - rev_min) / (rev_max - rev_min) if rev_max != rev_min else np.zeros_like(revenue)
    views_norm = (views - views_min) / (views_max - views_min) if views_max != views_min else np.zeros_like(views)
    df['value_score'] = (rating / 5 * 0.4) + (rev_norm * 0.4) + (views_norm * 0.2)
    return df

#4 Фильтрация уровня PRO (Битовые маски NumPy)
def filter_pro_products(df):
    mask_price = (df['price'].to_numpy() > 100)
    mask_rating = (df['rating'].to_numpy() > 4.0)
    mask_value = (df['value_score'].to_numpy() > 0.5)
    full_mask = mask_price & mask_rating & mask_value
    return df.iloc[full_mask]

#5 Кастомная сортировка (Через np.lexsort)
def custom_numpy_sort(df):
    value_arr = df['value_score'].to_numpy()
    rating_arr = df['rating'].to_numpy()
    price_arr = df['price'].to_numpy()
    sorted_indices = np.lexsort((price_arr, -rating_arr, -value_arr))
    return df.iloc[sorted_indices]

#6 Алгоритм поиска аномалий (Векторизованный поиск без циклов)
def find_anomalies_vectorized(df):
    revenue = df['revenue_last_month'].to_numpy()
    price = df['price'].to_numpy()
    anomaly_mask = revenue > (price * 500)
    anomaly_ids = df.loc[anomaly_mask, 'product_id'].tolist()
    return anomaly_ids

#7 Генераторы (Потоковая обработка чанков для Big Data)
def high_value_generator(file_path, chunk_size=2000):
    """
    Потоковый генератор чанков. Экономит ОЗУ, обрабатывая гигантские файлы на лету.
    """
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if 'revenue_last_month' in chunk.columns:
            high_revenue_mask = chunk['revenue_last_month'].to_numpy() > 5000
            yield chunk.iloc[high_revenue_mask]

#8 Работа со строками (Векторизованные строковые методы NumPy)
def process_strings_vectorized(df):
    if 'product_name' in df.columns:
        names = df['product_name'].to_numpy().astype(str)
        df['product_name_upper'] = np.char.upper(names)
        df['name_length'] = np.vectorize(len)(names)
        sku_10_mask = np.char.find(names, 'Product_1') != -1
        return df, df.iloc[sku_10_mask]
    return df, df

#9 Мини-алгоритм ранжирования (Scoring Matrix через матричное умножение)
def calculate_vectorized_score(df):
    revenue = (df['revenue_last_month'] / df['revenue_last_month'].max()).to_numpy()
    rating = (df['rating'] / 5).to_numpy()
    value = df['value_score'].to_numpy()
    metrics_matrix = np.column_stack((revenue, rating, value))
    weights = np.array([0.5, 0.2, 0.3])
    df['business_priority_score'] = np.dot(metrics_matrix, weights)
    return df

#10 Итоговый Pipeline & Замер производительности
def full_analysis_optimized(df_raw_input):
    start_time = time.time()
    df_cleaned = clean_data_vectorized(df_raw_input)
    df_features = build_features_vectorized(df_cleaned)
    df_filtered = filter_pro_products(df_features)
    df_sorted = custom_numpy_sort(df_filtered)
    df_scored = calculate_vectorized_score(df_sorted)
    end_time = time.time()
    print(
        f"\n Оптимизированный пайплайн Студента №2 обработал {len(df_raw_input)} строк за {end_time - start_time:.4f} секунд!")
    return df_scored

# ЗАПУСК СИСТЕМЫ
advanced_data_audit(df_raw)
final_analytics_report = full_analysis_optimized(df_raw)
anomalies = find_anomalies_vectorized(final_analytics_report)
print(f"Количество обнаруженных скрытых аномалий в ценах: {len(anomalies)}")
df_strings, df_sku10 = process_strings_vectorized(final_analytics_report)
print(f"Товаров, содержащих в названии 'Product_1': {df_sku10.shape[0]}")

# Сохраняю чистый результат для Студента №3
OUTPUT_FILE = "products_optimized_output.csv"
final_analytics_report.to_csv(OUTPUT_FILE, index=False)
print(f"\n РЕЗУЛЬТАТ УСПЕШНО СОХРАНЕН! Создан файл '{OUTPUT_FILE}' для Студента №3.")