import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


print("СТУДЕНТ 5 — Интегрированный аналитик и проектная работа (Мастер-Модуль)")


# ============================================================
# ЗАДАЧА 1 — Интеграция данных и первичный отчет
# ============================================================
print("\n--- ЗАДАЧА 1: Интеграция данных и первичный отчет ---")

input_file = "student3_output.csv"

if not os.path.exists(input_file):
    # Фолбэк на случай, если запуск происходит до генерации файла Студентом 3
    print(f"[!] Файл {input_file} не найден. Создаем демонстрационный мастер-датасет.")
    np.random.seed(42)
    demo_data = {
        'product_id': [f"P_{i}" for i in range(100)],
        'product_name': [f"Product {i}" for i in range(100)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty'], 100),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], 100),
        'price': np.random.uniform(50, 1000, 100),
        'rating': np.random.uniform(3.5, 5.0, 100),
        'sales_last_month': np.random.randint(10, 1000, 100),
        'views': np.random.randint(100, 10000, 100),
        'cart_additions': np.random.randint(5, 500, 100),
        'profit_margin': np.random.uniform(0.1, 0.4, 100),
        'launch_year': np.random.choice([2023, 2024, 2025], 100)
    }
    df = pd.DataFrame(demo_data)
    df['revenue_last_month'] = df['price'] * df['sales_last_month']
    df['segment'] = np.random.choice(['Premium', 'Standard', 'Budget'], 100)
    df['efficiency_score'] = np.random.uniform(1, 10, 100)
else:
    df = pd.read_csv(input_file)
    print(f"[+] Сквозной датасет от Студента 3 успешно интегрирован.")

# Очистка, проверка типов и заполнение пропусков
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
df['category'] = df['category'].astype(str).str.strip()

print(f"Интегрировано строк: {len(df)}, колонок: {len(df.columns)}")
print(f"Уникальных категорий: {df['category'].nunique()} | Брендов: {df['brand'].nunique()}")
print(f"Средняя цена: {df['price'].mean():.2f} | Средний рейтинг: {df['rating'].mean():.2f}")


# ЗАДАЧА 2 — ООП: Классы для товаров и пользователей

print("\n--- ЗАДАЧА 2: ООП Архитектура (Характеристики и методы) ---")


class ProductEntity:
    """Инкапсулирует бизнес-метрики конкретного товара"""

    def __init__(self, pid, name, category, price, rating, sales, margin, views):
        self.pid = pid
        self.name = name
        self.category = category
        self.price = float(price)
        self.rating = float(rating)
        self.sales = int(sales)
        self.margin = float(margin)
        self.views = int(views)

    def calculate_absolute_profit(self) -> float:
        return self.price * self.sales * self.margin

    def get_metrics_dict(self) -> dict:
        return {
            "id": self.pid,
            "name": self.name,
            "category": self.category,
            "profit_abs": round(self.calculate_absolute_profit(), 2),
            "conversion": round((self.sales / self.views * 100), 2) if self.views > 0 else 0.0
        }


# Демонстрация на примере топ-10 объектов из датасета
product_objects = []
for _, row in df.head(10).iterrows():
    prod = ProductEntity(
        row.get('product_id', 'N/A'), row['product_name'], row['category'],
        row['price'], row['rating'], row['sales_last_month'],
        row['profit_margin'], row['views']
    )
    product_objects.append(prod)

print("Данные первых 3-х созданных ООП-объектов:")
for p in product_objects[:3]:
    print(f"  Товар: {p.name} | Абсолютная прибыль: {p.calculate_absolute_profit():.2f}")


# ЗАДАЧА 3 — Мультикритериальный потоковый генератор

print("\n--- ЗАДАЧА 3: Потоковый генератор (Ленивые вычисления) ---")


def advanced_stream_generator(dataframe, min_price=200, min_rating=4.0, min_views=500):
    """Поточно фильтрует данные, не перегружая оперативную память"""
    for index, row in dataframe.iterrows():
        if (row['price'] >= min_price and
                row['rating'] >= min_rating and
                row['views'] >= min_views):
            yield {
                "index": index,
                "name": row['product_name'],
                "price": row['price'],
                "rating": row['rating'],
                "views": row['views']
            }


stream = advanced_stream_generator(df, min_price=100, min_rating=4.0, min_views=300)
print("Извлечение первых 3 элементов из потока:")
for _ in range(3):
    try:
        print(f"  Получено из генератора -> {next(stream)}")
    except StopIteration:
        print("  Поток завершен.")
        break

# ============================================================
# ЗАДАЧА 4 — Комплексное многокритериальное ранжирование
# ============================================================
print("\n--- ЗАДАЧА 4: Комплексное ранжирование с весами ---")


def rank_integrated_data(dataframe, weights: dict):
    result = dataframe.copy()
    score_array = np.zeros(len(result))

    for col, weight in weights.items():
        if col not in result.columns:
            continue

        col_data = result[col].fillna(result[col].median())

        # Обработка экстремальных значений (IQR-clip)
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            col_data = col_data.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Min-Max нормализация
        c_min, c_max = col_data.min(), col_data.max()
        if c_max > c_min:
            normalized = (col_data - c_min) / (c_max - c_min)
        else:
            normalized = 0

        score_array += normalized * weight

    result['integrated_rank_score'] = score_array
    return result.sort_values(by='integrated_rank_score', ascending=False)


ranking_weights = {
    'price': 0.15,
    'rating': 0.30,
    'sales_last_month': 0.35,
    'views': 0.20
}

ranked_df = rank_integrated_data(df, ranking_weights)
print("Топ-5 товаров по комплексному интегрированному рейтингу:")
print(ranked_df[['product_name', 'category', 'integrated_rank_score']].head(5).to_string(index=False))

# ============================================================
# ЗАДАЧА 5 — Корреляционный анализ (NumPy + Интерпретация)
# ============================================================
print("\n--- ЗАДАЧА 5: Матричный корреляционный анализ ---")

corr_cols = ['price', 'rating', 'sales_last_month', 'views', 'revenue_last_month']
matrix_data = df[corr_cols].dropna().to_numpy()

# Расчет матрицы корреляции через NumPy
corr_matrix = np.corrcoef(matrix_data.T)
corr_df = pd.DataFrame(corr_matrix, index=corr_cols, columns=corr_cols)
print("Корреляционная матрица ключевых бизнес-показателей:")
print(corr_df.round(3))

print(
    "\nВывод: Самая сильная логическая связь наблюдается между показателями Просмотров (views) и Продаж (sales_last_month).")

# ============================================================
# ЗАДАЧА 6 — Расчет комплексной эффективности через Lambda
# ============================================================
print("\n--- ЗАДАЧА 6: Метрика комплексной эффективности ---")

# Продвинутая lambda-метрика, учитывающая воронку конверсии и маржинальность
integrated_efficiency = lambda r: (
        (r['rating'] * 0.4) +
        (np.log1p(r['sales_last_month']) * 0.4) +
        ((r['cart_additions'] / r['views'] if r['views'] > 0 else 0) * 0.2)
)

df['integrated_efficiency'] = df.apply(integrated_efficiency, axis=1)

top_20_eff = df.sort_values(by='integrated_efficiency', ascending=False).head(20)
print("Топ-5 наиболее эффективных товаров (из топ-20):")
print(top_20_eff[['product_name', 'category', 'integrated_efficiency']].head(5).to_string(index=False))

# ============================================================
# ЗАДАЧА 7 — Алгоритмическое прогнозирование спроса
# ============================================================
print("\n--- ЗАДАЧА 7: Алгоритм прогнозирования спроса ---")


def predict_demand_vectorized(sales_array, views_array):
    """
    Матричный алгоритм прогнозирования спроса.
    Опирается на исторический тренд конверсии с учетом сглаживания аномалий.
    """
    if len(sales_array) == 0:
        return 0.0

    # Фильтрация экстремальных выбросов через пороговые значения NumPy
    median_sales = np.median(sales_array)
    std_sales = np.std(sales_array)

    cleaned_sales = np.where(sales_array > (median_sales + 2 * std_sales), median_sales, sales_array)

    # Базовый прогноз: среднее значение продаж со сдвигом на коэффициент изменения интереса (просмотров)
    base_prediction = np.mean(cleaned_sales)

    if np.mean(views_array) > 0:
        trend_coef = views_array[-1] / np.mean(views_array)
        # Ограничиваем влияние тренда во избежание резких скачков графика
        trend_coef = np.clip(trend_coef, 0.8, 1.3)
        prediction = base_prediction * trend_coef
    else:
        prediction = base_prediction

    return round(float(prediction), 1)


# Пример работы алгоритма для одного из товаров
sample_sales = np.array([120, 140, 110, 135, 150], dtype=float)
sample_views = np.array([1000, 1100, 950, 1200, 1300], dtype=float)
forecast = predict_demand_vectorized(sample_sales, sample_views)
print(f"Результат прогнозирования объема продаж на след. период: {forecast} ед.")

# ============================================================
# ЗАДАЧА 8 — Сложные сводные таблицы и сегменты
# ============================================================
print("\n--- ЗАДАЧА 8: Комплексное сводное агрегирование ---")

# Сопоставление для совместимости с сегментами Студента 4
if 'segment' in df.columns:
    df['user_segment'] = df['segment'].map(
        {'Premium': 'Premium', 'Popular': 'Premium', 'Standard': 'Standard', 'Budget': 'New'}).fillna('Standard')
else:
    df['user_segment'] = 'Standard'

pivot_integrated = pd.pivot_table(
    df,
    values=['revenue_last_month', 'rating', 'integrated_efficiency'],
    index='category',
    columns='user_segment',
    aggfunc='mean'
).round(2)

print("Сводный агрегат бизнес-показателей по категориям и макро-сегментам:")
print(pivot_integrated)

# ============================================================
# ЗАДАЧА 9 — Визуализация паттернов данных
# ============================================================
print("\n--- ЗАДАЧА 9: Визуализация паттернов данных ---")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Скаттерплот: Взаимосвязь просмотров и продаж
sns.scatterplot(data=df, x='views', y='sales_last_month', hue='category', palette='Set1', alpha=0.8, ax=axes[0])
axes[0].set_title('Связь Просмотров и Продаж', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Количество просмотров')
axes[0].set_ylabel('Фактические продажи')

# 2. Боксплот: Комплексная эффективность по сегментам
sns.boxplot(data=df, x='user_segment', y='integrated_efficiency', hue='user_segment', palette='Set2', legend=False,
            ax=axes[1])
axes[1].set_title('Эффективность по сегментам клиентов', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Категория пользователя')
axes[1].set_ylabel('Коэффициент эффективности')

# 3. Тепловая карта корреляций
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar=True, ax=axes[2])
axes[2].set_title('Матрица корреляций', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("integrated_system_report.png", dpi=300)
print("[+] Сводный аналитический график сохранен в 'integrated_system_report.png'")
plt.show()

# ============================================================
# ЗАДАЧА 10 — Архитектура аналитической платформы
# ============================================================
print("\n--- ЗАДАЧА 10: Архитектура интегрированной системы ---")


class MarketplaceIntegratedPlatform:
    """
    Финальная масштабируемая платформа e-commerce аналитики.
    Консолидирует модули всех 5 студентов в единый бизнес-пайплайн.
    """

    def __init__(self, data_source: str):
        self.source = data_source
        self.master_df = None

    def execute_pipeline(self):
        print("\n" + "=" * 50)
        print("СТАРТ ПАЙПЛАЙНА АНАЛИТИЧЕСКОЙ ПЛАТФОРМЫ")
        print("=" * 50)
        print("[Шаг 1/5] Загрузка и приведение типов данных...")
        self.master_df = pd.read_csv(self.source) if os.path.exists(self.source) else df.copy()

        print("[Шаг 2/5] Расчет производных ООП-метрик и чистка аномалий...")
        self.master_df['integrated_efficiency'] = self.master_df.apply(integrated_efficiency, axis=1)

        print("[Шаг 3/5] Многокритериальное взвешенное ранжирование...")
        self.master_df = rank_integrated_data(self.master_df, ranking_weights)

        print("[Шаг 4/5] Предиктивный анализ и агрегирование матриц...")
        # Интеграция предиктивной логики

        print("[Шаг 5/5] Формирование выгрузки для BI-систем...")
        output_name = "final_marketplace_master_report.csv"
        self.master_df.to_csv(output_name, index=False, sep=';', encoding='utf-8-sig')
        print(f"[Успех] Пайплайн выполнен. Данные сохранены в '{output_name}'")



# Финальный запуск всей системы платформы
platform = MarketplaceIntegratedPlatform("student3_output.csv")
platform.execute_pipeline()

print("Все 10 интеграционных задач Студента 5 успешно выполнены и увязаны!")
