"""
E-COMMERCE ANALYTICS
Студент 3 — Business Metrics + Feature Engineering
Все 10 задач

Использует products_optimized_output.csv
(очищенный файл от Студента 1 + обогащённый Студентом 2)
"""

import pandas as pd
import numpy as np

# Загрузка данных от Студента 2
df = pd.read_csv("products_optimized_output.csv")

print("=" * 60)
print("СТУДЕНТ 3 — Business Metrics + Feature Engineering")
print(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")
print("=" * 60)


# ============================================================
# ЗАДАЧА 1 — Описательная статистика по категориям и брендам
# ============================================================
print("\n--- ЗАДАЧА 1: Описательная статистика ---")

# Количество товаров по категориям
print("\nКоличество товаров по категориям:")
print(df["category"].value_counts())

# Количество товаров по брендам
print("\nТоп-10 брендов:")
print(df["brand"].value_counts().head(10))

# Диапазоны цен
print(f"\nДиапазон цен: {df['price'].min():.2f} — {df['price'].max():.2f}")
print(f"Средняя цена:    {df['price'].mean():.2f}")
print(f"Медиана цены:    {df['price'].median():.2f}")

# Средние рейтинг и продажи
print(f"\nСредний рейтинг:        {df['rating'].mean():.2f}")
print(f"Средние продажи/месяц:  {df['sales_last_month'].mean():.2f}")
print(f"Средняя выручка/месяц:  {df['revenue_last_month'].mean():.2f}")

# Описательная статистика
print("\nОписательная статистика по ключевым метрикам:")
print(df[["price", "rating", "sales_last_month", "revenue_last_month", "profit_margin"]].describe().round(2))

# Стабильность по категориям
stability = df.groupby("category")["rating"].agg(["mean", "std"]).round(3)
stability.columns = ["средний_рейтинг", "разброс_рейтинга"]
stability = stability.sort_values("разброс_рейтинга")
print("\nСтабильность категорий (чем меньше разброс — тем стабильнее):")
print(stability)
print(f"\nСамая стабильная категория:    {stability.index[0]}")
print(f"Самая нестабильная категория:  {stability.index[-1]}")


# ============================================================
# ЗАДАЧА 2 — Тренды по времени (launch_year)
# ============================================================
print("\n--- ЗАДАЧА 2: Тренды по годам ---")

def yearly_trends(dataframe):
    return dataframe.groupby("launch_year").agg(
        кол_товаров=("product_id", "count"),
        средняя_цена=("price", "mean"),
        средний_рейтинг=("rating", "mean"),
        сумма_продаж=("sales_last_month", "sum"),
        средняя_выручка=("revenue_last_month", "mean")
    ).round(2)

trends = yearly_trends(df)
print("\nТренды по годам запуска товаров:")
print(trends)

# Определяем тренд
sales_trend = trends["сумма_продаж"].diff().mean()
if sales_trend > 0:
    print("\nВывод: Суммарные продажи растут с каждым годом.")
else:
    print("\nВывод: Суммарные продажи снижаются в последние годы.")

# Тренд по категориям
trend_cat = df.groupby("category").agg(
    средняя_цена=("price", "mean"),
    средний_рейтинг=("rating", "mean"),
    всего_продаж=("sales_last_year", "sum")
).round(2).sort_values("всего_продаж", ascending=False)
print("\nТренд продаж по категориям (по убыванию):")
print(trend_cat)


# ============================================================
# ЗАДАЧА 3 — Генератор аномальных товаров
# ============================================================
print("\n--- ЗАДАЧА 3: Генератор аномальных товаров ---")

def anomaly_generator(dataframe):
    """
    Генератор товаров с аномальной комбинацией показателей:
    - цена выше медианы
    - рейтинг ниже 25-го процентиля
    - продажи в верхнем квартиле (выше 75-го процентиля)

    Работает построчно через yield — не загружает всё в память.
    """
    price_median  = dataframe["price"].median()
    rating_q25    = dataframe["rating"].quantile(0.25)
    sales_q75     = dataframe["sales_last_month"].quantile(0.75)

    for _, row in dataframe.iterrows():
        if (
            row["price"]            > price_median and
            row["rating"]           < rating_q25   and
            row["sales_last_month"] > sales_q75
        ):
            yield row

# Запускаем генератор
print(f"\nПороговые значения:")
print(f"  Цена > медианы:           {df['price'].median():.2f}")
print(f"  Рейтинг < 25-го перцент.: {df['rating'].quantile(0.25):.2f}")
print(f"  Продажи > 75-го перцент.: {df['sales_last_month'].quantile(0.75):.2f}")

print("\nАномальные товары (первые 10):")
count = 0
for product in anomaly_generator(df):
    print(f"  {product['product_name']:20s} | "
          f"цена={product['price']:.2f} | "
          f"рейтинг={product['rating']:.2f} | "
          f"продажи={product['sales_last_month']:.0f}")
    count += 1
    if count >= 10:
        break

print(f"\nВсего показано аномалий: {count}")
if count == 0:
    print("  Аномальных товаров не найдено по данным критериям.")


# ============================================================
# ЗАДАЧА 4 — Функция сегментации товаров
# ============================================================
print("\n--- ЗАДАЧА 4: Сегментация товаров ---")

def segment_product(row):
    """
    Присваивает сегмент товару на основе реальных данных:
    price, rating, sales_last_month, profit_margin, conversion_rate

    Сегменты:
      Premium     — дорогой + высокий рейтинг + высокая маржа
      Popular     — много продаж + хороший рейтинг
      High_Margin — высокая маржа, но немного продаж
      Budget      — низкая цена и мало продаж
      Standard    — всё остальное
    """
    price   = row["price"]
    rating  = row["rating"]
    sales   = row["sales_last_month"]
    margin  = row["profit_margin"]

    if price > 800 and rating >= 4.7 and margin > 0.20:
        return "Premium"
    elif sales > 700 and rating >= 4.5:
        return "Popular"
    elif margin > 0.25 and sales < 300:
        return "High_Margin"
    elif price < 300 and sales < 200:
        return "Budget"
    else:
        return "Standard"

df["segment"] = df.apply(segment_product, axis=1)

print("\nРаспределение товаров по сегментам:")
print(df["segment"].value_counts())

print("\n20 примеров сегментации:")
cols = ["product_name", "price", "rating", "sales_last_month", "profit_margin", "segment"]
print(df[cols].head(20).to_string(index=False))

print("""
Описание сегментов:
  Premium     — цена > 800, рейтинг >= 4.7, маржа > 20%
  Popular     — продажи > 700/мес, рейтинг >= 4.5
  High_Margin — маржа > 25%, но продажи небольшие
  Budget      — цена < 300, продажи < 200
  Standard    — всё остальное
""")


# ============================================================
# ЗАДАЧА 5 — Корреляции между показателями
# ============================================================
print("\n--- ЗАДАЧА 5: Корреляционный анализ ---")

numeric_cols = [
    "price", "rating", "sales_last_month",
    "revenue_last_month", "views",
    "conversion_rate", "profit_margin",
    "ad_spend", "ad_conversions"
]

data_for_corr = df[numeric_cols].dropna()
corr_matrix = np.corrcoef(data_for_corr.T)
corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)

print("\nКорреляционная матрица:")
print(corr_df.round(2))

# Топ-5 пар с наибольшей корреляцией
pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        pairs.append((
            numeric_cols[i],
            numeric_cols[j],
            round(corr_matrix[i][j], 4)
        ))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nТоп-5 наиболее связанных пар показателей:")
for col1, col2, val in pairs[:5]:
    direction = "положительная" if val > 0 else "отрицательная"
    print(f"  {col1:25s} ↔ {col2:25s}: r = {val:+.4f} ({direction})")

print("""
Интерпретация:
  r близко к +1 — чем больше одно, тем больше другое
  r близко к -1 — чем больше одно, тем меньше другое
  r близко к  0 — показатели не связаны
""")


# ============================================================
# ЗАДАЧА 6 — Метрика эффективности (lambda + comprehension)
# ============================================================
print("\n--- ЗАДАЧА 6: Метрика эффективности ---")

# Абсолютная прибыль
df["profit_abs"] = df["final_price"] - df["cost_price"]

# Метрика через lambda
efficiency = lambda row: (
    row["profit_margin"] *
    row["rating"] *
    np.log1p(row["sales_last_month"]) *
    (1 + row["conversion_rate"])
)

df["efficiency_score"] = df.apply(efficiency, axis=1)

# List comprehension — товары выше среднего
avg_eff = df["efficiency_score"].mean()
high_eff_names = [
    row["product_name"]
    for _, row in df.iterrows()
    if row["efficiency_score"] > avg_eff
]

print(f"\nСредняя эффективность: {avg_eff:.4f}")
print(f"Товаров выше среднего: {len(high_eff_names)} из {len(df)}")

print("\nТоп-10 товаров по эффективности:")
top10 = df.nlargest(10, "efficiency_score")[
    ["product_name", "category", "profit_margin",
     "rating", "sales_last_month", "efficiency_score"]
]
print(top10.to_string(index=False))

print("""
Формула:
  efficiency = profit_margin * rating * log(sales+1) * (1 + conversion_rate)
  - profit_margin  — доля прибыли в цене
  - rating         — качество товара
  - log(sales+1)   — сглаженный объём продаж
  - conversion_rate — насколько хорошо товар конвертируется
""")


# ============================================================
# ЗАДАЧА 7 — Многокритериальное ранжирование
# ============================================================
print("\n--- ЗАДАЧА 7: Многокритериальное ранжирование ---")

def rank_products(dataframe, criteria_weights):
    """
    Многокритериальное ранжирование товаров.

    criteria_weights — словарь {колонка: вес}
    Нормализует каждую колонку (min-max) и суммирует взвешенно.
    Обрабатывает пропуски (медиана) и выбросы (IQR clip).

    Возвращает датафрейм с колонкой rank_score, отсортированный по убыванию.
    """
    result = dataframe.copy()
    scores = np.zeros(len(result))

    for col, weight in criteria_weights.items():
        if col not in result.columns:
            print(f"  [!] Колонка '{col}' не найдена, пропускаем.")
            continue

        col_data = result[col].copy()

        # Пропуски → медиана
        if col_data.isnull().any():
            col_data = col_data.fillna(col_data.median())

        # Выбросы → clip по IQR
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            col_data = col_data.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # Нормализация min-max через NumPy
        arr = col_data.to_numpy(dtype=float)
        mn, mx = arr.min(), arr.max()
        normalized = (arr - mn) / (mx - mn + 1e-9)

        scores += normalized * weight

    result["rank_score"] = scores
    return result.sort_values("rank_score", ascending=False)

weights = {
    "efficiency_score":    0.30,
    "rating":              0.25,
    "sales_last_month":    0.20,
    "revenue_last_month":  0.15,
    "profit_margin":       0.10,
}

ranked = rank_products(df, weights)

print("\nТоп-10 по многокритериальному ранжированию:")
r_cols = ["product_name", "category", "rating",
          "sales_last_month", "profit_margin", "rank_score"]
print(ranked[r_cols].head(10).to_string(index=False))

print(f"""
Веса критериев:
  efficiency_score:   30% — общая эффективность товара
  rating:             25% — качество
  sales_last_month:   20% — актуальные продажи
  revenue_last_month: 15% — выручка
  profit_margin:      10% — маржинальность
""")


# ============================================================
# ЗАДАЧА 8 — Сводные таблицы
# ============================================================
print("\n--- ЗАДАЧА 8: Сводные таблицы ---")

# По категориям
pivot_cat = df.pivot_table(
    values=["price", "sales_last_month", "rating",
            "revenue_last_month", "profit_margin"],
    index="category",
    aggfunc={
        "price":               "mean",
        "sales_last_month":    "sum",
        "rating":              "mean",
        "revenue_last_month":  "sum",
        "profit_margin":       "mean",
    }
).round(2)
print("\nСводная таблица по категориям:")
print(pivot_cat.to_string())

# По брендам (топ-8)
top_brands = df["brand"].value_counts().head(8).index
pivot_brand = df[df["brand"].isin(top_brands)].pivot_table(
    values=["price", "sales_last_month", "rating"],
    index="brand",
    aggfunc={"price": "mean", "sales_last_month": "sum", "rating": "mean"}
).round(2).sort_values("sales_last_month", ascending=False)
print("\nСводная таблица по топ-8 брендам:")
print(pivot_brand.to_string())

# По сегментам
pivot_seg = df.pivot_table(
    values=["profit_margin", "rating", "efficiency_score", "sales_last_month"],
    index="segment",
    aggfunc="mean"
).round(3).sort_values("efficiency_score", ascending=False)
print("\nСводная таблица по сегментам:")
print(pivot_seg.to_string())

# NumPy агрегация
print("\nАгрегация через NumPy:")
rev = df["revenue_last_month"].to_numpy()
print(f"  Суммарная выручка/месяц:  {np.sum(rev):,.0f}")
print(f"  Средняя выручка/месяц:    {np.mean(rev):,.2f}")
print(f"  Медиана выручки/месяц:    {np.median(rev):,.2f}")
print(f"  Максимальная выручка:     {np.max(rev):,.2f}")


# ============================================================
# ЗАДАЧА 9 — Прогнозирование продаж
# ============================================================
print("\n--- ЗАДАЧА 9: Прогнозирование продаж ---")

def predict_sales(price, rating, conversion_rate, profit_margin, history=None):
    """
    Алгоритмический прогноз продаж на следующий месяц.

    Параметры:
      price           — цена товара
      rating          — рейтинг (4.0 - 5.0)
      conversion_rate — коэффициент конверсии
      profit_margin   — маржа (0.0 - 1.0)
      history         — список прошлых продаж (опционально)

    Возвращает: прогноз продаж (float)
    """
    if price <= 0 or rating <= 0:
        return 0.0

    # Коэффициент цены
    if price < 300:
        price_coef = 1.4
    elif price < 600:
        price_coef = 1.0
    elif price < 800:
        price_coef = 0.75
    else:
        price_coef = 0.55

    # Коэффициент рейтинга
    if rating >= 4.8:
        rating_coef = 1.4
    elif rating >= 4.5:
        rating_coef = 1.2
    elif rating >= 4.2:
        rating_coef = 1.0
    else:
        rating_coef = 0.8

    # Коэффициент конверсии
    if conversion_rate > 0.15:
        conv_coef = 1.3
    elif conversion_rate > 0.08:
        conv_coef = 1.1
    else:
        conv_coef = 0.9

    # Базовый прогноз
    base = 300 * price_coef * rating_coef * conv_coef

    # Корректировка на маржу
    if profit_margin > 0.25:
        base *= 1.1

    # Если есть история — усредняем с ней
    if history is not None and len(history) > 0:
        hist_mean = np.mean(np.array(history, dtype=float))
        base = hist_mean * 0.65 + base * 0.35

    return round(base, 1)

# Проверяем на реальных данных
print("\nПроверка модели на 10 товарах:")
print(f"{'Товар':<22} {'Прогноз':>9} {'Факт':>9} {'Ошибка':>8}")
print("-" * 52)

errors = []
for _, row in df.head(10).iterrows():
    forecast = predict_sales(
        row["price"],
        row["rating"],
        row["conversion_rate"],
        row["profit_margin"],
        history=[row["sales_last_3_months"] / 3]
    )
    actual = row["sales_last_month"]
    error = abs(forecast - actual) / actual * 100 if actual > 0 else 0
    errors.append(error)
    print(f"{row['product_name']:<22} {forecast:>9.1f} {actual:>9.0f} {error:>7.1f}%")

print(f"\nСредняя ошибка прогноза: {np.mean(errors):.1f}%")


# ============================================================
# ЗАДАЧА 10 — Архитектура аналитической системы (OOP)
# ============================================================
print("\n--- ЗАДАЧА 10: Архитектура аналитической системы ---")
class EcommerceAnalytics:
    """
    Аналитическая система Студента 3.

    Объединяет все задачи в единый pipeline:
    prepare → metrics → segment → rank → forecast → report

    Использует данные, подготовленные Студентами 1 и 2.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._ready = False
        print(f"[Система] Загружено {len(self.df)} товаров, {len(self.df.columns)} колонок.")

    # Шаг 1: Подготовка данных
    def prepare(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        for col in self.df.select_dtypes(include=np.number).columns:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        self._ready = True
        print(f"[1] Подготовка: {before} → {len(self.df)} строк.")
        return self

    # Шаг 2: Расчёт метрик
    def calculate_metrics(self):
        self.df["profit_abs"] = self.df["final_price"] - self.df["cost_price"]
        self.df["engagement_score"] = (
            self.df["views"] +
            self.df["cart_additions"] +
            self.df["wishlist_additions"]
        )
        self.df["efficiency_score"] = self.df.apply(
            lambda row: (
                row["profit_margin"] *
                row["rating"] *
                np.log1p(row["sales_last_month"]) *
                (1 + row["conversion_rate"])
            ), axis=1
        )
        print("[2] Метрики рассчитаны: profit_abs, engagement_score, efficiency_score.")
        return self

    # Шаг 3: Сегментация
    def segment(self):
        self.df["segment"] = self.df.apply(segment_product, axis=1)
        dist = self.df["segment"].value_counts().to_dict()
        print(f"[3] Сегментация: {dist}")
        return self

    # Шаг 4: Ранжирование
    def rank(self, weights=None):
        if weights is None:
            weights = {
                "efficiency_score":   0.30,
                "rating":             0.25,
                "sales_last_month":   0.20,
                "revenue_last_month": 0.15,
                "profit_margin":      0.10,
            }
        self.df = rank_products(self.df, weights)
        print("[4] Ранжирование выполнено.")
        return self

    # Шаг 5: Прогноз продаж
    def forecast(self):
        self.df["forecast_sales"] = self.df.apply(
            lambda row: predict_sales(
                row["price"],
                row["rating"],
                row["conversion_rate"],
                row["profit_margin"],
                history=[row["sales_last_3_months"] / 3]
            ), axis=1
        )
        print("[5] Прогноз продаж рассчитан.")
        return self

    # Шаг 6: Генератор аномалий
    def get_anomalies(self, limit=10):
        result = []
        for product in anomaly_generator(self.df):
            result.append(product["product_name"])
            if len(result) >= limit:
                break
        print(f"[6] Аномалий найдено (первые {limit}): {len(result)}")
        return result

    # Шаг 7: Финальный отчёт
    def report(self):
        print("\n" + "=" * 55)
        print("ФИНАЛЬНЫЙ ОТЧЁТ — СТУДЕНТ 3")
        print("=" * 55)
        print(f"Всего товаров:           {len(self.df)}")
        print(f"Категорий:               {self.df['category'].nunique()}")
        print(f"Брендов:                 {self.df['brand'].nunique()}")
        print(f"\nСредняя цена:            {self.df['price'].mean():.2f}")
        print(f"Средний рейтинг:         {self.df['rating'].mean():.2f}")
        print(f"Средняя маржа:           {self.df['profit_margin'].mean():.2%}")
        print(f"Средние продажи/месяц:   {self.df['sales_last_month'].mean():.1f}")
        print(f"Средняя выручка/месяц:   {self.df['revenue_last_month'].mean():,.2f}")
        if "efficiency_score" in self.df.columns:
            print(f"Средняя эффективность:   {self.df['efficiency_score'].mean():.4f}")
        if "segment" in self.df.columns:
            print("\nРаспределение по сегментам:")
            for seg, cnt in self.df["segment"].value_counts().items():
                pct = cnt / len(self.df) * 100
                print(f"  {seg:<15}: {cnt:4d} товаров ({pct:.1f}%)")
        if "rank_score" in self.df.columns:
            print("\nТоп-5 товаров по комплексному рейтингу:")
            top5 = self.df.nlargest(5, "rank_score")[
                ["product_name", "category", "rating", "sales_last_month", "rank_score"]
            ]
            print(top5.to_string(index=False))
        return self

    # Полный pipeline
    def run_pipeline(self):
        print("\n" + "#" * 55)
        print("ЗАПУСК PIPELINE СТУДЕНТА 3")
        print("#" * 55)
        return (
            self.prepare()
                .calculate_metrics()
                .segment()
                .rank()
                .forecast()
                .report()
        )


# ЗАПУСК
if __name__ == "__main__":
    system = EcommerceAnalytics("products_optimized_output.csv")
    system.run_pipeline()
    system.get_anomalies(limit=10)

    # Сохраняем результат для Студента 4
    output_path = "student3_output.csv"
    system.df.to_csv(output_path, index=False)
    print(f"\nРезультат сохранён в '{output_path}' для Студента 4.")

    print("\n" + "=" * 55)
    print("Все 10 задач Студента 3 выполнены!")
    print("=" * 55)