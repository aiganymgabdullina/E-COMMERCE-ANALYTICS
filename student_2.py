import numpy as np
import pandas as pd
# ЗАГРУЗКА ДАННЫХ ОТ СТУДЕНТА 1
df = pd.read_csv("products_optimized_output.csv")

#1 Векторизация и первичный обзор NumPy-массивов
def transform_to_numpy_and_audit(df: pd.DataFrame) -> tuple:
    """Переводит числовые метрики в NumPy-массив
    и выполняет базовый статистический аудит.
    """
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 1: NUMPY ВЕКТОРНЫЙ АУДИТ ---")
    target_cols = [
        "price",
        "revenue_last_month",
        "rating",
        "views",
        "stock",
    ]
    matrix_data = df[target_cols].astype(float).to_numpy()
    shape = matrix_data.shape
    print(
        f"Форма массива: {shape} "
        f"(Товаров: {shape[0]}, Признаков: {shape[1]})"
    )
    means = np.mean(matrix_data, axis=0)
    medians = np.median(matrix_data, axis=0)
    stds = np.std(matrix_data, axis=0)
    for i, col in enumerate(target_cols):
        print(
            f"{col}: "
            f"mean={means[i]:.2f}, "
            f"median={medians[i]:.2f}, "
            f"std={stds[i]:.2f}"
        )
    print("\nПоиск экстремальных значений:")
    for i, col in enumerate(target_cols):
        col_data = matrix_data[:, i]
        z_scores = np.abs((col_data - means[i]) / (stds[i] + 1e-9))
        outliers_count = np.sum(z_scores > 3)
        print(f"{col}: найдено выбросов -> {outliers_count}")
    return matrix_data, target_cols

#2 Производные метрики и сводный анализ
def analyze_categories_and_brands(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 2: АНАЛИЗ КАТЕГОРИЙ И БРЕНДОВ ---")
    if "brand" not in df.columns:
        brands = ["TechCorp", "InnoStyle", "EcoGoods", "PrimeKit"]
        df["brand"] = [
            brands[i % len(brands)]
            for i in range(len(df))
        ]
    price_array = df["price"].to_numpy()
    views_array = df["views"].to_numpy()
    rating_array = df["rating"].to_numpy()
    df["price_per_view"] = np.where(
        views_array > 0,
        price_array / views_array,
        0
    )
    df["profit_per_unit"] = price_array * 0.40
    df["rating_views_index"] = (
        rating_array * np.log1p(views_array)
    )
    pivot_table = df.pivot_table(
        index=["category", "brand"],
        values=[
            "profit_per_unit",
            "revenue_last_month",
            "rating",
        ],
        aggfunc={
            "profit_per_unit": "mean",
            "revenue_last_month": "sum",
            "rating": "mean",
        },
    )
    print("\nСводная таблица:")
    print(pivot_table.head(10))
    return pivot_table

#3 Поиск структурных аномалий
def detect_structural_anomalies(
    matrix_data: np.ndarray,
    target_cols: list
) -> list:
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 3: АНОМАЛИИ ---")
    price_idx = target_cols.index("price")
    rev_idx = target_cols.index("revenue_last_month")
    rating_idx = target_cols.index("rating")
    avg_price = np.mean(matrix_data[:, price_idx])
    anomalies_report = []
    i = 0
    total_items = matrix_data.shape[0]
    while i < total_items:
        row = matrix_data[i]
        price = row[price_idx]
        revenue = row[rev_idx]
        rating = row[rating_idx]
        if price > (avg_price * 3) and rating < 2.5:
            anomalies_report.append({
                "row_index": i,
                "reason":
                    f"Высокая цена ({price:.2f}) "
                    f"при низком рейтинге ({rating})"
            })
        elif revenue == 0 and rating == 5.0:
            anomalies_report.append({
                "row_index": i,
                "reason":
                    "Идеальный рейтинг при нулевой выручке"
            })
        i += 1
    print(f"Найдено аномалий: {len(anomalies_report)}")
    if anomalies_report:
        print("Пример:", anomalies_report[0])
    return anomalies_report

#4 Автоматизация расчета метрик
def calculate_item_health_score(
    price: float,
    rating: float,
    views: float,
    stock: int
) -> tuple:
    efficiency_index = (rating * 20) + (views * 0.05)
    flags = []
    if stock < 5:
        flags.append("CRITICAL_STOCK_LOW")
    if price > 5000 and rating < 4.0:
        flags.append("OVERPRICED_BAD_RATING")
    if views < 10 and stock > 100:
        flags.append("DEAD_STOCK_NO_VIEWS")
    status = (
        "OUT_OF_BOUNDS"
        if len(flags) > 0
        else "NORMAL"
    )
    return efficiency_index, status, flags
def run_health_check_on_sample(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 4: ПРОВЕРКА 10 ТОВАРОВ ---")
    sample_10 = df.head(10)
    for _, row in sample_10.iterrows():
        score, status, flags = calculate_item_health_score(
            row.get("price", 0),
            row.get("rating", 0),
            row.get("views", 0),
            int(row.get("stock", 0)),
        )
        print(
            f"{row['product_name'][:20]} | "
            f"Score={score:.2f} | "
            f"{status} | {flags}"
        )

#5 Сегментация
def advanced_multidimensional_segmentation(
    df: pd.DataFrame
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 5: СЕГМЕНТАЦИЯ ---")
    if "seasonality" not in df.columns:
        seasons = ["High", "Medium", "Low"]
        df["seasonality"] = [
            seasons[i % 3]
            for i in range(len(df))
        ]
    med_revenue = df["revenue_last_month"].median()
    med_views = df["views"].median()
    segment_logic = lambda row: (
        "Высокомаржинальный Хит"
        if row["revenue_last_month"] >= med_revenue
        and row["rating"] >= 4.2
        and row["seasonality"] == "High"
        else (
            "Трафикогенератор"
            if row["views"] > med_views
            and row["revenue_last_month"] >= med_revenue
            else (
                "Неликвид"
                if row["rating"] < 3.5
                and row["stock"] > 50
                else "Базовый сегмент"
            )
        )
    )
    df["advanced_segment"] = [
        segment_logic(row)
        for _, row in df.iterrows()
    ]
    print(df["advanced_segment"].value_counts())
    return df

#6 Генератор потоковой обработки
def smart_stream_generator(
    file_path: str,
    category_metrics: dict
):
    for chunk in pd.read_csv(file_path, chunksize=500):
        for _, row in chunk.iterrows():
            cat = row.get("category")
            metrics = category_metrics.get(cat)
            if metrics:
                if (
                    row["price"] > metrics["mean_price"]
                    and row["rating"] > metrics["median_rating"]
                    and row["revenue_last_month"]
                    >= metrics["top_20_rev_threshold"]
                ):
                    yield {
                        "product_id": row.get("product_id"),
                        "product_name": row.get("product_name"),
                        "category": cat,
                        "price": row["price"],
                        "rating": row["rating"],
                    }
def precalculate_category_thresholds(
    df: pd.DataFrame
) -> dict:
    thresholds = {}
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        thresholds[cat] = {
            "mean_price": sub["price"].mean(),
            "median_rating": sub["rating"].median(),
            "top_20_rev_threshold":
                sub["revenue_last_month"].quantile(0.80),
        }
    return thresholds

#7 Тренды и сводные таблицы
def analyze_trends_and_matrices(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 7: ТРЕНДЫ ---")
    if "month" not in df.columns:
        df["month"] = np.random.randint(
            1,
            5,
            size=len(df)
        )
    trend_summary = (
        df.groupby(["category", "month"])[
            ["price", "revenue_last_month", "rating"]
        ]
        .mean()
        .to_numpy()
    )
    print("Форма тренд-массива:", trend_summary.shape)
    brand_pivot = df.pivot_table(
        index="brand",
        values="revenue_last_month",
        aggfunc="mean"
    ).sort_values(
        by="revenue_last_month",
        ascending=False
    )
    print("\nРейтинг брендов:")
    print(brand_pivot)

#8 Многокритериальное ранжирование
def advanced_multi_criteria_ranking(
    df: pd.DataFrame,
    weights: dict = None
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 8: РАНЖИРОВАНИЕ ---")
    if weights is None:
        weights = {
            "profit": 0.4,
            "rating": 0.3,
            "sales": 0.2,
            "views": 0.1,
        }
    df_rank = df.copy()
    df_rank["revenue_last_month"] = (
        df_rank["revenue_last_month"]
        .apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    )
    df_rank["rating"] = (
        df_rank["rating"]
        .apply(lambda x: 0 if pd.isna(x) or x <= 0 else x)
    )
    df_rank["views"] = (
        df_rank["views"]
        .apply(lambda x: 1 if pd.isna(x) or x <= 0 else x)
    )
    norm = lambda col: (
        (df_rank[col] - df_rank[col].min())
        /
        (df_rank[col].max() - df_rank[col].min() + 1e-9)
    )
    r_rev = norm("revenue_last_month")
    r_rat = norm("rating")
    r_view = norm("views")
    r_stock = norm("stock")
    df_rank["final_weighted_score"] = (
        (r_rev * weights["profit"])
        + (r_rat * weights["rating"])
        + (r_stock * weights["sales"])
        + (r_view * weights["views"])
    )
    top_10 = (
        df_rank.sort_values(
            by="final_weighted_score",
            ascending=False
        )
        .head(10)
    )
    print(top_10[   [
            "product_name",
            "category",
            "brand",
            "final_weighted_score",
        ]
    ])
    return top_10

#9 Корреляции NumPy
def compute_pure_numpy_correlations(
    matrix_data: np.ndarray,
    target_cols: list
):
    print("\n" + "=" * 60)
    print("--- ЗАДАЧА 9: КОРРЕЛЯЦИИ ---")
    corr_matrix = np.corrcoef(
        matrix_data,
        rowvar=False
    )
    for i, col_i in enumerate(target_cols):
        for j, col_j in enumerate(target_cols):
            if i < j:
                coef = corr_matrix[i, j]
                print(
                    f"{col_i} <-> {col_j}: "
                    f"r = {coef:.4f}"
                )
    print("\nИнтерпретация:")
    print(
        "Высокая корреляция между views и revenue "
        "может говорить о зависимости продаж от трафика."
    )

#10 Архитектура аналитической системы
class AdvancedAnalyticsOrchestrator:
    def __init__(self, df_source: pd.DataFrame):
        self.raw_df = df_source.copy()
        self.processed_df = None
        self.numpy_matrix = None
        self.features_list = None
    def execute_pipeline(self):
        print("\n" + "#" * 70)
        print("ЗАПУСК PIPELINE СТУДЕНТА 2")
        print(
            "Используются очищенные данные, "
            "подготовленные Студентом 1"
        )
        print("#" * 70)
        self.processed_df = self.raw_df.copy()
        analyze_categories_and_brands(
            self.processed_df
        )
        self.numpy_matrix, self.features_list = (
            transform_to_numpy_and_audit(
                self.processed_df
            )
        )
        compute_pure_numpy_correlations(
            self.numpy_matrix,
            self.features_list
        )
        self.processed_df = (
            advanced_multidimensional_segmentation(
                self.processed_df
            )
        )
        detect_structural_anomalies(
            self.numpy_matrix,
            self.features_list
        )
        run_health_check_on_sample(
            self.processed_df
        )
        analyze_trends_and_matrices(
            self.processed_df
        )
        advanced_multi_criteria_ranking(
            self.processed_df
        )
        print("\nPipeline успешно завершен.")

# ЗАПУСК СИСТЕМЫ
if __name__ == "__main__":
    analytics_system = AdvancedAnalyticsOrchestrator(df)
    analytics_system.execute_pipeline()