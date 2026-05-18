#=====TASK 1=====
#Вы получили данные о поведении пользователей маркетплейса: просмотры страниц товаров, добавление в корзину, заказы, отзывы и рейтинги.
#Задача – провести первичный анализ пользовательской активности, подсчитать количество активных и неактивных пользователей, средний
#рейтинг и среднее количество действий на пользователя. Необходимо выявить закономерности, например, сколько пользователей совершают
#покупки после просмотра, и сделать текстовый отчет с объяснением важности этих показателей для бизнеса.
import pandas as pd

df = pd.read_csv('products_advanced_dataset.csv')
df = df.fillna(0)
df['total_activity'] = df['views'] + df['cart_additions'] + df['wishlist_additions']

active_users_df = df[df['total_activity'] > 0]
inactive_users_count = len(df[df['total_activity'] == 0])
active_users_count = len(active_users_df)

avg_rating = df[df['rating'] > 0]['rating'].mean()
avg_actions = active_users_df['total_activity'].mean()

viewers = df[df['views'] > 0]
buyers_after_view = len(viewers[viewers['sales_last_month'] > 0])
conversion_rate = (buyers_after_view / len(viewers)) * 100 if len(viewers) > 0 else 0

print(f"Количество активных пользователей: {active_users_count}")
print(f"Количество неактивных пользователей: {inactive_users_count}")
print(f"Средний рейтинг товаров: {avg_rating:.2f}")
print(f"Среднее кол-во действий на пользователя: {avg_actions:.2f}")
print(f"Конверсия из просмотра в покупку: {conversion_rate:.2f}%")

#=====TASK 2=====
#Вам нужно создать функцию для фильтрации пользователей по активности. Используя условия if/else и циклы, функция должна
#возвращать пользователей, которые соответствуют сложным критериям: например, совершили не менее трех покупок, оставили
#рейтинг выше 4 и просмотрели не менее 10 товаров. Необходимо протестировать функцию на примере 20 пользователей, объяснить
#логику отбора и возможные применения в маркетинговых кампаниях.
users_sample = [
    {"id": i, "purchases": i % 5, "avg_rating": (i % 5) + 0.5, "views": i * 2}
    for i in range(1, 21)
]

def filter_top_users(users):
    selected_users = []
    for u in users:
        if u['purchases'] >= 3 and u['avg_rating'] > 4 and u['views'] >= 10:
            selected_users.append(u)
        else:
            continue
    return selected_users

vip_users = filter_top_users(users_sample)
print(f"Найдено подходящих пользователей: {len(vip_users)}")
print(vip_users)

#=====TASK 3=====
#Для отдела персонализации требуется генератор, который по мере поступления данных возвращает товары, с которыми пользователи
#взаимодействовали на высоком уровне: добавляли в корзину, покупали или оставляли положительные отзывы. Генератор должен работать
#с большим массивом данных, выдавая товары по одному, чтобы не перегружать память. Результат – первые 15 товаров с подробной
#информацией о действиях пользователей.
def high_interaction_generator(dataframe):
    for index, row in dataframe.iterrows():
        if row['cart_additions'] > 0 or row['sales_last_month'] > 0 or row['rating'] >= 4:
            yield {
                "product": row['product_name'],
                "action": "Куплено" if row['sales_last_month'] > 0 else "В корзине/Высокий рейтинг",
                "rating": row['rating']
            }

product_gen = high_interaction_generator(df)

print("Топ-15 товаров с высоким взаимодействием:")
for i in range(15):
    print(next(product_gen))

#=====TASK 4=====
#Аналитика интересует распределение наград и бонусов пользователей, например, начисленных баллов за покупки. Задача – создать
#словарь, где ключ – возрастная категория или сегмент пользователя, а значение – количество наград. Необходимо обработать все
#данные, выявить топ-2 сегмента с наибольшей активностью, описать результаты текстом и предложить возможные причины различий.
category_rewards = {}

for index, row in df.iterrows():
    cat = row['category']
    reward = row['revenue_last_month'] * 0.10

    if cat in category_rewards:
        category_rewards[cat] += reward
    else:
        category_rewards[cat] = reward

top_segments = sorted(category_rewards.items(), key=lambda x: x[1], reverse=True)[:2]

print("Топ-2 сегмента по наградам:")
for segment, value in top_segments:
    print(f"Категория: {segment}, Сумма баллов: {value:.0f}")

#=====TASK 5=====
#Вам поручено создать уникальные комбинации действий и категорий товаров, которые совершали пользователи. Используя множества (set),
#нужно определить, сколько уникальных комбинаций “действие + категория” встречается среди активных пользователей. Результатом должна
#быть таблица или список этих комбинаций и текстовый анализ, объясняющий, какие комбинации наиболее распространены.
import pandas as pd

file_name = 'catalog_products.xlsx - Sheet1.csv'
raw_df = pd.read_csv(file_name)

parsed_data = []

for i in range(0, raw_df.shape[1], 3):
    if i + 2 < raw_df.shape[1]:
        sub_df = raw_df.iloc[:, i:i + 3].dropna()
        sub_df.columns = ['category', 'views', 'sales_last_month']
        parsed_data.append(sub_df)

df = pd.concat(parsed_data, ignore_index=True)

df = df.fillna(0)

df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
df['sales_last_month'] = pd.to_numeric(df['sales_last_month'], errors='coerce').fillna(0)

df['cart_additions'] = (df['sales_last_month'] * 1.4).astype(int)

combinations_list = []

for index, row in df.iterrows():
    cat = str(row['category']).strip()
    if cat == 'nan' or cat == '':
        continue

    if row['views'] > 0:
        combinations_list.append(f"view + {cat}")
    if row['cart_additions'] > 0:
        combinations_list.append(f"cart + {cat}")
    if row['sales_last_month'] > 0:
        combinations_list.append(f"buy + {cat}")

unique_combinations = set(combinations_list)

comp_series = pd.Series(combinations_list)
counts = comp_series.value_counts()

print(f"Количество уникальных комбинаций: {len(unique_combinations)}\n")
print("Список уникальных комбинаций:")
for item in sorted(unique_combinations):
    print(f" | {item}")

print("\nТоп-10 самых распространенных комбинаций по частоте:")
print(counts.head(10))

#=====TASK 6=====
#Разработайте метрику “успех на пользователя”, учитывающую количество просмотров, покупки, отзывы и рейтинги. Используйте lambda-функции
#для расчета новой колонки в DataFrame. Затем выведите топ-10 пользователей или товаров по этой метрике, объясните выбор формулы,
#используемые веса и влияние каждого параметра на итоговую оценку.
import pandas as pd
import numpy as np

file_name = 'catalog_products.xlsx - Sheet1.csv'
raw_df = pd.read_csv(file_name)

parsed_data = []
for i in range(0, raw_df.shape[1], 3):
    if i+2 < raw_df.shape[1]:
        sub_df = raw_df.iloc[:, i:i+3].dropna()
        sub_df.columns = ['category', 'views', 'sales_last_month']
        parsed_data.append(sub_df)

df = pd.concat(parsed_data, ignore_index=True).fillna(0)

df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
df['sales_last_month'] = pd.to_numeric(df['sales_last_month'], errors='coerce').fillna(0)

df['cart_additions'] = (df['sales_last_month'] * 1.4).astype(int)

df['rating'] = np.where(df['views'] > 0, np.round(np.random.uniform(3.5, 5.0, len(df)), 1), 0.0)


df['product_success_score'] = df.apply(
    lambda row: (0.1 * row['views']) +
                (1.5 * row['cart_additions']) +
                (5.0 * row['sales_last_month']) +
                (2.0 * row['rating'] if row['sales_last_month'] > 0 else 0),
    axis=1
)

top_10_products = df.sort_values(by='product_success_score', ascending=False).head(10)

print("--- ТОП-10 УСПЕШНЫХ ТОВАРОВ ---")
print(top_10_products[['category', 'views', 'sales_last_month', 'rating', 'product_success_score']])

#=====TASK 7=====
#Для быстрого анализа больших массивов создайте NumPy массивы, включающие показатели активности пользователей: количество заказов,
#просмотров, отзывов, рейтинги. Рассчитайте среднее, стандартное отклонение, медиану и найдите пользователя с максимальной
#активностью по нескольким показателям. Задача развивает навыки работы с многомерными массивами и статистикой.
import numpy as np

activity_matrix = df[['views', 'cart_additions', 'sales_last_month', 'rating']].to_numpy()

means = np.mean(activity_matrix, axis=0)
stds = np.std(activity_matrix, axis=0)
medians = np.median(activity_matrix, axis=0)

columns = ["Просмотры", "В корзине", "Продажи", "Рейтинг"]

print("--- СТАТИСТИЧЕСКИЙ АНАЛИЗ МАССИВА (Задача 7) ---")
for i, col in enumerate(columns):
    print(f"{col}: Среднее = {means[i]:.2f}, Станд. отклонение = {stds[i]:.2f}, Медиана = {medians[i]:.2f}")

max_sales_idx = np.argmax(activity_matrix[:, 2])
best_category = df.iloc[max_sales_idx]['category']
best_sales = df.iloc[max_sales_idx]['sales_last_month']
best_views = df.iloc[max_sales_idx]['views']

print("\n--- ЛИДЕР АКТИВНОСТИ ---")
print(f"Самый успешный товар находится в категории: '{best_category}'")
print(f"Показатели лидера -> Продажи: {best_sales:.0f} шт., Просмотры: {best_views:.0f}")

#=====TASK 8=====
#Создайте сводные таблицы, которые показывают средний пользовательский рейтинг и среднюю прибыль по категориям товаров и
#сегментам пользователей. Затем создайте вторую сводную таблицу для среднего показателя “успех на пользователя”.
#Сравните результаты, выделите сегменты с наибольшей ценностью для бизнеса и объясните, почему получились именно такие выводы.
import pandas as pd
import numpy as np

file_name = 'catalog_products.xlsx - Sheet1.csv'
raw_df = pd.read_csv(file_name)

parsed_data = []
for i in range(0, raw_df.shape[1], 3):
    if i+2 < raw_df.shape[1]:
        sub_df = raw_df.iloc[:, i:i+3].dropna()
        sub_df.columns = ['category', 'views', 'sales_last_month']
        parsed_data.append(sub_df)

df = pd.concat(parsed_data, ignore_index=True).fillna(0)
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
df['sales_last_month'] = pd.to_numeric(df['sales_last_month'], errors='coerce').fillna(0)
df['cart_additions'] = (df['sales_last_month'] * 1.4).astype(int)

np.random.seed(42)
df['rating'] = np.where(df['views'] > 0, np.round(np.random.uniform(3.5, 5.0, len(df)), 1), 0.0)
df['profit'] = df['sales_last_month'] * np.random.uniform(10, 50, len(df)) # Условная прибыль
df['user_segment'] = np.random.choice(['New', 'Standard', 'Premium'], size=len(df), p=[0.3, 0.5, 0.2])

df['product_success_score'] = (0.1 * df['views']) + (1.5 * df['cart_additions']) + (5.0 * df['sales_last_month'])

pivot_1 = pd.pivot_table(
    df,
    values=['rating', 'profit'],
    index='category',
    columns='user_segment',
    aggfunc='mean'
)

pivot_2 = pd.pivot_table(
    df,
    values='product_success_score',
    index='category',
    columns='user_segment',
    aggfunc='mean'
)

print("--- СВОДНАЯ ТАБЛИЦА 1 (Рейтинг и Прибыль) ---")
print(pivot_1.round(2))
print("\n--- СВОДНАЯ ТАБЛИЦА 2 (Метрика Успеха) ---")
print(pivot_2.round(2))

#=====TASK 9=====
#Для презентации руководству необходимо построить графики с Matplotlib и Seaborn: гистограмму пользовательских рейтингов,
#scatter plot количества сезонов и эпизодов или аналогичных показателей для товаров и пользователей, boxplot по сегментам.
#Важна визуальная наглядность и правильная маркировка осей, легенд и заголовков. Задача тренирует визуализацию сложных
#данных и интерпретацию графиков.

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df[df['rating'] > 0]['rating'], bins=15, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Распределение рейтингов товаров', fontsize=14)
axes[0].set_xlabel('Рейтинг', fontsize=12)
axes[0].set_ylabel('Количество товаров', fontsize=12)

sns.scatterplot(data=df, x='views', y='sales_last_month', hue='user_segment', alpha=0.7, ax=axes[1])
axes[1].set_title('Взаимосвязь Просмотров и Продаж', fontsize=14)
axes[1].set_xlabel('Количество просмотров', fontsize=12)
axes[1].set_ylabel('Количество продаж', fontsize=12)

sns.boxplot(
    data=df,
    x='user_segment',
    y='product_success_score',
    hue='user_segment',
    palette='Set2',
    legend=False,
    ax=axes[2]
)
axes[2].set_title('Разброс метрики успеха по сегментам', fontsize=14)
axes[2].set_xlabel('Сегмент пользователя', fontsize=12)
axes[2].set_ylabel('Балл успеха', fontsize=12)

plt.tight_layout()
plt.show()

#=====TASK 10=====
#Вы должны разработать систему аналитических функций и генераторов, объединяющую все предыдущие задачи: фильтры, метрики,
#генераторы потоковых данных, сводные таблицы и визуализацию. Задача заключается в проектировании повторно используемой
#архитектуры: функции для расчета активности, генераторы для потоковых данных, методы агрегирования и визуализации.
#Итог – масштабируемая система для анализа пользовательской активности.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MarketplaceAnalyticsPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_and_clean_data(self):
        """Пайплайн парсинга и предобработки нестандартного CSV"""
        raw_df = pd.read_csv(self.file_path)
        parsed_data = []
        for i in range(0, raw_df.shape[1], 3):
            if i + 2 < raw_df.shape[1]:
                sub_df = raw_df.iloc[:, i:i + 3].dropna()
                sub_df.columns = ['category', 'views', 'sales_last_month']
                parsed_data.append(sub_df)

        self.df = pd.concat(parsed_data, ignore_index=True).fillna(0)
        self.df['views'] = pd.to_numeric(self.df['views'], errors='coerce').fillna(0)
        self.df['sales_last_month'] = pd.to_numeric(self.df['sales_last_month'], errors='coerce').fillna(0)
        self.df['cart_additions'] = (self.df['sales_last_month'] * 1.4).astype(int)

        np.random.seed(42)
        self.df['rating'] = np.where(self.df['views'] > 0, np.round(np.random.uniform(3.5, 5.0, len(self.df)), 1), 0.0)
        self.df['user_segment'] = np.random.choice(['New', 'Standard', 'Premium'], size=len(self.df))
        return self

    def calculate_metrics(self):
        """Расчет комплексной метрики успеха через lambda"""
        self.df['success_score'] = self.df.apply(
            lambda r: (0.1 * r['views']) + (1.5 * r['cart_additions']) + (5.0 * r['sales_last_month']),
            axis=1
        )
        return self

    def data_stream_generator(self, chunk_size=5):
        """Генератор потоковых данных для экономии памяти (Задача 3/10)"""
        for i in range(0, len(self.df), chunk_size):
            yield self.df.iloc[i:i+chunk_size]

    def run_full_report(self):
        """Запуск финальной агрегации и построения сводных таблиц"""
        print("=== СВОДНЫЙ ОТЧЕТ СИСТЕМЫ ===")
        pivot = pd.pivot_table(self.df, values='success_score', index='category', columns='user_segment', aggfunc='mean')
        print(pivot.round(2))

        plt.figure(figsize=(8, 4))
        sns.barplot(data=self.df, x='category', y='success_score', hue='user_segment', palette='magma')
        plt.title('Итоговый успех категорий в разрезе сегментов')
        plt.tight_layout()
        plt.show()

pipeline = MarketplaceAnalyticsPipeline('catalog_products.xlsx - Sheet1.csv')
pipeline.load_and_clean_data().calculate_metrics().run_full_report()

stream = pipeline.data_stream_generator(chunk_size=3)
print("\nПервый потоковый пакет данных из генератора:")
print(next(stream)[['category', 'views', 'success_score']])