#%%
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import phik
from sqlalchemy import create_engine
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
from phik import resources, report
from phik.report import plot_correlation_matrix

# Загружаем переменные окружения из .env
load_dotenv()

# Параметры подключения
db_config = {
    'user': os.getenv('DB_USER'),
    'pwd':  os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 6432)),
    'db':   os.getenv('DB_NAME'),
}

# Создаем подключение
url = URL.create(
    drivername='postgresql+psycopg2',
    username=db_config['user'],
    password=db_config['pwd'],
    host=db_config['host'],
    port=db_config['port'],
    database=db_config['db'],
)

engine = create_engine(url)

# Проверка подключения
with engine.connect() as conn:
    print('Подключение успешно')

# Выгружаем данные
query = '''
WITH set_config_precode AS (
    SELECT set_config('synchronize_seqscans', 'off', true)
)
SELECT
    p.user_id,
    p.device_type_canonical,
    p.order_id,
    p.created_dt_msk AS order_dt,
    p.created_ts_msk AS order_ts,
    p.currency_code,
    p.revenue,
    p.tickets_count,
    p.created_dt_msk::date
      - LAG(p.created_dt_msk::date) OVER (
            PARTITION BY p.user_id ORDER BY p.created_dt_msk
        ) AS days_since_prev,
    p.event_id,
    e.event_name_code AS event_name,
    e.event_type_main,
    p.service_name,
    c.city_name,
    r.region_name
FROM afisha.purchases AS p
JOIN afisha.events AS e ON e.event_id = p.event_id
JOIN afisha.city AS c ON c.city_id = e.city_id
JOIN afisha.regions AS r ON r.region_id = c.region_id
WHERE p.device_type_canonical IN ('mobile', 'desktop')
  AND e.event_type_main != 'фильм'
ORDER BY p.user_id;
'''

df = pd.read_sql_query(query, con=engine)

# Выводим информацию о данных
print(df.info())

# Выводим данные
print(df.head())

# Выгружаем данные
info_ten = pd.read_csv("https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv")

# Выводим информацию о данных
print(info_ten.info())

# Выводим данные
print(info_ten.head())

# Переведем деньги из kzt в ru

# Приводим дату курса к типу datetime
info_ten['data'] = pd.to_datetime(info_ten['data'])

# Находим сколько ru в 1 kzt
info_ten['rub_per_kzt'] = info_ten['curs'] / info_ten['nominal']

rate_map = info_ten.set_index('data')['rub_per_kzt']
df['revenue_rub'] = df['revenue']
mask_kzt = df['currency_code'].str.lower() == 'kzt'

# Пересчитываем по дате
df.loc[mask_kzt, 'revenue_rub'] = (df.loc[mask_kzt, 'revenue'] * df.loc[mask_kzt, 'order_dt'].map(rate_map))
# Удаляем лишнее
df = df.drop(columns=['rub_per_kzt_x', 'rub_per_kzt_y', 'rub_per_kzt'], errors='ignore')

# Выводим данные
df.head()

# Выведем количество пропусков
df.isna().sum()

# Найдем процент пропусков
missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
round(missing_percent, 2)

# Сортируем по пользователю и дате
d = df.sort_values(['user_id', 'order_dt']).copy()
d['is_first'] = ~d.duplicated('user_id', keep='first')

# все NaN должны быть первыми строками
bad_nan = d[d['days_since_prev'].isna() & ~d['is_first']]
print('NaN не у первого заказа:', len(bad_nan))

# все первые строки должны быть NaN
bad_first = d[d['is_first'] & d['days_since_prev'].notna()]
print('Первый заказ без NaN:', len(bad_first))

# Приведем к типу float
for col in ['revenue', 'days_since_prev', 'revenue_rub']:
    df[col] = pd.to_numeric(df[col], downcast='float')

# Приведем к типу integer
for col in ['order_id', 'tickets_count', 'event_id']:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Выводим информацию о данных
df.info()

# Проверяем полные дубликаты
df.duplicated().sum()

# Проверяем неявные дубликаты
df.duplicated(subset='user_id').sum()

cols_lower = ['device_type_canonical', 'event_type_main','service_name', 'city_name', 'region_name']

for c in cols_lower:
    df[c] = df[c].astype(str).str.strip().str.lower()

for i in ['device_type_canonical', 'event_type_main', 'service_name', 'city_name', 'region_name']:
    print(df[i].unique().tolist())

# Проверяем полные дубликаты
df.duplicated().sum()

# Выведем статистику
df[['revenue_rub', 'tickets_count']].describe()

# Гистограмма
plt.figure(figsize=(12, 6))
df['revenue_rub'].hist(bins=100)
plt.title('Выручка')
plt.xlabel('Выручка, руб.')
plt.ylabel('Число заказов')
plt.show()

# Столбчатая диаграмма
plt.figure(figsize=(12, 6))
df['tickets_count'].value_counts().sort_index().plot(kind='bar')
plt.title('Количество билетов')
plt.xlabel('Билетов в заказе')
plt.ylabel('Число заказов')
plt.tight_layout()
plt.show()

# Диаграммы размаха (boxplot)
df.boxplot(column='revenue_rub', figsize=(10, 6))
plt.title('Выручка с заказа')
plt.ylabel('Выручка, руб.')
plt.show()

df.boxplot(column='tickets_count', figsize=(10, 6))
plt.title('Количество билетов')
plt.ylabel('Билетов в заказе')
plt.show()

q_99 = df['revenue_rub'].quantile(0.99)
print('99-й перцентиль:', round(q_99, 4))

# Число строк
n_before = len(df)

# Оставляем только строки с выручкой <= q_99 и неотрицательные
df = df[df['revenue_rub'] <= q_99].copy()
df = df[df['revenue_rub'] >= 0].copy()

# Число строк
n_after = len(df)

# Объём отфильтрованных данных
print('Отфильтровано строк:', n_before - n_after, f'({(n_before - n_after)/n_before*100:.2f}%)')

# Вывод статистики
df['revenue_rub'].describe()

# Сортировка по пользователю и дате
df = df.sort_values(['user_id', 'order_dt']).reset_index(drop=True)

first_orders = (df.groupby('user_id', as_index=False).first().rename(
    columns={
        'order_dt': 'first_order_dt',
        'device_type_canonical': 'first_device',
        'region_name': 'first_region',
        'service_name': 'first_service',
        'event_type_main': 'first_event_type',
    })[['user_id', 'first_order_dt', 'first_device',
        'first_region', 'first_service', 'first_event_type']]
)

last_orders = (df.groupby('user_id', as_index=False)['order_dt'].max().rename(columns={'order_dt': 'last_order_dt'}))

# Средние значения
agg = (df.groupby('user_id', as_index=False).agg(
    total_orders = ('order_id', 'count'),
    avg_revenue_rub = ('revenue_rub', 'mean'),
    avg_tickets_count = ('tickets_count', 'mean'),
    avg_days_between = ('days_since_prev', 'mean'),
)
)

# Бинарные признаки
agg['is_two']  = (agg['total_orders'] >= 2).astype(int)
agg['is_five'] = (agg['total_orders'] >= 5).astype(int)

# Собираем профиль пользователя
user_profile = (first_orders.merge(last_orders, on='user_id', how='left').merge(agg, on='user_id', how='left'))

# Выводим данные
user_profile.head()

# Число уникальных пользователей в профиле
n_users = user_profile['user_id'].nunique()

# Средняя выручка с одного заказа
avg_revenue = user_profile['avg_revenue_rub'].mean()

# Доля пользователей с 2+ заказами
share_two = user_profile['is_two'].mean()

# Доля пользователей с 5+ заказами
share_five = user_profile['is_five'].mean()

# Вывод
print('Пользователей:', n_users)
print('Средняя выручка с заказа:', round(avg_revenue, 2))
print('Доля 2+ заказов:', round(share_two*100, 2), '%')
print('Доля 5+ заказов:', round(share_five*100, 2), '%')

# Статистика
user_profile[['total_orders', 'avg_tickets_count', 'avg_days_between']].describe(percentiles=[0.5, 0.9, 0.95, 0.99])

# До фильтрации
n_before = len(user_profile)

# 99-й перцентиль
q_99_orders = user_profile['total_orders'].quantile(0.99)
print('q_99 total_orders:', q_99_orders)

# Оставляем только пользователей с числом заказов <= q_99
up_filt = user_profile[user_profile['total_orders'] <= q_99_orders].copy()

# После фильтрации
n_after = len(up_filt)

# Вывод
print('Удалено:', n_before - n_after, f'({(n_before - n_after) / n_before * 100:.2f}%)')

# Статистика
up_filt[['total_orders', 'avg_tickets_count', 'avg_days_between']].describe(percentiles=[0.5, 0.9, 0.95, 0.99])

def group(df, col, name):
    # Группируем и для уникаьных пользователей находим значения
    s = (df.groupby(col)['user_id'].nunique().sort_values(ascending=False).to_frame(name='users'))

    # Доля сегмента
    s['share_%'] = (s['users'] / s['users'].sum() * 100).round(2)

    # Вывод
    print(f"{name} ({col})")
    print(s)


# Применяем функцию
seg_event = group(user_profile, 'first_event_type', 'Тип первого мероприятия')
seg_device = group(user_profile, 'first_device', 'Устройство первого заказа')
seg_region = group(user_profile, 'first_region', 'Регион первого заказа')
seg_service = group(user_profile, 'first_service', 'Оператор первого заказа')

# Общая доля вернувшихся
overall = user_profile['is_two'].mean()
print(f'Средняя доля вернувшихся: {overall*100:.2f}%')

overall_5 = user_profile['is_five'].mean()
print(f'Средняя доля вернувшихся: {overall_5*100:.2f}%')

def segment_return(df, col, name, min_users=30):
    """
    Для каждого сегмента по признаку col считаем: число уникальных пользователей, долю вернувшихся
    """
    g = (df.groupby(col).agg(users=('user_id', 'nunique'), share_returned=('is_two', 'mean'))
         .sort_values('users', ascending=False))

    g['share_%'] = (g['share_returned'] * 100).round(2)

    print(f"{name} ({col})")
    print(g.head(10))
    return g


# Применяем функцию
seg_event = segment_return(user_profile, 'first_event_type', 'Тип первого мероприятия')
seg_device = segment_return(user_profile, 'first_device', 'Устройство первого заказа')
seg_region = segment_return(user_profile, 'first_region', 'Регион первого заказа')
seg_service = segment_return(user_profile, 'first_service', 'Оператор первого заказа')

def plot_top_return(g, title, top_n=10):
    # Берём top_n крупнейших сегментов и разворачиваем порядок, чтобы крупнейший был сверху
    d = g.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))

    # Горизонтальные столбцы: категории по оси Y, доля (%) по оси X
    bars = ax.barh(d.index, d['share_%'])

    # Красная пунктирная линия среднее по выборке
    ax.axvline(overall * 100, color='red', linestyle='--', label=f'Среднее по выборке: {overall*100:.1f}%')
    ax.set_title(title)
    ax.set_xlabel('Доля вернувшихся, %')
    ax.legend()

    # Подписи у каждого столбца
    for bar, v, n in zip(bars, d['share_%'], d['users']):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{v:.1f}% (n={n})', va='center', fontsize=8)
    plt.show()


# Строим 4 графика по признакам
plot_top_return(seg_event, 'Тип первого мероприятия: доля вернувшихся')
plot_top_return(seg_device, 'Устройство первого заказа: доля вернувшихся')
plot_top_return(seg_region, 'Топ-10 регионов: доля вернувшихся')
plot_top_return(seg_service, 'Топ-10 операторов: доля вернувшихся')

#Исходные данные
n_sport, n_conc = 805, 9648
p_sport, p_conc = 0.563975, 0.621787

# Число вернувшихся
count = np.array([round(p_sport * n_sport), round(p_conc  * n_conc)])

# Размеры групп
nobs  = np.array([n_sport, n_conc])

# z-тест для двух долей
stat, pval = proportions_ztest(count, nobs, alternative='larger')

# Вывод
print(f'z = {stat:.3f}, p-value = {pval:.4f}')

# Для каждого региона считаем число уникальных пользователей, сколько из них вернулись
reg = user_profile.groupby('first_region').agg(users=('user_id', 'nunique'), returned=('is_two', 'sum')).reset_index()

# Доля вернувшихся в каждом регионе
reg['share'] = reg['returned'] / reg['users']

# Деление регионов на группы по размеру
med = reg['users'].median()

# active - крупный,less_active — мелкий
reg['group'] = np.where(reg['users'] >= med, 'active', 'less_active')

# Взвешенное по пользователям доля вернувшихся
g = reg.groupby('group').agg(users_total=('users', 'sum'), returned_total=('returned', 'sum'))
g['share_weighted'] = (g['returned_total'] / g['users_total'] * 100).round(2)
print(g)

# Вывод
print(reg.groupby('group')[['users', 'share']].mean())

# Формируем две группы
one_order = up_filt[up_filt['is_two'] == 0]['avg_revenue_rub']
returned  = up_filt[up_filt['is_two'] == 1]['avg_revenue_rub']

# Вывод
print(f'Однозаказные: {len(one_order)}, медиана {one_order.median():.1f}, среднее {one_order.mean():.1f}')
print(f'Вернувшиеся:  {len(returned)}, медиана {returned.median():.1f}, среднее {returned.mean():.1f}')

# Общие границы интервалов
bins = np.histogram_bin_edges(np.concatenate([one_order, returned]), bins=50)

# Гистограммы
plt.figure(figsize=(10, 5))
plt.hist(one_order, bins=bins, alpha=0.5, density=True,
         label='1 заказ', color='tab:orange')
plt.hist(returned,  bins=bins, alpha=0.5, density=True,
         label='2+ заказа', color='tab:blue')

plt.xlabel('Средняя выручка с заказа')
plt.ylabel('Плотность')
plt.title('Распределение средней выручки: однократные vs вернувшиеся')
plt.legend()
plt.show()

# Формируем две группы по числу заказов
g_2_4 = up_filt[(up_filt['total_orders'] >= 2) & (up_filt['total_orders'] <= 4)]['avg_revenue_rub']
g_5p  = up_filt[up_filt['total_orders'] >= 5]['avg_revenue_rub']

# Вывод
print(f'2–4 заказа: n={len(g_2_4)}, медиана {g_2_4.median():.1f}, среднее {g_2_4.mean():.1f}')
print(f'5+ заказов: n={len(g_5p)}, медиана {g_5p.median():.1f}, среднее {g_5p.mean():.1f}')

# Общие границы интервалов
bins = np.histogram_bin_edges(np.concatenate([g_2_4, g_5p]), bins=50)

# Гистограммы
plt.figure(figsize=(10, 5))

plt.hist(g_2_4, bins=bins, alpha=0.5, density=True,
         label='2–4 заказа', color='tab:orange')
plt.hist(g_5p,  bins=bins, alpha=0.5, density=True,
         label='5+ заказов', color='tab:green')

plt.xlabel('Средняя выручка с заказа')
plt.ylabel('Плотность')
plt.title('Средняя выручка: 2–4 заказа vs 5+ заказов')
plt.legend()
plt.show()

# Статистика
up_filt['avg_tickets_count'].describe()

# Гистограмма
plt.figure(figsize=(12, 6))
plt.hist(up_filt['avg_tickets_count'], bins=30,
         color='tab:blue', alpha=0.8)

plt.xlabel('Среднее число билетов в заказе')
plt.ylabel('Пользователей')
plt.title('Распределение')
plt.show()

# Границы и метки сегментов
bins   = [1, 2, 3, 5, np.inf]
labels = ['1–2 билета', '2–3 билета', '3–5 билетов', '5+ билетов']

# Разбиваем пользователей на сегменты
up_filt['tickets_segment'] = pd.cut(up_filt['avg_tickets_count'], bins=bins, labels=labels, right=False, include_lowest=True)

# Агрегируем по сегментам
seg = (up_filt.groupby('tickets_segment', observed=True).agg(users=('user_id', 'nunique'), returned=('is_two', 'sum'))
       .reset_index())

# Доли
seg['share_returned'] = (seg['returned'] / seg['users'] * 100).round(2)
seg['share_users']    = (seg['users'] / seg['users'].sum() * 100).round(2)

# Вывод
print(seg)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))

# График 1: где сосредоточена аудитория
ax[0].bar(seg['tickets_segment'], seg['users'],
          color='tab:blue', alpha=0.8)
ax[0].set_title('Пользователей в сегменте')
ax[0].set_ylabel('Пользователей')

# График 2: доля вернувшихся по сегментам
bars = ax[1].bar(seg['tickets_segment'], seg['share_returned'],
                 color='tab:green', alpha=0.8)

ax[1].axhline(up_filt['is_two'].mean() * 100,
              color='red', linestyle='--',
              label=f'Среднее: {up_filt["is_two"].mean()*100:.1f}%')
ax[1].set_title('Доля вернувшихся по сегментам')
ax[1].set_ylabel('%')
ax[1].legend()

# Подписи значений над столбцами правого графика
for b, v in zip(bars, seg['share_returned']):
    ax[1].text(b.get_x() + b.get_width()/2, v + 0.5,
               f'{v:.1f}%', ha='center', fontsize=9)
plt.show()

# Выделяем день недели первого заказа
up_filt['first_dow'] = up_filt['first_order_dt'].dt.dayofweek

dow_names = {
    0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт',
    4: 'Пт', 5: 'Сб', 6: 'Вс'
}

up_filt['first_dow_name'] = up_filt['first_dow'].map(dow_names)

# Группируем по дню недели первого заказа
seg_dow = up_filt.groupby('first_dow_name', sort=False).agg(users=('user_id', 'nunique'), returned=('is_two', 'sum'))

# Доли
seg_dow['share_returned'] = (seg_dow['returned'] / seg_dow['users'] * 100).round(2)
seg_dow['share_users'] = (seg_dow['users'] / seg_dow['users'].sum() * 100).round(2)

print(seg_dow)

fig, ax = plt.subplots(1, 2, figsize=(16, 4))

# График 1: сколько пользователей начали с каждого дня недели
ax[0].bar(seg_dow.index, seg_dow['users'], color='tab:blue', alpha=0.8)
ax[0].set_title('Пользователей по дню недели первого заказа')
ax[0].set_ylabel('Пользователей')

# График 2: доля вернувшихся по дням
bars = ax[1].bar(seg_dow.index, seg_dow['share_returned'], color='tab:green', alpha=0.8)

overall = up_filt['is_two'].mean() * 100
ax[1].axhline(overall, color='red', linestyle='--',
              label=f'Среднее: {overall:.1f}%')
ax[1].set_title('Доля вернувшихся по дню недели')
ax[1].set_ylabel('%')
ax[1].legend()

# Подписи значений над столбцами
for b, v in zip(bars, seg_dow['share_returned']):
    ax[1].text(b.get_x() + b.get_width()/2, v + 0.5,
               f'{v:.1f}%', ha='center', fontsize=9)
plt.show()

# Формируем две группы по числу заказов
g_2_4 = up_filt[(up_filt['total_orders'] >= 2) & (up_filt['total_orders'] <= 4)]['avg_days_between']
g_5p  = up_filt[up_filt['total_orders'] >= 5]['avg_days_between']

# Вывод
print(f'2–4 заказа: n={len(g_2_4)}, медиана {g_2_4.median():.1f}, среднее {g_2_4.mean():.1f}')
print(f'5+ заказов: n={len(g_5p)},  медиана {g_5p.median():.1f},  среднее {g_5p.mean():.1f}')

# Квантили
for name, s in [('2–4', g_2_4), ('5+', g_5p)]:
    print(name, '25/50/75/90:', s.quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist())

bins = np.histogram_bin_edges(np.concatenate([g_2_4, g_5p]), bins=40)

# Гистогаммы
plt.figure(figsize=(16, 4))

plt.hist(g_2_4, bins=bins, alpha=0.5, density=True,
         label='2–4 заказа', color='tab:orange')
plt.hist(g_5p,  bins=bins, alpha=0.5, density=True,
         label='5+ заказов', color='tab:green')

plt.xlabel('Средний интервал между заказами, дней')
plt.ylabel('Плотность')
plt.title('Средний интервал между заказами: 2–4 vs 5+')
plt.legend()
plt.show()

# Список признаков для корреляционного анализа
features = [
    'first_event_type',
    'first_device',
    'first_region',
    'first_service',
    'first_dow_name',
    'avg_revenue_rub',
    'avg_tickets_count',
    'avg_days_between',
    'total_orders',
]

interval_cols = ['avg_revenue_rub', 'avg_tickets_count', 'avg_days_between', 'total_orders']
df_corr = up_filt[features].copy()

# Считаем матрицу phi_k
phik_matrix = df_corr.phik_matrix(interval_cols=interval_cols)

# Вывод
print(phik_matrix['total_orders'].sort_values(ascending=False))

# Тепловая карта матрицы
plot_correlation_matrix(
    phik_matrix.values,
    x_labels=phik_matrix.columns,
    y_labels=phik_matrix.index,
    vmin=0, vmax=1,
    figsize=(10, 8),
    title='Признаки профиля и число заказов'
)





