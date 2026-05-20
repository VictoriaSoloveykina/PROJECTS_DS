# Документация: Рекомендательная система для интернет-магазина

Учебный проект по Data Science. Рекомендательная система на датасете RetailRocket — предсказывает какие товары пользователь с наибольшей вероятностью купит при следующем визите.

---

## Быстрый старт

### 1. Требования к окружению

- Python 3.10+
- RAM: минимум 8 GB (датасет item_properties — 20 млн строк)
- Google Colab или Jupyter Notebook
- Docker (для запуска сервиса)

### 2. Расположение данных

CSV файлы должны лежать в **той же папке** что и ноутбук:

```
project/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
├── category_tree.csv
└── recsys_notebook.ipynb
```

### 3. Порядок запуска

Весь пайплайн реализован в одном ноутбуке `recsys_notebook.ipynb`. Запустить все ячейки по порядку.

CSV файлы должны лежать в **той же папке** что и ноутбук. Пути уже прописаны:
```python
file_path = ""        # CSV в той же папке
save_path = "processed/"  # промежуточные parquet
```

Ноутбук последовательно выполняет:
1. EDA — исследование данных
2. Temporal split на train / val / test
3. Feature Engineering — признаки товаров, пользователей, пар user-item
4. Candidate Generation — popularity + co-occurrence CF
5. Обучение и сравнение моделей (ALS, LightGBM v1-v5, XGBoost)
6. Оценку Precision@3 на val выборке
7. **Сохранение артефактов** в папку `models/` (раздел 8 ноутбука)

После выполнения в `processed/` появятся промежуточные файлы, в `models/` — артефакты для сервиса:
```
processed/
  train.parquet, val.parquet, test.parquet
  item_features.parquet, user_features.parquet
  popularity.parquet, item_similar.parquet, candidates.parquet

models/
  model.joblib             — обученная модель LightGBM v5
  item_features.parquet    — признаки товаров
  user_features.parquet    — признаки пользователей
  popularity.parquet       — топ популярных товаров
```

**Шаг 1 — Запустить ноутбук:**

Открыть `recsys_notebook.ipynb` и запустить все ячейки по порядку. После выполнения папки `processed/` и `models/` появятся автоматически.

**Шаг 2 — Запустить сервис:**

```bash
# Собрать и запустить Docker-образ
docker build -t recsys-service .
docker run -d -p 8000:8000 --name recsys recsys-service

# Проверить
curl http://localhost:8000/health
```

### 4. Структура проекта

```
project/
├── events.csv                      # Сырые данные
├── item_properties_part1.csv
├── item_properties_part2.csv
├── category_tree.csv
├── recsys_notebook.ipynb           # Единый ноутбук: EDA → признаки → модели → артефакты
├── processed/                      # Создаётся ноутбуком автоматически
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   ├── item_features.parquet
│   ├── user_features.parquet
│   ├── popularity.parquet
│   ├── item_similar.parquet
│   └── candidates.parquet
├── models/                         # Создаётся ноутбуком (раздел 8)
│   ├── model.joblib
│   ├── item_features.parquet
│   ├── user_features.parquet
│   └── popularity.parquet
├── app.py                          # FastAPI сервис
├── Dockerfile
└── requirements.txt
```

---

## 1. Формат входных данных для обучения

### Исходные файлы

| Файл | Строк | Описание |
|------|-------|----------|
| `events.csv` | 2 755 641 | События пользователей |
| `item_properties_part1.csv` + `item_properties_part2.csv` | 20 275 902 | Свойства товаров |
| `category_tree.csv` | 1 669 | Иерархия категорий |

### Структура `events.csv`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `timestamp` | int64 | Время события в миллисекундах |
| `visitorid` | int64 | Анонимный ID пользователя |
| `event` | string | Тип события: `view` / `addtocart` / `transaction` |
| `itemid` | int64 | ID товара |
| `transactionid` | float64 | ID транзакции (NaN для не-покупок) |

### Структура `item_properties`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `timestamp` | int64 | Время записи свойства |
| `itemid` | int64 | ID товара |
| `property` | string | Название свойства (`categoryid`, `available`, остальные захешированы) |
| `value` | string | Значение свойства |

### Структура `category_tree.csv`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `categoryid` | int64 | ID категории |
| `parentid` | float64 | ID родительской категории (NaN = корень) |

---

## 2. Трансформации датасета

### 2.1 Предобработка events.csv

```python
events = pd.read_csv("events.csv")

# Удаление дубликатов (459 точных копий по всем полям включая timestamp)
events = events.drop_duplicates()

# Конвертация timestamp из миллисекунд в datetime
events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
```

### 2.2 Temporal split

Данные разбиты по времени на три выборки. Границы выставлены по **понедельникам** — чтобы не разрывать недельный цикл сезонности. Первая неполная неделя (до 2015-05-03, всего 13К событий) отрезана.

Все признаки считаются **только по train** — это исключает data leakage.

| Выборка | Период | События | Назначение |
|---------|--------|---------|------------|
| Train | 2015-05-04 — 2015-08-16 | ~1.8 млн | Обучение модели и подсчёт признаков |
| Val | 2015-08-17 — 2015-08-30 | ~290К | Отбор модели и подбор гиперпараметров |
| Test | 2015-08-31 — 2015-09-18 | ~640К | Финальная оценка |

```python
train = events[(events["datetime"] >= "2015-05-04") & (events["datetime"] <= "2015-08-16")]
val   = events[(events["datetime"] >= "2015-08-17") & (events["datetime"] <= "2015-08-30")]
test  = events[(events["datetime"] >= "2015-08-31") & (events["datetime"] <= "2015-09-18")]
```

### 2.3 Подготовка item_properties

Свойства товаров берутся **только до конца train** — чтобы не было утечки из будущего:

```python
train_end_ts = pd.Timestamp("2015-08-16").value // 10**6  # в миллисекундах
props_train = properties[properties["timestamp"] <= train_end_ts]
```

Для каждого товара берётся **последнее значение** категории и статуса наличия:

```python
# Категория товара
item_category = (
    props_train[props_train["property"] == "categoryid"]
    .sort_values("timestamp")
    .groupby("itemid")["value"]
    .last()
    .reset_index()
)
item_category["categoryid"] = item_category["categoryid"].astype(int)

# Статус наличия
item_available = (
    props_train[props_train["property"] == "available"]
    .sort_values("timestamp")
    .groupby("itemid")["value"]
    .last()
    .reset_index()
)
item_available["available"] = item_available["available"].astype(int)
```

### 2.4 Признаки товаров (`item_features`)

Считаются по событиям из **train**:

```python
# Popularity-факторы
item_views     = train[train["event"] == "view"].groupby("itemid").size().reset_index(name="n_views")
item_carts     = train[train["event"] == "addtocart"].groupby("itemid").size().reset_index(name="n_carts")
item_purchases = train[train["event"] == "transaction"].groupby("itemid").size().reset_index(name="n_purchases")

item_features = item_views.merge(item_carts, on="itemid", how="left") \
                           .merge(item_purchases, on="itemid", how="left")
item_features["cart_to_view_ratio"] = item_features["n_carts"] / item_features["n_views"]

# Time-decay — популярность за последние 7 и 30 дней train
last_7d  = pd.Timestamp("2015-08-16") - pd.Timedelta(days=7)
last_30d = pd.Timestamp("2015-08-16") - pd.Timedelta(days=30)

item_views_7d  = train[(train["event"] == "view") & (train["datetime"] >= last_7d)] \
                     .groupby("itemid").size().reset_index(name="n_views_7d")
item_views_30d = train[(train["event"] == "view") & (train["datetime"] >= last_30d)] \
                     .groupby("itemid").size().reset_index(name="n_views_30d")
item_carts_7d  = train[(train["event"] == "addtocart") & (train["datetime"] >= last_7d)] \
                     .groupby("itemid").size().reset_index(name="n_carts_7d")

# Итоговая таблица: merge всех источников
item_features = item_features \
    .merge(item_category,  on="itemid", how="left") \
    .merge(item_available, on="itemid", how="left") \
    .merge(category[["categoryid", "depth", "parent_id"]], on="categoryid", how="left") \
    .merge(category_size,  on="categoryid", how="left") \
    .merge(item_views_7d,  on="itemid", how="left") \
    .merge(item_views_30d, on="itemid", how="left") \
    .merge(item_carts_7d,  on="itemid", how="left")
```

Итог: **210 510 товаров**, 13 колонок. Пропуски в категориальных признаках (~21% товаров без категории) заполняются `-1`.

| Признак | Источник | Описание |
|---------|----------|----------|
| `n_views` | events (train) | Количество просмотров |
| `n_carts` | events (train) | Количество добавлений в корзину |
| `n_purchases` | events (train) | Количество покупок |
| `cart_to_view_ratio` | вычисляемый | n_carts / n_views |
| `n_views_7d` | events (train, последние 7 дней) | Просмотры за последние 7 дней |
| `n_views_30d` | events (train, последние 30 дней) | Просмотры за последние 30 дней |
| `n_carts_7d` | events (train, последние 7 дней) | Корзины за последние 7 дней |
| `categoryid` | item_properties (до конца train) | Категория товара |
| `available` | item_properties (до конца train) | Статус наличия (0/1) |
| `depth` | category_tree | Глубина категории в дереве |
| `parent_id` | category_tree | Родительская категория |
| `category_size` | item_properties | Количество товаров в категории |

### 2.5 Признаки пользователей (`user_features`)

```python
user_views     = train[train["event"] == "view"].groupby("visitorid").size().reset_index(name="user_total_views")
user_purchases = train[train["event"] == "transaction"].groupby("visitorid").size().reset_index(name="user_total_purchases")

user_features = user_views.merge(user_purchases, on="visitorid", how="left")
user_features["user_total_purchases"] = user_features["user_total_purchases"].fillna(0).astype(int)
user_features["user_conversion_rate"] = user_features["user_total_purchases"] / user_features["user_total_views"]

# Любимая категория — категория с наибольшим числом просмотров
train_with_cat = train[train["event"] == "view"].merge(item_category, on="itemid", how="left")
user_fav_category = (
    train_with_cat.dropna(subset=["categoryid"])
    .groupby(["visitorid", "categoryid"]).size()
    .reset_index(name="cat_views")
    .sort_values("cat_views", ascending=False)
    .groupby("visitorid").first()
    .reset_index()[["visitorid", "categoryid"]]
    .rename(columns={"categoryid": "user_fav_category"})
)
user_features = user_features.merge(user_fav_category, on="visitorid", how="left")
```

### 2.6 Признаки пары пользователь–товар (`user_item_features`)

```python
event_rank = {"view": 1, "addtocart": 2, "transaction": 3}
train["event_rank"] = train["event"].map(event_rank)

user_item_views     = train[train["event"] == "view"].groupby(["visitorid", "itemid"]).size().reset_index(name="ui_n_views")
user_item_carts     = train[train["event"] == "addtocart"].groupby(["visitorid", "itemid"]).size().reset_index(name="ui_n_carts")
user_item_purchases = train[train["event"] == "transaction"].groupby(["visitorid", "itemid"]).size().reset_index(name="ui_n_purchases")
user_item_max_event = train.groupby(["visitorid", "itemid"])["event_rank"].max().reset_index(name="ui_max_event")

user_item_features = user_item_views \
    .merge(user_item_carts,     on=["visitorid", "itemid"], how="left") \
    .merge(user_item_purchases, on=["visitorid", "itemid"], how="left") \
    .merge(user_item_max_event, on=["visitorid", "itemid"], how="left")
```

### 2.7 Генерация кандидатов

**Popularity baseline** — топ-100 товаров по числу покупок, только `available=1`:

```python
popularity = train.groupby("itemid").agg(
    n_purchases=("event", lambda x: (x == "transaction").sum()),
    n_views=("event", lambda x: (x == "view").sum())
).reset_index().sort_values(["n_purchases", "n_views"], ascending=False)

available_items = set(item_features[item_features["available"] == 1]["itemid"])
top_items = popularity[popularity["itemid"].isin(available_items)].head(100)["itemid"].tolist()
```

**Item-based CF (co-occurrence)** — товары которые часто смотрят вместе в одной сессии. Сессия — события пользователя с перерывом не более 30 минут:

```python
# Определение сессий
train_sorted = train.sort_values(["visitorid", "timestamp"])
train_sorted["time_diff"] = train_sorted.groupby("visitorid")["timestamp"].diff()
train_sorted["new_session"] = (train_sorted["time_diff"] > 30 * 60 * 1000) | train_sorted["time_diff"].isna()
train_sorted["session_id"] = train_sorted["visitorid"].astype(str) + "_" + \
                              train_sorted.groupby("visitorid")["new_session"].cumsum().astype(str)

# Co-occurrence пар товаров
from itertools import combinations
from collections import defaultdict

cooccurrence = defaultdict(int)
session_items = train_sorted[train_sorted["event"] == "view"] \
    .groupby("session_id")["itemid"].apply(list)

for items in session_items:
    items = items[:10]  # ограничение длины сессии
    for a, b in combinations(set(items), 2):
        cooccurrence[(a, b)] += 1
        cooccurrence[(b, a)] += 1

# Топ-20 похожих товаров для каждого
item_similar = pd.DataFrame(
    [{"item_a": k[0], "item_b": k[1], "cooc_count": v} for k, v in cooccurrence.items()]
).sort_values("cooc_count", ascending=False).groupby("item_a").head(20)
```

### 2.8 Negative sampling и сборка обучающей выборки

Дисбаланс классов в кандидатах — 1:85 000. Применяется negative sampling 1:200:

```python
N_NEGATIVE = 200  # соотношение негативных к позитивным

pos_samples = candidates[candidates["purchased"] == 1]
neg_samples = candidates[candidates["purchased"] == 0].sample(
    n=len(pos_samples) * N_NEGATIVE, random_state=42
)
train_samples = pd.concat([pos_samples, neg_samples])

# Присоединение признаков
train_samples = train_samples \
    .merge(item_features,      on="itemid",                how="left") \
    .merge(user_features,      on="visitorid",             how="left") \
    .merge(user_item_features, on=["visitorid", "itemid"], how="left")

# Заполнение пропусков
train_samples[["user_total_views", "user_total_purchases", "user_conversion_rate"]] = \
    train_samples[["user_total_views", "user_total_purchases", "user_conversion_rate"]].fillna(0)
train_samples[["ui_n_views", "ui_n_carts", "ui_n_purchases", "ui_max_event"]] = \
    train_samples[["ui_n_views", "ui_n_carts", "ui_n_purchases", "ui_max_event"]].fillna(0)
train_samples[["categoryid", "depth", "parent_id", "category_size", "user_fav_category"]] = \
    train_samples[["categoryid", "depth", "parent_id", "category_size", "user_fav_category"]].fillna(-1)
```

---

## 3. Построение валидации

### Схема

Валидация построена на **val** выборке (2015-08-17 — 2015-08-30).

1. Кандидаты генерируются для всех пользователей из val.
2. Таргет — факт покупки товара пользователем в val (`event == 'transaction'`).
3. Модель ранжирует кандидатов и выдаёт топ-3.
4. Считается **Precision@3**.

### Метрика Precision@K

$$\text{Precision@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|\text{рекомендации}_u \cap \text{покупки}_u|}{K}$$

Где $U$ — множество пользователей с покупками в val, $K=3$.

### Важные свойства валидации

- **Temporal split:** val идёт строго после train — нет утечки данных из будущего.
- **100% покрытие:** все пользователи с покупками в val покрыты кандидатами (popularity baseline гарантирует это).
- **Cold start учтён:** 84% покупателей в val — новые пользователи без истории в train. Для них используется popularity fallback.

---

## 4. Эксперименты

| # | Модель | Negative sampling | n_estimators | num_leaves | Доп. параметры | Precision@3 |
|---|--------|-------------------|--------------|------------|-----------------|-------------|
| 1 | ALS (implicit) | — | — | — | factors=50, iterations=20, reg=0.1 | 0.0049 |
| 2 | LightGBM v1 | 1:200 | 100 | 31 | item фичи, scale_pos_weight | 0.0141 |
| 3 | LightGBM v2 | 1:200 | 100 | 31 | убраны has_prop_* и available | 0.0144 |
| 4 | LightGBM v3 | 1:200 | 300 | 63 | добавлены user фичи | 0.0185 |
| 5 | LightGBM v4 | 1:500 | 300 | 63 | добавлены user + user-item фичи | ~0.019 |
| 6 | **LightGBM v5** | **1:200** | **300** | **63** | **все фичи (18 признаков)** | **0.0202** |
| 7 | XGBoost | 1:200 | 300 | — | max_depth=6, subsample=0.8 | 0.0179 |

### Итоговые гиперпараметры лучшей модели (LightGBM v5)

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    scale_pos_weight=scale,  # вычисляется как (кол-во негативных) / (кол-во позитивных)
    random_state=42,
    n_jobs=-1,
)
```

### Набор признаков лучшей модели

```python
feature_cols = [
    # Item
    "n_views", "n_carts", "n_purchases", "cart_to_view_ratio",
    "n_views_7d", "n_views_30d", "n_carts_7d",
    "categoryid", "depth", "parent_id", "category_size",
    # User
    "user_total_views", "user_total_purchases", "user_conversion_rate",
    # User-Item
    "ui_n_views", "ui_n_carts", "ui_n_purchases", "ui_max_event",
]
```

### Выводы по экспериментам

- ALS слабый из-за разреженности матрицы (0.0007%) и большой доли cold start пользователей.
- Наиболее важные признаки: `n_purchases`, `n_views`, `categoryid` — popularity и категория товара.
- Добавление user и user-item признаков даёт прирост, но небольшой — у большинства пользователей мало истории.
- Признаки `has_prop_*` (захешированные свойства) убраны — importance ≈ 0.

---

## 5. Docker-сервис

### Пайплайн: от сырых данных до сервиса

Признаки товаров и пользователей — это **offline-признаки**: агрегаты по миллионам событий, которые нельзя пересчитывать при каждом запросе. Поэтому они считаются один раз в ноутбуке и сохраняются в parquet. Сервис загружает их при старте и использует для скоринга.

Полная цепочка:

```
Сырые данные (CSV)
  events.csv, item_properties_part1/2.csv, category_tree.csv
       │
       ▼
recsys_notebook.ipynb
  - EDA, очистка данных
  - temporal split (train / val / test)
  - подсчёт признаков только по train
  - генерация кандидатов (popularity + co-occurrence CF)
  - обучение моделей (ALS, LightGBM v1-v5, XGBoost)
  - оценка Precision@3 на val
  - раздел 8: сохранение артефактов в models/
       │
       ▼
models/
  ├── model.joblib
  ├── item_features.parquet
  ├── user_features.parquet
  └── popularity.parquet
       │
       ▼
Docker-сервис
  - загружает артефакты при старте
  - принимает запросы /recommend
```

Папка `models/` создаётся автоматически при выполнении раздела 8 ноутбука.

### Структура сервиса

```
project/
├── app.py                  # FastAPI приложение
├── Dockerfile
├── requirements.txt
└── models/                 # Артефакты из ноутбука
    ├── model.joblib
    ├── item_features.parquet
    ├── user_features.parquet
    └── popularity.parquet
```

### Сборка и запуск

```bash
# Собрать образ
docker build -t recsys-service .

# Запустить контейнер
docker run -d -p 8000:8000 --name recsys recsys-service

# Проверить что сервис поднялся
curl http://localhost:8000/health
```

### Экспорт и загрузка образа

```bash
# Экспортировать образ в файл
docker save recsys-service -o recsys-service.tar

# Загрузить образ из файла
docker load -i recsys-service.tar

# Запустить после загрузки
docker run -d -p 8000:8000 --name recsys recsys-service
```

### Устройство сервиса

При старте контейнера сервис загружает модель и данные в память. Все последующие запросы обрабатываются без обращения к диску.

| Компонент | Описание |
|-----------|----------|
| FastAPI | Веб-фреймворк, обработка запросов |
| Uvicorn | ASGI-сервер |
| LightGBM | Модель ранжирования |
| joblib | Загрузка сохранённой модели |
| pandas | Работа с признаками |

---

## 6. API сервиса

Base URL: `http://localhost:8000`

---

### `GET /health`

Проверка работоспособности сервиса.

**Запрос:**
```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "ok"
}
```

---

### `POST /recommend`

Получить персональные рекомендации для пользователя.

**Тело запроса:**

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `visitorid` | int | ✓ | ID пользователя |
| `viewed_items` | list[int] | — | Список просмотренных товаров (для фильтрации из рекомендаций) |
| `top_k` | int | — | Количество рекомендаций (по умолчанию 3, макс. 20) |

**Пример запроса — warm user:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "visitorid": 12345,
    "viewed_items": [111, 222, 333],
    "top_k": 3
  }'
```

**Пример запроса — cold start (нет истории):**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "visitorid": 99999,
    "viewed_items": [],
    "top_k": 3
  }'
```

**Ответ:**

| Поле | Тип | Описание |
|------|-----|----------|
| `visitorid` | int | ID пользователя из запроса |
| `recommendations` | list[int] | Список ID рекомендованных товаров |
| `source` | string | `"model"` — LightGBM, `"popularity"` — cold start fallback |

```json
{
  "visitorid": 12345,
  "recommendations": [456, 789, 101],
  "source": "model"
}
```

**Коды ответа:**

| Код | Описание |
|-----|----------|
| 200 | Успешный ответ |
| 422 | Ошибка валидации входных данных |
| 500 | Внутренняя ошибка сервиса |

**Пример некорректного запроса:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"visitorid": "not_a_number"}'
```
```json
{
  "detail": [
    {
      "type": "int_parsing_error",
      "loc": ["body", "visitorid"],
      "msg": "Input should be a valid integer"
    }
  ]
}
```

---

### `GET /metrics`

Операционные метрики сервиса.

**Запрос:**
```bash
curl http://localhost:8000/metrics
```

**Ответ:**

| Поле | Тип | Описание |
|------|-----|----------|
| `total_requests` | int | Общее число запросов к `/recommend` |
| `total_errors` | int | Число завершившихся ошибкой |
| `error_rate` | float | Доля ошибок |
| `avg_response_time_ms` | float | Среднее время ответа в миллисекундах |

```json
{
  "total_requests": 142,
  "total_errors": 1,
  "error_rate": 0.007,
  "avg_response_time_ms": 23.4
}
```
