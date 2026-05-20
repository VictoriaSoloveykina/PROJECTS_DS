import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# ─── Глобальные объекты ────────────────────────────────────────────────────
model = None
item_features = None
user_features_dict = None
top_items = None

# Метрики
metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "total_response_time_ms": 0,
}

FEATURE_COLS = [
    "n_views", "n_carts", "n_purchases", "cart_to_view_ratio",
    "n_views_7d", "n_views_30d", "n_carts_7d",
    "categoryid", "depth", "parent_id", "category_size",
    "user_total_views", "user_total_purchases", "user_conversion_rate",
    "ui_n_views", "ui_n_carts", "ui_n_purchases", "ui_max_event",
]

N_CANDIDATES = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели и данных при старте сервиса."""
    global model, item_features, user_features_dict, top_items

    model = joblib.load("models/model.joblib")

    item_features = pd.read_parquet("models/item_features.parquet")
    user_features = pd.read_parquet("models/user_features.parquet")
    user_features_dict = user_features.set_index("visitorid").to_dict("index")

    popularity = pd.read_parquet("models/popularity.parquet")
    available_items = set(
        item_features[item_features["available"] == 1]["itemid"].unique()
    )
    top_items = (
        popularity[popularity["itemid"].isin(available_items)]
        .head(N_CANDIDATES)["itemid"]
        .tolist()
    )

    print(f"Модель загружена. Товаров: {len(item_features):,}. Топ кандидатов: {len(top_items)}")
    yield


app = FastAPI(
    title="Recommendation Service",
    description="Сервис персональных рекомендаций товаров",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Схемы запросов/ответов ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    visitorid: int
    viewed_items: Optional[List[int]] = []
    top_k: Optional[int] = 3

class RecommendResponse(BaseModel):
    visitorid: int
    recommendations: List[int]
    source: str  # "model" или "popularity"


# ─── Эндпоинты ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Получить персональные рекомендации для пользователя.

    - Если viewed_items пустой — возвращает топ популярных товаров (cold start).
    - Если viewed_items передан — модель ранжирует кандидатов и возвращает топ-K.
    """
    start = time.time()
    metrics["total_requests"] += 1

    try:
        top_k = max(1, min(request.top_k or 3, 20))
        viewed = set(request.viewed_items or [])

        # Cold start — нет истории просмотров
        if not viewed:
            _record_time(start)
            return RecommendResponse(
                visitorid=request.visitorid,
                recommendations=top_items[:top_k],
                source="popularity",
            )

        # Кандидаты = топ популярных (фильтруем просмотренные)
        candidates = [i for i in top_items if i not in viewed]

        if not candidates:
            _record_time(start)
            return RecommendResponse(
                visitorid=request.visitorid,
                recommendations=top_items[:top_k],
                source="popularity",
            )

        # Собираем фичи для кандидатов
        cand_df = pd.DataFrame({"itemid": candidates})
        cand_df = cand_df.merge(item_features, on="itemid", how="left")

        # User-level фичи — быстрый поиск по dict
        user_row = user_features_dict.get(request.visitorid)
        if user_row:
            for col in ["user_total_views", "user_total_purchases", "user_conversion_rate"]:
                cand_df[col] = user_row[col]
        else:
            for col in ["user_total_views", "user_total_purchases", "user_conversion_rate"]:
                cand_df[col] = 0

        # User-item фичи — нет истории на уровне пары, заполняем нулями
        for col in ["ui_n_views", "ui_n_carts", "ui_n_purchases", "ui_max_event"]:
            cand_df[col] = 0

        # Заполняем пропуски в item-фичах
        X = cand_df[FEATURE_COLS].fillna(-1)

        # Скоринг
        scores = model.predict_proba(X)[:, 1]
        cand_df["score"] = scores

        # Фильтруем просмотренные из финальных рекомендаций
        top_recs = (
            cand_df[~cand_df["itemid"].isin(viewed)]
            .sort_values("score", ascending=False)
            .head(top_k)["itemid"]
            .tolist()
        )

        _record_time(start)
        return RecommendResponse(
            visitorid=request.visitorid,
            recommendations=top_recs,
            source="model",
        )

    except Exception as e:
        metrics["total_errors"] += 1
        _record_time(start)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Снятие операционных метрик сервиса."""
    total = metrics["total_requests"]
    avg_ms = (
        round(metrics["total_response_time_ms"] / total, 2)
        if total > 0 else 0
    )
    return {
        "total_requests": total,
        "total_errors": metrics["total_errors"],
        "error_rate": round(metrics["total_errors"] / total, 4) if total > 0 else 0,
        "avg_response_time_ms": avg_ms,
    }


# ─── Вспомогательные функции ───────────────────────────────────────────────

def _record_time(start: float):
    elapsed_ms = (time.time() - start) * 1000
    metrics["total_response_time_ms"] += elapsed_ms
