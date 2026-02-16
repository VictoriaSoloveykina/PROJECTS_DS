import numpy as np
import pandas as pd
from joblib import load


def get_predictions(city, floor, area, rooms, floors_total):
    """
    Предсказание цены квартиры с помощью RandomForest

    Args:
        city: название города (str)
        floor: этаж квартиры (int)
        area: общая площадь (float)
        rooms: количество комнат (int)
        floors_total: этажей в доме (int)

    Returns:
        Предсказанная цена (int)
    """
    # Загрузка модели и данных
    model = load('model_rf.joblib')
    city_mean_price = np.load('city_price_mapping.npy', allow_pickle=True).item()
    city_stats = pd.read_csv('city_stats.csv')
    feature_columns = np.load('feature_columns.npy', allow_pickle=True).tolist()

    # Функция категоризации города
    def categorize_city(city_name, stats_df):
        city_data = stats_df[stats_df['city'] == city_name]
        if len(city_data) == 0:
            return 'large_expensive'

        mean_price = city_data['mean_price'].values[0]
        count = city_data['count'].values[0]

        if city_name in ['Москва', 'Санкт-Петербург']:
            return 'capital'
        elif count > 5000 or mean_price > 20000000:
            return 'large_expensive'
        elif count > 2000 or mean_price > 15000000:
            return 'large'
        else:
            return 'large'

    # Получаем среднюю цену по городу (дефолт: средняя)
    city_mean_price_val = city_mean_price.get(city, 15343460)

    # Категория города
    city_category = categorize_city(city, city_stats)

    # Создаем признаки
    floor_ratio = floor / floors_total
    is_first_floor = 1 if floor == 1 else 0
    is_last_floor = 1 if floor == floors_total else 0
    is_middle_floor = 1 if (floor > 1 and floor < floors_total) else 0

    area_per_room = area / rooms
    is_studio = 1 if rooms == 1 else 0
    is_large_flat = 1 if rooms >= 4 else 0

    is_low_rise = 1 if floors_total <= 5 else 0
    is_mid_rise = 1 if (floors_total > 5 and floors_total <= 16) else 0
    is_high_rise = 1 if floors_total > 16 else 0

    city_cat_capital = 1 if city_category == 'capital' else 0
    city_cat_large = 1 if city_category == 'large' else 0
    city_cat_large_expensive = 1 if city_category == 'large_expensive' else 0

    # Формируем данные
    data = {
        'city_mean_price': city_mean_price_val,
        'floorNumber': floor,
        'totalArea': area,
        'rooms': rooms,
        'floorsTotal': floors_total,
        'floor_ratio': floor_ratio,
        'is_first_floor': is_first_floor,
        'is_last_floor': is_last_floor,
        'is_middle_floor': is_middle_floor,
        'area_per_room': area_per_room,
        'is_studio': is_studio,
        'is_large_flat': is_large_flat,
        'is_low_rise': is_low_rise,
        'is_mid_rise': is_mid_rise,
        'is_high_rise': is_high_rise,
        'city_cat_large_expensive': city_cat_large_expensive,
        'city_cat_capital': city_cat_capital,
        'city_cat_large': city_cat_large
    }

    # DataFrame с признаками в правильном порядке
    df = pd.DataFrame([data])[feature_columns]

    # Предсказание
    prediction = model.predict(df)[0]

    return int(round(prediction))


def get_available_cities():
    """Возвращает список доступных городов"""
    cities = np.load('cities_list.npy', allow_pickle=True)
    return sorted(cities.tolist())


# Тестирование
if __name__ == "__main__":
    print("Тест предсказаний:")
    print(f"Москва, 5 этаж, 60м², 2 комнаты, 14 этажей: {get_predictions('Москва', 5, 60, 2, 14):,} руб.")
    print(
        f"Санкт-Петербург, 3 этаж, 45м², 1 комната, 9 этажей: {get_predictions('Санкт-Петербург', 3, 45, 1, 9):,} руб.")
    print(f"Краснодар, 7 этаж, 80м², 3 комнаты, 16 этажей: {get_predictions('Краснодар', 7, 80, 3, 16):,} руб.")
    print(f"\nДоступные города: {get_available_cities()}")