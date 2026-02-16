from app import app, db
from app.models import Flat
import pandas as pd

# Читаем CSV
df = pd.read_csv('flat_dataset_2020.csv')

# Предполагаем, что в CSV есть колонки: city, rooms, totalArea, floorNumber, floorsTotal, price
# Подстройте названия колонок под ваш файл!

with app.app_context():
    for index, row in df.head(20).iterrows():  # загружаем первые 20 записей для примера
        flat = Flat(
            title=f"Квартира {row['rooms']}-комн в {row['city']}",
            city=row['city'],
            rooms=int(row['rooms']),
            area=str(row['totalArea']),
            floor=int(row['floorNumber']),
            floors_total=int(row['floorsTotal']),
            cost=int(row['price'])
        )
        db.session.add(flat)

    db.session.commit()
    print("Данные загружены!")