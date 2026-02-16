import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('flat_dataset_2020.csv')

print("Размер датасета:", df.shape)
print("\nПервые строки:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистика:")
print(df.describe())
print("\nПропущенные значения:")
print(df.isnull().sum())

# Проверка наличия выбросов в цене
print("\nРаспределение цен:")
print(f"Минимальная цена: {df['price'].min():,.0f}")
print(f"Максимальная цена: {df['price'].max():,.0f}")
print(f"Средняя цена: {df['price'].mean():,.0f}")
print(f"Медиана цены: {df['price'].median():,.0f}")