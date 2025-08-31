from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_beds_by_property = None
        self.median_sqft_by_property_beds = None
        self.global_median_sqft = None
        self.median_baths = None
        self.frequency_mappings_ = {}
    
    def fit(self, X, y=None):
        # Вычисляем статистики на исходных данных
        self.median_beds_by_property = X.groupby('property_type')['beds'].median()
        self.median_sqft_by_property_beds = X.groupby(['property_type', 'beds'])['sqft'].median()
        self.global_median_sqft = X['sqft'].median()
        self.median_baths = X['baths'].median()
        
        # Frequency encoding для категориальных признаков
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.frequency_mappings_[col] = X[col].value_counts().to_dict()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Заполнение пропусков в beds
        for prop_type, median_val in self.median_beds_by_property.items():
            mask = (X['property_type'] == prop_type) & (X['beds'].isna())
            X.loc[mask, 'beds'] = median_val
        
        # Заполнение пропусков в sqft
        for (prop_type, beds), median_val in self.median_sqft_by_property_beds.items():
            mask = (X['property_type'] == prop_type) & (X['beds'] == beds) & (X['sqft'].isna())
            X.loc[mask, 'sqft'] = median_val
        
        X['sqft'] = X['sqft'].fillna(self.global_median_sqft)
        X['baths'] = X['baths'].fillna(self.median_baths)
        
        # Ваша функция округления ванных
        def round_to_standard_bath(value):
            if pd.isna(value): return np.nan
            rounded = round(value * 2) / 2
            return rounded if rounded >= 1 else 1.0
        
        X['baths'] = X['baths'].apply(round_to_standard_bath)
        
        # Frequency encoding
        for col, mapping in self.frequency_mappings_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                avg_freq = np.mean(list(mapping.values()))
                X[col] = X[col].fillna(avg_freq)
        
        # Создаем производные признаки
        X['total_rooms'] = X['beds'] + X['baths']
        X['baths_per_bed'] = X['baths'] / X['beds'].replace(0, 1)
        X['sqft_per_room'] = X['sqft'] / X['total_rooms'].replace(0, 1)
        
        return X