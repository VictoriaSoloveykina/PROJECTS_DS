# === predictor.py ===
import joblib
import pandas as pd
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstatePredictor:
    def __init__(self, model_path='custom_model_pipeline.pkl'):
        """
        Инициализация предсказателя с загрузкой обученного пайплайна
        """
        try:
            self.pipeline = joblib.load(model_path)
            # Получаем правильный порядок признаков из самой модели XGBoost
            self.expected_features = self.pipeline.named_steps['regressor'].get_booster().feature_names
            logger.info(f"✅ Модель успешно загружена. Ожидает {len(self.expected_features)} признаков")
            logger.info(f"📋 Порядок признаков важен!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def _prepare_input_data(self, input_data):
        """Подготавливает входные данные в правильном порядке признаков"""
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
        
        # Создаем DataFrame с правильным порядком колонок
        prepared_data = pd.DataFrame(columns=self.expected_features)
        
        # Заполняем значениями из входных данных
        for feature in self.expected_features:
            if feature in input_data.columns:
                prepared_data[feature] = input_data[feature].values
            else:
                # Для отсутствующих признаков используем значения по умолчанию
                if feature in ['beds', 'baths', 'sqft', 'year_built', 'stories']:
                    prepared_data[feature] = 0  # Числовые по умолчанию
                else:
                    prepared_data[feature] = 0  # Остальные тоже нулями
                logger.warning(f"⚠️ Признак {feature} заполнен значением по умолчанию: 0")
        
        return prepared_data
    
    def predict(self, new_data):
        """
        Предсказание стоимости недвижимости для новых данных
        """
        try:
            # Подготавливаем данные с правильным порядком признаков
            prepared_data = self._prepare_input_data(new_data)
            logger.info(f"📊 Обработка {len(prepared_data)} записей")
            
            # Проверяем порядок признаков
            if list(prepared_data.columns) != self.expected_features:
                logger.error("❌ Порядок признаков не совпадает!")
                return None
            
            # Преобразуем данные через пайплайн
            log_predictions = self.pipeline.predict(prepared_data)
            
            # Преобразуем обратно в доллары
            predictions = np.expm1(log_predictions)
            
            logger.info("✅ Предсказание завершено успешно")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Ошибка при предсказании: {e}")
            raise

# Пример использования с ВСЕМИ признаками в правильном порядке
if __name__ == "__main__":
    # Данные должны содержать ВСЕ признаки в правильном порядке
    sample_data = {
        'status': 1.0,
        'baths': 2.0,
        'sqft': 1500.0,
        'beds': 3.0,
        'state': 1.0,
        'stories': 2.0,
        'has_pool': 1.0,
        'property_type': 1.0,
        'stories_was_missing': 0.0,
        'has_fireplace': 1.0,
        'year_built': 1990.0,
        'avg_school_rating': 8.5,
        'zipcode_density': 1500.0,
        'is_urban': 1.0,
        'is_coastal': 1.0,
        'baths_per_bed': 0.67,
        'sqft_per_room': 300.0,
        'is_luxury': 0.0,
        'is_new_property': 0.0,
        'school_count': 5.0,
        'has_top_school': 1.0,
        'pool_and_fireplace': 0.0,
        'street_type': 1.0,
        'very_old_property': 0.0,
        'zipcode_region': 900.0,
        'is_major_city': 1.0,
        'city_size': 5000.0,
        'is_major_region': 1.0,
        'total_rooms': 5.0
    }
    
    # Инициализация и предсказание
    predictor = RealEstatePredictor()
    prediction = predictor.predict(sample_data)
    print(f"Предсказанная стоимость: ${prediction[0]:,.2f}")