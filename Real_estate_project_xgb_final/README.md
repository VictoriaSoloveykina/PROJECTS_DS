#  Модель предсказания цены на недвижимость

ML-сервис для предсказания стоимости недвижимости с точностью **74.8% (R²)**.

##  Быстрый старт

###  Шаг 1: Скачайте данные
Перед запуском скачайте файл с данными:
- **data.csv** (~300 МБ): [ Скачать с Google Drive](https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view?usp=share_link)

Поместите скачанный файл `data.csv` в корневую папку проекта.

### Шаг 2: Запустите сервис
```bash
# 1. Активируйте виртуальное окружение
source venv/bin/activate

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Запустите сервер
python app.py

# 4. Сделайте тестовый запрос
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"status":1.0,"baths":2.0,"sqft":1500.0,"beds":3.0,"garage":1.0}'

