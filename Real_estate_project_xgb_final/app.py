# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                                
from flask import Flask, request, jsonify
from predictor import RealEstatePredictor
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
predictor = None

def create_app():
    global predictor
    predictor = RealEstatePredictor('custom_model_pipeline.pkl')
    return app

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Real estate predictor is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        predictions = predictor.predict(data)
        result = {
            "predicted_price": float(predictions[0]),
            "formatted_price": "${:,.2f}".format(predictions[0]),
            "currency": "USD"
        }
        return jsonify(result)
        
    except Exception as e:
        logger.error("Prediction error: {}".format(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)