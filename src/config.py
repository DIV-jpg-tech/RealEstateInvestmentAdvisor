import os

#Base os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Data Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "india_housing_prices.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

#Model Paths
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "investment_classifier.pkl")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "price_predictor.pkl")

#Random Seed
RANDOM_STATE = 42
