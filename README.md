# 🏠 Real Estate Investment Advisor

An AI-powered real estate analytics and investment decision support system built using Machine Learning and Streamlit.

This project helps users analyze residential properties, predict future prices, estimate ROI, and identify whether a property is a good investment opportunity.

---

# 🚀 Live Features

## 📈 Investment Advisor
- Predicts whether a property is a **Good Investment** or **Risky Investment**
- Confidence score for prediction
- Risk level analysis
- Suggested action (Buy / Negotiate / Avoid)

## 💰 Future Price Predictor
- Predicts estimated property value after **5 years**
- Calculates:
  - Appreciation amount
  - ROI %
  - CAGR %

## 📊 Market Analytics Dashboard
- Total properties
- States covered
- Cities covered
- Average market price
- Top expensive cities
- Price distribution charts

---

# 🧠 Machine Learning Models Used

## Classification Models
- Logistic Regression
- Random Forest Classifier

## Regression Models
- Linear Regression
- Random Forest Regressor

Best performing models were selected based on evaluation metrics.

---

# 📌 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

---

# 📂 Project Structure

```text
RealEstateInvestmentAdvisor/
│── data/
│   ├── raw/
│   └── processed/
│
│── models/
│   ├── investment_classifier.pkl
│   └── price_predictor.pkl
│
│── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_classification.py
│   └── train_regression.py
│
│── streamlit_app/
│   └── app.py
│
│── requirements.txt
│── README.md
