import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------
# BASE PATH
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
CLF_MODEL_PATH = BASE_DIR / "models" / "investment_classifier.pkl"
REG_MODEL_PATH = BASE_DIR / "models" / "price_predictor.pkl"

# ------------------------------------------------
# LOAD DATA + MODELS
# ------------------------------------------------
df = pd.read_csv(DATA_PATH)

clf_model = joblib.load(CLF_MODEL_PATH)
reg_model = joblib.load(REG_MODEL_PATH)

# ------------------------------------------------
# PREMIUM CSS
# ------------------------------------------------
st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */
.stApp {
    background: linear-gradient(135deg,#1e293b,#334155,#475569);
    color: white !important;
}

/* ---------- GLOBAL TEXT ---------- */
html, body, p, span, div, label, small, h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.96);
    backdrop-filter: blur(14px);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ---------- METRIC CARDS ---------- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

[data-testid="metric-container"] * {
    color: white !important;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    font-weight: bold;
}

/* ---------- INPUTS ---------- */
input, textarea {
    color: white !important;
    background-color: rgba(255,255,255,0.08) !important;
}

/* ---------- SELECTBOX CLOSED ---------- */
[data-baseweb="select"] > div {
    background-color: #334155 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* ---------- DROPDOWN MENU OPEN ---------- */
ul[role="listbox"] {
    background: black !important;
}

/* ---------- DROPDOWN OPTIONS ---------- */
ul[role="listbox"] li {
    color: black !important;
    background: black !important;
}

/* ---------- HOVER OPTION ---------- */
ul[role="listbox"] li:hover {
    background: #e5e7eb !important;
    color: black !important;
}

/* ---------- NUMBER INPUT ---------- */
[data-testid="stNumberInput"] input {
    color: white !important;
}

/* ---------- SLIDER ---------- */
[data-testid="stSlider"] * {
    color: white !important;
}

/* ---------- TABS ---------- */
button[data-baseweb="tab"] {
    color: white !important;
}

/* ---------- ALERTS ---------- */
.stAlert * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("🏠 Real Estate AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📈 Investment Advisor",
        "💰 Future Price Predictor",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------
if menu == "🏠 Dashboard":

    st.title("🏠 Real Estate Investment Advisor")
    st.caption("AI Powered Property Decision Intelligence")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Properties", f"{len(df):,}")
    c2.metric("Cities", df["City"].nunique())
    c3.metric("States", df["State"].nunique())
    c4.metric("Avg Price", f"₹ {df['Price_in_Lakhs'].mean():.1f} L")

    st.markdown("---")

    fig = px.histogram(
        df,
        x="Price_in_Lakhs",
        nbins=40,
        title="Property Price Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------
# INVESTMENT ADVISOR
# ------------------------------------------------
elif menu == "📈 Investment Advisor":

    st.title("📈 AI Investment Advisor")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", sorted(df["State"].unique()))
        city = st.selectbox("City", sorted(df["City"].unique()))
        bhk = st.slider("BHK", 1, 6, 2)
        sqft = st.number_input("Size in SqFt", 500, 5000, 1200)
        price = st.number_input("Current Price (Lakhs)", 10, 1000, 100)

    with col2:
        schools = st.slider("Nearby Schools", 1, 10, 5)
        hospitals = st.slider("Nearby Hospitals", 1, 10, 5)
        age = st.slider("Age of Property", 0, 30, 5)
        furnished = st.selectbox(
            "Furnished Status",
            sorted(df["Furnished_Status"].unique())
        )

    if st.button("Analyze Property"):

        sample = df.iloc[[0]].copy()

        sample["State"] = state
        sample["City"] = city
        sample["BHK"] = bhk
        sample["Size_in_SqFt"] = sqft
        sample["Price_in_Lakhs"] = price
        sample["Nearby_Schools"] = schools
        sample["Nearby_Hospitals"] = hospitals
        sample["Age_of_Property"] = age
        sample["Furnished_Status"] = furnished

        pred = clf_model.predict(sample)[0]
        prob = clf_model.predict_proba(sample)[0][1]

        if pred == 1:
            verdict = "🟢 Good Investment"
            action = "BUY / SHORTLIST"
            risk = "Low Risk"
        else:
            verdict = "🔴 Risky Investment"
            action = "NEGOTIATE / AVOID"
            risk = "Higher Risk"

        st.success(verdict)

        a, b, c = st.columns(3)
        a.metric("Confidence", f"{prob*100:.1f}%")
        b.metric("Risk Level", risk)
        c.metric("Suggested Action", action)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Investment Confidence"},
            gauge={'axis': {'range': [0,100]}}
        ))

        gauge.update_layout(template="plotly_dark")
        st.plotly_chart(gauge, width="stretch")

# ------------------------------------------------
# FUTURE PRICE PREDICTOR
# ------------------------------------------------
elif menu == "💰 Future Price Predictor":

    st.title("💰 AI Future Price Predictor")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State ", sorted(df["State"].unique()))
        city = st.selectbox("City ", sorted(df["City"].unique()))
        bhk = st.slider("BHK ", 1, 6, 2)
        sqft = st.number_input("Size in SqFt ", 500, 5000, 1200)
        current_price = st.number_input("Current Price ", 10, 1000, 100)

    with col2:
        schools = st.slider("Nearby Schools ", 1, 10, 5)
        hospitals = st.slider("Nearby Hospitals ", 1, 10, 5)
        age = st.slider("Age of Property ", 0, 30, 5)
        furnished = st.selectbox(
            "Furnished Status ",
            sorted(df["Furnished_Status"].unique())
        )

    if st.button("Predict Future Value"):

        sample = df.iloc[[0]].copy()

        sample["State"] = state
        sample["City"] = city
        sample["BHK"] = bhk
        sample["Size_in_SqFt"] = sqft
        sample["Price_in_Lakhs"] = current_price
        sample["Nearby_Schools"] = schools
        sample["Nearby_Hospitals"] = hospitals
        sample["Age_of_Property"] = age
        sample["Furnished_Status"] = furnished

        future_price = reg_model.predict(sample)[0]

        appreciation = future_price - current_price
        roi = (appreciation / current_price) * 100
        cagr = ((future_price / current_price) ** (1/5) - 1) * 100

        if roi >= 40:
            advice = "🟢 Strong Buy Opportunity"
        elif roi >= 20:
            advice = "🟡 Moderate Growth Potential"
        else:
            advice = "🔴 Low Growth Potential"

        st.success(f"Predicted Price After 5 Years: ₹ {future_price:,.2f} Lakhs")

        c1, c2, c3 = st.columns(3)
        c1.metric("Appreciation", f"₹ {appreciation:,.2f} L")
        c2.metric("ROI", f"{roi:.2f}%")
        c3.metric("CAGR", f"{cagr:.2f}%")

        st.info(advice)

# ------------------------------------------------
# ANALYTICS
# ------------------------------------------------
elif menu == "📊 Analytics":

    st.title("📊 Market Analytics")

    city_avg = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig = px.bar(
        x=city_avg.values,
        y=city_avg.index,
        orientation="h",
        title="Top Expensive Cities",
        template="plotly_dark"
    )

    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------
# ABOUT
# ------------------------------------------------
else:

    st.title("ℹ️ About This Project")

    st.markdown("""
### 🚀 Real Business Features

✔ Good Investment Detection  
✔ Future Price Prediction  
✔ ROI Analysis  
✔ CAGR Calculation  
✔ Buy / Avoid Recommendation  
✔ Premium Interactive Dashboard  

### 🛠 Built With

Python • Streamlit • Plotly • Scikit-learn
""")
