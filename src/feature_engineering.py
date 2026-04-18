import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)

    # Use sample for faster development
    df = df.sample(n=50000, random_state=42)

    return df


def create_features(df):

    # =====================================================
    # FUTURE PRICE TARGET (Proper Multi-Factor Logic)
    # =====================================================

    appreciation = 1.20

    # Infrastructure impact
    appreciation += df["Nearby_Schools"] * 0.01
    appreciation += df["Nearby_Hospitals"] * 0.008

    # Property size / demand impact
    appreciation += df["BHK"] * 0.015
    appreciation += (df["Size_in_SqFt"] / 1000) * 0.02

    # Older properties appreciate less
    appreciation -= df["Age_of_Property"] * 0.002

    # Premium facilities boost value
    appreciation += (df["Parking_Space"] == "Yes").astype(int) * 0.03
    appreciation += (df["Security"] == "Yes").astype(int) * 0.025
    appreciation += (df["Amenities"] == "Yes").astype(int) * 0.04

    # Final Future Price after 5 years
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * appreciation

    # =====================================================
    # CLASSIFICATION TARGET (Good Investment)
    # =====================================================

    median_price_sqft = df["Price_per_SqFt"].median()

    df["Good_Investment"] = (
        (df["Price_per_SqFt"] <= median_price_sqft) &
        (df["Nearby_Schools"] >= 5) &
        (df["Nearby_Hospitals"] >= 5) &
        (df["Age_of_Property"] <= 10)
    ).astype(int)

    # =====================================================
    # EXTRA FEATURES
    # =====================================================

    # Luxury Property
    df["Luxury_Property"] = (df["Price_in_Lakhs"] > 150).astype(int)

    # Spacious Property
    df["Spacious_Property"] = (df["Size_in_SqFt"] > 1800).astype(int)

    return df


def save_data(df):
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Processed file saved successfully.")


if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    save_data(df)

    print(df.head())