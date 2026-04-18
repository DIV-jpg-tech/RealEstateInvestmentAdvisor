import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from config import PROCESSED_DATA_PATH, REGRESSION_MODEL_PATH, RANDOM_STATE

# Load Data
df = pd.read_csv(PROCESSED_DATA_PATH)

#Features/Target

drop_cols = ["ID", "Good_Investment", "Future_Price_5Y", "Price_in_Lakhs"]
X = df.drop(columns=drop_cols)

y = df["Future_Price_5Y"]

# Column Types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

#Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    
    ]
)

# Models
models= {
    "Linear Regression": LinearRegression(),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=RANDOM_STATE
)

best_model = None
best_r2 = -999

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print("MAE:", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R2 Score:", round(r2,4))

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

    #Save
    joblib.dump(best_model, REGRESSION_MODEL_PATH)

    print("\nBest regression model saved successfully.")

