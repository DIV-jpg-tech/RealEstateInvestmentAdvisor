import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import PROCESSED_DATA_PATH, CLASSIFICATION_MODEL_PATH, RANDOM_STATE

# Load Data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Drop unnecessary columns
drop_cols = ["ID", "Future_Price_5Y", "Good_Investment"]
X = df.drop(columns=drop_cols)

# Target
y = df["Good_Investment"]

# Separate column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
                ]
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
   "Random Forest": RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

}

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

best_model = None
best_score = 0

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))

    if f1 > best_score:
        best_score = f1
        best_model = pipeline

# Save best model
joblib.dump(best_model, CLASSIFICATION_MODEL_PATH)

print("\nBest classification model saved successfully.")