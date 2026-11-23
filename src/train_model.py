import pandas as pd
import hashlib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Load cleaned dataset
df = pd.read_csv("data/processed/train_clean.csv")

# Select input features
features = [
    "distance_km",
    "wait_min",
    "passengers",
    "hour",
    "luggage_bags",
    "fuel_price_rs",
    "booking",
    "zone"
]

X = df[features]
y = df["fare_rs"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
# Categorical columns
categorical_features = ["booking", "zone"]

# Convert categorical values into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_val)

# Calculate validation error
mae = mean_absolute_error(y_val, predictions)

print("Validation MAE:", round(mae, 2))
# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("G095_MP041")

# Calculate SHA-256 of original dataset
with open("data/train.csv", "rb") as f:
    data_hash = hashlib.sha256(f.read()).hexdigest()

# Record baseline model experiment
# Log and register the best model
with mlflow.start_run(run_name="RandomForest_100_v1"):
    mlflow.log_param("data_sha256", data_hash)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("validation_mae", mae)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="fare_model",
        registered_model_name="G095_MP041_Fare_Model",
        serialization_format="cloudpickle"
    )

    print("Model registered successfully!")

    if model_info.registered_model_version is None:
        raise RuntimeError("Model registration did not return a version")

    # Assign the champion alias
    client = MlflowClient()

    client.set_registered_model_alias(
        name="G095_MP041_Fare_Model",
        alias="champion",
        version=str(model_info.registered_model_version)
    )

    print("Champion alias assigned successfully!")
    print("Registered version:", model_info.registered_model_version)