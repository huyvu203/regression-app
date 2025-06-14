from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from preprocess import preprocessor

# Train RandomForestRegressor model
def train_random_forest_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    kfold):

  # RandomForestRegressor
  rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
  ])

  rf_grid = {
      "model__n_estimators": [100, 200],
      "model__max_depth": [None, 10, 20]
  }

  rf_grid_search = GridSearchCV(
    estimator = rf_pipeline,
    param_grid = rf_grid,
    scoring = "neg_mean_squared_error",
    cv = kfold,
    verbose = 2,
    n_jobs = -1
  )

  print("Fitting RandomForestRegressor...")
  rf_grid_search.fit(X_train, y_train)

  rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_grid_search.best_estimator_, X_test, y_test)
  rf_metrics = {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2}

  log_model_mlflow(rf_grid_search, "RandomForestRegressor", X_train, rf_metrics)

  return rf_grid_search, rf_rmse, rf_mae, rf_r2

# Train XGBoost model
def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    kfold):

  # XGBoost
  xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb.XGBRegressor(random_state=42))
  ])

  xgb_grid = {
      "model__n_estimators": [100, 200],
      "model__max_depth": [6, 10, 20]
  }
  xgb_grid_search = GridSearchCV(
      estimator = xgb_pipeline,
      param_grid = xgb_grid,
      scoring = "neg_mean_squared_error",
      cv = kfold,
      verbose = 2,
      n_jobs = -1
  )

  print("Fitting XGBoost...")
  xgb_grid_search.fit(X_train, y_train)

  xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(xgb_grid_search.best_estimator_, X_test, y_test)
  xgb_metrics = {"rmse": xgb_rmse, "mae": xgb_mae, "r2": xgb_r2}

  log_model_mlflow(xgb_grid_search, "XGBoost", X_train, xgb_metrics)

  return xgb_grid_search, xgb_rmse, xgb_mae, xgb_r2

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)  # Predict
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  return rmse, mae, r2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
  best_model = None
  best_rmse = float("inf")
  best_model_name = ""

  kfold = KFold(n_splits=5, shuffle=True, random_state=42)

  # Train models
  rf_grid_search, rf_rmse, rf_mae, rf_r2 = train_random_forest_regressor(X_train, y_train, X_test, y_test, preprocessor, kfold)
  xgb_grid_search, xgb_rmse, xgb_mae, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test, preprocessor, kfold)

  print("\nRandomForestRegressor metrics:")
  print(f"RMSE: {rf_rmse}\nMAE: {rf_mae}\nR2: {rf_r2}")
  print("\nXGBoost metrics:")
  print(f"RMSE: {xgb_rmse}\nMAE: {xgb_mae}\nR2: {xgb_r2}")

  # Choose best performing model
  if rf_rmse < best_rmse:
    best_rmse = rf_rmse
    best_model = rf_grid_search.best_estimator_
    best_model_name = "RandomForestRegressor"
  if xgb_rmse < best_rmse:
    best_rmse = xgb_rmse
    best_model = xgb_grid_search.best_estimator_
    best_model_name = "XGBoost"
  print(f"\nBest performing model: {best_model_name} with RMSE: {best_rmse}")

  # Package best performing model
  if best_model is not None:
    joblib.dump(best_model, "model.pkl")

# Logging using mlflow
def log_model_mlflow(grid_search_result, model_name: str, X_train, model_metrics: dict):
  with mlflow.start_run(run_name=model_name):
    mlflow.log_param("n_estimators", grid_search_result.best_params_["model__n_estimators"])
    mlflow.log_param("max_depth", grid_search_result.best_params_["model__max_depth"])

    mlflow.log_metrics(model_metrics)

    signature = mlflow.models.infer_signature(X_train, grid_search_result.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        grid_search_result.best_estimator_,
        name=model_name,
        signature=signature,
        input_example=X_train.iloc[:2]
    )

    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training info": f"{model_name} for California Housing Data"}
    )


if __name__ == "__main__":
    train_model()