import pandas as pd
import pickle
import kagglehub
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
path = kagglehub.dataset_download("nanditapore/healthcare-diabetes")
file_path = os.path.join(path, "Healthcare-Diabetes.csv")
data = pd.read_csv(file_path)

# Clean
data = data.drop("Id", axis=1)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# MODEL 1: Logistic Regression
# ======================
lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Cross-validation
lr_cv = cross_val_score(lr_pipeline, X, y, cv=5)
print("LR Cross-validation Accuracy:", lr_cv.mean())

# ======================
# MODEL 2: Random Forest
# ======================
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

rf_cv = cross_val_score(rf_pipeline, X, y, cv=5)
print("RF Cross-validation Accuracy:", rf_cv.mean())

# ======================
# SELECT BEST MODEL
# ======================
if rf_cv.mean() > lr_cv.mean():
    best_model = rf_pipeline
    model_name = "Random Forest"
else:
    best_model = lr_pipeline
    model_name = "Logistic Regression"

print(f"\nBest Model Selected: {model_name}")

# ======================
# FEATURE IMPORTANCE (if RF)
# ======================
if model_name == "Random Forest":
    importances = best_model.named_steps["model"].feature_importances_
    feature_importance = dict(zip(X.columns, importances))
else:
    coefs = best_model.named_steps["model"].coef_[0]
    feature_importance = dict(zip(X.columns, coefs))

print("\nFeature Importance:")
for k, v in feature_importance.items():
    print(k, ":", v)

# Save everything
pickle.dump({
    "model": best_model,
    "features": list(X.columns),
    "importance": feature_importance,
    "model_name": model_name,
    "data": data
}, open("diabetes_pipeline.pkl", "wb"))

print("\nModel trained & saved successfully!")