
import os
from pathlib import Path

import joblib
import kagglehub
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATA
# ==========================================

path = kagglehub.dataset_download(
    "nanditapore/healthcare-diabetes"
)
file_path = os.path.join(path, "Healthcare-Diabetes.csv")

data = pd.read_csv(file_path)

X = data.drop(columns=["Outcome", "Id"], errors="ignore")
y = data["Outcome"]

print("Dataset shape:", data.shape)
print("Features:", list(X.columns))
print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# All model selection happens within the training data.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}


# ==========================================
# 4. CROSS-VALIDATION ON TRAINING DATA
# ==========================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = {}
comparison_rows = []

for name, pipeline in models.items():
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    cv_results[name] = scores

    comparison_rows.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean()
    })

    print(f"\n{name}:")
    print(
        "Mean CV ROC-AUC:",
        round(scores["test_roc_auc"].mean(), 4)
    )


# Select the model using training-set CV ROC-AUC.
comparison = pd.DataFrame(comparison_rows)

best_index = comparison["CV ROC-AUC"].idxmax()
model_name = comparison.loc[best_index, "Model"]

print("\nCross-validation comparison:")
print(comparison.round(3).to_string(index=False))
print("\nSelected model:", model_name)


# ==========================================
# 5. FIT AND SAVE EACH MODEL
# ==========================================

Path("models").mkdir(exist_ok=True)

test_results = {}
fitted_models = {}

for name, pipeline in models.items():
    # Fit on training data only.
    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline

    # Preserve individual model files.
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, Path("models") / filename)

    # Evaluate on the same held-out test set.
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    test_results[name] = metrics

    print(f"\n{'=' * 12} {name} TEST RESULTS {'=' * 12}")
    print(pd.Series(metrics).round(4))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


# ==========================================
# 6. ANALYSE ERRORS FOR SELECTED MODEL
# ==========================================

best_model = fitted_models[model_name]

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

error_analysis = X_test.reset_index(drop=True).copy()
error_analysis["Actual"] = y_test.reset_index(drop=True)
error_analysis["Predicted"] = best_predictions
error_analysis["Probability"] = best_probabilities

error_analysis["Error Type"] = "Correct"

error_analysis.loc[
    (error_analysis["Actual"] == 0)
    & (error_analysis["Predicted"] == 1),
    "Error Type"
] = "False Positive"

error_analysis.loc[
    (error_analysis["Actual"] == 1)
    & (error_analysis["Predicted"] == 0),
    "Error Type"
] = "False Negative"

error_analysis = error_analysis[
    error_analysis["Error Type"] != "Correct"
].copy()


# ==========================================
# 7. FEATURE ANALYSIS
# ==========================================

if model_name == "Random Forest":
    feature_importance = dict(zip(
        X.columns,
        best_model.named_steps["model"].feature_importances_
    ))

elif model_name == "Logistic Regression":
    # Signed coefficients, not universal feature importance.
    feature_importance = dict(zip(
        X.columns,
        best_model.named_steps["model"].coef_[0]
    ))

else:
    # KNN has no native feature-importance attribute.
    feature_importance = None


# ==========================================
# 8. SAVE SELECTED MODEL AND RESULTS
# ==========================================

bundle = {
    "model": best_model,
    "features": list(X.columns),
    "model_name": model_name,
    "importance": feature_importance,
    "data": data,

    # Cross-validation comparison for model selection.
    "model_comparison": comparison,

    # Held-out test metrics for all three models.
    "test_results": test_results,

    # Selected model's held-out test predictions.
    "y_test": y_test.reset_index(drop=True),
    "test_predictions": best_predictions,
    "test_probabilities": best_probabilities,

    # Misclassified held-out examples.
    "error_analysis": error_analysis
}

joblib.dump(bundle, "diabetes_pipeline.pkl")

print("\nSelected model saved to diabetes_pipeline.pkl")
print("Individual models saved in models/")
print("Training and evaluation completed.")
