"""
First Machine Learning Project: Supervised Learning with Scikit-Learn
=====================================================================
Part 1 - REGRESSION: predict house prices (Linear Regression vs Decision Tree)
         Metrics: MSE, RMSE, R^2
Part 2 - CLASSIFICATION: predict benign/malignant tumours (Decision Tree)
         Metrics: Accuracy, Confusion Matrix, Classification Report

Run:  python first_ml_classifier.py
Needs: pip install scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # saves plots to files; remove this line to show windows
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# PART 1: REGRESSION - HOUSE PRICES
# ----------------------------------------------------------------------
def load_house_data():
    """Load California housing (price in $100,000s). If the download is
    unavailable (offline), fall back to a synthetic house dataset."""
    try:
        data = fetch_california_housing()
        print("Dataset: California Housing (20,640 districts)")
        return data.data, data.target, list(data.feature_names)
    except Exception:
        print("Dataset: synthetic houses (California data could not be downloaded)")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 2000
        sqft = rng.uniform(500, 4000, n)
        bedrooms = rng.integers(1, 6, n)
        age = rng.uniform(0, 60, n)
        X = np.column_stack([sqft, bedrooms, age])
        y = (0.0006 * sqft + 0.15 * bedrooms - 0.01 * age
             + rng.normal(0, 0.4, n) + 1.0)  # price in $100k
        return X, y, ["sqft", "bedrooms", "age"]


def run_regression():
    print("\n" + "=" * 60)
    print("PART 1: REGRESSION - PREDICTING HOUSE PRICES")
    print("=" * 60)

    X, y, feature_names = load_house_data()
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]} -> {feature_names}")

    # 1) Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 2) Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree (max_depth=6)": DecisionTreeRegressor(
            max_depth=6, random_state=RANDOM_STATE
        ),
    }

    # 3) Train + 4) Evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "pred": y_pred,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_test, y_pred),
        }
        # Compare with training error to spot overfitting
        train_mse = mean_squared_error(y_train, model.predict(X_train))

        print(f"\n{name}")
        print(f"  Train MSE : {train_mse:.4f}")
        print(f"  Test  MSE : {mse:.4f}")
        print(f"  Test  RMSE: {np.sqrt(mse):.4f}  (~${np.sqrt(mse) * 100_000:,.0f} typical error)")
        print(f"  Test  R^2 : {results[name]['r2']:.4f}")

    # Baseline: always predict the mean price. Any real model must beat this.
    baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
    print(f"\nBaseline (predict mean) MSE: {baseline_mse:.4f}")

    # Show what the linear model learned
    lin = models["Linear Regression"]
    print("\nLinear Regression coefficients:")
    for fname, coef in zip(feature_names, lin.coef_):
        print(f"  {fname:<12} {coef:+.4f}")

    # Plot: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, r) in zip(axes, results.items()):
        ax.scatter(y_test, r["pred"], alpha=0.3, s=8)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual price ($100k)")
        ax.set_ylabel("Predicted price ($100k)")
        ax.set_title(f"{name}\nMSE={r['mse']:.3f}  R²={r['r2']:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("regression_results.png", dpi=120)
    plt.close()
    print("\nSaved plot: regression_results.png")


# ----------------------------------------------------------------------
# PART 2: CLASSIFICATION - TUMOUR DIAGNOSIS
# ----------------------------------------------------------------------
def run_classification():
    print("\n" + "=" * 60)
    print("PART 2: CLASSIFICATION - DECISION TREE")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {dict(zip(data.target_names, np.bincount(y)))}")

    # stratify=y keeps the class ratio the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Unrestricted tree: tends to overfit (memorises training data)
    deep = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    # Depth-limited tree: usually generalises better
    shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(
        X_train, y_train
    )

    for name, model in [("Unrestricted tree", deep), ("Tree max_depth=3", shallow)]:
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<20} train acc = {train_acc:.3f} | test acc = {test_acc:.3f}")

    # Detailed evaluation of the depth-limited model
    y_pred = shallow.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True negatives  (malignant correctly found): {tn}")
    print(f"  False positives (malignant called benign)  : {fp}  <- dangerous error")
    print(f"  False negatives (benign called malignant)  : {fn}")
    print(f"  True positives  (benign correctly found)   : {tp}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 5-fold cross-validation: a more reliable accuracy estimate than one split
    cv = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        X, y, cv=5, scoring="accuracy",
    )
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    # Most important features
    order = np.argsort(shallow.feature_importances_)[::-1][:5]
    print("\nTop 5 features:")
    for i in order:
        print(f"  {data.feature_names[i]:<25} {shallow.feature_importances_[i]:.3f}")

    # Plot confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Blues")
    plt.title("Decision Tree (max_depth=3) - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("\nSaved plot: confusion_matrix.png")


if __name__ == "__main__":
    run_regression()
    run_classification()
