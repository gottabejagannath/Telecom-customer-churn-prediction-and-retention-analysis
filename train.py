"""Train a churn model. Uses the IBM Telco CSV if present, else synthetic data."""
import os, numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV = "customer_churn_prediction_dataset.csv"
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
       "OnlineSecurity", "PaperlessBilling", "PhoneService"]

def synthetic(n=5000, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "tenure": r.integers(1, 73, n),
        "MonthlyCharges": r.uniform(20, 120, n).round(2),
        "SeniorCitizen": r.integers(0, 2, n),
        "Contract": r.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2]),
        "InternetService": r.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .2]),
        "PaymentMethod": r.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "TechSupport": r.choice(["Yes", "No"], n),
        "OnlineSecurity": r.choice(["Yes", "No"], n),
        "PaperlessBilling": r.choice(["Yes", "No"], n),
        "PhoneService": r.choice(["Yes", "No"], n, p=[.9, .1]),
    })
    d["TotalCharges"] = d.tenure * d.MonthlyCharges
    z = (-1 + 1.6 * (d.Contract == "Month-to-month") - 0.03 * d.tenure
         + 0.015 * d.MonthlyCharges + 0.6 * (d.InternetService == "Fiber optic")
         + 0.5 * (d.PaymentMethod == "Electronic check") - 0.5 * (d.TechSupport == "Yes"))
    d["Churn"] = np.where(r.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return d

def load():
    if os.path.exists(CSV):
        d = pd.read_csv(CSV)
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
        return d
    print("CSV not found -> using synthetic data")
    return synthetic()

if __name__ == "__main__":
    df = load()
    X, y = df[NUM + CAT], (df["Churn"] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    model = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42))])
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(yte, p), 3))
    print(classification_report(yte, p > .5))
    joblib.dump(model, "model.joblib")
    print("saved model.joblib")
