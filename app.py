"""Streamlit app: predict churn risk and suggest retention actions."""
import joblib, os, pandas as pd, streamlit as st
from train import NUM, CAT

st.set_page_config(page_title="Churn Predictor", page_icon="📡")
if not os.path.exists("model.joblib"):
    os.system("python train.py")  # train on first run
model = joblib.load("model.joblib")

def actions(r, p):
    out = []
    if r["Contract"] == "Month-to-month": out.append("Offer a discounted 1-2 year contract")
    if r["TechSupport"] == "No": out.append("Give 3 months free tech support")
    if r["InternetService"] == "Fiber optic" and p > .5: out.append("Proactive call about service quality")
    if r["PaymentMethod"] == "Electronic check": out.append("Incentivise auto-pay (card/bank)")
    if r["tenure"] < 12: out.append("Onboarding check-in and welcome offer")
    return out or ["Standard loyalty communication"]

st.title("📡 Telecom Churn Prediction & Retention")
tab1, tab2 = st.tabs(["Single customer", "Batch (CSV)"])

with tab1:
    c1, c2 = st.columns(2)
    r = {
        "tenure": c1.slider("Tenure (months)", 0, 72, 6),
        "MonthlyCharges": c1.number_input("Monthly charges", 18.0, 130.0, 85.0),
        "SeniorCitizen": int(c1.checkbox("Senior citizen")),
        "Contract": c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "InternetService": c2.selectbox("Internet", ["DSL", "Fiber optic", "No"]),
        "PaymentMethod": c2.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]),
        "TechSupport": c2.selectbox("Tech support", ["Yes", "No"]),
        "OnlineSecurity": c2.selectbox("Online security", ["Yes", "No"]),
        "PaperlessBilling": c2.selectbox("Paperless billing", ["Yes", "No"]),
        "PhoneService": c2.selectbox("Phone service", ["Yes", "No"]),
    }
    r["TotalCharges"] = r["tenure"] * r["MonthlyCharges"]
    if st.button("Predict"):
        p = model.predict_proba(pd.DataFrame([r])[NUM + CAT])[0, 1]
        level = "HIGH" if p > .6 else "MEDIUM" if p > .3 else "LOW"
        st.metric("Churn probability", f"{p:.0%}", level + " risk")
        st.subheader("Recommended retention actions")
        for a in actions(r, p): st.write("•", a)

with tab2:
    f = st.file_uploader("Upload customers CSV (same columns)", type="csv")
    if f:
        d = pd.read_csv(f)
        d["TotalCharges"] = pd.to_numeric(d.get("TotalCharges", d.tenure * d.MonthlyCharges), errors="coerce").fillna(0)
        d["churn_prob"] = model.predict_proba(d[NUM + CAT])[:, 1]
        d = d.sort_values("churn_prob", ascending=False)
        st.dataframe(d.head(50))
        st.download_button("Download scored CSV", d.to_csv(index=False), "scored.csv")
