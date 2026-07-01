import os
import json
import joblib
import pandas as pd
from pipeline import DataPreprocessor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class CreditScorePredictor:
    """Wrapper inferencing: preprocess -> model -> label + probabilitas."""

    def __init__(self, model_dir=MODEL_DIR):
        self.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
        self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        self.metadata = joblib.load(os.path.join(model_dir, "metadata.pkl"))
        self.class_labels = self.metadata["class_labels"]

    def _to_frame(self, records):
        if isinstance(records, dict):
            records = [records]
        return pd.DataFrame(records)

    def predict(self, records):
        """records: dict atau list-of-dict -> list hasil prediksi."""
        df = self._to_frame(records)
        X = self.preprocessor.transform(df)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        results = []
        for i, p in enumerate(preds):
            prob_map = {cls: float(probs[i][j]) for j, cls in enumerate(self.model.classes_)}
            results.append({
                "credit_score": str(p),
                "confidence": round(max(prob_map.values()), 4),
                "probabilities": {k: round(v, 4) for k, v in prob_map.items()},
            })
        return results

    def predict_one(self, record):
        return self.predict(record)[0]

TEST_CASES = {
    "Expected GOOD": {
        "Age": 35, "Occupation": "Engineer", "Annual_Income": 95000,
        "Monthly_Inhand_Salary": 7800, "Num_Bank_Accounts": 3, "Num_Credit_Card": 4,
        "Interest_Rate": 6, "Num_of_Loan": 1, "Type_of_Loan": "Auto Loan",
        "Delay_from_due_date": 2, "Num_of_Delayed_Payment": 1, "Changed_Credit_Limit": 8.5,
        "Num_Credit_Inquiries": 1, "Credit_Mix": "Good", "Outstanding_Debt": 300,
        "Credit_Utilization_Ratio": 28, "Credit_History_Age": "25 Years and 2 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 120,
        "Amount_invested_monthly": 500, "Payment_Behaviour": "High_spent_Large_value_payments",
        "Monthly_Balance": 900, "Month": "March",
    },
    "Expected STANDARD": {
        "Age": 29, "Occupation": "Teacher", "Annual_Income": 45000,
        "Monthly_Inhand_Salary": 3600, "Num_Bank_Accounts": 5, "Num_Credit_Card": 5,
        "Interest_Rate": 14, "Num_of_Loan": 3, "Type_of_Loan": "Personal Loan, Auto Loan",
        "Delay_from_due_date": 14, "Num_of_Delayed_Payment": 9, "Changed_Credit_Limit": 10,
        "Num_Credit_Inquiries": 4, "Credit_Mix": "Standard", "Outstanding_Debt": 900,
        "Credit_Utilization_Ratio": 33, "Credit_History_Age": "18 Years and 5 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 180,
        "Amount_invested_monthly": 200, "Payment_Behaviour": "Low_spent_Medium_value_payments",
        "Monthly_Balance": 350, "Month": "April",
    },
    "Expected POOR": {
        "Age": 24, "Occupation": "Mechanic", "Annual_Income": 18000,
        "Monthly_Inhand_Salary": 1400, "Num_Bank_Accounts": 9, "Num_Credit_Card": 8,
        "Interest_Rate": 28, "Num_of_Loan": 6, "Type_of_Loan": "Payday Loan, Home Equity Loan, Auto Loan",
        "Delay_from_due_date": 45, "Num_of_Delayed_Payment": 20, "Changed_Credit_Limit": 18,
        "Num_Credit_Inquiries": 9, "Credit_Mix": "Bad", "Outstanding_Debt": 3800,
        "Credit_Utilization_Ratio": 38, "Credit_History_Age": "9 Years and 1 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 130,
        "Amount_invested_monthly": 60, "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 210, "Month": "January",
    },
}


if __name__ == "__main__":
    predictor = CreditScorePredictor()
    print("Best model:", predictor.metadata["best_model"])
    print("=" * 60)
    for name, case in TEST_CASES.items():
        res = predictor.predict_one(case)
        print(f"\n[{name}]")
        print(f"  -> Prediksi : {res['credit_score']}  (conf={res['confidence']})")
        print(f"  -> Probabilitas: {json.dumps(res['probabilities'])}")
