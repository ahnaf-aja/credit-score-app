"""
pipeline.py  (versi RINGAN untuk Streamlit Community Cloud)
==========================================================
Hanya berisi class `DataPreprocessor` + konstanta yang dibutuhkan untuk
MEMUAT (unpickle) preprocessor.pkl saat inferensi. Kode training & MLflow
sengaja dibuang supaya deploy di Streamlit Cloud ringan dan tidak perlu
menginstal mlflow.

Isi class ini SAMA PERSIS dengan pipeline.py asli, jadi preprocessor.pkl
bisa dimuat tanpa error.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    "Num_Loan_Types",
]
CATEGORICAL_FEATURES = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount",
    "Payment_Behaviour", "Month",
]
NUMERIC_BOUNDS = {
    "Age": (14, 100), "Annual_Income": (0, 1e7), "Monthly_Inhand_Salary": (0, 1e6),
    "Num_Bank_Accounts": (0, 20), "Num_Credit_Card": (0, 15), "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 15), "Delay_from_due_date": (-5, 100),
    "Num_of_Delayed_Payment": (0, 60), "Changed_Credit_Limit": (-50, 50),
    "Num_Credit_Inquiries": (0, 50), "Outstanding_Debt": (0, 1e5),
    "Credit_Utilization_Ratio": (0, 100), "Credit_History_Age": (0, 600),
    "Total_EMI_per_month": (0, 1e5), "Amount_invested_monthly": (0, 1e5),
    "Monthly_Balance": (0, 1e6),
}
PLACEHOLDERS = {"_______", "_", "!@9#%8", "#F%$D@*&8", "NM", "nan", "NaN", ""}


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Membersihkan dataset Credit Score yang noisy menjadi fitur siap model."""

    def __init__(self):
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.medians_ = {}
        self.modes_ = {}
        self.fitted_ = False

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace("_", "", regex=False).str.strip()
        s = s.replace(list(PLACEHOLDERS), np.nan)
        s = s.apply(lambda x: re.sub(r"[^0-9.\-]", "", x) if isinstance(x, str) else x)
        s = s.replace("", np.nan)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _parse_history_age(series: pd.Series) -> pd.Series:
        def conv(v):
            if not isinstance(v, str):
                return np.nan
            m = re.search(r"(\d+)\s*Year", v)
            n = re.search(r"(\d+)\s*Month", v)
            if m is None and n is None:
                return np.nan
            years = int(m.group(1)) if m else 0
            months = int(n.group(1)) if n else 0
            return years * 12 + months
        return series.apply(conv)

    @staticmethod
    def _count_loan_types(series: pd.Series) -> pd.Series:
        def cnt(v):
            if not isinstance(v, str) or v.strip() in PLACEHOLDERS:
                return 0
            v = v.replace(" and ", ",")
            parts = [p.strip() for p in v.split(",") if p.strip()
                     and p.strip().lower() != "not specified"]
            return len(parts)
        return series.apply(cnt)

    def _raw_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = self._parse_history_age(df["Credit_History_Age"])
        if "Type_of_Loan" in df.columns:
            df["Num_Loan_Types"] = self._count_loan_types(df["Type_of_Loan"])
        else:
            df["Num_Loan_Types"] = 0
        for col in self.numeric_features:
            if col in df.columns and col != "Credit_History_Age":
                df[col] = self._to_numeric(df[col])
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col in df.columns:
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(list(PLACEHOLDERS), np.nan)
        return df

    def fit(self, X: pd.DataFrame, y=None):
        df = self._raw_clean(X)
        for col in self.numeric_features:
            self.medians_[col] = df[col].median() if col in df.columns else 0.0
        for col in self.categorical_features:
            if col in df.columns and not df[col].dropna().empty:
                self.modes_[col] = df[col].mode(dropna=True).iloc[0]
            else:
                self.modes_[col] = "Unknown"
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("DataPreprocessor belum di-fit.")
        df = self._raw_clean(X)
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_features:
            vals = df[col] if col in df.columns else np.nan
            out[col] = pd.Series(vals, index=df.index).fillna(self.medians_[col])
        for col in self.categorical_features:
            vals = df[col] if col in df.columns else np.nan
            out[col] = pd.Series(vals, index=df.index).fillna(self.modes_[col]).astype(str)
        return out

    def get_feature_names(self):
        return self.numeric_features + self.categorical_features
