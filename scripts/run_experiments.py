"""Запуск усіх режимів застосунку на реальних наборах і фіксація результатів для звіту.

Використовує ті самі функції `src/`, що й інтерфейс Streamlit, з random_state=42.
Результати: reports/results.json та графіки reports/figures/*.png.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly import run_anomaly_detection
from src.classification import create_classifier_pipeline, evaluate_classifier_model
from src.data_loader import analyze_dataset, load_dataset, sample_dataframe
from src.plots import (
    plot_confusion_matrix,
    plot_pca_anomalies,
    plot_regression_predictions,
    plot_regression_residuals,
)
from src.preprocessing import build_preprocessor, filter_rare_classes, split_data
from src.regression import create_regression_pipeline, evaluate_regression_model
from src.validation import validate_classification_target, validate_regression_target

SEED = 42
OUT = "reports"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R: dict = {}


def save(fig, name: str) -> str:
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def split_types(info, feats):
    return (
        [c for c in feats if c in info.numeric_columns],
        [c for c in feats if c in info.categorical_columns],
    )


# ============================================================ 1. House Prices — регресія
t0 = time.time()
df = load_dataset("data/House Prices.csv", "csv", "utf-8", "auto")
info = analyze_dataset(df)
feats = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt",
         "LotArea", "Neighborhood", "ExterQual", "KitchenQual", "BldgType"]
num, cat = split_types(info, feats)
ok, msgs, meta = validate_regression_target(df["SalePrice"])
X_tr, X_te, y_tr, y_te, desc = split_data(df, feats, "SalePrice", 0.2, None, "regression", SEED)
house = {"shape": list(df.shape), "split": desc, "features": feats, "numeric": num,
         "categorical": cat, "models": {}}
for mt, name in [("ridge", "Ridge"), ("random_forest", "RandomForestRegressor")]:
    pipe = create_regression_pipeline(build_preprocessor(num, cat), mt, alpha=1.0, random_state=SEED)
    res = evaluate_regression_model(pipe, X_tr, y_tr, X_te, y_te, mt, "SalePrice")
    house["models"][name] = {"MAE": res.metrics.mae, "RMSE": res.metrics.rmse, "R2": res.metrics.r2}
    save(plot_regression_predictions(res.y_test, res.y_pred, f"Факт vs Прогноз: {name}", "SalePrice, USD"),
         f"house_pred_vs_fact_{mt}.png")
    save(plot_regression_residuals(res.y_test, res.y_pred, f"Залишки: {name}"), f"house_residuals_{mt}.png")
    if mt == "random_forest":
        err = (res.y_test - res.y_pred).abs().sort_values(ascending=False).head(5)
        house["worst_errors"] = [
            {"index": int(i), "fact": float(res.y_test.loc[i]), "pred": float(res.y_pred[list(res.y_test.index).index(i)]),
             "Neighborhood": str(X_te.loc[i, "Neighborhood"]), "GrLivArea": float(X_te.loc[i, "GrLivArea"])}
            for i in err.index
        ]
house["y_stats"] = {k: float(v) for k, v in df["SalePrice"].describe().items()}
house["seconds"] = round(time.time() - t0, 1)
R["house_prices"] = house

# Некоректна ціль для регресії на реальних даних
ok, msgs, _ = validate_regression_target(df["Neighborhood"])
R["house_prices"]["dirty_text_target"] = {"valid": ok, "message": msgs[0] if msgs else ""}

# ============================================================ 2. Реєстр ЗВО — класифікація
t0 = time.time()
df = load_dataset("data/reiestr-zakladiv-vishchoyi-osviti.csv", "csv", "windows-1251", "auto")
info = analyze_dataset(df)
target = "university_financing_type_name"
ok0, msgs0, meta0 = validate_classification_target(df[target])
R["hei"] = {"shape": list(df.shape), "skipped_rows": int(df.attrs["skipped_rows"]),
            "id_like": info.id_like_columns, "class_counts_raw": {str(k): int(v) for k, v in df[target].value_counts().items()},
            "validation_without_filter": {"valid": ok0, "message": msgs0[0] if msgs0 else ""}}
df_f, removed = filter_rare_classes(df, target, 2)
feats = ["education_type_name", "university_type_name", "region_name", "registration_year"]
num, cat = split_types(info, feats)
ok1, msgs1, meta1 = validate_classification_target(df_f[target])
X_tr, X_te, y_tr, y_te, desc = split_data(df_f, feats, target, 0.2, None, "classification", SEED)
hei = R["hei"]
hei.update({"removed_classes": removed, "rows_after_filter": int(len(df_f)), "split": desc,
            "features": feats, "validation_after_filter": msgs1, "test_class_counts":
            {str(k): int(v) for k, v in y_te.value_counts().items()}, "models": {}})
for mt, name in [("logistic_regression", "LogisticRegression"), ("random_forest", "RandomForestClassifier")]:
    pipe = create_classifier_pipeline(build_preprocessor(num, cat), mt, True, SEED)
    res = evaluate_classifier_model(pipe, X_tr, y_tr, X_te, y_te, mt)
    m = res.metrics
    hei["models"][name] = {"accuracy": m.accuracy, "precision_macro": m.precision_macro,
                           "recall_macro": m.recall_macro, "f1_macro": m.f1_macro,
                           "classes": res.classes, "confusion_matrix": m.confusion_matrix.tolist(),
                           "report": m.report_df.round(3).to_dict()}
    save(plot_confusion_matrix(m.confusion_matrix, res.classes, f"Confusion Matrix: {name}"), f"hei_cm_{mt}.png")
    if mt == "logistic_regression":
        wrong = X_te[(res.y_test.values != res.y_pred)].copy()
        wrong["fact"] = res.y_test[res.y_test.values != res.y_pred].values
        wrong["pred"] = res.y_pred[res.y_test.values != res.y_pred]
        hei["error_examples"] = wrong.head(6).astype(str).to_dict("records")
hei["seconds"] = round(time.time() - t0, 1)

# ============================================================ 3. Реєстр ТЗ — аномалії
t0 = time.time()
df = load_dataset("data/reestrTZ2022.csv", "csv", "utf-8", "auto")
info = analyze_dataset(df)
R["tz"] = {"shape": list(df.shape), "load_seconds": round(time.time() - t0, 1),
           "datetime_columns": info.datetime_columns, "id_like": info.id_like_columns,
           "high_cardinality": info.high_cardinality_columns}
d = pd.to_datetime(df["D_REG"], format="%d.%m.%Y", errors="coerce")
R["tz"]["period"] = [str(d.min().date()), str(d.max().date())]
R["tz"]["capacity_stats"] = {k: float(v) for k, v in df["CAPACITY"].describe().items()}
sample = sample_dataframe(df, 20000, SEED).reset_index(drop=True)
feats = ["MAKE_YEAR", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "KIND", "FUEL"]
num, cat = split_types(info, feats)
R["tz"].update({"sample_rows": len(sample), "features": feats, "contamination": 0.01, "models": {}})
masks = {}
show = ["BRAND", "MODEL", "KIND", "FUEL", "MAKE_YEAR", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]
for mt, name in [("isolation_forest", "IsolationForest"), ("local_outlier_factor", "LocalOutlierFactor")]:
    t1 = time.time()
    a = run_anomaly_detection(sample, feats, build_preprocessor(num, cat, numeric_scaler="robust"),
                              mt, contamination=0.01, n_neighbors=20, top_n=10, random_state=SEED)
    masks[name] = a.is_anomaly_mask
    top = a.top_anomalies_df[show + ["anomaly_score", "anomaly_reasons"]].copy()
    R["tz"]["models"][name] = {"n_anomalies": a.n_anomalies, "seconds": round(time.time() - t1, 1),
                               "top10": top.astype(str).to_dict("records")}
    save(plot_pca_anomalies(a.pca_coords, a.is_anomaly_mask, a.anomaly_scores,
                            f"PCA-проєкція: {name} (вибірка {len(sample)} ТЗ)"), f"tz_pca_{mt}.png")
inter = int((masks["IsolationForest"] & masks["LocalOutlierFactor"]).sum())
union = int((masks["IsolationForest"] | masks["LocalOutlierFactor"]).sum())
R["tz"]["overlap"] = {"intersection": inter, "union": union, "jaccard": round(inter / union, 3)}

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=str)
print("DONE")
