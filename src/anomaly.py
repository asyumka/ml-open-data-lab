"""Модуль виявлення аномалій (Isolation Forest, Local Outlier Factor) та z-score інтерпретації."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class AnomalyDetectionResult:
    """Результат роботи алгоритму виявлення аномалій."""

    model_name: str
    model_type: str
    n_samples: int
    n_anomalies: int
    anomaly_percentage: float  # частка аномалій у % (не плутати з accuracy!)
    top_anomalies_df: pd.DataFrame
    enriched_df: pd.DataFrame
    pca_coords: np.ndarray
    is_anomaly_mask: np.ndarray
    anomaly_scores: np.ndarray
    interpretation: str
    limitations: str


def get_anomaly_docs(model_type: str) -> Tuple[str, str]:
    """Повертає опис 'Як інтерпретувати' та 'Обмеження методу' для виявлення аномалій."""
    if model_type == "isolation_forest":
        interpretation = (
            "**Ізоляційний ліс (Isolation Forest)** — ансамблевий алгоритм, що ізолює аномалії "
            "шляхом випадкового розбиття простору ознак. Оскільки нетипові спостереження (викиди) "
            "знаходяться на периферії простору, для їх ізоляції потрібно значно менше розбиттів "
            "(менша середня глибина шляху в дереві). "
            "Чим вищий бал аномальності (Anomaly Score), тим сильніше запис відрізняється від загальної маси."
        )
        limitations = (
            "- Чутливий до вибору гіперпараметра contamination (очікуваної частки аномалій).\n"
            "- Може гірше працювати при наявності великої кількості іррелевантних (шумових) ознак.\n"
            "- Не враховує локальну щільність скупчень (clusters) різного розміру."
        )
    else:  # local_outlier_factor
        interpretation = (
            "**Локальний фактор викиду (Local Outlier Factor, LOF)** — метод оцінки аномальності на основі "
            "локальної щільності сусідів. Запис вважається аномальним, якщо його локальна щільність "
            "суттєво нижча за щільність його найближчих k сусідів (k-nearest neighbors). "
            "Це дозволяє ефективно знаходити аномалії навіть у датасетах із кластерами різної щільності."
        )
    return interpretation, (
        "- Має вищу обчислювальну складність $O(N^2)$, тому повільніший на великих наборах даних.\n"
        "- Не дозволяє легко застосовувати навчену модель до нових записів без перерахунку графів сусідів.\n"
        "- Чутливий до кількості сусідів `n_neighbors`."
    )


def compute_zscore_explanations(
    X_df: pd.DataFrame,
    is_anomaly_mask: np.ndarray,
    numeric_cols: List[str],
    top_k_features: int = 3,
) -> List[str]:
    """Обчислює z-score для кожної ознаки запису та генерує текстове пояснення причин аномальності.

    Args:
        X_df: Таблиця з числовими даними.
        is_anomaly_mask: Булевий масив (True для аномалій).
        numeric_cols: Список числових стовпців.
        top_k_features: Скільки найбільш відхилених ознак включати у пояснення.

    Returns:
        Список рядків-пояснень для кожного запису датасету.
    """
    if not numeric_cols:
        return ["Немає числових ознак для z-score" if is_anom else "Норма" for is_anom in is_anomaly_mask]

    num_data = X_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = num_data.mean()
    stds = num_data.std().replace(0.0, 1.0).fillna(1.0)

    z_scores = (num_data - means) / stds

    reasons: List[str] = []
    for i, is_anom in enumerate(is_anomaly_mask):
        if not is_anom:
            reasons.append("Типове спостереження")
            continue

        row_z = z_scores.iloc[i]
        # Сортуємо ознаки за абсолютним значенням z-score (спадно)
        sorted_feats = row_z.abs().sort_values(ascending=False)

        top_feats = []
        for feat in sorted_feats.index[:top_k_features]:
            val_z = row_z[feat]
            sign = "+" if val_z >= 0 else ""
            top_feats.append(f"{feat} (z={sign}{val_z:.2f}σ)")

        reasons.append("; ".join(top_feats) if top_feats else "Аномалія у комбінації ознак")

    return reasons


def run_anomaly_detection(
    df: pd.DataFrame,
    feature_cols: List[str],
    preprocessor: ColumnTransformer,
    model_type: str = "isolation_forest",
    contamination: float = 0.05,
    n_neighbors: int = 20,
    top_n: int = 10,
    random_state: int = 42,
) -> AnomalyDetectionResult:
    """Виконує пошук аномалій, PCA-2D трансформацію та z-score пояснення без витоку даних.

    Args:
        df: Вхідний DataFrame.
        feature_cols: Список обраних ознак.
        preprocessor: ColumnTransformer для підготовки ознак.
        model_type: 'isolation_forest' або 'local_outlier_factor'.
        contamination: Очікувана частка аномалій (наприклад, 0.05 = 5%).
        n_neighbors: Кількість сусідів для LOF.
        top_n: Кількість записів для виведення в топ аномалій.
        random_state: Фіксований сід для відтворюваності.

    Returns:
        Об'єкт AnomalyDetectionResult.
    """
    X_subset = df[feature_cols].copy()

    # Трансформація ознак через preprocessor
    X_trans = preprocessor.fit_transform(X_subset)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    n_samples = len(df)

    if model_type == "isolation_forest":
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        labels = model.fit_predict(X_trans)
        raw_scores = model.decision_function(X_trans)
        # У Isolation Forest чим нижчий raw_score, тим сильніша аномалія.
        # Інвертуємо, щоб вищий бал відповідав сильнішій аномальності:
        anomaly_scores = -raw_scores
        display_name = "Ізоляційний ліс (Isolation Forest)"

    elif model_type == "local_outlier_factor":
        k_neighbors = min(n_neighbors, max(2, n_samples - 1))
        model = LocalOutlierFactor(
            n_neighbors=k_neighbors,
            contamination=contamination,
        )
        labels = model.fit_predict(X_trans)
        # negative_outlier_factor_: чим більше від'ємне, тим сильніша аномалія
        anomaly_scores = -model.negative_outlier_factor_
        display_name = "Локальний фактор викиду (Local Outlier Factor)"
    else:
        raise ValueError(f"Невідомий тип моделі аномалій: {model_type}")

    is_anomaly = labels == -1
    n_anomalies = int(np.sum(is_anomaly))
    anomaly_pct = float(n_anomalies / n_samples * 100) if n_samples > 0 else 0.0

    # 2D PCA проекція
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_trans)

    # Обчислення z-score пояснень для числових колонок
    numeric_in_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    reasons = compute_zscore_explanations(
        X_df=df[feature_cols],
        is_anomaly_mask=is_anomaly,
        numeric_cols=numeric_in_features,
        top_k_features=3,
    )

    # Збагачення вихідного датасету
    enriched_df = df.copy()
    enriched_df["is_anomaly"] = np.where(is_anomaly, "Так (Аномалія)", "Ні (Типовий)")
    enriched_df["anomaly_score"] = np.round(anomaly_scores, 4)
    enriched_df["anomaly_reasons"] = reasons

    # Формування топ-N найбільш аномальних записів (сортування за спаданням anomaly_score)
    top_indices = np.argsort(anomaly_scores)[::-1][:top_n]
    top_anomalies_df = enriched_df.iloc[top_indices].copy()

    interpretation, limitations = get_anomaly_docs(model_type)

    return AnomalyDetectionResult(
        model_name=display_name,
        model_type=model_type,
        n_samples=n_samples,
        n_anomalies=n_anomalies,
        anomaly_percentage=anomaly_pct,
        top_anomalies_df=top_anomalies_df,
        enriched_df=enriched_df,
        pca_coords=pca_coords,
        is_anomaly_mask=is_anomaly,
        anomaly_scores=anomaly_scores,
        interpretation=interpretation,
        limitations=limitations,
    )
