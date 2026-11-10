"""Модуль навчання, оцінювання та порівняння моделей класифікації."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


@dataclass
class ClassificationMetrics:
    """Метрики якості класифікації на тестовій вибірці."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    report_dict: Dict[str, Any]
    report_df: pd.DataFrame


@dataclass
class ClassificationResult:
    """Повний результат навчання та тестування моделі класифікації."""

    model_name: str
    model_type: str
    pipeline: Pipeline
    metrics: ClassificationMetrics
    classes: List[str]
    y_test: pd.Series
    y_pred: np.ndarray
    interpretation: str
    limitations: str


def get_model_docs(model_type: str) -> Tuple[str, str]:
    """Повертає опис 'Як інтерпретувати' та 'Обмеження методу' для обраної моделі."""
    if model_type == "logistic_regression":
        interpretation = (
            "**Логістична регресія** моделює ймовірність належності до кожного класу через лінійну "
            "комбінацію ознак із логістичною функцією активації. "
            "Високий F1-macro свідчить про рівномірну якість розпізнавання як частих, так і рідкісних класів. "
            "Матриця невідповідностей показує, які саме класи модель схильна плутати між собою."
        )
        limitations = (
            "- Припускає лінійну роздільність класів у просторі ознак.\n"
            "- Чутлива до мультиколінеарності (сильної кореляції між ознаками).\n"
            "- Вимагає ретельного масштабування числових ознак (вирішено в нашому Pipeline).\n"
            "- Складно моделює нелінійні взаємодії без спеціального проєктування ознак."
        )
    else:  # random_forest
        interpretation = (
            "**Випадковий ліс (Random Forest)** — це ансамбль із десятків незалежних дерев рішень, "
            "що приймають рішення більшістю голосів. Модель здатна виявляти складні нелінійні закономірності "
            "та не вимагає нормальності розподілу ознак."
        )
        limitations = (
            "- Схильна до перенавчання (overfitting) на зашумлених вибірках невеликого розміру.\n"
            "- Працює як «чорна скринька» (інтерпретація конкретних рішень складніша, ніж у лінійних моделей).\n"
            "- Вимагає більше пам'яті та обчислювальних ресурсів при великій кількості дерев та глибині."
        )
    return interpretation, limitations


def create_classifier_pipeline(
    preprocessor: ColumnTransformer,
    model_type: str = "random_forest",
    use_balanced_weights: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Створює повний конвеєр (Pipeline = Preprocessor + Classifier).

    Args:
        preprocessor: Налаштований ColumnTransformer.
        model_type: 'logistic_regression' або 'random_forest'.
        use_balanced_weights: Чи використовувати автоматичне зважування класів.
        random_state: Фіксований сід для відтворюваності.

    Returns:
        Об'єкт sklearn Pipeline.
    """
    class_weight = "balanced" if use_balanced_weights else None

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            class_weight=class_weight,
            random_state=random_state,
            max_iter=1000,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Невідомий тип моделі класифікації: {model_type}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return pipeline


def evaluate_classifier_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    model_name: Optional[str] = None,
) -> ClassificationResult:
    """Навчає конвеєр виключно на train і обчислює метрики на test без витоку даних.

    Args:
        pipeline: Сконфігурований Pipeline.
        X_train: Ознаки навчальної вибірки.
        y_train: Ціль навчальної вибірки.
        X_test: Ознаки тестової вибірки.
        y_test: Ціль тестової вибірки.
        model_type: 'logistic_regression' або 'random_forest'.
        model_name: Опціональна назва для відображення.

    Returns:
        Об'єкт ClassificationResult з метриками та навченим пайплайном.
    """
    # Навчання строго на X_train
    pipeline.fit(X_train, y_train)

    # Прогноз на X_test
    y_pred = pipeline.predict(X_test)

    # Отримуємо унікальні класи у фіксованому порядку
    classes = [str(c) for c in pipeline.classes_]

    # Розрахунок метрик
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    # Звіт класифікації
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=pipeline.classes_,
        output_dict=True,
        zero_division=0,
    )
    # Формуємо зручний DataFrame зі звіту
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)

    metrics = ClassificationMetrics(
        accuracy=acc,
        precision_macro=prec_macro,
        recall_macro=rec_macro,
        f1_macro=f1_mac,
        confusion_matrix=cm,
        report_dict=report_dict,
        report_df=report_df,
    )

    display_name = (
        model_name
        or ("Логістична регресія" if model_type == "logistic_regression" else "Випадковий ліс (Random Forest)")
    )
    interpretation, limitations = get_model_docs(model_type)

    return ClassificationResult(
        model_name=display_name,
        model_type=model_type,
        pipeline=pipeline,
        metrics=metrics,
        classes=classes,
        y_test=y_test,
        y_pred=y_pred,
        interpretation=interpretation,
        limitations=limitations,
    )


def compare_classification_models(
    results: Dict[str, ClassificationResult]
) -> pd.DataFrame:
    """Формує порівняльну таблицю метрик для кількох моделей.

    Args:
        results: Словник з результатами моделей {model_name: ClassificationResult}.

    Returns:
        DataFrame з порівнянням метрик Accuracy, Precision (macro), Recall (macro), F1 (macro).
    """
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append(
            {
                "Модель": name,
                "Accuracy (Точність)": f"{m.accuracy:.3f}",
                "Precision (Macro)": f"{m.precision_macro:.3f}",
                "Recall (Macro)": f"{m.recall_macro:.3f}",
                "F1-Score (Macro)": f"{m.f1_macro:.3f}",
            }
        )
    return pd.DataFrame(rows)
