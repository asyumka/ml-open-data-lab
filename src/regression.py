"""Модуль навчання, оцінювання та порівняння моделей регресії."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@dataclass
class RegressionMetrics:
    """Метрики якості регресії на тестовій вибірці."""

    mae: float
    rmse: float
    r2: float


@dataclass
class RegressionResult:
    """Повний результат навчання та тестування моделі регресії."""

    model_name: str
    model_type: str
    pipeline: Pipeline
    metrics: RegressionMetrics
    y_test: pd.Series
    y_pred: np.ndarray
    target_name: str
    interpretation: str
    limitations: str


def get_regression_docs(model_type: str) -> Tuple[str, str]:
    """Повертає опис 'Як інтерпретувати' та 'Обмеження методу' для обраної регресійної моделі."""
    if model_type == "ridge":
        interpretation = (
            "**Гребенева регресія (Ridge)** — це лінійна регресія з L2-регуляризацією (штрафом на квадрат "
            "величини коефіцієнтів). Вона запобігає роздуванню ваг при мультиколінеарності ознак. "
            "Коефіцієнт детермінації R² показує частку дисперсії цільової змінної, яку пояснює модель "
            "(1.0 — ідеальний прогноз, 0.0 — рівень середнього значення цілі, менше 0 — модель гірша за константу). "
            "MAE та RMSE вимірюють абсолютну середню помилку у вихідних одиницях цільової величини."
        )
        limitations = (
            "- Моделює лише лінійні зв'язки між ознаками та ціллю.\n"
            "- Чутлива до грубих викидів (outliers), хоча L2-штраф робить її стабільнішою за OLS.\n"
            "- Не зануляє коефіцієнти незначущих ознак повністю (на відміну від Lasso)."
        )
    else:  # random_forest
        interpretation = (
            "**Випадковий ліс для регресії (Random Forest Regressor)** — ансамбль багатьох дерев регресії, "
            "де фінальний прогноз є усередненням прогнозів усіх дерев. Відмінно схоплює складні нелінійні залежності "
            "та взаємодії високих порядків між факторами."
        )
        limitations = (
            "- Не здатний екстраполювати значення за межі діапазону навчальної вибірки (min/max target).\n"
            "- Значно повільніший у навчанні порівняно з лінійною регресією на великих вибірках.\n"
            "- Схильний до перенавчання за відсутності обмеження глибини дерев (`max_depth`)."
        )
    return interpretation, limitations


def create_regression_pipeline(
    preprocessor: ColumnTransformer,
    model_type: str = "ridge",
    alpha: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Створює повний конвеєр регресії (Pipeline = Preprocessor + Regressor).

    Args:
        preprocessor: Налаштований ColumnTransformer.
        model_type: 'ridge' або 'random_forest'.
        alpha: Коефіцієнт регуляризації для Ridge.
        random_state: Фіксований сід для відтворюваності.

    Returns:
        Об'єкт sklearn Pipeline.
    """
    if model_type == "ridge":
        reg = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == "random_forest":
        reg = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Невідомий тип регресійної моделі: {model_type}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])
    return pipeline


def evaluate_regression_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    target_name: str,
    model_name: Optional[str] = None,
) -> RegressionResult:
    """Навчає регресійний конвеєр на train та розраховує MAE, RMSE, R² на test без витоку даних.

    Args:
        pipeline: Сконфігурований Pipeline.
        X_train: Ознаки навчальної вибірки.
        y_train: Ціль навчальної вибірки.
        X_test: Ознаки тестової вибірки.
        y_test: Ціль тестової вибірки.
        model_type: 'ridge' або 'random_forest'.
        target_name: Назва цільового показника.
        model_name: Опціональна назва для відображення.

    Returns:
        Об'єкт RegressionResult з метриками та навченою моделлю.
    """
    # Навчання строго на train
    pipeline.fit(X_train, y_train)

    # Прогноз на test
    y_pred = pipeline.predict(X_test)

    # Розрахунок метрик
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    metrics = RegressionMetrics(mae=mae, rmse=rmse, r2=r2)

    display_name = (
        model_name
        or ("Гребенева регресія (Ridge)" if model_type == "ridge" else "Випадковий ліс (Random Forest Regressor)")
    )
    interpretation, limitations = get_regression_docs(model_type)

    return RegressionResult(
        model_name=display_name,
        model_type=model_type,
        pipeline=pipeline,
        metrics=metrics,
        y_test=y_test,
        y_pred=y_pred,
        target_name=target_name,
        interpretation=interpretation,
        limitations=limitations,
    )


def compare_regression_models(
    results: Dict[str, RegressionResult]
) -> pd.DataFrame:
    """Формує порівняльну таблицю регресійних метрик для кількох моделей.

    Args:
        results: Словник результатів {model_name: RegressionResult}.

    Returns:
        DataFrame з колонками 'Модель', 'MAE', 'RMSE', 'R² (Коефіцієнт детермінації)'.
    """
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append(
            {
                "Модель": name,
                "MAE (Середня абс. помилка)": f"{m.mae:,.2f}",
                "RMSE (Корінь із середньокв. помилки)": f"{m.rmse:,.2f}",
                "R² (Якість апроксимації)": f"{m.r2:.3f}",
            }
        )
    return pd.DataFrame(rows)
