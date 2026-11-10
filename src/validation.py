"""Модуль валідації вхідних даних та генерації зрозумілих повідомлень про помилки українською мовою."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Спеціалізований виняток для помилок валідації даних користувача."""

    def __init__(self, message: str, details: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or []


def validate_dataframe(
    df: Optional[pd.DataFrame], min_rows: int = 30
) -> Tuple[bool, List[str]]:
    """Перевіряє базову коректність завантаженого датасету.

    Args:
        df: Завантажений DataFrame або None.
        min_rows: Мінімально допустима кількість рядків (за замовчуванням 30).

    Returns:
        Кортеж (is_valid, list_of_messages): прапорець валідності та список попереджень/помилок.
    """
    messages: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Файл не містить табличних даних або не може бути прочитаний."]

    if df.empty or len(df) == 0:
        return False, ["Таблиця порожня (0 рядків). Завантажте файл із даними."]

    if len(df.columns) == 0:
        return False, ["У таблиці відсутні стовпці."]

    if len(df) < min_rows:
        messages.append(
            f"Увага: у таблиці лише {len(df)} рядків (рекомендований мінімум — {min_rows}). "
            "Навчання моделей може бути нестабільним або неточним."
        )

    # Перевірка стовпців, що повністю складаються з NaN
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        messages.append(
            f"Стовпці, які повністю складаються з пропущених значень (100% NaN): {', '.join(map(str, all_nan_cols))}."
        )

    # Перевірка наявності дубльованих назв стовпців
    if len(df.columns) != len(set(df.columns)):
        duplicated = [col for col in df.columns if list(df.columns).count(col) > 1]
        messages.append(
            f"Знайдено дубльовані назви стовпців: {', '.join(set(duplicated))}. Рекомендується перейменувати їх."
        )

    return True, messages


def validate_features_and_target(
    features: List[str],
    target: Optional[str] = None,
    require_target: bool = True,
) -> Tuple[bool, List[str]]:
    """Перевіряє вибір ознак та цільової змінної.

    Args:
        features: Список назв обраних ознак (X).
        target: Назва цільової змінної (y) або None (для задач без вчителя).
        require_target: Чи обов'язкова наявність цілі.

    Returns:
        Кортеж (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not features or len(features) == 0:
        errors.append("Не обрано жодної ознаки для навчання моделі. Оберіть хоча б одну ознаку.")

    if require_target:
        if not target:
            errors.append("Не обрано цільову змінну. Будь ласка, оберіть стовпець-ціль.")
        elif target in features:
            errors.append(
                f"Цільова змінна '{target}' не повинна входити до списку ознак (це призведе до штучного витоку даних)."
            )

    return len(errors) == 0, errors


def validate_classification_target(
    y: pd.Series, min_samples_per_class: int = 2
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Перевіряє придатність цільової змінної для задачі класифікації.

    Args:
        y: Вектор цільової змінної.
        min_samples_per_class: Мінімальна кількість зразків на клас для можливості поділу/стратифікації.

    Returns:
        Кортеж (is_valid, errors_and_warnings, metadata).
    """
    messages: List[str] = []
    meta: Dict[str, Any] = {}

    y_clean = y.dropna()
    if len(y_clean) == 0:
        return False, ["Цільова змінна не містить жодного заповненого значення."], meta

    counts = y_clean.value_counts()
    n_classes = len(counts)
    meta["n_classes"] = n_classes
    meta["class_counts"] = counts.to_dict()

    if n_classes < 2:
        return (
            False,
            [
                f"У цільовій змінній виявлено лише один унікальний клас ({counts.index[0]}). "
                "Для класифікації необхідно щонайменше 2 класи."
            ],
            meta,
        )

    # Перевірка на надто рідкісні класи
    rare_classes = counts[counts < min_samples_per_class]
    if not rare_classes.empty:
        rare_info = ", ".join([f"'{cls}' ({cnt} зр.)" for cls, cnt in rare_classes.items()])
        return (
            False,
            [
                f"Виявлено класи з надто малою кількістю зразків (< {min_samples_per_class}): {rare_info}. "
                "Неможливо виконати стратифікований поділ на train/test. Об'єднайте або відфільтруйте рідкісні класи."
            ],
            meta,
        )

    # Оцінка дисбалансу класів
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    meta["imbalance_ratio"] = imbalance_ratio

    if imbalance_ratio > 3.0:
        messages.append(
            f"Попередження: виявлено помітний дисбаланс класів (співвідношення мажоритарного до міноритарного — "
            f"{imbalance_ratio:.1f}:1). Рекомендується увімкнути зважування класів (class_weight='balanced')."
        )

    return True, messages, meta


def validate_regression_target(
    y: pd.Series,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Перевіряє придатність цільової змінної для задачі регресії.

    Args:
        y: Вектор цільової змінної.

    Returns:
        Кортеж (is_valid, errors_and_warnings, metadata).
    """
    messages: List[str] = []
    meta: Dict[str, Any] = {}

    y_clean = y.dropna()
    if len(y_clean) == 0:
        return False, ["Цільова змінна не містить жодного заповненого значення."], meta

    # Перевірка на числовість
    if not pd.api.types.is_numeric_dtype(y_clean):
        # Спроба конвертації
        converted = pd.to_numeric(y_clean, errors="coerce")
        if converted.isna().sum() > 0.5 * len(y_clean):
            return (
                False,
                [
                    f"Цільова змінна '{y.name}' не є числовою і містить текстові значення. "
                    "Для регресії цільова змінна обов'язково має бути числовим показником (наприклад, ціна, площа, кількість)."
                ],
                meta,
            )
        else:
            messages.append(
                f"Стовпець '{y.name}' містив рядковий тип, але був успішно перетворений на числовий."
            )
            y_clean = converted.dropna()

    variance = float(y_clean.var())
    meta["variance"] = variance
    meta["std"] = float(y_clean.std())
    meta["mean"] = float(y_clean.mean())

    if variance == 0 or np.isnan(variance):
        return (
            False,
            [
                f"Цільова змінна '{y.name}' має нульову дисперсію (усі значення однакові: {y_clean.iloc[0]}). "
                "Регресія для константного значення неможлива."
            ],
            meta,
        )

    return True, messages, meta


def validate_clean_rows_count(
    n_rows: int, min_required: int = 30
) -> Tuple[bool, Optional[str]]:
    """Перевіряє кількість рядків після видалення/обробки пропусків у цілі."""
    if n_rows < min_required:
        return (
            False,
            f"Замало рядків після підготовки даних ({n_rows} рядків, мінімум {min_required}). "
            "Перевірте якість даних або оберіть інші ознаки.",
        )
    return True, None
