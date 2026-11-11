"""Модуль побудови Pipeline та ColumnTransformer без витоку даних (Data Leakage Free)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
    numeric_scaler: str = "standard",
    numeric_imputer: str = "median",
) -> ColumnTransformer:
    """Створює ColumnTransformer для обробки числових і категоріальних ознак.

    Усі трансформації виконуються виключно в межах конвеєра, що запобігає витоку даних.

    Args:
        numeric_features: Список числових стовпців.
        categorical_features: Список категоріальних стовпців.
        scale_numeric: Чи масштабувати числові ознаки.
        numeric_scaler: 'standard' (StandardScaler) або 'robust' (RobustScaler).
        numeric_imputer: 'median', 'mean' або 'most_frequent'.

    Returns:
        Сконфігурований ColumnTransformer.
    """
    transformers = []

    # Числовий конвеєр: імпутація пропусків + масштабування
    if numeric_features:
        num_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]
        if scale_numeric:
            if numeric_scaler == "robust":
                num_steps.append(("scaler", RobustScaler()))
            else:
                num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numeric_features))

    # Категоріальний конвеєр: заповнення пропусків константою/модою + OneHotEncoder
    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def filter_rare_classes(
    df: pd.DataFrame, target_col: str, min_count: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Відкидає рядки, клас цілі яких зустрічається менше `min_count` разів.

    Фільтрація залежить лише від частот у цілі (не від тесту), тому не створює
    витоку; її слід застосовувати ДО поділу на train/test.

    Args:
        df: Вхідний DataFrame.
        target_col: Назва цільового стовпця.
        min_count: Мінімальна кількість записів у класі; значення ≤ 1 вимикає фільтр.

    Returns:
        Кортеж (відфільтрований DataFrame, словник {клас: кількість прибраних рядків}).
    """
    if min_count <= 1:
        return df, {}
    counts = df[target_col].value_counts()
    rare = counts[counts < min_count]
    if rare.empty:
        return df, {}
    filtered = df[~df[target_col].isin(rare.index)]
    return filtered, {str(k): int(v) for k, v in rare.items()}


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    date_col: Optional[str] = None,
    task_type: str = "classification",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Виконує поділ датасету на навчальну та тестову вибірки без витоку даних.

    Підтримує:
    1. Хронологічний поділ («минуле -> майбутнє» без перемішування), якщо вказано `date_col`.
    2. Стратифікований випадковий поділ (для класифікації).
    3. Звичайний випадковий поділ (для регресії або коли стратифікація неможлива).

    Args:
        df: Вхідний DataFrame.
        feature_cols: Список колонок-ознак.
        target_col: Назва колонки-цілі.
        test_size: Частка тестової вибірки (наприклад, 0.2).
        date_col: Стовпець із датою для хронологічного поділу (якщо вказано).
        task_type: 'classification' або 'regression'.
        random_state: Сід генератора псевдовипадкових чисел.

    Returns:
        Кортеж (X_train, X_test, y_train, y_test, split_description).
    """
    # Відкидаємо рядки з пропусками в цільовій змінній
    df_clean = df.dropna(subset=[target_col]).copy()

    # Якщо вказано колонку дати для хронологічного поділу
    if date_col and date_col in df_clean.columns:
        # Конвертуємо дату для правильного сортування
        df_clean["__parsed_date__"] = pd.to_datetime(df_clean[date_col], errors="coerce")
        df_clean = df_clean.sort_values(by="__parsed_date__").reset_index(drop=True)
        df_clean = df_clean.drop(columns=["__parsed_date__"])

        n_samples = len(df_clean)
        split_idx = int(n_samples * (1.0 - test_size))

        train_data = df_clean.iloc[:split_idx]
        test_data = df_clean.iloc[split_idx:]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        split_desc = (
            f"Хронологічний поділ за стовпцем '{date_col}' (без shuffle): "
            f"Train={len(X_train)} зр., Test={len(X_test)} зр."
        )
        return X_train, X_test, y_train, y_test, split_desc

    # Випадковий поділ
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    stratify_target = None
    if task_type == "classification":
        # Перевіряємо, чи всі класи мають щонайменше 2 зразки для стратифікації
        counts = y.value_counts()
        if (counts >= 2).all() and len(counts) > 1:
            stratify_target = y

    if stratify_target is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
        split_desc = (
            f"Випадковий стратифікований поділ (random_state={random_state}): "
            f"Train={len(X_train)} зр., Test={len(X_test)} зр."
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        split_desc = (
            f"Випадковий поділ без стратифікації (random_state={random_state}): "
            f"Train={len(X_train)} зр., Test={len(X_test)} зр."
        )

    return X_train, X_test, y_train, y_test, split_desc
