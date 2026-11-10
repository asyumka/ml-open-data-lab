"""Модуль побудови графіків та візуалізацій (matplotlib / seaborn) з коректними підписами та одиницями."""
from __future__ import annotations

from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Налаштування загального стилю
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Матриця невідповідностей (Confusion Matrix)",
) -> plt.Figure:
    """Будує матрицю невідповідностей із підписами класів та числовими значеннями.

    Args:
        cm: Матриця невідповідностей (2D numpy array).
        class_names: Список назв класів.
        title: Заголовок графіку.

    Returns:
        Об'єкт matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Прогнозований клас", fontsize=11, labelpad=8)
    ax.set_ylabel("Фактичний (істинний) клас", fontsize=11, labelpad=8)
    plt.tight_layout()
    return fig


def plot_regression_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = "Факт vs Прогноз (Регресія)",
    target_name: str = "Цільова змінна",
) -> plt.Figure:
    """Будує графік розсіювання 'Факт vs Прогноз' з ідеальною лінією тренду y=x.

    Args:
        y_true: Фактичні значення цілі.
        y_pred: Прогнозовані значення моделі.
        title: Заголовок графіку.
        target_name: Назва цільової змінної для підпису осей.

    Returns:
        Об'єкт matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    ax.scatter(y_t, y_p, color="#1f77b4", alpha=0.65, edgecolors="k", s=45, label="Спостереження (тест)")

    # Ідеальна лінія y = x
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    margin = (max_val - min_val) * 0.05
    line_range = np.linspace(min_val - margin, max_val + margin, 100)

    ax.plot(line_range, line_range, color="#d62728", linestyle="--", linewidth=2, label="Ідеальний прогноз (y = x)")

    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.set_xlabel(f"Фактичні значення ({target_name})", fontsize=11, labelpad=8)
    ax.set_ylabel(f"Прогнозовані значення ({target_name})", fontsize=11, labelpad=8)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    return fig


def plot_regression_residuals(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = "Розподіл залишків регресії (Факт - Прогноз)",
) -> plt.Figure:
    """Будує гістограму залишків (residuals) з лінією нульової помилки.

    Args:
        y_true: Фактичні значення.
        y_pred: Прогнозовані значення.
        title: Заголовок графіку.

    Returns:
        Об'єкт matplotlib Figure.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax, bins=15, edgecolor="black")

    ax.axvline(0, color="#d62728", linestyle="--", linewidth=2, label="Нульова помилка")
    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Величина залишку (y_true - y_pred)", fontsize=11, labelpad=8)
    ax.set_ylabel("Частота (кількість спостережень)", fontsize=11, labelpad=8)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    return fig


def plot_pca_anomalies(
    pca_coords: np.ndarray,
    is_anomaly: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    title: str = "2D PCA проекція виявлених аномалій",
) -> plt.Figure:
    """Будує 2D-графік простору головних компонент (PCA) із виділенням аномалій.

    Args:
        pca_coords: Масив розмірності (N, 2) координат перших двох головних компонент.
        is_anomaly: Булевий масив (True — аномалія, False — норма).
        anomaly_scores: Опціональний масив скорів аномалій.
        title: Заголовок графіку.

    Returns:
        Об'єкт matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    normal_mask = ~is_anomaly
    anomaly_mask = is_anomaly

    # Звичайні точки
    ax.scatter(
        pca_coords[normal_mask, 0],
        pca_coords[normal_mask, 1],
        c="#1f77b4",
        alpha=0.6,
        s=35,
        label=f"Типові записи ({np.sum(normal_mask)})",
        edgecolors="none",
    )

    # Аномалії
    ax.scatter(
        pca_coords[anomaly_mask, 0],
        pca_coords[anomaly_mask, 1],
        c="#d62728",
        alpha=0.9,
        s=70,
        marker="^",
        edgecolors="black",
        label=f"Аномалії ({np.sum(anomaly_mask)})",
    )

    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Перша головна компонента (PC 1)", fontsize=11, labelpad=8)
    ax.set_ylabel("Друга головна компонента (PC 2)", fontsize=11, labelpad=8)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    return fig
