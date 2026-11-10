"""Тести для модуля anomaly (виявлення аномалій та z-score пояснення)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.anomaly import compute_zscore_explanations, run_anomaly_detection
from src.plots import plot_pca_anomalies
from src.preprocessing import build_preprocessor


@pytest.fixture
def sample_anomaly_data():
    """Синтетичні дані з вбудованими явними аномаліями."""
    np.random.seed(42)
    n = 80
    # Звичайні спостереження
    area = np.random.normal(50, 5, size=n)
    cost = np.random.normal(1000, 100, size=n)
    category = np.random.choice(["A", "B"], size=n)

    # Додаємо 3 явні аномалії
    area[-1] = 300.0  # дуже велика площа
    cost[-2] = 20000.0  # аномальна вартість
    area[-3] = 400.0
    cost[-3] = 35000.0

    return pd.DataFrame({"area": area, "cost": cost, "category": category})


def test_isolation_forest_detection(sample_anomaly_data):
    """Перевірка виявлення аномалій за допомогою Isolation Forest."""
    df = sample_anomaly_data
    preprocessor = build_preprocessor(
        numeric_features=["area", "cost"],
        categorical_features=["category"],
        numeric_scaler="robust",
    )

    res = run_anomaly_detection(
        df=df,
        feature_cols=["area", "cost", "category"],
        preprocessor=preprocessor,
        model_type="isolation_forest",
        contamination=0.05,
        top_n=5,
        random_state=42,
    )

    assert res.n_samples == 80
    assert res.n_anomalies > 0
    assert len(res.top_anomalies_df) == 5
    assert res.pca_coords.shape == (80, 2)
    assert "anomaly_score" in res.enriched_df.columns
    assert "anomaly_reasons" in res.enriched_df.columns

    # Перевірка впорядкованості за спаданням балу аномальності
    scores = res.top_anomalies_df["anomaly_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_local_outlier_factor_detection(sample_anomaly_data):
    """Перевірка виявлення аномалій за допомогою Local Outlier Factor."""
    df = sample_anomaly_data
    preprocessor = build_preprocessor(
        numeric_features=["area", "cost"],
        categorical_features=["category"],
        numeric_scaler="robust",
    )

    res = run_anomaly_detection(
        df=df,
        feature_cols=["area", "cost", "category"],
        preprocessor=preprocessor,
        model_type="local_outlier_factor",
        contamination=0.05,
        n_neighbors=15,
        top_n=5,
        random_state=42,
    )

    assert res.n_samples == 80
    assert res.n_anomalies > 0
    assert len(res.top_anomalies_df) == 5


def test_zscore_explanations():
    """Перевірка генерації z-score пояснень для аномальних рядків."""
    df = pd.DataFrame(
        {
            "feat1": [10.0, 10.0, 10.0, 100.0],
            "feat2": [5.0, 5.0, 5.0, 5.0],
        }
    )
    is_anomaly = np.array([False, False, False, True])
    reasons = compute_zscore_explanations(df, is_anomaly, numeric_cols=["feat1", "feat2"])

    assert len(reasons) == 4
    assert reasons[0] == "Типове спостереження"
    # Для аномального рядка перевіряємо наявність z-score для feat1
    assert "feat1" in reasons[3]
    assert "z=" in reasons[3]


def test_pca_plot():
    """Перевірка генерації графіка PCA проекції аномалій."""
    pca_coords = np.random.randn(20, 2)
    is_anomaly = np.array([False] * 18 + [True, True])
    scores = np.random.rand(20)

    fig = plot_pca_anomalies(pca_coords, is_anomaly, scores, title="Test PCA")
    assert fig is not None
    plt.close(fig)
