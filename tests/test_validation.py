"""Тести для модуля validation на чистих та 'брудних' даних."""
import numpy as np
import pandas as pd
import pytest

from src.validation import (
    validate_classification_target,
    validate_clean_rows_count,
    validate_dataframe,
    validate_features_and_target,
    validate_regression_target,
)


def test_validate_dataframe_empty_and_corrupt():
    """Перевірка обробки порожнього або пошкодженого датасету."""
    is_valid, msgs = validate_dataframe(None)
    assert not is_valid
    assert any("не містить табличних даних" in m for m in msgs)

    is_valid, msgs = validate_dataframe(pd.DataFrame())
    assert not is_valid
    assert any("порожня" in m for m in msgs)


def test_validate_dataframe_dirty_data():
    """Перевірка датасету з пропусками, 100% NaN колонками та малою кількістю рядків."""
    df_dirty = pd.DataFrame(
        {
            "valid_col": [1, 2, 3],
            "all_nan": [np.nan, np.nan, np.nan],
        }
    )
    is_valid, msgs = validate_dataframe(df_dirty, min_rows=30)
    assert is_valid  # базово валідний, але має попередження
    assert any("100% NaN" in m for m in msgs)
    assert any("лише 3 рядків" in m for m in msgs)


def test_validate_features_and_target():
    """Перевірка вибору ознак та цільової змінної."""
    # Не обрано ознак
    valid, errors = validate_features_and_target([], target="target")
    assert not valid
    assert any("Не обрано жодної ознаки" in e for e in errors)

    # Ціль входить до ознак
    valid, errors = validate_features_and_target(["col1", "target"], target="target")
    assert not valid
    assert any("не повинна входити до списку ознак" in e for e in errors)

    # Не обрано ціль
    valid, errors = validate_features_and_target(["col1", "col2"], target=None, require_target=True)
    assert not valid
    assert any("Не обрано цільову змінну" in e for e in errors)


def test_validate_classification_target():
    """Перевірка цілі класифікації: один клас, рідкісний клас, дисбаланс."""
    # Один клас
    y_single = pd.Series(["A", "A", "A", "A"])
    valid, msgs, _ = validate_classification_target(y_single)
    assert not valid
    assert any("лише один унікальний клас" in m for m in msgs)

    # Надто рідкісний клас (< 2 зразків)
    y_rare = pd.Series(["A", "A", "A", "A", "B"])
    valid, msgs, _ = validate_classification_target(y_rare, min_samples_per_class=2)
    assert not valid
    assert any("надто малою кількістю зразків" in m for m in msgs)

    # Дисбаланс класів
    y_imbalanced = pd.Series(["A"] * 20 + ["B"] * 4)
    valid, msgs, meta = validate_classification_target(y_imbalanced)
    assert valid
    assert any("дисбаланс класів" in m for m in msgs)
    assert meta["imbalance_ratio"] == 5.0


def test_validate_regression_target():
    """Перевірка цілі регресії: нечислова, нульова дисперсія."""
    # Текстова ціль
    y_str = pd.Series(["так", "ні", "можливо", "так"])
    valid, msgs, _ = validate_regression_target(y_str)
    assert not valid
    assert any("не є числовою" in m for m in msgs)

    # Нульова дисперсія
    y_const = pd.Series([42.0, 42.0, 42.0, 42.0])
    valid, msgs, _ = validate_regression_target(y_const)
    assert not valid
    assert any("нульову дисперсію" in m for m in msgs)

    # Валідна регресійна ціль
    y_valid = pd.Series([10.0, 20.0, 30.0, 40.0])
    valid, msgs, meta = validate_regression_target(y_valid)
    assert valid
    assert meta["variance"] > 0
