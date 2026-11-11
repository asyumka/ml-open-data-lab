"""Тести нових можливостей: пропущені рядки, ID-стовпці, вибірка, фільтр рідкісних класів."""
import io

import numpy as np
import pandas as pd

from src.data_loader import analyze_dataset, load_dataset, sample_dataframe
from src.preprocessing import filter_rare_classes


def test_skipped_rows_are_reported():
    """Рядок із зайвим полем пропускається, але кількість пропущених фіксується."""
    text = "a;b\n1;2\n3;4;5\n6;7\n"
    df = load_dataset(io.BytesIO(text.encode("utf-8")), "csv", "utf-8", ";")
    assert len(df) == 2
    assert df.attrs["skipped_rows"] == 1


def test_clean_csv_has_no_skipped_rows():
    """Коректний файл не має пропущених рядків."""
    df = load_dataset(io.BytesIO(b"a,b\n1,2\n3,4\n"), "csv", "utf-8", ",")
    assert df.attrs["skipped_rows"] == 0


def test_id_like_columns_detected():
    """Унікальні цілі та коди розпізнаються як ідентифікатори, звичайні величини — ні."""
    n = 50
    df = pd.DataFrame(
        {
            "row_id": range(n),
            "region_code": np.random.RandomState(0).randint(1, 5, n),
            "area": np.random.RandomState(1).uniform(10, 100, n),
            "rooms": np.random.RandomState(2).randint(1, 5, n),
        }
    )
    info = analyze_dataset(df)
    assert "row_id" in info.id_like_columns
    assert "region_code" in info.id_like_columns
    assert "area" not in info.id_like_columns
    assert "rooms" not in info.id_like_columns


def test_sample_dataframe():
    """Вибірка відтворювана, нуль вимикає обмеження."""
    df = pd.DataFrame({"x": range(1000)})
    a = sample_dataframe(df, 100, random_state=42)
    b = sample_dataframe(df, 100, random_state=42)
    assert len(a) == 100 and a.equals(b)
    assert sample_dataframe(df, 0) is df
    assert sample_dataframe(df, 5000) is df


def test_filter_rare_classes():
    """Рідкісні класи відкидаються, фільтр повідомляє, скільки рядків прибрано."""
    df = pd.DataFrame({"y": ["A"] * 10 + ["B"] * 5 + ["C"], "x": range(16)})
    out, removed = filter_rare_classes(df, "y", 2)
    assert removed == {"C": 1}
    assert len(out) == 15 and "C" not in set(out["y"])
    same, none = filter_rare_classes(df, "y", 0)
    assert none == {} and len(same) == 16

def test_decimal_comma_text_becomes_numeric():
    """Текстовий стовпець із десятковою комою та пропусками стає числовим."""
    from src.data_loader import coerce_numeric_like_columns

    df = pd.DataFrame({"w": ["1250,5", "980", None, "1,25", "300"] * 10, "name": ["a", "b", "c", "d", "e"] * 10})
    out = coerce_numeric_like_columns(df)
    assert pd.api.types.is_numeric_dtype(out["w"])
    assert out["w"].iloc[0] == 1250.5
    assert not pd.api.types.is_numeric_dtype(out["name"])
    assert out.attrs["numeric_converted"] == ["w"]


def test_codes_with_leading_zeros_stay_text():
    """Коди з провідними нулями не перетворюються на числа."""
    from src.data_loader import coerce_numeric_like_columns

    df = pd.DataFrame({"code": ["0123", "0456", "0789"] * 10})
    assert not pd.api.types.is_numeric_dtype(coerce_numeric_like_columns(df)["code"])


def test_dayfirst_dates_detected_and_sorted():
    """Формат дд.мм.рррр розпізнається як дата й коректно сортується."""
    from src.data_loader import parse_dates_flexible

    s = pd.Series(["31.01.2021", "05.02.2021", "15.03.2021", "02.01.2021"] * 5)
    parsed = parse_dates_flexible(s)
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2021-01-31")
    info = analyze_dataset(pd.DataFrame({"d": s, "v": range(20)}))
    assert "d" in info.datetime_columns


def test_encoder_handles_mixed_types_in_categorical_column():
    """Змішані типи (рядки, числа, NaN) в одній категоріальній ознаці не ламають конвеєр."""
    from src.preprocessing import build_preprocessor

    df = pd.DataFrame({"c": ["a", 1, 2.5, None, "a", 1] * 5})
    out = build_preprocessor([], ["c"]).fit_transform(df)
    assert out.shape[0] == len(df)
