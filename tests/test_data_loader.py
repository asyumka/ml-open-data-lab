"""Тести для модуля data_loader."""
import io
import pandas as pd
import pytest

from src.data_loader import analyze_dataset, detect_csv_delimiter, get_data_summary_table, load_dataset


def test_detect_csv_delimiter():
    """Тестування автовизначення роздільників."""
    comma_csv = "col1,col2,col3\n1,2,3\n4,5,6"
    assert detect_csv_delimiter(comma_csv) == ","

    semicolon_csv = "colA;colB;colC\n10;20;30\n40;50;60"
    assert detect_csv_delimiter(semicolon_csv) == ";"

    tab_csv = "id\tname\tscore\n1\tA\t100\n2\tB\t200"
    assert detect_csv_delimiter(tab_csv) == "\t"


def test_load_csv_with_autodetection():
    """Тестування завантаження CSV з автодетекцією роздільника."""
    csv_content = "name;age;salary\nІван;30;25000\nОлена;25;32000"
    buf = io.BytesIO(csv_content.encode("utf-8"))
    df = load_dataset(buf, file_type="csv", delimiter="auto", encoding="utf-8")
    assert df.shape == (2, 3)
    assert list(df.columns) == ["name", "age", "salary"]


def test_load_csv_windows_1251():
    """Тестування підтримки кодування windows-1251."""
    text = "Місто,Населення\nКиїв,3000000\nЛьвів,720000"
    buf = io.BytesIO(text.encode("windows-1251"))
    df = load_dataset(buf, file_type="csv", encoding="windows-1251", delimiter=",")
    assert df.shape == (2, 2)
    assert df.iloc[0]["Місто"] == "Київ"


def test_analyze_dataset_types():
    """Тестування автоматичної ідентифікації числових, категоріальних та дат."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "price": [10.5, 20.0, 15.2, 30.1, 22.0],
            "status": ["активно", "архів", "активно", "новий", "архів"],
            "created_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "high_card": [f"item_{i}" for i in range(5)],
        }
    )
    info = analyze_dataset(df)
    assert "price" in info.numeric_columns
    assert "id" in info.numeric_columns
    assert "status" in info.categorical_columns
    assert "created_date" in info.datetime_columns
    assert info.n_rows == 5
    assert info.n_cols == 5

    # Перевірка зведеної таблиці
    summary = get_data_summary_table(df, info)
    assert len(summary) == 5
    assert "Стовпець" in summary.columns
