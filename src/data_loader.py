"""Модуль завантаження та первинного аналізу відкритих даних."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DatasetInfo:
    """Структурована інформація про завантажений датасет."""

    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    high_cardinality_columns: List[str]  # > 50 унікальних значень
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    unique_counts: Dict[str, int]
    column_dtypes: Dict[str, str]


def detect_csv_delimiter(sample_text: str) -> str:
    """Визначає роздільник у CSV за текстовим фрагментом.

    Args:
        sample_text: Перші кілька рядків файлу.

    Returns:
        Знайдений символ-роздільник (за замовчуванням ',').
    """
    candidates = [",", ";", "\t", "|"]
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=",;\t|")
        if dialect.delimiter in candidates:
            return dialect.delimiter
    except Exception:
        pass

    # Евристичний аналіз: рахуємо частоту роздільників у перших рядках
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:10]
    if not lines:
        return ","

    best_delim = ","
    max_score = -1.0

    for cand in candidates:
        counts = [ln.count(cand) for ln in lines]
        # Перевіряємо стабільність кількості роздільників у рядках
        if counts and counts[0] > 0 and all(c == counts[0] for c in counts):
            score = counts[0] * 10
        else:
            score = float(np.mean(counts)) if counts else 0.0

        if score > max_score:
            max_score = score
            best_delim = cand

    return best_delim if max_score > 0 else ","


def load_dataset(
    file_source: Union[str, BinaryIO, io.BytesIO],
    file_type: str = "csv",
    encoding: str = "utf-8",
    delimiter: str = "auto",
) -> pd.DataFrame:
    """Завантажує датасет із файлу або байтового потоку (CSV або XLSX).

    Args:
        file_source: Шлях до файлу або об'єкт файлового потоку (наприклад, з st.file_uploader).
        file_type: 'csv' або 'xlsx' (або 'xls').
        encoding: 'utf-8', 'windows-1251' тощо (для CSV).
        delimiter: 'auto', ',', ';', '\t', '|' (для CSV).

    Returns:
        Завантажений pandas.DataFrame.

    Raises:
        ValueError: При некоректному форматі або неможливості прочитати дані.
    """
    file_type = file_type.lower().strip().lstrip(".")

    try:
        if file_type in ["xlsx", "xls"]:
            df = pd.read_excel(file_source)
            return df

        # Для CSV:
        if delimiter == "auto":
            # Спроба зчитати перші байти для детекції
            if hasattr(file_source, "read") and hasattr(file_source, "seek"):
                sample_bytes = file_source.read(4096)
                file_source.seek(0)
                try:
                    sample_text = sample_bytes.decode(encoding, errors="ignore")
                except Exception:
                    sample_text = sample_bytes.decode("utf-8", errors="ignore")
                chosen_delim = detect_csv_delimiter(sample_text)
            elif isinstance(file_source, str):
                with open(file_source, "r", encoding=encoding, errors="ignore") as f:
                    sample_text = f.read(4096)
                chosen_delim = detect_csv_delimiter(sample_text)
            else:
                chosen_delim = ","
        else:
            chosen_delim = delimiter

        # Завантажуємо CSV
        if hasattr(file_source, "seek"):
            file_source.seek(0)

        df = pd.read_csv(
            file_source,
            sep=chosen_delim,
            encoding=encoding,
            on_bad_lines="skip",
        )
        return df

    except UnicodeDecodeError as err:
        raise ValueError(
            f"Помилка кодування файлу: {err}. Спробуйте змінити кодування на 'windows-1251' або 'utf-8'."
        ) from err
    except Exception as err:
        raise ValueError(f"Не вдалося завантажити файл: {err}") from err


def analyze_dataset(df: pd.DataFrame) -> DatasetInfo:
    """Виконує автоматичний аналіз та типізацію стовпців датасету без захардкоджених назв.

    Args:
        df: Вхідний DataFrame.

    Returns:
        Об'єкт DatasetInfo з метаінформацією про структуру та типи даних.
    """
    n_rows, n_cols = df.shape
    columns = list(df.columns)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    high_cardinality_cols: List[str] = []

    missing_counts: Dict[str, int] = {}
    missing_percentages: Dict[str, float] = {}
    unique_counts: Dict[str, int] = {}
    column_dtypes: Dict[str, str] = {}

    for col in columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows * 100), 2) if n_rows > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))

        missing_counts[str(col)] = missing_count
        missing_percentages[str(col)] = missing_pct
        unique_counts[str(col)] = n_unique

        # 1. Якщо тип вже datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(str(col))
            column_dtypes[str(col)] = "datetime"
            continue

        # 2. Якщо числовий тип
        if pd.api.types.is_numeric_dtype(series):
            # Якщо мало унікальних значень і це int/bool — може бути категорією або цільовою міткою
            numeric_cols.append(str(col))
            column_dtypes[str(col)] = "numeric"
            continue

        # 3. Перевірка об'єктних/рядкових колонок на дату
        sample_non_null = series.dropna().astype(str).head(50)
        is_date = False

        if not sample_non_null.empty and n_unique > 2:
            # Перевіряємо типові роздільники дати
            has_date_symbols = any(
                ("-" in val or "/" in val or "." in val) for val in sample_non_null
            )
            if has_date_symbols:
                try:
                    converted = pd.to_datetime(sample_non_null, errors="coerce")
                    if converted.notna().sum() >= 0.8 * len(sample_non_null):
                        is_date = True
                except Exception:
                    is_date = False

        if is_date:
            datetime_cols.append(str(col))
            column_dtypes[str(col)] = "datetime"
        else:
            categorical_cols.append(str(col))
            column_dtypes[str(col)] = "categorical"
            if n_unique > 50:
                high_cardinality_cols.append(str(col))

    return DatasetInfo(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=[str(c) for c in columns],
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        high_cardinality_columns=high_cardinality_cols,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        unique_counts=unique_counts,
        column_dtypes=column_dtypes,
    )


def get_data_summary_table(df: pd.DataFrame, info: DatasetInfo) -> pd.DataFrame:
    """Формує зведену таблицю характеристик колонок для відображення в UI.

    Args:
        df: Вхідний DataFrame.
        info: Метадані DatasetInfo.

    Returns:
        DataFrame з колонками ['Стовпець', 'Тип даних', 'Унікальних значень', 'Пропусків (шт)', 'Пропусків (%)', 'Приклад значень'].
    """
    rows = []
    for col in info.columns:
        series = df[col]
        sample_vals = series.dropna().unique()[:3]
        sample_str = ", ".join(map(str, sample_vals))
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."

        dtype_ukr = {
            "numeric": "Числовий",
            "categorical": "Категоріальний",
            "datetime": "Дата / Час",
        }.get(info.column_dtypes.get(col, ""), "Інший")

        is_high_card = col in info.high_cardinality_columns
        note = " ⚠️ (>50 значень)" if is_high_card else ""

        rows.append(
            {
                "Стовпець": col,
                "Тип ознаки": dtype_ukr + note,
                "Унікальних": info.unique_counts.get(col, 0),
                "Пропуски (шт)": info.missing_counts.get(col, 0),
                "Пропуски (%)": f"{info.missing_percentages.get(col, 0.0):.1f}%",
                "Приклади": sample_str if sample_str else "—",
            }
        )

    return pd.DataFrame(rows)
