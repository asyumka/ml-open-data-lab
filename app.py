"""Головний модуль веб-застосунку Streamlit: ML-лабораторія для відкритих даних."""
from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import streamlit as st

from src.data_loader import analyze_dataset, get_data_summary_table, load_dataset
from src.validation import validate_dataframe

st.set_page_config(
    page_title="ML-лабораторія для відкритих даних",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Ініціалізація стану сесії Streamlit."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None


def render_sidebar():
    """Відображає бічну панель із навігацією та завантаженням файлів."""
    st.sidebar.title("🔬 ML-Лабораторія")
    st.sidebar.caption("Навчальний проєкт (КАІ: «Python для ML»)")

    mode = st.sidebar.radio(
        "Оберіть режим роботи:",
        (
            "📊 Дані та огляд",
            "🎯 Класифікація",
            "📈 Регресія",
            "🔍 Виявлення аномалій",
        ),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Джерело даних")

    # Демо-датасет
    demo_path = os.path.join("data", "kyiv_district_housing.csv")
    if os.path.exists(demo_path):
        if st.sidebar.button("Завантажити демо-дані (Нерухомість Києва)", use_container_width=True):
            try:
                df = load_dataset(demo_path, file_type="csv", encoding="utf-8", delimiter=",")
                st.session_state.df = df
                st.session_state.dataset_info = analyze_dataset(df)
                st.session_state.dataset_name = "kyiv_district_housing.csv (Демо)"
                st.sidebar.success("Демо-дані завантажено!")
            except Exception as e:
                st.sidebar.error(f"Помилка завантаження демо-даних: {e}")

    uploaded_file = st.sidebar.file_uploader(
        "Або завантажте власний файл (CSV / XLSX):",
        type=["csv", "xlsx", "xls"],
        help="Підтримуються відкриті табличні дані з data.gov.ua або інших джерел.",
    )

    encoding = "utf-8"
    delimiter = "auto"

    if uploaded_file is not None and uploaded_file.name.lower().endswith(".csv"):
        st.sidebar.markdown("##### Параметри CSV:")
        encoding = st.sidebar.selectbox(
            "Кодування:",
            ["utf-8", "windows-1251"],
            index=0,
            help="windows-1251 часто використовується в українських реєстрах.",
        )
        delim_choice = st.sidebar.selectbox(
            "Роздільник:",
            ["Автовизначення", "Кома (,)", "Крапка з комою (;)", "Табуляція (Tab)", "Вертикальна риска (|)"],
            index=0,
        )
        delim_map = {
            "Автовизначення": "auto",
            "Кома (,)": ",",
            "Крапка з комою (;)": ";",
            "Табуляція (Tab)": "\t",
            "Вертикальна риска (|)": "|",
        }
        delimiter = delim_map[delim_choice]

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        try:
            bytes_data = uploaded_file.getvalue()
            buffer = io.BytesIO(bytes_data)
            df = load_dataset(buffer, file_type=file_ext, encoding=encoding, delimiter=delimiter)
            st.session_state.df = df
            st.session_state.dataset_info = analyze_dataset(df)
            st.session_state.dataset_name = uploaded_file.name
        except Exception as e:
            st.sidebar.error(f"Не вдалося зчитати файл: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 **Підказка:** Усі етапи підготовки використовують конвеєри "
        "`scikit-learn` без витоку інформації з тесту."
    )

    return mode


def render_data_page():
    """Відображає вкладку 'Дані та огляд'."""
    st.header("📊 Огляд та аналіз відкритих даних")

    df = st.session_state.df
    info = st.session_state.dataset_info
    dataset_name = st.session_state.dataset_name

    if df is None or info is None:
        st.info("👈 Завантажте файл (CSV/XLSX) через бічну панель або натисніть «Завантажити демо-дані».")

        st.markdown(
            """
            ### 📌 Вимоги до даних
            - **Формат:** CSV (кома, крапка з комою, табуляція) або XLSX.
            - **Кодування:** UTF-8 або Windows-1251 (поширене в українських реєстрах).
            - **Структура:** Таблиця з заголовками стовпців у першому рядку.
            - **Обсяг:** Рекомендовано від 30 рядків.
            - **Типи ознак:** Числові, текстові/категоріальні та дати розпізнаються автоматично.
            """
        )
        return

    # Валідація датасету
    is_valid, validation_msgs = validate_dataframe(df, min_rows=30)
    for msg in validation_msgs:
        if "Увага" in msg or "рекомендований мінімум" in msg:
            st.warning(f"⚠️ {msg}")
        elif "Стовпці, які повністю складаються" in msg:
            st.error(f"❌ {msg}")
        else:
            st.info(f"ℹ️ {msg}")

    # Метричні картки
    st.subheader(f"Джерело: `{dataset_name}`")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_missing = sum(info.missing_counts.values())
    total_cells = info.n_rows * info.n_cols
    missing_pct_overall = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    col1.metric("Рядків", f"{info.n_rows:,}")
    col2.metric("Стовпців", f"{info.n_cols}")
    col3.metric("Числових ознак", f"{len(info.numeric_columns)}")
    col4.metric("Категоріальних", f"{len(info.categorical_columns)}")
    col5.metric("Загалом пропусків", f"{total_missing:,} ({missing_pct_overall:.1f}%)")

    # Попередження про високу кардинальність
    if info.high_cardinality_columns:
        st.warning(
            f"⚠️ Виявлено високо-кардинальні категоріальні стовпці (>50 унікальних значень): "
            f"**{', '.join(info.high_cardinality_columns)}**. "
            "При навчанні моделей такі ознаки можуть суттєво роздути простір ознак (OneHot) або вимагатимуть виключення."
        )

    # Вкладки детального перегляду
    tab1, tab2, tab3 = st.tabs(["📋 Попередній перегляд (Preview)", "🔍 Структура та типи", "📈 Описова статистика"])

    with tab1:
        st.markdown("##### Перші рядки таблиці")
        n_preview = st.slider("Кількість рядків для перегляду:", min_value=5, max_value=50, value=10, step=5)
        st.dataframe(df.head(n_preview), use_container_width=True)

    with tab2:
        st.markdown("##### Автоматично визначені типи ознак та пропуски")
        summary_table = get_data_summary_table(df, info)
        st.dataframe(summary_table, use_container_width=True)

        if info.datetime_columns:
            st.success(
                f"🕒 Знайдено стовпці з датами: **{', '.join(info.datetime_columns)}**. "
                "Їх можна використовувати для хронологічного поділу на навчальну й тестову вибірки."
            )

    with tab3:
        st.markdown("##### Описова статистика `describe()`")
        desc_mode = st.radio(
            "Тип статистичного огляду:",
            ["Числові показники", "Категоріальні ознаки", "Усі стовпці"],
            horizontal=True,
        )
        if desc_mode == "Числові показники":
            if info.numeric_columns:
                st.dataframe(df[info.numeric_columns].describe(), use_container_width=True)
            else:
                st.info("У датасеті немає числових стовпців.")
        elif desc_mode == "Категоріальні ознаки":
            if info.categorical_columns:
                st.dataframe(df[info.categorical_columns].describe(include=["O", "category"]), use_container_width=True)
            else:
                st.info("У датасеті немає категоріальних стовпців.")
        else:
            st.dataframe(df.describe(include="all"), use_container_width=True)


def main():
    """Головна функція запуску застосунку."""
    init_session_state()
    mode = render_sidebar()

    if mode == "📊 Дані та огляд":
        render_data_page()
    elif mode == "🎯 Класифікація":
        st.header("🎯 Класифікація")
        st.info("Модуль класифікації буде реалізовано на Етапі 3.")
    elif mode == "📈 Регресія":
        st.header("📈 Регресія")
        st.info("Модуль регресії буде реалізовано на Етапі 4.")
    elif mode == "🔍 Виявлення аномалій":
        st.header("🔍 Виявлення аномалій")
        st.info("Модуль виявлення аномалій буде реалізовано на Етапі 5.")


if __name__ == "__main__":
    main()
