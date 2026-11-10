"""Головний модуль веб-застосунку Streamlit: ML-лабораторія для відкритих даних."""
from __future__ import annotations

import io
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.anomaly import AnomalyDetectionResult, run_anomaly_detection
from src.classification import (
    ClassificationResult,
    compare_classification_models,
    create_classifier_pipeline,
    evaluate_classifier_model,
)
from src.data_loader import analyze_dataset, get_data_summary_table, load_dataset
from src.plots import (
    plot_confusion_matrix,
    plot_pca_anomalies,
    plot_regression_predictions,
    plot_regression_residuals,
)
from src.preprocessing import build_preprocessor, split_data
from src.regression import (
    RegressionResult,
    compare_regression_models,
    create_regression_pipeline,
    evaluate_regression_model,
)
from src.validation import (
    validate_classification_target,
    validate_dataframe,
    validate_features_and_target,
    validate_regression_target,
)

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
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None
    if "regression_results" not in st.session_state:
        st.session_state.regression_results = None
    if "anomaly_results" not in st.session_state:
        st.session_state.anomaly_results = None


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

    is_valid, validation_msgs = validate_dataframe(df, min_rows=30)
    for msg in validation_msgs:
        if "Увага" in msg or "рекомендований мінімум" in msg:
            st.warning(f"⚠️ {msg}")
        elif "Стовпці, які повністю складаються" in msg:
            st.error(f"❌ {msg}")
        else:
            st.info(f"ℹ️ {msg}")

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

    if info.high_cardinality_columns:
        st.warning(
            f"⚠️ Виявлено високо-кардинальні категоріальні стовпці (>50 унікальних значень): "
            f"**{', '.join(info.high_cardinality_columns)}**. "
            "При OneHot-кодуванні такі ознаки можуть суттєво роздути розмірність."
        )

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


def render_classification_page():
    """Відображає сторінку навчання та оцінювання класифікації."""
    st.header("🎯 Класифікація відкритих даних")

    df = st.session_state.df
    info = st.session_state.dataset_info

    if df is None or info is None:
        st.warning("⚠️ Спочатку завантажте дані у вкладці «Дані та огляд» або оберіть демо-набір у бічному меню.")
        return

    st.markdown(
        "Налаштуйте вибір ознак, цільової змінної та параметрів моделювання. "
        "Усі перетворення виконуються всередині `sklearn.pipeline.Pipeline` без витоку даних."
    )

    candidate_targets = [
        c for c in info.columns
        if (c in info.categorical_columns) or (c in info.numeric_columns and info.unique_counts.get(c, 0) <= 20)
    ]
    if not candidate_targets:
        candidate_targets = info.columns

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        target_col = st.selectbox(
            "Оберіть цільову змінну (класи, які потрібно передбачити):",
            candidate_targets,
            index=0 if candidate_targets else None,
            help="Рекомендовано обирати стовпці з дискретними класами або категоріями.",
        )
    with col_t2:
        if target_col:
            st.caption(f"Унікальних класів у цілі: **{df[target_col].nunique(dropna=True)}**")

    if target_col:
        is_target_valid, target_msgs, meta = validate_classification_target(df[target_col])
        for msg in target_msgs:
            if not is_target_valid:
                st.error(f"❌ {msg}")
            elif "Попередження" in msg:
                st.warning(f"⚠️ {msg}")
        if not is_target_valid:
            return

    available_features = [c for c in info.columns if c != target_col]
    default_features = [
        c for c in available_features
        if c not in info.datetime_columns and c not in info.high_cardinality_columns
    ]

    selected_features = st.multiselect(
        "Оберіть ознаки (X) для навчання моделі:",
        available_features,
        default=default_features if default_features else available_features[:5],
        help="Ви можете комбінувати числові та категоріальні ознаки.",
    )

    is_feat_valid, feat_errors = validate_features_and_target(selected_features, target_col, require_target=True)
    if not is_feat_valid:
        for err in feat_errors:
            st.error(f"❌ {err}")
        return

    selected_high_cards = [c for c in selected_features if c in info.high_cardinality_columns]
    if selected_high_cards:
        st.warning(
            f"⚠️ Обрано високо-кардинальні ознаки: **{', '.join(selected_high_cards)}**. "
            "Це може створити велику кількість стовпців після OneHotEncoder."
        )

    st.subheader("⚙️ Параметри поділу вибірки та моделювання")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        test_size = st.slider(
            "Частка тестової вибірки (Test Size):",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Зазвичай 20-30% даних залишають для фінальної перевірки.",
        )

    with col_p2:
        date_options = ["Не використовувати (стратифікований поділ)"] + info.datetime_columns
        chosen_date_option = st.selectbox(
            "Хронологічний поділ за датою:",
            date_options,
            help="Поділ «минуле -> майбутнє» без shuffle для часових рядів/реєстрів.",
        )
        date_col = None if chosen_date_option.startswith("Не використовувати") else chosen_date_option

    with col_p3:
        model_choice = st.selectbox(
            "Модель класифікації:",
            [
                "Порівняти обидві (Logistic Regression vs Random Forest)",
                "Логістична регресія (Logistic Regression)",
                "Випадковий ліс (Random Forest Classifier)",
            ],
        )

    use_balanced = st.checkbox(
        "Враховувати дисбаланс класів (class_weight='balanced')",
        value=True,
        help="Призначає більшу вагу міноритарним (рідкісним) класам.",
    )

    if st.button("🚀 Навчити модель(і)", type="primary", use_container_width=True):
        with st.spinner("Триває підготовка конвеєра та навчання..."):
            num_feats = [c for c in selected_features if c in info.numeric_columns]
            cat_feats = [c for c in selected_features if c in info.categorical_columns]

            try:
                X_train, X_test, y_train, y_test, split_desc = split_data(
                    df=df,
                    feature_cols=selected_features,
                    target_col=target_col,
                    test_size=test_size,
                    date_col=date_col,
                    task_type="classification",
                    random_state=42,
                )

                st.success(f"✅ {split_desc}")

                models_to_run = []
                if "Логістична регресія" in model_choice:
                    models_to_run.append(("logistic_regression", "Логістична регресія"))
                elif "Випадковий ліс" in model_choice:
                    models_to_run.append(("random_forest", "Випадковий ліс (Random Forest)"))
                else:
                    models_to_run.append(("logistic_regression", "Логістична регресія"))
                    models_to_run.append(("random_forest", "Випадковий ліс (Random Forest)"))

                results: dict[str, ClassificationResult] = {}

                for m_type, m_name in models_to_run:
                    preprocessor = build_preprocessor(
                        numeric_features=num_feats,
                        categorical_features=cat_feats,
                        scale_numeric=True,
                        numeric_scaler="standard",
                        numeric_imputer="median",
                    )
                    pipeline = create_classifier_pipeline(
                        preprocessor=preprocessor,
                        model_type=m_type,
                        use_balanced_weights=use_balanced,
                        random_state=42,
                    )
                    res = evaluate_classifier_model(
                        pipeline=pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        model_type=m_type,
                        model_name=m_name,
                    )
                    results[m_name] = res

                st.session_state.classification_results = results

            except Exception as e:
                st.error(f"❌ Виникла помилка під час навчання моделі: {e}")
                return

    if st.session_state.classification_results is not None:
        results = st.session_state.classification_results
        st.markdown("---")
        st.subheader("📊 Результати оцінювання на тестовій вибірці")

        if len(results) > 1:
            st.markdown("##### 🏆 Порівняння моделей")
            comp_df = compare_classification_models(results)
            st.dataframe(comp_df, use_container_width=True)

        for name, res in results.items():
            with st.expander(f"Детальні результати: {name}", expanded=True):
                m = res.metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy (Загальна точність)", f"{m.accuracy:.1%}")
                c2.metric("Precision (Macro)", f"{m.precision_macro:.3f}")
                c3.metric("Recall (Macro)", f"{m.recall_macro:.3f}")
                c4.metric("F1-Score (Macro)", f"{m.f1_macro:.3f}")

                g_col1, g_col2 = st.columns([1, 1])

                with g_col1:
                    st.markdown("##### Матриця невідповідностей (Confusion Matrix)")
                    fig_cm = plot_confusion_matrix(m.confusion_matrix, res.classes, title=f"Confusion Matrix: {name}")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with g_col2:
                    st.markdown("##### Звіт класифікації (Classification Report)")
                    st.dataframe(m.report_df, use_container_width=True)

                st.markdown("---")
                int_col, lim_col = st.columns(2)
                with int_col:
                    st.info(f"💡 **Як інтерпретувати результати:**\n\n{res.interpretation}")
                with lim_col:
                    st.warning(f"⚠️ **Обмеження методу:**\n\n{res.limitations}")


def render_regression_page():
    """Відображає сторінку навчання та оцінювання регресії."""
    st.header("📈 Регресія відкритих даних")

    df = st.session_state.df
    info = st.session_state.dataset_info

    if df is None or info is None:
        st.warning("⚠️ Спочатку завантажте дані у вкладці «Дані та огляд» або оберіть демо-набір у бічному меню.")
        return

    st.markdown(
        "Прогнозування безперервних числових показників (наприклад, цін, площ, бюджетних витрат). "
        "Усі етапи масштабування та кодування захищені від витоку даних завдяки `Pipeline`."
    )

    candidate_targets = info.numeric_columns
    if not candidate_targets:
        st.error("❌ У завантаженому наборі даних не виявлено числових стовпців для регресії.")
        return

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        default_idx = 0
        for i, c in enumerate(candidate_targets):
            if any(kw in c.lower() for kw in ["price", "ціна", "сума", "cost", "target", "val"]):
                default_idx = i
                break

        target_col = st.selectbox(
            "Оберіть числову цільову змінну (y):",
            candidate_targets,
            index=default_idx,
            help="Стовпець обов'язково має бути числовим із ненульовою дисперсією.",
        )
    with col_t2:
        if target_col:
            st.caption(
                f"Діапазон: `[{df[target_col].min():,.2f} ... {df[target_col].max():,.2f}]`, "
                f"Сер.: `{df[target_col].mean():,.2f}`"
            )

    if target_col:
        is_target_valid, target_msgs, meta = validate_regression_target(df[target_col])
        for msg in target_msgs:
            if not is_target_valid:
                st.error(f"❌ {msg}")
            else:
                st.info(f"ℹ️ {msg}")
        if not is_target_valid:
            return

    available_features = [c for c in info.columns if c != target_col]
    default_features = [
        c for c in available_features
        if c not in info.datetime_columns and c not in info.high_cardinality_columns
    ]

    selected_features = st.multiselect(
        "Оберіть ознаки (X) для прогнозування:",
        available_features,
        default=default_features if default_features else available_features[:5],
        help="Можна комбінувати числові та категоріальні ознаки.",
    )

    is_feat_valid, feat_errors = validate_features_and_target(selected_features, target_col, require_target=True)
    if not is_feat_valid:
        for err in feat_errors:
            st.error(f"❌ {err}")
        return

    st.subheader("⚙️ Параметри поділу вибірки та регресійних моделей")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        test_size = st.slider(
            "Частка тестової вибірки (Test Size):",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="reg_test_size",
        )

    with col_p2:
        date_options = ["Не використовувати (випадковий поділ)"] + info.datetime_columns
        chosen_date_option = st.selectbox(
            "Хронологічний поділ за датою:",
            date_options,
            key="reg_date_split",
            help="Поділ без shuffle за часовою міткою.",
        )
        date_col = None if chosen_date_option.startswith("Не використовувати") else chosen_date_option

    with col_p3:
        model_choice = st.selectbox(
            "Модель регресії:",
            [
                "Порівняти обидві (Ridge vs Random Forest Regressor)",
                "Гребенева регресія (Ridge)",
                "Випадковий ліс (Random Forest Regressor)",
            ],
            key="reg_model_choice",
        )

    ridge_alpha = 1.0
    if "Ridge" in model_choice:
        ridge_alpha = st.slider(
            "Сила L2-регуляризації Ridge (alpha):",
            min_value=0.01,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="Більше значення = сильніший штраф на ваги коефіцієнтів.",
        )

    if st.button("🚀 Навчити регресійну модель(і)", type="primary", use_container_width=True):
        with st.spinner("Навчання конвеєра регресії..."):
            num_feats = [c for c in selected_features if c in info.numeric_columns]
            cat_feats = [c for c in selected_features if c in info.categorical_columns]

            try:
                X_train, X_test, y_train, y_test, split_desc = split_data(
                    df=df,
                    feature_cols=selected_features,
                    target_col=target_col,
                    test_size=test_size,
                    date_col=date_col,
                    task_type="regression",
                    random_state=42,
                )

                st.success(f"✅ {split_desc}")

                models_to_run = []
                if "Ridge" in model_choice and "Порівняти" not in model_choice:
                    models_to_run.append(("ridge", "Гребенева регресія (Ridge)"))
                elif "Random Forest" in model_choice and "Порівняти" not in model_choice:
                    models_to_run.append(("random_forest", "Випадковий ліс (Random Forest Regressor)"))
                else:
                    models_to_run.append(("ridge", "Гребенева регресія (Ridge)"))
                    models_to_run.append(("random_forest", "Випадковий ліс (Random Forest Regressor)"))

                results: dict[str, RegressionResult] = {}

                for m_type, m_name in models_to_run:
                    preprocessor = build_preprocessor(
                        numeric_features=num_feats,
                        categorical_features=cat_feats,
                        scale_numeric=True,
                        numeric_scaler="standard",
                        numeric_imputer="median",
                    )
                    pipeline = create_regression_pipeline(
                        preprocessor=preprocessor,
                        model_type=m_type,
                        alpha=ridge_alpha,
                        random_state=42,
                    )
                    res = evaluate_regression_model(
                        pipeline=pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        model_type=m_type,
                        target_name=target_col,
                        model_name=m_name,
                    )
                    results[m_name] = res

                st.session_state.regression_results = results

            except Exception as e:
                st.error(f"❌ Помилка під час навчання регресійної моделі: {e}")
                return

    if st.session_state.regression_results is not None:
        results = st.session_state.regression_results
        st.markdown("---")
        st.subheader("📊 Результати оцінювання регресії на тестовій вибірці")

        if len(results) > 1:
            st.markdown("##### 🏆 Порівняльна таблиця регресійних моделей")
            comp_df = compare_regression_models(results)
            st.dataframe(comp_df, use_container_width=True)

        for name, res in results.items():
            with st.expander(f"Детальні результати: {name}", expanded=True):
                m = res.metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("R² (Коефіцієнт детермінації)", f"{m.r2:.3f}")
                c2.metric("MAE (Середня абс. помилка)", f"{m.mae:,.2f}")
                c3.metric("RMSE (Корінь із середньокв. помилки)", f"{m.rmse:,.2f}")

                g_col1, g_col2 = st.columns(2)

                with g_col1:
                    st.markdown("##### Графік «Факт vs Прогноз»")
                    fig_pvf = plot_regression_predictions(
                        y_true=res.y_test,
                        y_pred=res.y_pred,
                        title=f"Факт vs Прогноз: {name}",
                        target_name=res.target_name,
                    )
                    st.pyplot(fig_pvf)
                    plt.close(fig_pvf)

                with g_col2:
                    st.markdown("##### Розподіл залишків (Помилок)")
                    fig_res = plot_regression_residuals(
                        y_true=res.y_test,
                        y_pred=res.y_pred,
                        title=f"Залишки: {name}",
                    )
                    st.pyplot(fig_res)
                    plt.close(fig_res)

                st.markdown("---")
                int_col, lim_col = st.columns(2)
                with int_col:
                    st.info(f"💡 **Як інтерпретувати результати:**\n\n{res.interpretation}")
                with lim_col:
                    st.warning(f"⚠️ **Обмеження методу:**\n\n{res.limitations}")


def render_anomaly_page():
    """Відображає сторінку виявлення аномалій (Isolation Forest / LOF)."""
    st.header("🔍 Виявлення аномалій у відкритих даних")

    df = st.session_state.df
    info = st.session_state.dataset_info

    if df is None or info is None:
        st.warning("⚠️ Спочатку завантажте дані у вкладці «Дані та огляд» або оберіть демо-набір у бічному меню.")
        return

    st.markdown(
        "Автоматичний пошук нетипових спостережень, підозрілих витрат або рідкісних випадків (без вчителя). "
        "Включає розрахунок балу аномальності, проекцію в 2D PCA простір та z-score аналіз відхилень."
    )

    # 1. Вибір ознак
    available_features = [
        c for c in info.columns
        if c not in info.datetime_columns and c not in info.high_cardinality_columns
    ]

    selected_features = st.multiselect(
        "Оберіть ознаки для аналізу аномалій:",
        info.columns,
        default=available_features if available_features else info.columns[:5],
        help="Рекомендовано обирати числові та важливі категоріальні показники.",
    )

    if not selected_features:
        st.error("❌ Оберіть хоча б одну ознаку для виявлення аномалій.")
        return

    # 2. Налаштування параметрів
    st.subheader("⚙️ Параметри методів виявлення аномалій")
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        model_choice = st.selectbox(
            "Алгоритм пошуку аномалій:",
            [
                "Ізоляційний ліс (Isolation Forest)",
                "Локальний фактор викиду (Local Outlier Factor)",
            ],
        )

    with col_a2:
        contamination_pct = st.slider(
            "Очікувана частка аномалій (% Contamination):",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Частка найбільш віддалених записів, які алгоритм вважатиме аномальними.",
        )
        contamination = contamination_pct / 100.0

    with col_a3:
        top_n = st.slider(
            "Кількість аномалій для детального перегляду (Топ-N):",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

    n_neighbors = 20
    if "Local Outlier Factor" in model_choice:
        n_neighbors = st.slider(
            "Кількість сусідів для LOF (n_neighbors):",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Розмір локального околу для розрахунку щільності.",
        )

    # 3. Запуск аналізу
    if st.button("🚀 Виявити аномалії", type="primary", use_container_width=True):
        with st.spinner("Триває розрахунок аномалій, PCA проекції та z-score відхилень..."):
            m_type = "isolation_forest" if "Isolation Forest" in model_choice else "local_outlier_factor"
            num_feats = [c for c in selected_features if c in info.numeric_columns]
            cat_feats = [c for c in selected_features if c in info.categorical_columns]

            preprocessor = build_preprocessor(
                numeric_features=num_feats,
                categorical_features=cat_feats,
                scale_numeric=True,
                numeric_scaler="robust",  # RobustScaler стійкіший до викидів при пошуку аномалій
                numeric_imputer="median",
            )

            try:
                res = run_anomaly_detection(
                    df=df,
                    feature_cols=selected_features,
                    preprocessor=preprocessor,
                    model_type=m_type,
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    top_n=top_n,
                    random_state=42,
                )
                st.session_state.anomaly_results = res
            except Exception as e:
                st.error(f"❌ Помилка під час пошуку аномалій: {e}")
                return

    # 4. Відображення результатів
    if st.session_state.anomaly_results is not None:
        res: AnomalyDetectionResult = st.session_state.anomaly_results
        st.markdown("---")
        st.subheader(f"📊 Результати виявлення аномалій: {res.model_name}")

        # ЖОРСТКА ВИМОГА: НЕ показувати accuracy!
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всього записів у датасеті", f"{res.n_samples:,}")
        c2.metric("Виявлено аномалій (шт)", f"{res.n_anomalies:,}")
        c3.metric("Частка аномалій у вибірці", f"{res.anomaly_percentage:.1f}%")
        max_score = float(np.max(res.anomaly_scores)) if len(res.anomaly_scores) > 0 else 0.0
        c4.metric("Максимальний бал аномальності", f"{max_score:.3f}")

        # Графік PCA та Топ аномалій
        g_col1, g_col2 = st.columns([1, 1])

        with g_col1:
            st.markdown("##### 2D PCA проекція простору ознак")
            fig_pca = plot_pca_anomalies(
                pca_coords=res.pca_coords,
                is_anomaly=res.is_anomaly_mask,
                anomaly_scores=res.anomaly_scores,
                title=f"PCA проекція ({res.model_name})",
            )
            st.pyplot(fig_pca)
            plt.close(fig_pca)

        with g_col2:
            st.markdown(f"##### Топ-{len(res.top_anomalies_df)} найбільш нетипових записів")
            st.caption("Відсортовано за балом аномальності спадно. Стовпець `anomaly_reasons` містить z-score відхилення.")
            # Показуємо зручний зріз
            preview_cols = [c for c in ["anomaly_score", "anomaly_reasons"] if c in res.top_anomalies_df.columns]
            other_cols = [c for c in res.top_anomalies_df.columns if c not in ["is_anomaly", "anomaly_score", "anomaly_reasons"]]
            show_df = res.top_anomalies_df[preview_cols + other_cols]
            st.dataframe(show_df, use_container_width=True)

        st.markdown("##### Повний збагачений набір даних")
        with st.expander("Переглянути всю таблицю з доданими мітками `is_anomaly` та `anomaly_score`"):
            st.dataframe(res.enriched_df, use_container_width=True)

        st.markdown("---")
        int_col, lim_col = st.columns(2)
        with int_col:
            st.info(f"💡 **Як інтерпретувати результати:**\n\n{res.interpretation}")
        with lim_col:
            st.warning(f"⚠️ **Обмеження методу:**\n\n{res.limitations}")


def main():
    """Головна функція запуску застосунку."""
    init_session_state()
    mode = render_sidebar()

    if mode == "📊 Дані та огляд":
        render_data_page()
    elif mode == "🎯 Класифікація":
        render_classification_page()
    elif mode == "📈 Регресія":
        render_regression_page()
    elif mode == "🔍 Виявлення аномалій":
        render_anomaly_page()


if __name__ == "__main__":
    main()
