"""Головний модуль веб-застосунку Streamlit: ML-лабораторія для відкритих даних."""
import streamlit as st

st.set_page_config(
    page_title="ML-лабораторія для відкритих даних",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Точка входу в застосунок."""
    st.title("🔬 ML-лабораторія для відкритих даних")
    st.caption("Навчальний проєкт (КАІ, курс «Python для машинного навчання»)")
    st.info("Ласкаво просимо! Завантажте табличні дані для початку роботи.")


if __name__ == "__main__":
    main()
