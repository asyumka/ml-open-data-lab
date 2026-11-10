@echo off
chcp 65001 > nul
echo ========================================================
echo   ML-лабораторія для відкритих даних (КАІ)
echo ========================================================
echo.

IF NOT EXIST ".venv" (
    echo [1/3] Створення віртуального середовища .venv...
    python -m venv .venv
    IF ERRORLEVEL 1 (
        echo [ПОМИЛКА] Python не знайдено! Встановіть Python 3.9+ та додайте його в PATH.
        pause
        exit /b 1
    )
    echo [2/3] Встановлення залежностей із requirements.txt...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo [ПОМИЛКА] Не вдалося встановити бібліотеки.
        pause
        exit /b 1
    )
) ELSE (
    call .venv\Scripts\activate.bat
)

echo [3/3] Запуск Streamlit застосунку...
streamlit run app.py

pause
