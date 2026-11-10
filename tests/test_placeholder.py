"""Базовий тест структури проєкту."""
import src


def test_package_structure():
    """Перевірка завантаження кореневого пакету src."""
    assert src.__version__ == "0.1.0"
