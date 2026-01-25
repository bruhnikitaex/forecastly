# Contributing to Forecastly

Спасибо за интерес к проекту Forecastly! Мы приветствуем вклад от сообщества.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

### Наши обязательства

Мы стремимся создать открытое и дружелюбное сообщество. Мы ожидаем от всех участников:

- ✅ Уважительного отношения к другим
- ✅ Конструктивной критики
- ✅ Фокуса на улучшении проекта
- ❌ Неприемлемого поведения (оскорбления, харассмент, дискриминация)

---

## Getting Started

### 1. Fork и Clone

```bash
# Fork репозитория через GitHub UI
# Затем клонируйте свой fork
git clone https://github.com/YOUR_USERNAME/forecastly.git
cd forecastly
```

### 2. Настройка окружения

```bash
# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установите зависимости для разработки
make install-dev

# Или вручную
pip install -e ".[dev]"
pre-commit install
```

### 3. Создайте feature branch

```bash
git checkout -b feature/your-feature-name
# или
git checkout -b fix/bug-description
```

---

## Development Workflow

### Шаги разработки

1. **Убедитесь, что ваш fork актуален**
   ```bash
   git remote add upstream https://github.com/bruhnikita/forecastly.git
   git fetch upstream
   git merge upstream/main
   ```

2. **Внесите изменения**
   - Пишите чистый, читаемый код
   - Следуйте стилю проекта
   - Добавляйте комментарии для сложной логики

3. **Запустите тесты**
   ```bash
   make test
   ```

4. **Проверьте качество кода**
   ```bash
   make check  # format + lint + test
   ```

5. **Commit изменения**
   ```bash
   git add .
   git commit -m "feat: add new forecasting feature"
   ```

### Commit Message Convention

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Новая функциональность
- `fix`: Исправление бага
- `docs`: Изменения в документации
- `style`: Форматирование (без изменения логики)
- `refactor`: Рефакторинг кода
- `test`: Добавление тестов
- `chore`: Обновление зависимостей, конфигурации

**Примеры:**
```bash
feat(api): add endpoint for SKU filtering
fix(models): correct Prophet seasonality parameter
docs(readme): update installation instructions
test(etl): add tests for data validation
```

---

## Coding Standards

### Python Code Style

Мы используем следующие инструменты:

- **Black** - форматирование кода (line length: 100)
- **isort** - сортировка импортов
- **flake8** - линтинг
- **mypy** - проверка типов

```bash
# Автоформатирование
make format

# Проверка
make lint
```

### Правила написания кода

1. **Именование**
   ```python
   # Хорошо
   def calculate_forecast_metrics(predictions: pd.DataFrame) -> dict:
       """Calculate MAPE and other metrics."""
       pass

   # Плохо
   def calc(df):
       pass
   ```

2. **Docstrings**
   ```python
   def train_model(data: pd.DataFrame, horizon: int = 14) -> Prophet:
       """
       Train Prophet model on historical data.

       Args:
           data: Historical sales data with 'ds' and 'y' columns
           horizon: Forecast horizon in days

       Returns:
           Trained Prophet model

       Raises:
           ValueError: If data is empty or invalid
       """
       pass
   ```

3. **Type Hints**
   ```python
   from typing import Optional, List, Dict

   def get_skus(db: Session, limit: int = 100) -> List[str]:
       """Get list of SKU IDs."""
       pass
   ```

4. **Обработка ошибок**
   ```python
   # Хорошо
   try:
       result = risky_operation()
   except SpecificException as e:
       logger.error(f"Operation failed: {e}")
       raise

   # Плохо
   try:
       result = risky_operation()
   except:
       pass
   ```

---

## Testing

### Написание тестов

Все новые функции должны иметь тесты.

```python
# tests/test_models.py

import pytest
from src.models.train_prophet import train_prophet_model

class TestProphetModel:
    """Tests for Prophet model training."""

    def test_train_with_valid_data(self):
        """Should successfully train with valid data."""
        # Arrange
        data = create_test_data()

        # Act
        model = train_prophet_model(data)

        # Assert
        assert model is not None
        assert hasattr(model, 'predict')

    def test_train_with_empty_data(self):
        """Should raise ValueError with empty data."""
        with pytest.raises(ValueError):
            train_prophet_model(pd.DataFrame())
```

### Запуск тестов

```bash
# Все тесты
make test

# Конкретный файл
pytest tests/test_models.py -v

# Конкретный тест
pytest tests/test_models.py::TestProphetModel::test_train_with_valid_data -v

# С покрытием
make test-cov
```

### Минимальное покрытие

Стремитесь к покрытию >80% для нового кода.

---

## Pull Request Process

### 1. Убедитесь, что все проверки проходят

```bash
make check  # format + lint + test
```

### 2. Обновите документацию

- Обновите README.md если изменился API
- Обновите docs/api.md для новых endpoints
- Добавьте docstrings к новым функциям

### 3. Создайте Pull Request

**Хороший PR:**
- ✅ Имеет четкое описание
- ✅ Ссылается на issue (если есть)
- ✅ Содержит тесты
- ✅ Проходит все CI/CD проверки
- ✅ Имеет осмысленные коммиты

**Шаблон описания PR:**

```markdown
## Описание
Краткое описание изменений

## Мотивация и контекст
Почему эти изменения необходимы? Какую проблему они решают?

Fixes #(issue)

## Типы изменений
- [ ] Bug fix (не ломает существующую функциональность)
- [ ] New feature (добавляет функциональность)
- [ ] Breaking change (изменения, которые ломают совместимость)
- [ ] Documentation update

## Чек-лист
- [ ] Код следует стилю проекта
- [ ] Добавлены тесты
- [ ] Все тесты проходят
- [ ] Обновлена документация
- [ ] CHANGELOG.md обновлен (если релевантно)

## Скриншоты (если применимо)
```

### 4. Code Review

- Будьте открыты к feedback
- Отвечайте на комментарии
- Вносите запрошенные изменения
- Будьте терпеливы - review может занять время

---

## Reporting Bugs

### Перед созданием bug report

1. Проверьте [существующие issues](https://github.com/bruhnikita/forecastly/issues)
2. Обновите зависимости до последних версий
3. Воспроизведите баг в чистом окружении

### Создание bug report

Используйте шаблон:

```markdown
**Описание бага**
Четкое описание что не работает

**Шаги для воспроизведения**
1. Перейти на '...'
2. Нажать на '....'
3. Скроллить до '....'
4. Ошибка появляется

**Ожидаемое поведение**
Что должно было произойти

**Скриншоты**
Если применимо

**Окружение:**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.11.5]
- Forecastly version: [e.g. 1.1.0]

**Логи**
```
Вставьте релевантные логи
```

**Дополнительный контекст**
```

---

## Suggesting Features

### Feature Request Template

```markdown
**Проблема**
Четкое описание проблемы, которую решает feature

**Предлагаемое решение**
Как вы видите реализацию

**Альтернативы**
Какие альтернативные решения вы рассматривали

**Дополнительный контекст**
Скриншоты, примеры из других проектов
```

---

## Questions?

Если у вас есть вопросы:

1. Проверьте [документацию](README.md)
2. Поищите в [Issues](https://github.com/bruhnikita/forecastly/issues)
3. Создайте новый issue с меткой `question`

---

Спасибо за ваш вклад! 🎉
