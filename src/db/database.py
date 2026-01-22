"""
Конфигурация подключения к базе данных.

Поддерживает PostgreSQL (production) и SQLite (development/testing).
"""

import os
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.utils.logger import logger
from src.db.models import Base


def get_database_url() -> str:
    """
    Получает URL подключения к БД из переменных окружения.

    Приоритет:
    1. DATABASE_URL (полный URL)
    2. Составной URL из POSTGRES_* переменных
    3. SQLite для локальной разработки

    Returns:
        Строка подключения к базе данных.
    """
    # Проверяем полный URL
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        # Heroku использует postgres://, SQLAlchemy требует postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        return database_url

    # Составляем URL из отдельных переменных
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'forecastly')
    db_user = os.getenv('POSTGRES_USER', 'forecastly')
    db_password = os.getenv('POSTGRES_PASSWORD', '')

    # Если есть пароль, используем PostgreSQL
    if db_password:
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Fallback на SQLite для локальной разработки
    sqlite_path = os.getenv('SQLITE_PATH', 'data/forecastly.db')
    logger.info(f"Используется SQLite: {sqlite_path}")
    return f"sqlite:///{sqlite_path}"


# Получаем URL базы данных
DATABASE_URL = get_database_url()

# Определяем, это SQLite или PostgreSQL
is_sqlite = DATABASE_URL.startswith('sqlite')

# Настройки engine
if is_sqlite:
    # SQLite требует особых настроек для многопоточности
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
    )

    # Включаем поддержку внешних ключей для SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Проверка соединения перед использованием
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
    )

# Фабрика сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency для FastAPI - создаёт сессию БД для каждого запроса.

    Использование:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

    Yields:
        Session: Сессия SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Контекстный менеджер для работы с БД вне FastAPI.

    Использование:
        with get_db_session() as db:
            db.query(Item).all()

    Yields:
        Session: Сессия SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Инициализирует базу данных, создаёт все таблицы.

    Безопасно вызывать повторно - существующие таблицы не затрагиваются.
    """
    logger.info(f"Инициализация базы данных: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Таблицы успешно созданы")


def drop_all_tables():
    """
    Удаляет все таблицы. ОСТОРОЖНО: уничтожает все данные!

    Использовать только для тестирования.
    """
    logger.warning("Удаление всех таблиц...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("✓ Все таблицы удалены")


def check_connection() -> bool:
    """
    Проверяет соединение с базой данных.

    Returns:
        True если соединение успешно, False иначе.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        return False
