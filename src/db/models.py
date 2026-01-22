"""
SQLAlchemy модели для Forecastly.

Определяет структуру таблиц базы данных:
- SKU: товары
- Prediction: прогнозы продаж
- ForecastRun: история запусков прогнозирования
- Metric: метрики качества моделей
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date,
    ForeignKey, Boolean, Text, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class SKU(Base):
    """
    Модель товара (SKU - Stock Keeping Unit).

    Attributes:
        id: Первичный ключ
        sku_id: Уникальный идентификатор товара (SKU001, SKU002, ...)
        name: Название товара
        category: Категория товара
        store_id: Идентификатор магазина
        is_active: Флаг активности
        created_at: Дата создания записи
        updated_at: Дата обновления записи
    """
    __tablename__ = 'skus'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    category = Column(String(100), nullable=True)
    store_id = Column(String(50), default='default')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Отношения
    predictions = relationship('Prediction', back_populates='sku', cascade='all, delete-orphan')
    metrics = relationship('Metric', back_populates='sku', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<SKU(sku_id='{self.sku_id}', name='{self.name}')>"


class ForecastRun(Base):
    """
    История запусков прогнозирования.

    Каждый запуск predict.py создаёт новую запись.

    Attributes:
        id: Первичный ключ
        run_id: Уникальный идентификатор запуска (UUID)
        horizon: Горизонт прогноза в днях
        model_type: Тип модели (prophet, xgboost, ensemble)
        status: Статус выполнения (running, completed, failed)
        started_at: Время начала
        completed_at: Время завершения
        error_message: Сообщение об ошибке (если failed)
        records_count: Количество сгенерированных прогнозов
    """
    __tablename__ = 'forecast_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False, index=True)
    horizon = Column(Integer, nullable=False, default=14)
    model_type = Column(String(50), default='ensemble')
    status = Column(String(20), default='running')  # running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    records_count = Column(Integer, default=0)

    # Отношения
    predictions = relationship('Prediction', back_populates='forecast_run', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<ForecastRun(run_id='{self.run_id}', status='{self.status}')>"


class Prediction(Base):
    """
    Прогноз продаж.

    Содержит прогнозные значения от разных моделей для конкретного SKU и даты.

    Attributes:
        id: Первичный ключ
        sku_id: Внешний ключ на SKU
        forecast_run_id: Внешний ключ на ForecastRun
        date: Дата прогноза
        prophet: Прогноз модели Prophet
        xgb: Прогноз модели XGBoost
        ensemble: Ансамблевый прогноз (среднее)
        p_low: Нижняя граница доверительного интервала
        p_high: Верхняя граница доверительного интервала
        actual: Фактическое значение (если известно)
        created_at: Дата создания записи
    """
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(Integer, ForeignKey('skus.id'), nullable=False, index=True)
    forecast_run_id = Column(Integer, ForeignKey('forecast_runs.id'), nullable=True)
    date = Column(Date, nullable=False, index=True)
    prophet = Column(Float, nullable=True)
    xgb = Column(Float, nullable=True)
    ensemble = Column(Float, nullable=True)
    p_low = Column(Float, nullable=True)
    p_high = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)  # Фактическое значение для сравнения
    created_at = Column(DateTime, default=datetime.utcnow)

    # Отношения
    sku = relationship('SKU', back_populates='predictions')
    forecast_run = relationship('ForecastRun', back_populates='predictions')

    # Уникальное ограничение: один прогноз на SKU+дату в рамках одного запуска
    __table_args__ = (
        UniqueConstraint('sku_id', 'date', 'forecast_run_id', name='uq_prediction_sku_date_run'),
        Index('ix_predictions_sku_date', 'sku_id', 'date'),
    )

    def __repr__(self):
        return f"<Prediction(sku_id={self.sku_id}, date='{self.date}', ensemble={self.ensemble})>"


class Metric(Base):
    """
    Метрики качества прогнозирования.

    Содержит MAPE и другие метрики для каждого SKU и модели.

    Attributes:
        id: Первичный ключ
        sku_id: Внешний ключ на SKU
        forecast_run_id: Внешний ключ на ForecastRun (опционально)
        mape_prophet: MAPE для Prophet
        mape_xgboost: MAPE для XGBoost
        mape_ensemble: MAPE для ансамбля
        mape_naive: MAPE для наивного прогноза (бенчмарк)
        best_model: Лучшая модель по MAPE
        created_at: Дата создания записи
    """
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(Integer, ForeignKey('skus.id'), nullable=False, index=True)
    forecast_run_id = Column(Integer, ForeignKey('forecast_runs.id'), nullable=True)
    mape_prophet = Column(Float, nullable=True)
    mape_xgboost = Column(Float, nullable=True)
    mape_ensemble = Column(Float, nullable=True)
    mape_naive = Column(Float, nullable=True)
    best_model = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Отношения
    sku = relationship('SKU', back_populates='metrics')

    def __repr__(self):
        return f"<Metric(sku_id={self.sku_id}, best_model='{self.best_model}')>"


class SalesHistory(Base):
    """
    История продаж.

    Хранит исторические данные о продажах для обучения моделей.

    Attributes:
        id: Первичный ключ
        sku_id: Внешний ключ на SKU
        date: Дата продажи
        units: Количество проданных единиц
        revenue: Выручка
        price: Цена за единицу
        promo_flag: Флаг акции
        created_at: Дата загрузки данных
    """
    __tablename__ = 'sales_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(Integer, ForeignKey('skus.id'), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    units = Column(Float, nullable=False, default=0)
    revenue = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    promo_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Уникальное ограничение
    __table_args__ = (
        UniqueConstraint('sku_id', 'date', name='uq_sales_sku_date'),
        Index('ix_sales_sku_date', 'sku_id', 'date'),
    )

    def __repr__(self):
        return f"<SalesHistory(sku_id={self.sku_id}, date='{self.date}', units={self.units})>"


class User(Base):
    """
    Пользователь системы.

    Attributes:
        id: Первичный ключ
        email: Email (уникальный, используется для логина)
        username: Имя пользователя
        hashed_password: Хэш пароля (bcrypt)
        is_active: Активен ли аккаунт
        is_superuser: Является ли суперпользователем
        role: Роль пользователя (admin, analyst, viewer)
        company: Название компании
        created_at: Дата регистрации
        last_login: Последний вход
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    role = Column(String(20), default='viewer')  # admin, analyst, viewer
    company = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Отношения
    api_keys = relationship('APIKey', back_populates='user', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role}')>"


class APIKey(Base):
    """
    API ключ для программного доступа.

    Attributes:
        id: Первичный ключ
        key: Хэш API ключа (сам ключ показывается только при создании)
        key_prefix: Первые 8 символов ключа для идентификации
        name: Название ключа (для удобства пользователя)
        user_id: Владелец ключа
        is_active: Активен ли ключ
        expires_at: Дата истечения (опционально)
        last_used_at: Последнее использование
        created_at: Дата создания
        permissions: JSON с разрешениями (read, write, admin)
    """
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(8), nullable=False)  # Первые 8 символов для идентификации
    name = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    permissions = Column(Text, default='["read"]')  # JSON array: read, write, admin

    # Отношения
    user = relationship('User', back_populates='api_keys')

    def __repr__(self):
        return f"<APIKey(prefix='{self.key_prefix}...', name='{self.name}')>"


class RefreshToken(Base):
    """
    Refresh токен для обновления JWT.

    Attributes:
        id: Первичный ключ
        token: Хэш refresh токена
        user_id: Владелец токена
        expires_at: Дата истечения
        is_revoked: Отозван ли токен
        created_at: Дата создания
    """
    __tablename__ = 'refresh_tokens'

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RefreshToken(user_id={self.user_id}, revoked={self.is_revoked})>"
