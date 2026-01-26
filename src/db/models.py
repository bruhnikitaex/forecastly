"""
SQLAlchemy модели для Forecastly.

Определяет структуру таблиц базы данных:
- SKU: товары
- Prediction: прогнозы продаж
- ForecastRun: история запусков прогнозирования
- Metric: метрики качества моделей
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date,
    ForeignKey, Boolean, Text, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base


def utcnow():
    """Возвращает текущее время в UTC (timezone-aware)."""
    return datetime.now(timezone.utc)

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
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

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
    started_at = Column(DateTime, default=utcnow)
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
    created_at = Column(DateTime, default=utcnow)

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
    created_at = Column(DateTime, default=utcnow)

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
    created_at = Column(DateTime, default=utcnow)

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
        failed_login_attempts: Счётчик неудачных попыток входа
        locked_until: Время до которого аккаунт заблокирован
        last_failed_login: Время последней неудачной попытки
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
    created_at = Column(DateTime, default=utcnow)
    last_login = Column(DateTime, nullable=True)

    # Account lockout fields
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    last_failed_login = Column(DateTime, nullable=True)

    # Отношения
    api_keys = relationship('APIKey', back_populates='user', cascade='all, delete-orphan')

    @property
    def is_locked(self) -> bool:
        """Проверяет, заблокирован ли аккаунт."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

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
    created_at = Column(DateTime, default=utcnow)
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
    created_at = Column(DateTime, default=utcnow)

    def __repr__(self):
        return f"<RefreshToken(user_id={self.user_id}, revoked={self.is_revoked})>"


class SecurityAuditLog(Base):
    """
    Журнал событий безопасности.

    Хранит все важные события для аудита и анализа инцидентов.

    Attributes:
        id: Первичный ключ
        event_type: Тип события (login_success, login_failed, account_locked, etc.)
        user_id: ID пользователя (если применимо)
        user_email: Email пользователя (для поиска даже если пользователь удалён)
        ip_address: IP адрес клиента
        user_agent: User-Agent браузера/клиента
        details: JSON с дополнительными деталями
        severity: Уровень важности (info, warning, critical)
        created_at: Время события
    """
    __tablename__ = 'security_audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    user_email = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 может быть до 45 символов
    user_agent = Column(String(500), nullable=True)
    details = Column(Text, nullable=True)  # JSON
    severity = Column(String(20), default='info')  # info, warning, critical
    created_at = Column(DateTime, default=utcnow, index=True)

    # Индексы для быстрого поиска
    __table_args__ = (
        Index('ix_audit_event_created', 'event_type', 'created_at'),
        Index('ix_audit_user_created', 'user_id', 'created_at'),
        Index('ix_audit_severity_created', 'severity', 'created_at'),
    )

    def __repr__(self):
        return f"<SecurityAuditLog(event='{self.event_type}', user='{self.user_email}', severity='{self.severity}')>"


# Константы для типов событий
class AuditEventType:
    """Типы событий для аудита."""
    # Аутентификация
    LOGIN_SUCCESS = 'login_success'
    LOGIN_FAILED = 'login_failed'
    LOGOUT = 'logout'
    TOKEN_REFRESH = 'token_refresh'

    # Блокировка аккаунта
    ACCOUNT_LOCKED = 'account_locked'
    ACCOUNT_UNLOCKED = 'account_unlocked'

    # Регистрация и профиль
    USER_REGISTERED = 'user_registered'
    PASSWORD_CHANGED = 'password_changed'
    PASSWORD_CHANGE_FAILED = 'password_change_failed'
    PROFILE_UPDATED = 'profile_updated'

    # API ключи
    API_KEY_CREATED = 'api_key_created'
    API_KEY_DELETED = 'api_key_deleted'
    API_KEY_USED = 'api_key_used'

    # Администрирование
    USER_ROLE_CHANGED = 'user_role_changed'
    USER_DEACTIVATED = 'user_deactivated'
    USER_ACTIVATED = 'user_activated'

    # Безопасность
    SUSPICIOUS_ACTIVITY = 'suspicious_activity'
    RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded'

    # OAuth2
    OAUTH2_LOGIN = 'oauth2_login'
    OAUTH2_LINK = 'oauth2_link'
    OAUTH2_UNLINK = 'oauth2_unlink'

    # MFA
    MFA_ENABLED = 'mfa_enabled'
    MFA_DISABLED = 'mfa_disabled'
    MFA_VERIFIED = 'mfa_verified'
    MFA_FAILED = 'mfa_failed'
    MFA_BACKUP_CODE_USED = 'mfa_backup_code_used'


class Tenant(Base):
    """
    Тенант (организация/компания) для multi-tenancy.

    Attributes:
        id: Первичный ключ
        tenant_id: Уникальный идентификатор тенанта (UUID)
        name: Название организации
        slug: URL-friendly идентификатор
        plan: Тарифный план (free, starter, pro, enterprise)
        is_active: Активен ли тенант
        settings: JSON с настройками тенанта
        usage_limits: JSON с лимитами использования
        created_at: Дата создания
    """
    __tablename__ = 'tenants'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(36), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    plan = Column(String(50), default='free')  # free, starter, pro, enterprise
    is_active = Column(Boolean, default=True)
    settings = Column(Text, default='{}')  # JSON
    usage_limits = Column(Text, default='{}')  # JSON: {skus: 10, users: 1, api_calls: 1000}
    current_usage = Column(Text, default='{}')  # JSON: текущее использование
    billing_email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Отношения
    users = relationship('TenantUser', back_populates='tenant', cascade='all, delete-orphan')
    webhooks = relationship('Webhook', back_populates='tenant', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Tenant(name='{self.name}', plan='{self.plan}')>"


class TenantUser(Base):
    """
    Связь пользователя с тенантом.

    Пользователь может быть членом нескольких тенантов с разными ролями.
    """
    __tablename__ = 'tenant_users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    role = Column(String(20), default='member')  # owner, admin, member, viewer
    is_primary = Column(Boolean, default=False)  # Основной тенант пользователя
    created_at = Column(DateTime, default=utcnow)

    # Отношения
    tenant = relationship('Tenant', back_populates='users')

    __table_args__ = (
        UniqueConstraint('tenant_id', 'user_id', name='uq_tenant_user'),
    )


class OAuth2Connection(Base):
    """
    Связь аккаунта пользователя с OAuth2 провайдером.

    Позволяет входить через Google, Azure AD, GitHub и др.
    """
    __tablename__ = 'oauth2_connections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # google, azure, github
    provider_user_id = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    picture_url = Column(String(500), nullable=True)
    access_token = Column(Text, nullable=True)  # Encrypted
    refresh_token = Column(Text, nullable=True)  # Encrypted
    token_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint('provider', 'provider_user_id', name='uq_oauth2_provider_user'),
        Index('ix_oauth2_user_provider', 'user_id', 'provider'),
    )


class UserMFA(Base):
    """
    Настройки MFA для пользователя.
    """
    __tablename__ = 'user_mfa'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, nullable=False, index=True)
    is_enabled = Column(Boolean, default=False)
    secret = Column(String(255), nullable=True)  # TOTP secret (encrypted)
    backup_codes = Column(Text, nullable=True)  # JSON array (encrypted)
    enabled_at = Column(DateTime, nullable=True)
    last_verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Webhook(Base):
    """
    Webhook для уведомлений о событиях.
    """
    __tablename__ = 'webhooks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    webhook_id = Column(String(36), unique=True, nullable=False, index=True)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=True)  # For signature verification
    events = Column(Text, nullable=False)  # JSON array: ["forecast.completed", "model.trained"]
    is_active = Column(Boolean, default=True)
    headers = Column(Text, default='{}')  # JSON: custom headers
    retry_count = Column(Integer, default=3)
    timeout_seconds = Column(Integer, default=30)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Statistics
    last_triggered_at = Column(DateTime, nullable=True)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)

    # Отношения
    tenant = relationship('Tenant', back_populates='webhooks')
    deliveries = relationship('WebhookDelivery', back_populates='webhook', cascade='all, delete-orphan')


class WebhookDelivery(Base):
    """
    История доставки webhook.
    """
    __tablename__ = 'webhook_deliveries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    webhook_id = Column(Integer, ForeignKey('webhooks.id'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    payload = Column(Text, nullable=False)  # JSON
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    attempt_count = Column(Integer, default=1)
    status = Column(String(20), default='pending')  # pending, success, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    delivered_at = Column(DateTime, nullable=True)

    # Отношения
    webhook = relationship('Webhook', back_populates='deliveries')

    __table_args__ = (
        Index('ix_webhook_delivery_status', 'webhook_id', 'status'),
    )


class BackgroundJob(Base):
    """
    Фоновая задача для асинхронной обработки.
    """
    __tablename__ = 'background_jobs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), unique=True, nullable=False, index=True)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    job_type = Column(String(50), nullable=False)  # model_train, predict, export, import
    status = Column(String(20), default='pending')  # pending, running, completed, failed, cancelled
    priority = Column(Integer, default=0)  # Higher = more important
    params = Column(Text, default='{}')  # JSON: job parameters
    result = Column(Text, nullable=True)  # JSON: job result
    progress = Column(Integer, default=0)  # 0-100
    progress_message = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('ix_job_status_priority', 'status', 'priority'),
        Index('ix_job_tenant_status', 'tenant_id', 'status'),
    )

    def __repr__(self):
        return f"<BackgroundJob(job_id='{self.job_id}', type='{self.job_type}', status='{self.status}')>"


class UsageRecord(Base):
    """
    Запись использования для биллинга и квот.
    """
    __tablename__ = 'usage_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)  # api_call, prediction, model_train, storage
    quantity = Column(Integer, default=1)
    metadata = Column(Text, default='{}')  # JSON: additional info
    recorded_at = Column(DateTime, default=utcnow, index=True)

    __table_args__ = (
        Index('ix_usage_tenant_resource_date', 'tenant_id', 'resource_type', 'recorded_at'),
    )
