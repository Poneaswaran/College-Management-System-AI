import pytest

from rag.config import Settings
from rag.errors import AppError
from rag.security import verify_internal_headers


def _settings() -> Settings:
    return Settings(
        INTERNAL_SOURCE_VALUE="django",
        INTERNAL_API_SECRET="secret",
    )


def test_verify_internal_headers_success() -> None:
    verify_internal_headers("django", "secret", _settings())


def test_verify_internal_headers_missing_headers() -> None:
    with pytest.raises(AppError) as exc:
        verify_internal_headers(None, None, _settings())

    assert exc.value.status_code == 401
    assert exc.value.code == "missing_internal_auth"


def test_verify_internal_headers_invalid_secret() -> None:
    with pytest.raises(AppError) as exc:
        verify_internal_headers("django", "wrong", _settings())

    assert exc.value.status_code == 403
    assert exc.value.code == "invalid_internal_auth"


def test_verify_internal_headers_uses_allowed_sources_list() -> None:
    settings = Settings(
        INTERNAL_SOURCE_VALUE="django-default",
        INTERNAL_ALLOWED_SOURCES=["django-a", "django-b"],
        INTERNAL_API_SECRET="secret",
    )
    verify_internal_headers("django-b", "secret", settings)


def test_verify_internal_headers_secret_optional_when_disabled() -> None:
    settings = Settings(
        INTERNAL_SOURCE_VALUE="django",
        INTERNAL_API_SECRET="",
    )
    verify_internal_headers("django", None, settings)


def test_verify_internal_headers_validation_can_be_disabled() -> None:
    settings = Settings(
        INTERNAL_HEADERS_ENABLED=False,
        INTERNAL_SOURCE_VALUE="django",
        INTERNAL_API_SECRET="secret",
    )
    verify_internal_headers(None, None, settings)
