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
