import hmac
import logging

logger = logging.getLogger(__name__)

from fastapi import Depends, Header

from rag.config import Settings, get_settings
from rag.errors import AppError


def verify_internal_headers(
    source_header: str | None,
    secret_header: str | None,
    settings: Settings,
) -> None:
    """Validate trusted caller headers from Django."""

    if not settings.INTERNAL_HEADERS_ENABLED:
        return

    logger.info("Verifying internal headers: source=%s", source_header)

    if not source_header:
        raise AppError(
            status_code=401,
            detail="Missing internal authentication headers",
            code="missing_internal_auth",
        )

    is_valid_source = source_header in settings.allowed_internal_sources
    if not is_valid_source:
        logger.warning("Invalid internal source: %s (Allowed: %s)", source_header, settings.allowed_internal_sources)
        raise AppError(
            status_code=403,
            detail="Invalid internal authentication headers",
            code="invalid_internal_auth",
        )

    if not settings.require_internal_secret:
        return

    if not secret_header:
        raise AppError(
            status_code=401,
            detail="Missing internal authentication headers",
            code="missing_internal_auth",
        )

    is_valid_secret = hmac.compare_digest(secret_header, settings.INTERNAL_API_SECRET)

    if not is_valid_secret:
        logger.warning("Invalid internal secret provided")
        raise AppError(
            status_code=403,
            detail="Invalid internal authentication headers",
            code="invalid_internal_auth",
        )


async def validate_internal_request(
    x_internal_source: str | None = Header(default=None, alias="X-Internal-Source"),
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
    settings: Settings = Depends(get_settings),
) -> None:
    verify_internal_headers(
        source_header=x_internal_source,
        secret_header=x_internal_secret,
        settings=settings,
    )
