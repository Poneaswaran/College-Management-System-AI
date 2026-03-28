from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    """Application-level error with API-safe metadata."""

    status_code: int
    detail: str
    code: str


def build_error_payload(detail: str, code: str, correlation_id: str | None = None) -> dict:
    payload: dict[str, str] = {"detail": detail, "code": code}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    return payload
