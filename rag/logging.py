import logging
from contextvars import ContextVar


CORRELATION_ID_HEADER = "X-Correlation-Id"
_correlation_id_context: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(value: str) -> None:
    _correlation_id_context.set(value)


def get_correlation_id() -> str:
    return _correlation_id_context.get()


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


def configure_logging(log_level: str) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s %(levelname)s %(name)s "
                "correlation_id=%(correlation_id)s message=\"%(message)s\""
            )
        )
    )
    handler.addFilter(CorrelationIdFilter())

    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
