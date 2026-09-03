"""Logging helpers for removing Telegram bot credentials from log records.

The module deliberately has no logging of its own.  Applications can register a
token as soon as they read it and attach :class:`TelegramTokenFilter` to a
handler or logger:

    register_telegram_bot_token(token)
    handler.addFilter(TelegramTokenFilter())

The token is kept only in memory.  Registration returns ``None`` and the
redactor replaces registered tokens everywhere in a message, in addition to
redacting the token-shaped path segment of Telegram Bot API URLs.
"""

from __future__ import annotations

import logging
import re
import traceback
from threading import RLock
from typing import Any


_REDACTED = "[REDACTED_TELEGRAM_BOT_TOKEN]"
_TELEGRAM_API_URL = re.compile(
    r"(?P<prefix>(?:https?://)?api\.telegram\.org(?::\d+)?/bot)"
    r"(?P<token>[A-Za-z0-9_-]+:[A-Za-z0-9_-]+)"
    r"(?P<suffix>(?=[/?#\s]|$))",
    re.IGNORECASE,
)


class TelegramTokenRedactor:
    """Thread-safe in-memory redactor for Telegram bot tokens."""

    def __init__(self) -> None:
        self._tokens: set[str] = set()
        self._lock = RLock()

    def register(self, token: str) -> None:
        """Register a raw token without returning or logging it.

        Telegram tokens are normally read from a protected file or environment
        variable.  Whitespace around a token is ignored because callers often
        pass the contents of a token file directly.
        """

        if not isinstance(token, str):
            return
        normalized = token.strip()
        if not normalized:
            return
        with self._lock:
            self._tokens.add(normalized)

    def redact(self, value: Any) -> str:
        """Return ``value`` as text with Telegram bot tokens removed."""

        text = value if isinstance(value, str) else str(value)
        with self._lock:
            # Longest-first avoids leaving a suffix visible if a caller
            # accidentally registers overlapping values.
            tokens = tuple(sorted(self._tokens, key=len, reverse=True))
        for token in tokens:
            text = text.replace(token, _REDACTED)
        return _TELEGRAM_API_URL.sub(
            lambda match: f"{match.group('prefix')}{_REDACTED}{match.group('suffix') or ''}",
            text,
        )


_DEFAULT_REDACTOR = TelegramTokenRedactor()


def register_telegram_bot_token(token: str) -> None:
    """Register a token for process-wide logging redaction.

    This function intentionally has no return value, so callers cannot obtain
    a token through the registration API.
    """

    _DEFAULT_REDACTOR.register(token)


def redact_sensitive_text(value: Any) -> str:
    """Redact registered tokens and Telegram Bot API URL tokens from ``value``."""

    return _DEFAULT_REDACTOR.redact(value)


class TelegramTokenFilter(logging.Filter):
    """A standard logging filter that sanitizes a record's rendered message."""

    def __init__(self, redactor: TelegramTokenRedactor | None = None) -> None:
        super().__init__()
        self.redactor = redactor or _DEFAULT_REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        # Render first so tokens passed as %-format arguments are sanitized too.
        # Clearing args prevents a later formatter from reintroducing the raw
        # values through record.msg/record.args.
        record.msg = self.redactor.redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = self.redactor.redact(
                "".join(traceback.format_exception(*record.exc_info))
            )
        if record.stack_info:
            record.stack_info = self.redactor.redact(record.stack_info)
        return True


def install_telegram_logging_redaction() -> None:
    """Install idempotent redaction on current root handlers.

    HTTPX request logs include the full Telegram Bot API URL at INFO level;
    suppress those routine request lines as a second layer of protection.
    The filter remains attached so warnings and exception records are also
    sanitized.
    """

    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(item, TelegramTokenFilter) for item in handler.filters):
            handler.addFilter(TelegramTokenFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)


__all__ = [
    "TelegramTokenFilter",
    "TelegramTokenRedactor",
    "redact_sensitive_text",
    "install_telegram_logging_redaction",
    "register_telegram_bot_token",
]
