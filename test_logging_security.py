import io
import logging
import unittest

from logging_security import (
    TelegramTokenFilter,
    TelegramTokenRedactor,
    redact_sensitive_text,
    register_telegram_bot_token,
)


TOKEN = "123456789:AAExampleToken_012345678901234567890"
OTHER_TOKEN = "987654321:BBAnotherToken_012345678901234567890"


class TelegramTokenRedactorTest(unittest.TestCase):
    def test_telegram_api_url_is_redacted_without_registration(self):
        value = f"POST https://api.telegram.org/bot{TOKEN}/sendMessage"

        result = TelegramTokenRedactor().redact(value)

        self.assertNotIn(TOKEN, result)
        self.assertEqual(
            result,
            "POST https://api.telegram.org/bot[REDACTED_TELEGRAM_BOT_TOKEN]/sendMessage",
        )

    def test_registered_token_is_redacted_in_plain_text_and_url(self):
        redactor = TelegramTokenRedactor()
        redactor.register(TOKEN)

        result = redactor.redact(f"token={TOKEN} url=https://example.test/{TOKEN}")

        self.assertNotIn(TOKEN, result)
        self.assertEqual(result.count("[REDACTED_TELEGRAM_BOT_TOKEN]"), 2)

    def test_registration_does_not_return_or_log_token(self):
        redactor = TelegramTokenRedactor()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("logging-security-registration-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            self.assertIsNone(redactor.register(TOKEN))
            self.assertEqual(stream.getvalue(), "")
        finally:
            logger.handlers = []

    def test_process_wide_registration_is_used_by_text_helper(self):
        register_telegram_bot_token(OTHER_TOKEN)

        result = redact_sensitive_text(f"https://example.test/{OTHER_TOKEN}")

        self.assertNotIn(OTHER_TOKEN, result)
        self.assertIn("[REDACTED_TELEGRAM_BOT_TOKEN]", result)


class TelegramTokenFilterTest(unittest.TestCase):
    def test_filter_redacts_formatted_arguments(self):
        redactor = TelegramTokenRedactor()
        redactor.register(TOKEN)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        handler.addFilter(TelegramTokenFilter(redactor))
        logger = logging.getLogger("logging-security-filter-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            logger.info("request failed for %s", f"https://api.telegram.org/bot{TOKEN}/getMe")
        finally:
            logger.handlers = []

        output = stream.getvalue()
        self.assertNotIn(TOKEN, output)
        self.assertIn("[REDACTED_TELEGRAM_BOT_TOKEN]", output)

    def test_filter_keeps_log_record_usable_after_sanitizing_args(self):
        redactor = TelegramTokenRedactor()
        redactor.register(TOKEN)
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "url=%s", (TOKEN,), None
        )

        self.assertTrue(TelegramTokenFilter(redactor).filter(record))
        self.assertEqual(record.getMessage(), "url=[REDACTED_TELEGRAM_BOT_TOKEN]")
        self.assertNotIn(TOKEN, record.msg)
        self.assertEqual(record.args, ())

    def test_filter_redacts_exception_text(self):
        redactor = TelegramTokenRedactor()
        redactor.register(TOKEN)
        try:
            raise RuntimeError(f"request URL contained {TOKEN}")
        except RuntimeError:
            record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed", (), None)
            record.exc_info = __import__("sys").exc_info()

        TelegramTokenFilter(redactor).filter(record)

        self.assertNotIn(TOKEN, record.exc_text)
        self.assertIn("[REDACTED_TELEGRAM_BOT_TOKEN]", record.exc_text)


if __name__ == "__main__":
    unittest.main()
