import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException

from app import telegram_webhook


class TelegramWebhookTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_secret_fails_closed(self):
        with patch("app.telegram_webhook_secret", return_value=""):
            with self.assertRaises(HTTPException) as raised:
                await telegram_webhook({}, BackgroundTasks(), None)

        self.assertEqual(raised.exception.status_code, 503)

    async def test_configured_secret_mismatch_is_unauthorized(self):
        with patch("app.telegram_webhook_secret", return_value="configured-secret"):
            with self.assertRaises(HTTPException) as raised:
                await telegram_webhook({}, BackgroundTasks(), "wrong-secret")

        self.assertEqual(raised.exception.status_code, 401)

    async def test_matching_secret_is_accepted(self):
        with patch("app.telegram_webhook_secret", return_value="configured-secret"):
            result = await telegram_webhook({}, BackgroundTasks(), "configured-secret")

        self.assertEqual(result, {"ok": True})


if __name__ == "__main__":
    unittest.main()
