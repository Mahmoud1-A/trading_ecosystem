from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram not configured; skip alert: %s", message)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = httpx.post(url, json={"chat_id": chat_id, "text": message}, timeout=10.0)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Telegram alert failed: %s", exc)
        return False
