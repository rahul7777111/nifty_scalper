import logging
import requests
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Sends asynchronous Telegram notifications."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        if self.enabled:
            logger.info("Telegram notifications enabled.")

    def send_message(self, text: str, parse_mode: str = "HTML") -> None:
        if not self.enabled:
            return
            
        def _send():
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            try:
                resp = requests.post(url, json=payload, timeout=5.0)
                if not resp.ok:
                    logger.error(f"Telegram notification failed: {resp.text}")
            except Exception as e:
                logger.error(f"Telegram exception: {e}")
                
        # Fire and forget
        threading.Thread(target=_send, daemon=True).start()
