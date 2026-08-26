from typing import TYPE_CHECKING

import requests
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from utils.pipeline import Pipeline


class TelegramAPI(BaseModel):
    token: str = Field(..., description="Telegram bot token")
    chat_id: str = Field(..., description="Telegram chat ID to send to")

    def message(self, message: str):
        # Send a message to the Telegram chat
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": None}
        # Just use the requests library to send the message
        response = requests.post(url, data=data)
