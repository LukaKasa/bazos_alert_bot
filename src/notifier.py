import os
import json
import logging
from typing import Optional
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("Discord webhook URL not set.")

    def send_vehicle_notification(
        self,
        title: str,
        url: str,
        price: str,
        year: Optional[str] = None,
        mileage: Optional[str] = None,
        location: Optional[str] = None,
        image_url: Optional[str] = None,
        description: Optional[str] = None,
        ai_reason: Optional[str] = None,
        discount: Optional[float] = None,
        color: int = 0x3498DB,
    ) -> bool:
        """
        Send a listing notification to Discord with clear AI assessment.
        """
        if not self.webhook_url:
            logger.error("Cannot send notification: webhook URL not configured")
            return False

        fields = []

        if price:
            fields.append({"name": "💰 Cena", "value": str(price), "inline": True})

        if location:
            fields.append({"name": "📍 Lokalita", "value": str(location), "inline": True})

        if discount is not None:
            fields.append({
                "name": "📉 Sleva",
                "value": f"{abs(discount):.0f} % pod trhem",
                "inline": True
            })

        # AI hodnocení – vždy celé, neořezané
        if ai_reason:
            fields.append({
                "name": "🤖 AI hodnocení",
                "value": ai_reason[:1000],  # Discord field limit je 1024
                "inline": False
            })

        embed = {
            "title": title[:256],
            "url": url,
            "color": color,
            "fields": fields,
            "footer": {"text": "New Listing Alert"},
            "timestamp": self._get_timestamp(),
        }

        # Krátký popis inzerátu (max ~300 znaků)
        if description:
            desc = description.strip()
            if len(desc) > 300:
                desc = desc[:300].rstrip() + "..."
            embed["description"] = desc

        if image_url:
            embed["thumbnail"] = {"url": image_url}

        discord_data = {"embeds": [embed]}
        return self._send_webhook(discord_data)

    def send_notification(
        self, title: str, message: str, color: int = 0x3498DB
    ) -> bool:
        if not self.webhook_url:
            logger.error("Cannot send notification: webhook URL not configured")
            return False

        discord_data = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                }
            ]
        }
        return self._send_webhook(discord_data)

    def _send_webhook(self, data: dict) -> bool:
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.webhook_url,
                data=json.dumps(data),
                headers=headers,
                timeout=10,
            )
            if response.status_code == 204:
                logger.info("Discord notification sent successfully")
                return True
            else:
                logger.error(
                    f"Failed to send Discord notification: {response.status_code}, {response.text}"
                )
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.utcnow().isoformat()
