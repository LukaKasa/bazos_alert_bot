import os
import json
import logging
import re
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class AIEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = "gemini-3.5-flash-lite"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def evaluate_deal(
        self,
        title: str,
        price_str: str,
        description: str = "",
        location: str = "",
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Returns:
            (should_notify, reason, discount_percent)
        """
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set – skipping AI evaluation")
            return False, "Gemini API key missing", None

        # Extrahuj číslo z ceny
        price_clean = re.sub(r"[^\d]", "", price_str or "")
        if not price_clean:
            return False, "Nelze přečíst cenu", None

        price = int(price_clean)

        prompt = f"""Jsi expert na český bazarový trh (Bazoš, Sbazar, Vinted, Facebook Marketplace) v roce 2026.
Specializuješ se na flipping elektroniky, herních konzolí a sběratelských předmětů.

Úkol: Rozhodni, jestli je tato nabídka výhodná pro rychlý flipping (koupit levně → prodat dráž).

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:500]}

Pravidla hodnocení:

1. Odhadni realistickou tržní cenu použitého kusu v dobrém stavu v ČR (2026).
2. Spočítej slevu v procentech (záporné číslo = pod tržní cenou).
3. Doporuč koupit POUZE pokud je sleva mezi 8 % a 28 % pod trhem.
4. Sleva větší než 30 % = velmi rizikové (podvod / vadné / kradené).
5. Sleva menší než 8 % = málo zajímavé.

Speciální znalosti:
- PlayStation 3: Hledej rare modely CECHAxx, CECHExx (4× USB + PS2 kompatibilita), speciální edice (Metal Gear Solid 4 Gunmetal, Final Fantasy XIII atd.). Ty mají vyšší hodnotu.
- PlayStation 4 Fat a PS5 Fat: Sleduj kompletní sety (krabička + hry).
- Pokémon TCG: Moderní sety, graded karty (PSA/CGC), starší rare/holo mají dobrou likviditu.
- iPhone: Důležitá je kondice baterie (ideálně 85 %+).
- Podezřele nízké ceny u nových modelů (iPhone 16/17, Galaxy Z Fold 8 atd.) = téměř vždy podvod.

Odpověz VÝHRADNĚ platným JSON objektem (žádný markdown, žádný další text):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "krátké zdůvodnění česky (1-2 věty)"
}}
"""

        try:
            headers = {
                "Content-Type": "application/json",
            }
            params = {
                "key": self.api_key
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 450,
                    "responseMimeType": "application/json"
                }
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                params=params,
                json=payload,
                timeout=35
            )
            response.raise_for_status()
            data = response.json()

            content = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            # Vyčisti případný markdown
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            result = json.loads(content)

            should_buy = bool(result.get("should_buy", False))
            discount = result.get("discount_percent")
            reason = result.get("reason", "bez důvodu")

            # Dodatečná kontrola rozsahu 8–28 %
            if should_buy and discount is not None:
                try:
                    disc = float(discount)
                    if not (8 <= abs(disc) <= 28 and disc < 0):
                        should_buy = False
                        reason += " (mimo rozsah 8–28 %)"
                except (TypeError, ValueError):
                    should_buy = False

            logger.info(
                f"Gemini eval: {title[:55]}... → buy={should_buy}, discount={discount}%, reason={reason}"
            )

            # Pauza kvůli rate limitu free tieru
            time.sleep(1.6)

            return should_buy, reason, discount

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            time.sleep(2.0)  # při chybě delší pauza
            return False, f"Chyba Gemini: {str(e)[:130]}", None
