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

        price_clean = re.sub(r"[^\d]", "", price_str or "")
        if not price_clean:
            return False, "Nelze přečíst cenu", None

        price = int(price_clean)

        prompt = f"""Jsi expert na český a slovenský bazarový trh (Bazoš → Vinted + Facebook Marketplace) v roce 2026.
Specializuješ se na rychlý flipping věcí s vysokou likviditou: iPhony, AirPods, PlayStation, Nintendo Switch, Pokémon karty a značkové tenisky (Nike, Adidas, New Balance).

Úkol: Rozhodni, jestli je tato nabídka výhodná a dostatečně důvěryhodná pro rychlý prodej (ideálně do několika dnů).

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

Pravidla hodnocení:

1. Odhadni realistickou tržní cenu použitého kusu v dobrém stavu v ČR/SK (2026).
2. Spočítej slevu v procentech (záporné číslo = pod tržní cenou).
3. Doporuč koupit, pokud je sleva mezi 8 % a 50 % pod trhem A nabídka působí důvěryhodně.
4. U slev 8–30 %: běžně doporučuj, pokud nevypadá na podvod.
5. U slev 30–50 %: doporučuj POUZE pokud nabídka působí velmi důvěryhodně.

Kontrola důvěryhodnosti:
- Podezřelé formulace: „cenu nabídněte“, „jen dnes“, „rychle“, „nutno prodat“, „odvoz ihned“
- Neexistující nebo budoucí modely = vždy podvod
- Stock fotky / příliš dokonalé fotky u drahých věcí = vyšší riziko
- Chybějící detaily o stavu = opatrnost

Speciální znalosti pro rychlý prodej:
- iPhone: klíčová je kondice baterie (ideálně 85 %+), originální krabička zvyšuje cenu
- AirPods: originál vs. padělek – u podezřele levných kousků buď opatrný
- PlayStation 4/5: kompletní set (krabice + kabely + ovladač) = lepší prodejnost
- Nintendo Switch: OLED verze je žádanější, sleduj stav joy-conů a krabičku
- Pokémon TCG: graded karty (PSA/CGC) a moderní sety mají dobrou likviditu
- Nike / Adidas tenisky: stav podrážky, krabička a originalita rozhodují. Čisté, málo nošené páry jdou nejrychleji
- Notebooky: jen kvalitní značky (MacBook, ThinkPad, XPS) v dobrém stavu

Odpověz VÝHRADNĚ platným JSON objektem (žádný markdown, žádný další text):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "krátké zdůvodnění česky (1-2 věty). Uveď proč je nebo není důvěryhodné a proč do toho jít / nejít."
}}
"""

        try:
            headers = {"Content-Type": "application/json"}
            params = {"key": self.api_key}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 500,
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

            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            result = json.loads(content)

            should_buy = bool(result.get("should_buy", False))
            discount = result.get("discount_percent")
            reason = result.get("reason", "bez důvodu")

            # Kontrola rozsahu 8–50 %
            if should_buy and discount is not None:
                try:
                    disc = float(discount)
                    if not (8 <= abs(disc) <= 50 and disc < 0):
                        should_buy = False
                        reason += " (mimo rozsah 8–50 %)"
                except (TypeError, ValueError):
                    should_buy = False

            logger.info(
                f"Gemini eval: {title[:55]}... → buy={should_buy}, discount={discount}%, reason={reason}"
            )

            time.sleep(1.7)
            return should_buy, reason, discount

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            time.sleep(2.2)
            return False, f"Chyba Gemini: {str(e)[:130]}", None
