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

        prompt = f"""Jsi přísný expert na český a slovenský bazarový trh (Bazoš → Vinted + Facebook Marketplace) v roce 2026.
Specializuješ se na rychlý flipping s vysokou likviditou. Posíláš notifikace POUZE u opravdu zajímavých a důvěryhodných nabídek.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

ZÁKLADNÍ PRAVIDLA:
1. Odhadni realistickou tržní cenu použitého kusu v dobrém stavu v ČR/SK (2026).
2. Spočítej slevu (záporné číslo = pod trhem).
3. Doporuč koupit POUZE pokud je sleva 8–50 % pod trhem A nabídka je atraktivní + důvěryhodná.

PŘÍSNÁ PRAVIDLA PODLE KATEGORIE:

**Konzole (Nintendo Switch / PlayStation):**
- Preferuj kompletní sety (krabice, kabely, alespoň 1 ovladač).
- Switch OLED je žádanější než klasický V1/V2.
- Samostatné levné hry (Just Dance, sportovní, méně žádané tituly) → VŽDY odmítnout.
- Posílej jen pokud je cena výrazně lepší než běžný trh.

**Hry:**
- Posílej POUZE žádané tituly: Mario, Zelda, Animal Crossing, Super Smash, Pokémon (hry), God of War, Spider-Man, Horizon, The Last of Us, Ghost of Tsushima, GTA, kvalitní FIFA/FC, Call of Duty atd.
- Levné / méně žádané hry (Just Dance, různé taneční, sportovní low-tier) → odmítnout.

**Tenisky (Nike / Adidas):**
- Posílej jen pokud jsou v dobrém stavu (čisté, málo nošené, dobrá podrážka).
- Preferuj populární modely (Air Force, Dunk, Samba, Gazelle, Campus, Ultraboost atd.).
- Velmi ošoupané nebo podezřele levné = odmítnout.

**iPhone / AirPods:**
- iPhone: důležitá kondice baterie (ideálně 85 %+).
- AirPods: originál vs. padělek – u velmi nízkých cen buď opatrný.

**Pokémon karty:**
- Preferuj graded (PSA/CGC) nebo moderní žádané sety / rare karty.
- Běžné bulk karty za pár korun → odmítnout.

**Obecně odmítni:**
- Podezřelé formulace („cenu nabídněte“, „jen dnes“, „nutno prodat“)
- Neexistující / budoucí modely
- Stock fotky + podezřele nízká cena
- Nedostatečný popis u drahých věcí
- Jednotlivé levné hry a low-demand věci

Odpověz VÝHRADNĚ platným JSON objektem (žádný markdown):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "krátké zdůvodnění česky (1-2 věty). Uveď proč ano/ne a jestli je to atraktivní pro rychlý prodej."
}}
"""

        try:
            headers = {"Content-Type": "application/json"}
            params = {"key": self.api_key}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.12,
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
