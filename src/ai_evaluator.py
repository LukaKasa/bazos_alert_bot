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
    ) -> Tuple[bool, str, Optional[float], Optional[int]]:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set – skipping AI evaluation")
            return False, "Gemini API key missing", None, None

        price_clean = re.sub(r"[^\d]", "", price_str or "")
        if not price_clean:
            return False, "Nelze přečíst cenu", None, None

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
4. Pokud chybí popis, buď opatrnější, ale u jasného žádaného modelu (iPhone, Samba, Lego set s číslem) můžeš doporučit, pokud cena dává smysl.

PŘÍSNÁ PRAVIDLA PODLE KATEGORIE:

**Konzole (Nintendo Switch / PlayStation):**
- Preferuj kompletní sety (krabice, kabely, alespoň 1 ovladač).
- Switch OLED je žádanější než klasický V1/V2.
- Samostatné levné hry (Just Dance, sportovní, méně žádané tituly) → VŽDY odmítnout.

**Hry:**
- Posílej POUZE žádané tituly: Mario, Zelda, Animal Crossing, Super Smash, Pokémon (hry), God of War, Spider-Man, Horizon, The Last of Us, Ghost of Tsushima, GTA, Call of Duty atd.
- Levné / méně žádané hry → odmítnout.

**Tenisky (Nike / Adidas):**
- Posílej POUZE žádané modely:
  • Nike: Air Force 1, Dunk Low/High, Jordan 1, Jordan 4, Blazer, Cortez
  • Adidas: Samba, Gazelle, Campus, Spezial, Ultraboost, Superstar, Stan Smith, Handball Spezial
- Méně žádané (Kamanda, starší běžecké, neznámé collaby) → VŽDY odmítnout.

**iPhone / AirPods / Apple Watch / iPad:**
- iPhone: kondice baterie ideálně 85 %+.
- AirPods: u velmi nízkých cen opatrně na padělky.

**Lego:**
- Preferuj kompletní / nové sety, Star Wars, Technic, Icons, Harry Potter, Duplo konkrétní sety.
- Bulk bez specifikace → odmítnout.

**Pokémon karty:**
- Preferuj graded nebo moderní žádané sety / rare karty.
- Bulk běžných karet → odmítnout.

**Šperky:**
- Hlavně zlato a stříbro. Podezřele levné = opatrnost.

**Obecně odmítni:**
- Podezřelé formulace, stock fotky + podezřele nízká cena
- Neexistující / budoucí modely
- Jednotlivé levné hry a low-demand věci

Odpověz VÝHRADNĚ platným JSON objektem (žádný markdown):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "krátké zdůvodnění česky (1-2 věty)."
}}
"""

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.12,
                "maxOutputTokens": 500,
                "responseMimeType": "application/json",
            },
        }

        last_error = None
        for attempt in range(2):  # 1. pokus + 1 retry
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=55,  # delší timeout
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

                market_price = result.get("market_price_estimate")
                try:
                    market_price = int(market_price) if market_price is not None else None
                except (TypeError, ValueError):
                    market_price = None

                if should_buy and discount is not None:
                    try:
                        disc = float(discount)
                        # Povolit i silnější slevy, pokud AI řekla should_buy
                        # (ochrana jen proti absurdním kladným "slevám")
                        if disc >= 0:
                            should_buy = False
                            reason += " (cena není pod trhem)"
                        elif abs(disc) < 8:
                            should_buy = False
                            reason += " (sleva pod 8 %)"
                    except (TypeError, ValueError):
                        should_buy = False

                logger.info(
                    f"Gemini eval: {title[:55]}... → buy={should_buy}, discount={discount}%, market={market_price}, reason={reason}"
                )

                time.sleep(1.5)
                return should_buy, reason, discount, market_price

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Gemini timeout (attempt {attempt + 1}/2): {e}")
                time.sleep(2.5)
            except Exception as e:
                last_error = e
                logger.error(f"Gemini evaluation failed: {e}")
                time.sleep(2.0)
                break

        return False, f"Chyba Gemini: {str(last_error)[:130]}", None, None
