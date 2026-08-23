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
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

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

        prompt = f"""Jsi přísný expert na český bazarový flipping (Bazoš/Sbazar → Vinted + FB Marketplace) v roce 2026.
Cíl: koupit pod cenou a prodat do 1–7 dní. Posílej notifikace POUZE u vysoce likvidních věcí.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

POVOLENÉ KATEGORIE (vše ostatní → should_buy=false):
1) Tenisky – jen žádané siluety
2) Vintage / streetwear Nike a Adidas (mikiny, bundy, tepláky, dresy – ne obyčejné sportovní kalhoty)
3) Lego – konkrétní sety s číslem (ne bulk kg bez specifikace)
4) Herní konzole (PS4/PS5, Nintendo Switch) + žádané hry

TENISKY – whitelist:
- Nike: Air Force 1, Dunk Low/High, Jordan 1, Jordan 4, Blazer, Cortez
- Adidas: Samba, Gazelle, Campus, Spezial, Handball Spezial, Ultraboost, Superstar, Stan Smith
- Odmítni: Revolution, VS Pace, generic běžecké, neznámé collaby, dětské kopačky, repliky

VINTAGE / STREETWEAR Nike–Adidas:
- Preferuj: starší mikiny, bundy, dresy, tepláky s dobrým stavem a velikostí
- Odmítni: běžné nové sportovní kalhoty, ponožky, čepice bez hodnoty, dětské low-end

LEGO:
- Preferuj: set s číslem (Star Wars, Technic, Icons, Minecraft, Harry Potter, Creator, City konkrétní set)
- Nové/nerozbalené = bonus
- Odmítni: bulk kg bez obsahu, nekompletní bez čísla, čínské kopie

KONZOLE + HRY:
- Konzole: ideálně s ovladačem/kabely
- Hry jen žádané: Mario, Zelda, Animal Crossing, Smash, Odyssey, BOTW/TOTK, Pokémon;
  God of War, Spider-Man, Horizon, TLOU, Ghost of Tsushima, RDR, GTA, CoD
- Odmítni: Just Dance, levné sportovní, neznámé low-demand tituly, čínské handheldy

PRAVIDLA CENY:
1. Odhadni realistickou tržní cenu použitého kusu v ČR (2026) pro rychlý prodej na Vinted.
2. discount_percent: ZÁPORNÉ číslo = cena POD trhem (např. -25 znamená 25 % pod trhem). KLADNÉ = nad trhem.
3. should_buy=true jen pokud:
   - cena je pod trhem (discount_percent záporné, ideálně ≤ -10, minimum cca -8)
   - věc patří do povolených kategorií
   - není zjevný scam / replika / neexistující model
4. U silné slevy (>40 % pod trhem) buď opatrný na podvod, ale pokud model sedí a nabídka působí OK, můžeš doporučit.
5. Pokud chybí popis: u jasného žádaného modelu (Samba, Dunk, Lego s číslem) můžeš doporučit; jinak buď přísnější.

Odpověz VÝHRADNĚ platným JSON (žádný markdown):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "1–2 věty česky: proč koupit / proč ne, včetně odhadu trhu."
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
        for attempt in range(2):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=55,
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

                # Přepočet slevy z reálné ceny vs. odhad trhu (AI často vrací špatné znaménko)
                if market_price and market_price > 0:
                    real_discount = ((price - market_price) / market_price) * 100.0
                    discount = round(real_discount, 1)

                if should_buy:
                    if discount is None:
                        should_buy = False
                        reason += " (chybí sleva)"
                    else:
                        try:
                            disc = float(discount)
                            if disc >= 0:
                                should_buy = False
                                reason += " (cena není pod trhem)"
                            elif disc > -8:
                                should_buy = False
                                reason += " (sleva pod 8 %)"
                        except (TypeError, ValueError):
                            should_buy = False

                logger.info(
                    f"Gemini eval: {title[:55]}... → buy={should_buy}, "
                    f"discount={discount}%, market={market_price}, reason={reason}"
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
