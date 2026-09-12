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

        prompt = f"""Jsi PŘÍSNÝ a KONZERVATIVNÍ expert na český bazarový flipping (Bazoš/Sbazar → Vinted + FB Marketplace) v roce 2026.
Cíl: koupit pod cenou a prodat do 1–7 dní. Notifikuj POUZE reálně výhodné věci z povolených kategorií.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

=== ODHA TRŽNÍ CENY (KONZERVATIVNÍ) ===
market_price_estimate = cena, za kterou se to v ČR REÁLNĚ prodá na Vinted/FB do 7 dnů.
NE nová maloobchodní cena, NE ideální sběratelská cena, NE EU průměr.

Pravidla:
1. Počítej s použitým stavem, opotřebením, chybějící krabicí a konkurencí v ČR.
2. Když si nejsi jistý, sniž odhad o 15–25 %.
3. Raději podstřel trh než přestřel.
4. Podezřele levné telefony/notebooky bez popisu baterie/iCloud = vysoké riziko podvodu → should_buy false.

POVOLENÉ KATEGORIE (vše ostatní → should_buy=false):

1) SBĚRATELSKÉ / ŽÁDANÉ TENISKY A DOPLŇKY
- Nike: Air Force 1, Dunk, Jordan 1, Jordan 4, Blazer, Cortez, žádané collaby
- Jordan (samostatně i Nike Jordan)
- Adidas: Samba, Gazelle, Campus, Spezial, Handball Spezial, Ultraboost, Superstar, Stan Smith, žádané collaby
- DC Shoes: žádané skate siluety (Court Graffik, Legacy, pure apod.) v dobrém stavu
- New Era: originální kšiltovky 59FIFTY / 9FORTY (ne no-name)
ODMÍTNI: Revolution, VS Pace, generické běžecké, kopačky, dětské low-end, jasné repliky

2) iPhone – POUZE řady 14, 15, 16 (včetně Plus / Pro / Pro Max / mini kde dává smysl)
- ODMÍTNI: iPhone 13 a starší, SE, nejasný model, extrémně nízká cena bez baterie/popisu (scam)
- Sleduj: % baterie, iCloud lock, Face ID, stav displeje

3) MacBook
- Air / Pro – preferuj Apple Silicon (M1/M2/M3/M4) pokud je v popisu
- ODMÍTNI: mrtvé kusy, silně poškozené, podezřele levné bez specifikace
- Sleduj: rok/čip/RAM/SSD pokud jsou uvedené

4) SBĚRATELSKÉ LEGO
- Konkrétní set s číslem (Star Wars, Technic, Icons, Creator Expert, Modular, žádané City/Minecraft/HP)
- Nové/nerozbalené = bonus
- ODMÍTNI: bulk kg bez čísla, Duplo na váhu, CHEVA, nekompletní bez figurek, čínské kopie

PRAVIDLA ROZHODNUTÍ:
1. discount_percent: záporné = pod trhem (např. -20 = 20 % pod).
2. should_buy=true JEN pokud:
   - cena je pod konzervativním trhem (ideálně ≤ -12 %, minimum cca -10 %)
   - kategorie je povolená
   - není scam / replika / špatný model iPhonu
3. U slevy >35 % u telefonů a MacBooků buď velmi opatrný.
4. Bez popisu: u jasného modelu tenisky/Lego s číslem možné; u iPhone/MacBook spíš reject.

Odpověz VÝHRADNĚ platným JSON (žádný markdown):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "1–2 věty česky: proč koupit / proč ne."
}}
"""

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 500,
                "responseMimeType": "application/json",
            },
        }

        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=55,
                )

                if response.status_code == 429:
                    wait = 20 + (attempt * 15)
                    logger.warning(
                        f"Gemini 429 rate limit (attempt {attempt + 1}/3), čekám {wait}s"
                    )
                    time.sleep(wait)
                    last_error = Exception("429 Too Many Requests")
                    continue

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

                # Konzervativní korekce trhu (−8 %)
                if market_price and market_price > 0:
                    market_price = int(round(market_price * 0.92))
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
                            elif disc > -10:
                                should_buy = False
                                reason += " (sleva pod 10 % po konzervativním odhadu)"
                        except (TypeError, ValueError):
                            should_buy = False

                logger.info(
                    f"Gemini eval: {title[:55]}... → buy={should_buy}, "
                    f"discount={discount}%, market={market_price}, reason={reason}"
                )
                time.sleep(3.0)
                return should_buy, reason, discount, market_price

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Gemini timeout (attempt {attempt + 1}/3): {e}")
                time.sleep(5 + attempt * 3)
            except Exception as e:
                last_error = e
                logger.error(f"Gemini evaluation failed: {e}")
                time.sleep(3.0)
                break

        return False, f"Chyba Gemini: {str(last_error)[:130]}", None, None
