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

Úkol: Rozhodni, jestli je tato nabídka výhodná a dostatečně důvěryhodná pro rychlý flipping.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

Pravidla hodnocení:

1. Odhadni realistickou tržní cenu použitého kusu v dobrém stavu v ČR (2026).
2. Spočítej slevu v procentech (záporné číslo = pod tržní cenou).
3. Doporuč koupit, pokud je sleva mezi 8 % a 50 % pod trhem A zároveň nabídka působí důvěryhodně.
4. U slev 8–30 %: běžně doporučuj, pokud nevypadá na podvod.
5. U slev 30–50 %: doporučuj POUZE pokud nabídka působí velmi důvěryhodně.

Kontrola důvěryhodnosti (velmi důležité):
- Podezřelé formulace: „cenu nabídněte“, „jen dnes“, „rychle“, „nutno prodat“, „nemám čas“, „odvoz ihned“
- Neexistující nebo budoucí modely (iPhone 17, Galaxy Z Fold 8, iPhone Air atd.) = vždy podvod
- Příliš dokonalé / stock fotky (zejména u drahých telefonů a konzolí) = vyšší riziko
- Chybějící detaily o stavu, baterii, příslušenství = opatrnost
- Velmi nízká cena + minimální popis = vysoké riziko
- Rare PS3 (CECHAxx, CECHExx, speciální edice) a graded Pokémon karty posuzuj přísněji, ale pozitivně, pokud sedí detaily

Speciální znalosti:
- PlayStation 3: CECHAxx / CECHExx (4× USB + PS2 kompatibilita) a speciální edice mají vyšší hodnotu.
- PlayStation 4 Fat a PS5 Fat: sleduj kompletní sety.
- Pokémon TCG: moderní sety, graded karty (PSA/CGC), starší rare/holo.
- iPhone: důležitá je kondice baterie (ideálně 85 %+).

Odpověz VÝHRADNĚ platným JSON objektem (žádný markdown, žádný další text):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "krátké zdůvodnění česky (1-2 věty). Uveď proč je nebo není důvěryhodné a proč do toho jít / nejít."
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

            # Vyčisti případný markdown
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            result = json.loads(content)

            should_buy = bool(result.get("should_buy", False))
            discount = result.get("discount_percent")
            reason = result.get("reason", "bez důvodu")

            # Dodatečná kontrola rozsahu 8–50 %
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

            # Pauza kvůli rate limitu free tieru
            time.sleep(1.7)

            return should_buy, reason, discount

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            time.sleep(2.2)
            return False, f"Chyba Gemini: {str(e)[:130]}", None
