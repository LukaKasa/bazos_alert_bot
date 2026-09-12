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

        prompt = f"""Jsi PŘÍSNÝ expert na RÝCHLÝ prodej z druhé ruky v České republice (2026).
Flipping: Bazoš/Sbazar → Vinted + Facebook Marketplace, prodej do 1–7 dnů.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

=== TRŽNÍ CENA – NEJDŮLEŽITĚJŠÍ PRAVIDLO ===
market_price_estimate = částka, za kterou by to BĚŽNÝ kupující v ČR reálně koupil v příštích dnech na Vinted/FB.

VŽDY PODSTŘELUJ. Typická chyba je přestřelit o 20–40 %. Tomu se vyhni.

ZAKÁZANÉ kotvy (nepoužívej):
- nová cena v Alze/Datartu
- „zánovní / jako nové“ pokud to není výslovně top stav s krabicí
- zahraniční eBay / US / DE ceny bez ohledu na DPH a dopravu
- sběratelské maximum

POVINNÉ kotvy:
- ceny POUŽITÝCH kusů v ČR, které se opravdu točí
- chybějící krabice, běžné škrábance, nižší baterie = NIŽŠÍ odhad
- když chybí specifikace (u MacBooku čip/RAM/SSD, u iPhonu baterie) = odhaduj SPODNÍ pásmo daného modelu

MacBook:
- použité Air/Pro s M1/M2 často jdou výrazně levněji než lidé čekají
- bez přesné konfigurace ber SPODNÍ reálnou CZ cenu dané generace
- nepřirovnávej k novému kusům

iPhone 12–16:
- rozhoduje % baterie a stav
- baterie pod 85 % = výrazně nižší trh
- bez uvedení baterie = konzervativní (nižší) odhad

Tenisky / Lego:
- nošené bez krabice = nižší pásmo
- Lego nekompletní / bez figurek = výrazně níž nebo reject

POVOLENÉ KATEGORIE (jinak should_buy=false):
1) Nike/Jordan, Adidas (žádané siluety), DC Shoes, New Era
2) iPhone 12, 13, 14, 15, 16 (+ Plus/Pro/Pro Max/mini)
3) MacBook Air/Pro
4) Lego set s číslem (sběratelské)

ODMÍTNI: iPhone 11 a starší, generické běžecké boty, kopačky, bulk Lego bez čísla, zjevné scamy.

ROZHODNUTÍ:
- discount_percent záporné = pod trhem
- should_buy=true jen při reálné slevě cca 15 %+ pod TVÝM konzervativním odhadem
- u telefonů/MacBooků při slevě >35 % silně zvaž scam

Odpověz VÝHRADNĚ JSON:
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "1–2 věty česky"
}}
"""

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.05,
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

                # Silná konzervativní korekce: AI u CZ bazaru často přestřelí 20–30 %
                if market_price and market_price > 0:
                    title_l = (title or "").lower()
                    desc_l = (description or "").lower()
                    is_electronics = any(
                        k in title_l or k in desc_l
                        for k in (
                            "macbook",
                            "iphone",
                            "mac book",
                            "air m",
                            "pro m",
                        )
                    )
                    factor = 0.78 if is_electronics else 0.88
                    market_price = int(round(market_price * factor))
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
                            elif disc > -15:
                                should_buy = False
                                reason += " (sleva pod 15 % po konzervativním odhadu)"
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
