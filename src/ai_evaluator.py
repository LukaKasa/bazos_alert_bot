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
Cíl: koupit pod cenou a prodat do 1–7 dní. Notifikuj POUZE reálně výhodné a likvidní věci.

Nabídka:
- Titulek: {title}
- Cena: {price} Kč
- Lokalita: {location or "neznámá"}
- Popis: {(description or "bez popisu")[:550]}

=== ODHA TRŽNÍ CENY (KRITICKÉ – BUĎ KONZERVATIVNÍ) ===
market_price_estimate = cena, za kterou se to v ČR REÁLNĚ prodá na Vinted/FB do 7 dnů, NE:
- nová maloobchodní cena
- ideální „sběratelská“ cena
- evropský průměr BrickLink bez dopravy
- cena za kus ve perfektním stavu s krabicí, pokud to inzerát nemá

Pravidla odhadu:
1. Vždy počítej s použitým stavem, běžným opotřebením, chybějící krabicí a lokální konkurencí v ČR.
2. Když si nejsi jistý, sniž odhad o 15–25 %.
3. U konzolí bez her/příslušenství sniž odhad. U her samotných buď přísný.
4. U tenisek bez krabice / nošených sniž odhad. Repliky a podezřelé collaby = should_buy false.
5. U Lega: nekompletní, bez figurek, bulk bez čísla = nízká cena nebo reject.
6. Raději podstřel trh než přestřel – lepší minout slabý deal než poslat falešný „good deal“.

POVOLENÉ KATEGORIE (vše ostatní → should_buy=false):
1) Tenisky – JEN whitelist níže
2) Vintage / streetwear Nike a Adidas (mikiny, bundy, dresy – ne obyčejné fleecové mikiny)
3) Lego – konkrétní sety s číslem
4) Konzole PS4/PS5/Nintendo Switch (OLED/Lite) + žádané hry pro tyto platformy

TENISKY – whitelist:
- Nike: Air Force 1, Dunk Low, Dunk High, Jordan 1, Jordan 4, Blazer, Cortez
- Adidas: Samba, Gazelle, Campus, Spezial, Handball Spezial, Ultraboost, Superstar, Stan Smith
ODMÍTNI: Revolution, VS Pace, Huarache, Monarch, Satire, ACG, Terrex, trail/outdoor,
kopačky, sálovky, dětské boty, generické „Nike Air“, low-end běžecké, většinu Air Max mimo whitelist.

VINTAGE / STREETWEAR:
- Preferuj: starší mikiny, bundy, dresy v dobrém stavu
- Odmítni: obyčejné fleecové mikiny, ponožky, čepice, dětské low-end

LEGO:
- Preferuj: set s číslem (Star Wars, Technic, Icons, Minecraft, Harry Potter, Creator, City)
- Odmítni: bulk bez čísla, Duplo na váhu, CHEVA, nekompletní bez figurek

KONZOLE + HRY:
- POVOLENO: PS4, PS5, Nintendo Switch (OLED/Lite)
- HRY: Mario, Zelda, Animal Crossing, Smash, Odyssey, BOTW, TOTK, Pokémon,
  God of War, Spider-Man, Horizon, The Last of Us, Ghost of Tsushima, RDR, GTA, CoD
- ODMÍTNI: Wii, Wii U, hry na Wii/Wii U, Just Dance, samotné Fifa, low-demand tituly,
  čínské handheldy, samotné příslušenství bez konzole

PRAVIDLA ROZHODNUTÍ:
1. Spočítej slevu: záporné % = pod trhem (např. -20 = 20 % pod).
2. should_buy=true JEN pokud:
   - cena je pod KONZERVATIVNÍM odhadem trhu (ideálně ≤ -12 %, minimum cca -10 %)
   - věc je v povolených kategoriích
   - není scam / replika / neexistující model
3. U slevy >35 % pod trhem buď velmi opatrný (často podvod) – doporuč jen při jasné důvěryhodnosti.
4. Bez popisu: u jasného žádaného modelu můžeš doporučit, ale s ještě konzervativnějším trhem; jinak reject.

Odpověz VÝHRADNĚ platným JSON (žádný markdown):
{{
  "market_price_estimate": číslo,
  "discount_percent": číslo,
  "should_buy": true/false,
  "reason": "1–2 věty česky: proč koupit / proč ne, s důrazem na reálnou prodejní cenu v ČR."
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

                # Konzervativní korekce: AI občas stále přestřelí → mírně stáhneme odhad
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
