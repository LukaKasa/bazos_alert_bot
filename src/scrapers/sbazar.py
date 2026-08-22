import logging
import random
import time
from typing import Dict, List, Optional

import requests

from .base import BaseScraper, Listing

logger = logging.getLogger(__name__)


class SbazarScraper(BaseScraper):
    """Scraper pro Sbazar.cz přes interní JSON API."""

    API_BASE = "https://www.sbazar.cz/api/v1"

    def __init__(self, source_name: str = "sbazar_cz"):
        super().__init__(source_name)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8",
            }
        )

    def scrape(self, search_config: Dict) -> List[Listing]:
        phrase = search_config.get("search_term") or search_config.get("name", "")
        if not phrase and not search_config.get("url"):
            self.logger.error("Sbazar: chybí search_term")
            return []

        # Pokud je URL typu https://www.sbazar.cz/hledat/iphone → vezmi poslední část
        if search_config.get("url") and not search_config.get("search_term"):
            url = search_config["url"].rstrip("/")
            phrase = url.split("/")[-1] if "/hledat/" in url else phrase

        max_pages = search_config.get("max_pages", 1)
        limit = 20  # API default page size
        price_min = search_config.get("price_min")
        price_max = search_config.get("price_max")

        all_listings: List[Listing] = []

        for page in range(max_pages):
            offset = page * limit
            self.logger.info(
                f"Scraping Sbazar page {page + 1}/{max_pages}: phrase={phrase}, offset={offset}"
            )

            params = {
                "limit": limit,
                "offset": offset,
                "phrase": phrase,
            }
            # API podporuje i category – zatím stačí phrase

            try:
                resp = self.session.get(
                    f"{self.API_BASE}/adverts/search",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                self.logger.error(f"Sbazar API error: {e}")
                break
            except ValueError as e:
                self.logger.error(f"Sbazar JSON parse error: {e}")
                break

            results = data.get("results") or []
            if not results:
                self.logger.info(f"No more Sbazar listings on page {page + 1}")
                break

            for item in results:
                listing = self._parse_item(item, price_min, price_max)
                if listing:
                    all_listings.append(listing)

            self.logger.info(f"Found {len(results)} listings on page {page + 1}")

            # Anti-bot pauza mezi stránkami
            if page < max_pages - 1:
                delay = random.uniform(2.5, 5.5)
                time.sleep(delay)

        self.logger.info(f"Total Sbazar listings found: {len(all_listings)}")
        return all_listings

    def _parse_item(
        self,
        item: dict,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
    ) -> Optional[Listing]:
        try:
            listing_id = str(item.get("id") or "")
            if not listing_id:
                return None

            title = item.get("name") or "Bez názvu"
            price_val = item.get("price")
            if item.get("price_by_agreement"):
                price = "Dohodou"
            elif price_val is not None:
                price = f"{int(price_val)} Kč"
            else:
                price = "N/A"

            # Filtr ceny (pokud je nastavený)
            if price_val is not None:
                try:
                    p = int(price_val)
                    if price_min is not None and p < int(price_min):
                        return None
                    if price_max is not None and p > int(price_max):
                        return None
                except (TypeError, ValueError):
                    pass

            seo = item.get("seo_name") or listing_id
            url = f"https://www.sbazar.cz/inzerat/{seo}"

            # Lokalita
            loc = item.get("locality") or {}
            city = loc.get("municipality") or loc.get("district") or ""
            zip_code = loc.get("zip") or ""
            location = f"{city}, {zip_code}".strip(", ") if city or zip_code else city or None

            # Obrázek
            image_url = None
            images = item.get("images") or []
            if images:
                img = images[0].get("url") or ""
                if img.startswith("//"):
                    image_url = "https:" + img
                elif img.startswith("http"):
                    image_url = img

            category = None
            cat = item.get("category") or {}
            if cat.get("name"):
                category = cat["name"]

            date_posted = item.get("create_date") or item.get("sorting_date")

            return Listing(
                listing_id=listing_id,
                source=self.source_name,
                title=title,
                url=url,
                price=price,
                location=location,
                image_url=image_url,
                description=None,  # detail lze doplnit později
                category=category,
                date_posted=date_posted,
                view_count=None,
            )
        except Exception as e:
            self.logger.debug(f"Error parsing Sbazar item: {e}")
            return None
