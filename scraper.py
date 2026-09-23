"""Generic, filter-aware website scraper.

Fetches paginated, filterable listing pages (a "search archive" style page)
and writes the results to CSV. The site to scrape, its filters, and how to
read each result item are all described by a `SiteConfig` object, so the
same scraper works for other websites too -- just point it at a different
URL and describe the page structure (see `FLN_PRAKSIS` below for a worked
example against https://fln.dk/praksis/).

Usage as a library:

    from scraper import WebsiteScraper, FLN_PRAKSIS

    scraper = WebsiteScraper(FLN_PRAKSIS)
    scraper.print_filters()                       # inspect available filters
    rows = scraper.run(
        "output.csv",
        filters={"categorizations": ["67433", "60912"]},  # AND of two terms
        max_pages=5,
    )

Usage from the command line:

    python3 scraper.py --list-filters
    python3 scraper.py --filter categorizations=67433 --filter month=9 \
        --pages 3 --output praksis.csv
    python3 scraper.py --config my_site.json --output other_site.csv

A `--config path/to/site.json` file can override every field of
`SiteConfig` (see the dataclass below for field names) to point the same
script at a different website.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

FilterValue = Union[str, Sequence[str]]


@dataclass
class FilterField:
    """A single discovered filter (a <select> or text <input>) on the page."""

    name: str
    label: str
    kind: str  # "select" | "text"
    options: List[Dict[str, str]] = field(default_factory=list)  # [{"value", "text"}]


@dataclass
class SiteConfig:
    """Describes one website: where its filter form is and how to read result items.

    Every selector is a CSS selector (as understood by BeautifulSoup's
    `.select` / `.select_one`). Update these fields (or supply a JSON file
    with the same keys via --config) to point the scraper at a different
    website.
    """

    name: str
    base_url: str

    # Where to find the filter <form> when auto-discovering filters.
    filter_form_selector: str = "form"

    # One CSS selector that matches each individual result "card"/row.
    item_selector: str = "article"

    # Selectors evaluated *within* each item element found above.
    title_selector: str = "a"
    link_selector: Optional[str] = None  # defaults to title_selector if omitted
    id_selector: Optional[str] = None  # explicit unique-id element, if any
    date_selector: Optional[str] = None
    date_attr: Optional[str] = None  # e.g. "datetime" to read an attribute instead of text
    tags_selector: Optional[str] = None  # matches *multiple* elements, joined with "|"
    description_selector: Optional[str] = None

    # Pagination + request shape.
    pagination_param: str = "page"
    start_page: int = 1
    request_method: str = "GET"
    static_params: Dict[str, str] = field(default_factory=dict)
    request_delay: float = 0.5

    headers: Dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ConfigurableScraper/1.0; "
                "+https://github.com/)"
            )
        }
    )


# ---------------------------------------------------------------------------
# Default site: https://fln.dk/praksis/ (Flygtningenaevnet case-law archive)
# ---------------------------------------------------------------------------
FLN_PRAKSIS = SiteConfig(
    name="FLN Praksis (Flygtningenaevnet)",
    base_url="https://fln.dk/praksis/",
    filter_form_selector="div.archive-filters form",
    item_selector="ul.results-list li.results-list__item",
    title_selector="h2.results-item__title a.results-item__inner-link",
    date_selector="time",
    date_attr="datetime",
    tags_selector="li.results-item__tag",
    description_selector="p.results-item__description",
    pagination_param="pageNumber",
    start_page=1,
)


class WebsiteScraper:
    """Discovers filters, fetches pages, parses items, and writes CSV output."""

    def __init__(self, config: SiteConfig, session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update(config.headers)

    # ------------------------------------------------------------------
    # Filter discovery
    # ------------------------------------------------------------------
    def discover_filters(self) -> List[FilterField]:
        """Parse the site's filter form and return each filter with its options."""
        resp = self.session.get(self.config.base_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.select_one(self.config.filter_form_selector) or soup.find("form")
        if form is None:
            return []

        fields: List[FilterField] = []
        for select in form.find_all("select"):
            name = select.get("name")
            if not name:
                continue
            options = [
                {"value": opt.get("value", ""), "text": opt.get_text(strip=True)}
                for opt in select.find_all("option")
            ]
            fields.append(
                FilterField(name=name, label=self._label_for(form, select), kind="select", options=options)
            )

        for inp in form.find_all("input"):
            if (inp.get("type") or "text").lower() not in ("text", "search"):
                continue
            name = inp.get("name")
            if not name:
                continue
            fields.append(FilterField(name=name, label=self._label_for(form, inp), kind="text"))

        return fields

    @staticmethod
    def _label_for(form, element) -> str:
        elid = element.get("id")
        if elid:
            label_tag = form.find("label", attrs={"for": elid})
            if label_tag:
                return label_tag.get_text(strip=True)
        return element.get("name", "")

    def print_filters(self, max_options: int = 25) -> None:
        """Human-readable dump of discovered filters, for picking values by hand."""
        fields = self.discover_filters()
        if not fields:
            print("No filters discovered on this page.")
            return
        for f in fields:
            print(f"\n[{f.kind}] name={f.name!r}  label={f.label!r}")
            if f.kind == "select":
                for opt in f.options[:max_options]:
                    print(f"    value={opt['value']!r:>10}  text={opt['text']}")
                if len(f.options) > max_options:
                    print(f"    ... and {len(f.options) - max_options} more options")

    # ------------------------------------------------------------------
    # Fetching / parsing
    # ------------------------------------------------------------------
    def _build_params(self, filters: Dict[str, FilterValue], page: int) -> List[Tuple[str, Any]]:
        params: List[Tuple[str, Any]] = list(self.config.static_params.items())
        for key, value in (filters or {}).items():
            if isinstance(value, (list, tuple, set)):
                params.extend((key, v) for v in value if v not in (None, ""))
            elif value not in (None, ""):
                params.append((key, value))
        params.append((self.config.pagination_param, page))
        return params

    def fetch_page(self, filters: Dict[str, FilterValue], page: int) -> str:
        params = self._build_params(filters, page)
        resp = self.session.request(
            self.config.request_method, self.config.base_url, params=params, timeout=30
        )
        resp.raise_for_status()
        return resp.text

    def parse_items(self, html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        return [self._parse_item(item) for item in soup.select(self.config.item_selector)]

    def _parse_item(self, item) -> Dict[str, Any]:
        cfg = self.config

        title_el = item.select_one(cfg.title_selector)
        title = title_el.get_text(strip=True) if title_el else ""

        link_el = item.select_one(cfg.link_selector) if cfg.link_selector else title_el
        link = urljoin(cfg.base_url, link_el["href"]) if link_el and link_el.has_attr("href") else ""

        published_date = ""
        if cfg.date_selector:
            date_el = item.select_one(cfg.date_selector)
            if date_el:
                published_date = (
                    date_el.get(cfg.date_attr, "") if cfg.date_attr else date_el.get_text(strip=True)
                )

        categories = ""
        if cfg.tags_selector:
            categories = "|".join(t.get_text(strip=True) for t in item.select(cfg.tags_selector))

        description = ""
        if cfg.description_selector:
            desc_el = item.select_one(cfg.description_selector)
            if desc_el:
                description = desc_el.get_text(" ", strip=True)

        item_id = ""
        if cfg.id_selector:
            id_el = item.select_one(cfg.id_selector)
            if id_el:
                item_id = id_el.get_text(strip=True)
        if not item_id:
            item_id = self._derive_id(link, title)

        return {
            "item_id": item_id,
            "title": title,
            "url": link,
            "published_date": published_date,
            "categories": categories,
            "description": description,
        }

    @staticmethod
    def _derive_id(link: str, title: str) -> str:
        if link:
            slug = link.rstrip("/").rsplit("/", 1)[-1]
            if slug:
                return slug
        slug = re.sub(r"\s+", "_", title.strip().lower())
        return slug or "unknown"

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def scrape(
        self,
        filters: Optional[Dict[str, FilterValue]] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Page through results (applying `filters`) until an empty/repeat page or `max_pages`."""
        filters = filters or {}
        all_rows: List[Dict[str, Any]] = []
        seen_ids: set = set()
        page = self.config.start_page
        pages_done = 0

        while max_pages is None or pages_done < max_pages:
            html = self.fetch_page(filters, page)
            rows = self.parse_items(html)
            if not rows:
                break

            new_rows = [r for r in rows if r["item_id"] not in seen_ids]
            seen_ids.update(r["item_id"] for r in new_rows)
            all_rows.extend(new_rows)
            pages_done += 1

            if not new_rows:
                # Site ignored pagination or we hit the last page.
                break

            page += 1
            time.sleep(self.config.request_delay)

        return all_rows

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
        fieldnames = list(rows[0].keys()) if rows else [
            "item_id", "title", "url", "published_date", "categories", "description",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run(
        self,
        output_path: str,
        filters: Optional[Dict[str, FilterValue]] = None,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scrape and write CSV in one call. Returns the scraped rows too."""
        rows = self.scrape(filters=filters, max_pages=max_pages)
        self.save_to_csv(rows, output_path)
        return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_config(path: Optional[str], url_override: Optional[str]) -> SiteConfig:
    config = FLN_PRAKSIS
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f.name for f in dataclasses.fields(SiteConfig)}
        unknown = set(data) - valid_fields
        if unknown:
            raise ValueError(f"Unknown SiteConfig field(s) in {path}: {sorted(unknown)}")
        config = SiteConfig(**data)
    if url_override:
        config = dataclasses.replace(config, base_url=url_override)
    return config


def parse_filter_args(pairs: List[str]) -> Dict[str, FilterValue]:
    filters: Dict[str, FilterValue] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --filter value {pair!r}, expected name=value")
        name, value = pair.split("=", 1)
        if name in filters:
            existing = filters[name]
            filters[name] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            filters[name] = value
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON file overriding SiteConfig fields, to target a different site")
    parser.add_argument("--url", help="Override the site's base URL")
    parser.add_argument("--list-filters", action="store_true", help="Print available filters and exit")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="name=value",
        help="Filter to apply, e.g. --filter categorizations=67433. Repeat the flag for multi-value filters.",
    )
    parser.add_argument("--pages", type=int, default=None, help="Max number of pages to scrape (default: all)")
    parser.add_argument("--output", default="scraped_data.csv", help="Output CSV path")
    args = parser.parse_args()

    config = load_config(args.config, args.url)
    scraper = WebsiteScraper(config)

    if args.list_filters:
        scraper.print_filters()
        return

    filters = parse_filter_args(args.filter)
    rows = scraper.run(args.output, filters=filters, max_pages=args.pages)
    print(f"Scraped {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
