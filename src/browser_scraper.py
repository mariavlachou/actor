"""Browser-driven scraper for JS/AJAX "callback" search pages.

`scraper.py` works for sites like https://fln.dk/praksis/, where filters and
pagination are encoded as plain URL query parameters -- a `requests` GET is
enough. Some sites don't work that way: https://caselaw.euaa.europa.eu/Pages/search.aspx
is a SharePoint page built on a DevExpress ASPxGridView, where every filter
selection and every "next batch of results" fires a JavaScript callback (an
AJAX POST carrying ASP.NET ViewState) that patches the page in place. There is
no URL that represents a filtered/paged state, so a plain HTTP client cannot
reproduce it. This module drives a real (headless) browser with Playwright
instead: it clicks the filter control the same way a person would, waits for
the callback to finish, and repeatedly scrolls to the bottom to trigger the
site's "load more" behaviour until every matching row has been loaded.

This targets the specific interaction pattern used by DevExpress
ASPxGridView/ASPxDropDownEdit combos (a `combobox`-role trigger that opens a
checkbox list of options, where picking an option immediately re-runs the
search), plus a scroll-to-bottom "load more". Other JS-heavy sites that use a
different pattern (an explicit "Next page" button, infinite scroll on a
different container, etc.) would need `load_more_strategy` extended or a new
`BrowserSiteConfig`, but the overall shape -- open filter, pick value, drain
results, extract fields by selector -- carries over.

Usage as a library:

    from browser_scraper import EUAA_CASELAW, BrowserWebsiteScraper

    scraper = BrowserWebsiteScraper(EUAA_CASELAW)
    rows = scraper.run(
        "euaa.csv",
        filter_label="Input Provider",
        filter_value="EUAA Asylum Report",
    )

Usage from the command line:

    python3 browser_scraper.py --filter-label "Input Provider" \
        --filter-value "EUAA Asylum Report" --output euaa.csv
    python3 browser_scraper.py --config other_site.json --output other.csv

Requires Playwright with the Chromium browser installed:

    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass
class BrowserSiteConfig:
    """Describes one JS-driven site: how to open its filter and read its result rows."""

    name: str
    base_url: str

    # --- Filter interaction (DevExpress-style dropdown + checkbox listbox) ---
    filter_trigger_role: str = "combobox"  # ARIA role of the control that opens the filter's dropdown

    # --- Result list ---
    item_selector: str = "tr.dxgvDataRow_Moderno"
    link_selector: str = "a[href]"
    # {output_column: id_suffix} -- matches elements via `[id$='<suffix>']` within each item,
    # which is how DevExpress names its per-row label spans (e.g. "..._row3_CourtLabel").
    field_id_suffixes: Dict[str, str] = field(default_factory=dict)
    # {output_column: css_selector} -- for fields that aren't exposed via a stable id suffix.
    field_css_selectors: Dict[str, str] = field(default_factory=dict)

    # --- "Load more" pagination triggered by scrolling to the bottom of the page ---
    load_more_strategy: str = "scroll"  # "scroll" | "none"
    load_more_wait_ms: int = 1500
    load_more_max_attempts_without_growth: int = 5

    # --- Misc ---
    post_filter_wait_ms: int = 1500
    goto_timeout_ms: int = 60000
    headless: bool = True
    user_agent: Optional[str] = None


EUAA_CASELAW = BrowserSiteConfig(
    name="EUAA Case Law Database",
    base_url="https://caselaw.euaa.europa.eu/Pages/search.aspx",
    item_selector="tr.dxgvDataRow_Moderno",
    link_selector="a[href*='viewcaselaw.aspx']",
    field_id_suffixes={
        "country": "countryOfDecision",
        "published_date": "DecisionDateLabel",
        "court": "CourtLabel",
        "case_type": "TypeLabel",
        "case_number": "CaseNumberLabel",
    },
    field_css_selectors={
        "description": ".abstract-ellipsis-twoLines",
    },
)


class BrowserWebsiteScraper:
    """Drives a headless browser to apply a filter and scrape the resulting rows."""

    def __init__(self, config: BrowserSiteConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Browser / navigation
    # ------------------------------------------------------------------
    def _launch(self, pw):
        browser = pw.chromium.launch(headless=self.config.headless)
        context_kwargs = {"user_agent": self.config.user_agent} if self.config.user_agent else {}
        page = browser.new_page(**context_kwargs)
        page.goto(self.config.base_url, wait_until="networkidle", timeout=self.config.goto_timeout_ms)
        return browser, page

    # ------------------------------------------------------------------
    # Filter interaction
    # ------------------------------------------------------------------
    def apply_filter(self, page, filter_label: str, filter_value: str) -> None:
        combo = page.get_by_role(self.config.filter_trigger_role, name=filter_label)
        if combo.count() == 0:
            raise ValueError(f"No {self.config.filter_trigger_role!r} control found with accessible name {filter_label!r}")
        combo.first.click()
        page.wait_for_timeout(500)

        option = self._find_visible_option(page, filter_value)
        if option is None:
            raise ValueError(f"No visible filter option found with text {filter_value!r}")

        base = self.config.base_url.split("?")[0]
        try:
            with page.expect_response(
                lambda r: r.request.method == "POST" and base in r.url, timeout=15000
            ):
                option.click()
        except PlaywrightTimeoutError:
            option.click()
        page.wait_for_timeout(self.config.post_filter_wait_ms)

    @staticmethod
    def _find_visible_option(page, text: str):
        # DevExpress mirrors the currently-active listbox item into a hidden
        # accessibility "active descendant" node (id contains "_AcAs") that
        # also matches on text and can report is_visible() == True. Scope the
        # search to the real listbox item cells so that mirror is never
        # picked instead of the actual clickable row.
        candidates = page.locator(".dxeListBoxItem_Moderno")
        for i in range(candidates.count()):
            candidate = candidates.nth(i)
            if candidate.is_visible() and candidate.inner_text().strip() == text:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Pagination ("load more" via scroll)
    # ------------------------------------------------------------------
    def _drain_results(self, page, max_items: Optional[int]) -> None:
        if self.config.load_more_strategy != "scroll":
            return
        rows = page.locator(self.config.item_selector)
        stable_attempts = 0
        while stable_attempts < self.config.load_more_max_attempts_without_growth:
            if max_items is not None and rows.count() >= max_items:
                break
            before = rows.count()
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            page.wait_for_timeout(self.config.load_more_wait_ms)
            stable_attempts = 0 if rows.count() > before else stable_attempts + 1

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_rows(self, page) -> List[Dict[str, Any]]:
        cfg = self.config
        js = """
        (args) => {
            const { itemSelector, linkSelector, idFields, cssFields } = args;
            const rows = Array.from(document.querySelectorAll(itemSelector));
            return rows.map(row => {
                const data = {};
                for (const [col, suffix] of Object.entries(idFields)) {
                    const el = row.querySelector(`[id$='${suffix}']`);
                    data[col] = el ? el.innerText.trim() : '';
                }
                for (const [col, selector] of Object.entries(cssFields)) {
                    const el = row.querySelector(selector);
                    data[col] = el ? el.innerText.trim() : '';
                }
                const link = row.querySelector(linkSelector);
                data.url = link ? link.href : '';
                data.title = link ? link.innerText.trim() : '';
                return data;
            });
        }
        """
        raw_rows = page.evaluate(
            js,
            {
                "itemSelector": cfg.item_selector,
                "linkSelector": cfg.link_selector,
                "idFields": cfg.field_id_suffixes,
                "cssFields": cfg.field_css_selectors,
            },
        )

        rows: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for raw in raw_rows:
            item_id = self._derive_id(raw.get("url", ""))
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            row = {"item_id": item_id, "title": raw.get("title", ""), "url": raw.get("url", "")}
            for key, value in raw.items():
                if key not in ("title", "url"):
                    row[key] = value
            rows.append(row)
        return rows

    @staticmethod
    def _derive_id(url: str) -> str:
        m = re.search(r"[?&]([A-Za-z]+ID)=(\d+)", url)
        if m:
            return m.group(2)
        return url.rstrip("/").rsplit("/", 1)[-1] or "unknown"

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def scrape(
        self,
        filter_label: Optional[str] = None,
        filter_value: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with sync_playwright() as pw:
            browser, page = self._launch(pw)
            try:
                if filter_label and filter_value:
                    self.apply_filter(page, filter_label, filter_value)
                self._drain_results(page, max_items)
                rows = self.extract_rows(page)
            finally:
                browser.close()

        if max_items is not None:
            rows = rows[:max_items]
        return rows

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["item_id", "title", "url"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run(
        self,
        output_path: str,
        filter_label: Optional[str] = None,
        filter_value: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply the filter, drain + scrape all matching rows, and write CSV. Returns the rows too."""
        rows = self.scrape(filter_label=filter_label, filter_value=filter_value, max_items=max_items)
        self.save_to_csv(rows, output_path)
        return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_config(path: Optional[str]) -> BrowserSiteConfig:
    if not path:
        return EUAA_CASELAW
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid_fields = {f.name for f in dataclasses.fields(BrowserSiteConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"Unknown BrowserSiteConfig field(s) in {path}: {sorted(unknown)}")
    return BrowserSiteConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON file overriding BrowserSiteConfig fields, to target a different site")
    parser.add_argument("--filter-label", help="Accessible name of the filter control, e.g. 'Input Provider'")
    parser.add_argument("--filter-value", help="Option text to select within that filter, e.g. 'EUAA Asylum Report'")
    parser.add_argument("--max-items", type=int, default=None, help="Stop after collecting this many rows")
    parser.add_argument("--output", default="scraped_data.csv", help="Output CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    scraper = BrowserWebsiteScraper(config)
    rows = scraper.run(
        args.output,
        filter_label=args.filter_label,
        filter_value=args.filter_value,
        max_items=args.max_items,
    )
    print(f"Scraped {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
