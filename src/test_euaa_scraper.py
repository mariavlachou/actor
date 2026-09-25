"""Test run of browser_scraper.py against a configurable JS/AJAX-callback site + filter.

Defaults to https://caselaw.euaa.europa.eu/Pages/search.aspx filtered by
Input Provider = "EUAA Asylum Report".

The target URL and filter can be overridden, either from the command line
(--url, --filter-label, --filter-value) or programmatically (pass `url=`,
`filter_label=`, `filter_value=` to main()).

Only the URL is a free parameter here -- unlike scraper.py's fln.dk config,
BrowserSiteConfig's result-row selectors (item_selector, field_id_suffixes,
etc.) are specific to the DevExpress markup of EUAA_CASELAW, so pointing
--url at a genuinely different site will still use EUAA's row-parsing rules.
To scrape a different JS-driven site's own markup, build a new
BrowserSiteConfig in browser_scraper.py (or via --config there) instead.
"""

import argparse
import dataclasses

from browser_scraper import EUAA_CASELAW, BrowserSiteConfig, BrowserWebsiteScraper

DEFAULT_OUTPUT_PATH = "data/euaa_asylum_report.csv"
DEFAULT_FILTER_LABEL = "Input Provider"
DEFAULT_FILTER_VALUE = "EUAA Asylum Report"


def main(
    url: str = EUAA_CASELAW.base_url,
    filter_label: str = DEFAULT_FILTER_LABEL,
    filter_value: str = DEFAULT_FILTER_VALUE,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    config: BrowserSiteConfig = dataclasses.replace(EUAA_CASELAW, base_url=url)
    scraper = BrowserWebsiteScraper(config)

    rows = scraper.run(
        output_path,
        filter_label=filter_label,
        filter_value=filter_value,
    )
    print(f"Scraped {len(rows)} rows -> {output_path}")
    for row in rows[:3]:
        print(row["item_id"], "|", row["published_date"], "|", row["country"], "|", row["court"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=EUAA_CASELAW.base_url, help="Site URL to scrape")
    parser.add_argument("--filter-label", default=DEFAULT_FILTER_LABEL, help="Accessible name of the filter control, e.g. 'Input Provider'")
    parser.add_argument("--filter-value", default=DEFAULT_FILTER_VALUE, help="Option text to select within that filter")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()
    main(url=args.url, filter_label=args.filter_label, filter_value=args.filter_value, output_path=args.output)
