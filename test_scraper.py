"""Test run of scraper.py against a configurable site + filter.

Defaults to https://fln.dk/praksis/ filtered by decision year ("Afgoerelsesaar") = 2026.

Filters on this kind of site are exposed as `<select>` options with opaque
values (WordPress taxonomy term ids here) that can change over time or
between sites, so the option is looked up at runtime -- by its human-readable
filter label and option text -- via WebsiteScraper.discover_filters() rather
than hardcoded.

The target URL and filter can be overridden, either from the command line
(--url, --filter-label, --filter-value) or programmatically (pass `url=`,
`filter_label=`, `filter_value=` to main()).
"""

import argparse
import dataclasses
from typing import Tuple

from scraper import FLN_PRAKSIS, SiteConfig, WebsiteScraper

DEFAULT_OUTPUT_PATH = "fln_praksis_2026.csv"
DEFAULT_FILTER_LABEL = "Afgørelsesår"
DEFAULT_FILTER_VALUE = "2026"


def find_filter_param(scraper: WebsiteScraper, label: str, option_text: str) -> Tuple[str, str]:
    """Resolve a human-readable (filter label, option text) pair to a (query param name, value) pair."""
    for filter_field in scraper.discover_filters():
        if filter_field.label == label:
            for option in filter_field.options:
                if option["text"] == option_text:
                    return filter_field.name, option["value"]
            raise ValueError(f"No option {option_text!r} found under filter {label!r}")
    raise ValueError(f"No filter labelled {label!r} found on the page")


def main(
    url: str = FLN_PRAKSIS.base_url,
    filter_label: str = DEFAULT_FILTER_LABEL,
    filter_value: str = DEFAULT_FILTER_VALUE,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    config: SiteConfig = dataclasses.replace(FLN_PRAKSIS, base_url=url)
    scraper = WebsiteScraper(config)

    param_name, param_value = find_filter_param(scraper, filter_label, filter_value)
    print(f"Resolved {filter_label!r} = {filter_value!r} -> {param_name}={param_value}")

    rows = scraper.run(output_path, filters={param_name: param_value})
    print(f"Scraped {len(rows)} rows -> {output_path}")
    for row in rows[:3]:
        print(row["item_id"], "|", row["published_date"], "|", row["categories"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=FLN_PRAKSIS.base_url, help="Site URL to scrape")
    parser.add_argument("--filter-label", default=DEFAULT_FILTER_LABEL, help="Human-readable filter label, e.g. 'Land'")
    parser.add_argument("--filter-value", default=DEFAULT_FILTER_VALUE, help="Option text within that filter, e.g. 'Afghanistan (I)'")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()
    main(url=args.url, filter_label=args.filter_label, filter_value=args.filter_value, output_path=args.output)
