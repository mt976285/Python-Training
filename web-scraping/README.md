Web Scraping Project

Overview
- Small scraper that fetches a URL, extracts title, meta description, and links, and writes JSON output.

Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Run

- Quick run (adds `src` to PYTHONPATH):

```bash
PYTHONPATH=./src python ./src/webscraper/scraper.py --url https://example.com --output out.json --delay 1.0 --user-agent webscraper
```

- Or run as module after exporting PYTHONPATH:

```bash
export PYTHONPATH=./src
python -m webscraper.scraper --url https://example.com --output out.json --delay 1.0 --user-agent webscraper
```

Notes about politeness

- The scraper now respects `robots.txt` and will refuse to fetch pages disallowed for the configured `--user-agent`.
- A per-host rate limit is enforced. Use `--delay` to set the minimum seconds between requests to the same host. If the site's `robots.txt` specifies a larger `Crawl-delay`, that value will be honored instead.

Notes
- Replace the `--url` value with the URL you want to scrape.
- The script saves JSON to `--output` (defaults to stdout if not provided).
