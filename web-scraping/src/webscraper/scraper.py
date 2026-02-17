"""Simple web scraper CLI.

Usage:
    python scraper.py --url <URL> [--output out.json]

The script extracts page title, meta description, and all links.
"""
import argparse
import json
import sys
import time
import urllib.parse
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser


_last_request_time: Dict[str, float] = {}

# Standard headers to fetch a website
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def fetch_website_contents(url) -> Dict:
    """
    Return the title and contents of the website at the given url;
    truncate to 2,000 characters as a sensible limit
    """
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    description = ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if not desc_tag:
        desc_tag = soup.find("meta", attrs={"property": "og:description"})
    if desc_tag and desc_tag.get("content"):
        description = desc_tag["content"].strip()
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return {"url": url, "title": title, "description": description, "text": text[:2000]}
    #return (title + "\n\n" + text)[:2_000]

def _get_netloc(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _enforce_rate_limit(netloc: str, delay: float) -> None:
    """Sleep if requests to this netloc are too frequent."""
    if delay <= 0:
        return
    now = time.time()
    last = _last_request_time.get(netloc)
    if last is not None:
        elapsed = now - last
        if elapsed < delay:
            to_sleep = delay - elapsed
            time.sleep(to_sleep)
    _last_request_time[netloc] = time.time()


def _fetch_robots_for(url: str, user_agent: str, timeout: float = 5.0) -> (Optional[RobotFileParser], Optional[float]):
    """Fetch and parse robots.txt for the given URL's host.

    Returns (parser, crawl_delay_seconds). If robots.txt cannot be fetched, returns (None, None).
    """
    parts = urllib.parse.urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, timeout=timeout)
    except requests.RequestException:
        return None, None
    if r.status_code != 200:
        return None, None

    parser = RobotFileParser()
    # parse expects an iterable of lines
    parser.parse(r.text.splitlines())
    try:
        crawl_delay = parser.crawl_delay(user_agent)
    except Exception:
        crawl_delay = None
    return parser, crawl_delay


def scrape_url(url: str, user_agent: str = "webscraper", delay: float = 1.0) -> Dict:
    """Fetch URL and extract title, meta description, and links.

    This function respects robots.txt (checks `can_fetch`) and enforces a per-host rate limit.
    """
    netloc = _get_netloc(url)

    # Fetch robots.txt and determine effective delay
    parser, robots_delay = _fetch_robots_for(url, user_agent)
    effective_delay = delay
    if robots_delay is not None and robots_delay > effective_delay:
        effective_delay = robots_delay

    # Check robots permission
    if parser is not None:
        try:
            allowed = parser.can_fetch(user_agent, url)
        except Exception:
            allowed = True
        if not allowed:
            raise PermissionError(f"Fetching disallowed by robots.txt for {url}")

    # Enforce rate limit per host
    _enforce_rate_limit(netloc, effective_delay)
    return fetch_website_contents(url)
    """ resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    description = ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if not desc_tag:
        desc_tag = soup.find("meta", attrs={"property": "og:description"})
    if desc_tag and desc_tag.get("content"):
        description = desc_tag["content"].strip()

    links: List[Dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        links.append({"href": href, "text": text})

    return {"url": url, "title": title, "description": description, "links": links} """


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple web scraper")
    parser.add_argument("--url", required=True, help="URL to scrape")
    parser.add_argument("--output", help="Output JSON file (if omitted, prints to stdout)")
    args = parser.parse_args(argv)

    result = scrape_url(args.url)

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
