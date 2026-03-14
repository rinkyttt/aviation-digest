"""
fetch_articles.py

Workflow 1 — runs every 2 hours via GitHub Actions.
1. Parse Simple Flying RSS feed.
2. Skip URLs already in Supabase.
3. Batch-classify new titles as aviation/non-aviation (1 LLM call).
4. For aviation articles: fetch HTML, extract text, summarise EN+ZH (1 LLM call each).
5. Upsert all results into the `articles` table.
"""

import json
import os
import sys

import feedparser
import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import db

RSS_URL = "https://simpleflying.com/feed/"
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def classify_titles(titles: list[str]) -> list[bool]:
    """
    Ask the LLM to decide which titles are about aviation.
    Returns a list of booleans in the same order as `titles`.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))
    prompt = (
        "You are an aviation news classifier.\n"
        "Given the following article titles (one per line), decide whether each is\n"
        "primarily about aviation (airlines, airports, aircraft, air travel, aviation\n"
        "safety, aviation technology, etc.).\n\n"
        f"{numbered}\n\n"
        "Reply with a JSON array of booleans, one per title, in the same order.\n"
        "Example: [true, false, true]\n"
        "Reply with ONLY the JSON array, no other text."
    )
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=len(titles) * 10 + 20,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarise_article(title: str, text: str) -> dict:
    """
    Summarise an article in English and Chinese.
    Returns {"en": "...", "zh": "..."}.
    """
    # Truncate body to keep token costs low
    body = text[:4000]
    prompt = (
        "You are an aviation journalist.\n"
        f"Title: {title}\n\n"
        f"Article text:\n{body}\n\n"
        "Write a concise 2-3 sentence summary of this article.\n"
        "Reply with a JSON object with keys 'en' (English) and 'zh' (Simplified Chinese).\n"
        "Reply with ONLY the JSON object, no other text."
    )
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def fetch_article_text(url: str) -> str:
    """Download a page and return visible text via BeautifulSoup."""
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "aviation-digest-bot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as exc:
        print(f"  [warn] Could not fetch {url}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== fetch_articles.py starting ===")

    # 1. Parse RSS
    feed = feedparser.parse(RSS_URL)
    entries = feed.entries
    print(f"RSS entries fetched: {len(entries)}")

    if not entries:
        print("No entries in feed — exiting.")
        sys.exit(0)

    # 2. Skip duplicates
    existing_urls = db.get_existing_urls()
    new_entries = [e for e in entries if e.get("link") not in existing_urls]
    print(f"New entries (not in DB): {len(new_entries)}")

    if not new_entries:
        print("No new articles — exiting.")
        sys.exit(0)

    # 3. Batch-classify titles
    titles = [e.get("title", "") for e in new_entries]
    print(f"Classifying {len(titles)} titles via LLM…")
    is_aviation_flags = classify_titles(titles)

    # Guard against length mismatch (model occasionally adds/drops items)
    if len(is_aviation_flags) != len(new_entries):
        print(f"[warn] Classification length mismatch ({len(is_aviation_flags)} vs {len(new_entries)}). "
              "Padding with False.")
        is_aviation_flags = (is_aviation_flags + [False] * len(new_entries))[:len(new_entries)]

    aviation_count = sum(is_aviation_flags)
    print(f"Aviation articles identified: {aviation_count}/{len(new_entries)}")

    # 4 & 5. Fetch HTML + summarise + upsert
    for entry, is_aviation in zip(new_entries, is_aviation_flags):
        url = entry.get("link", "")
        title = entry.get("title", "")
        published_raw = entry.get("published", None)

        # Parse published date
        published_at = None
        if published_raw:
            import email.utils, datetime
            try:
                parsed_dt = email.utils.parsedate_to_datetime(published_raw)
                published_at = parsed_dt.isoformat()
            except Exception:
                published_at = None

        article: dict = {
            "url": url,
            "title": title,
            "published_at": published_at,
            "is_aviation": bool(is_aviation),
        }

        if is_aviation:
            print(f"  [aviation] {title[:80]}")
            raw_text = fetch_article_text(url)
            article["raw_content"] = raw_text[:8000]  # store up to 8k chars

            if raw_text:
                try:
                    summaries = summarise_article(title, raw_text)
                    article["summary_en"] = summaries.get("en", "")
                    article["summary_zh"] = summaries.get("zh", "")
                except Exception as exc:
                    print(f"  [warn] Summarisation failed for {url}: {exc}")
        else:
            print(f"  [skip]    {title[:80]}")

        db.upsert_article(article)

    print("=== fetch_articles.py done ===")


if __name__ == "__main__":
    main()
