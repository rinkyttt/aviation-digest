"""Shared Supabase client and helper functions."""

import os
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def get_existing_urls() -> set[str]:
    """Return all article URLs already in the database."""
    client = get_client()
    result = client.table("articles").select("url").execute()
    return {row["url"] for row in result.data}


def upsert_article(article: dict) -> None:
    """Insert or update an article row (keyed on url)."""
    client = get_client()
    client.table("articles").upsert(article, on_conflict="url").execute()


def get_recent_aviation_articles(hours: int = 24) -> list[dict]:
    """Return aviation articles published in the last `hours` hours."""
    client = get_client()
    from datetime import datetime, timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = (
        client.table("articles")
        .select("id, url, title, published_at, summary_en, summary_zh")
        .eq("is_aviation", True)
        .gte("published_at", cutoff)
        .order("published_at", desc=True)
        .execute()
    )
    return result.data


def upsert_digest(digest: dict) -> None:
    """Insert or update a digest row (keyed on date)."""
    client = get_client()
    client.table("digests").upsert(digest, on_conflict="date").execute()


def get_digest_by_date(date_str: str) -> dict | None:
    """Return digest row for a given YYYY-MM-DD date string, or None."""
    client = get_client()
    result = (
        client.table("digests")
        .select("*")
        .eq("date", date_str)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def mark_email_sent(date_str: str) -> None:
    """Set email_sent=True for a digest row."""
    client = get_client()
    client.table("digests").update({"email_sent": True}).eq("date", date_str).execute()
