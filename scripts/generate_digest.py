"""
generate_digest.py

Workflow 2 — runs daily at 14:00 UTC (8 AM Chicago CST) via GitHub Actions.
1. Query recent aviation articles from Supabase.
2. Rank top 5 via LLM.
3. Generate full podcast show notes in EN + ZH via LLM.
4. Synthesise MP3s with edge-tts.
5. Upload MP3s to Supabase Storage.
6. Upsert digest record.
7. Write docs/episodes/YYYY-MM-DD.json and update docs/episodes/index.json.
8. Send email via Resend.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import date, datetime, timezone
from pathlib import Path

import edge_tts
import httpx
import resend
from openai import OpenAI
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential

import db

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"

RESEND_API_KEY = os.environ["RESEND_API_KEY"]
RESEND_TO = os.environ["RESEND_TO_EMAIL"]
RESEND_FROM = "Aviation Digest <onboarding@resend.dev>"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
STORAGE_BUCKET = "podcast-audio"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DOCS_DIR = Path(__file__).parent.parent / "docs"

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()
EPISODES_DIR = DOCS_DIR / "episodes"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def rank_top_articles(articles: list[dict]) -> list[dict]:
    """
    Ask the LLM to rank articles and return top 5.
    Returns list of {article_id, rank, score, reason}.
    """
    items = "\n".join(
        f"{i+1}. ID={a['id']} | {a['title']}\n   Summary: {(a.get('summary_en') or '')[:200]}"
        for i, a in enumerate(articles)
    )
    prompt = load_prompt("rank_articles.md").replace("{articles}", items)
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.strip("```json").strip("```").strip()
    return json.loads(raw)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_show_notes(top_articles: list[dict], article_map: dict) -> dict:
    """
    Generate podcast show notes for the top articles.
    Returns {"en": "...", "zh": "..."}.
    """
    stories = []
    for item in top_articles:
        art = article_map.get(item["article_id"])
        if art:
            stories.append(
                f"#{item['rank']}: {art['title']}\n"
                f"   {art.get('summary_en', '')}"
            )
    stories_text = "\n\n".join(stories)

    today = date.today().strftime("%B %d, %Y")
    prompt = (
        load_prompt("generate_shownotes.md")
        .replace("{today}", today)
        .replace("{stories}", stories_text)
    )
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=3000,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.strip("```json").strip("```").strip()
    return json.loads(raw)


def generate_no_news_notes() -> dict:
    today = date.today().strftime("%B %d, %Y")
    return {
        "en": (
            f"Welcome to Aviation Digest for {today}. "
            "There are no major new aviation stories in the last 24 hours. "
            "We'll be back tomorrow with the latest from the skies. Stay tuned!"
        ),
        "zh": (
            f"欢迎收听 {today} 的航空文摘。"
            "过去24小时内没有重要的航空新闻。"
            "我们明天将继续为您带来最新航空资讯，敬请关注！"
        ),
    }


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

async def _tts(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def synthesise(text: str, voice: str, output_path: str) -> None:
    asyncio.run(_tts(text, voice, output_path))


# ---------------------------------------------------------------------------
# Supabase Storage upload
# ---------------------------------------------------------------------------

def upload_audio(local_path: str, remote_name: str) -> str:
    """Upload file to Supabase Storage and return public URL."""
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    with open(local_path, "rb") as f:
        data = f.read()
    # upsert so re-runs overwrite
    supa.storage.from_(STORAGE_BUCKET).upload(
        path=remote_name,
        file=data,
        file_options={"content-type": "audio/mpeg", "upsert": "true"},
    )
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{remote_name}"
    return public_url


# ---------------------------------------------------------------------------
# Resend email
# ---------------------------------------------------------------------------

def send_email(
    date_str: str,
    shownotes_en: str,
    shownotes_zh: str,
    audio_en_url: str,
    audio_zh_url: str,
    top_articles: list[dict],
    article_map: dict,
) -> None:
    resend.api_key = RESEND_API_KEY

    story_items = ""
    for item in top_articles:
        art = article_map.get(item["article_id"], {})
        title = art.get("title", "Unknown")
        url = art.get("url", "#")
        story_items += f'<li><a href="{url}">{title}</a></li>\n'

    html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Aviation Digest {date_str}</title></head>
<body style="font-family: Arial, sans-serif; max-width: 680px; margin: auto; padding: 24px;">
  <h1 style="color: #1a3c5e;">✈ Aviation Digest — {date_str}</h1>

  <h2>Top Stories</h2>
  <ol>{story_items}</ol>

  <h2>English Show Notes</h2>
  <p style="white-space: pre-wrap;">{shownotes_en}</p>
  <p><a href="{audio_en_url}" style="background:#1a3c5e;color:white;padding:8px 16px;
     border-radius:4px;text-decoration:none;">▶ Listen in English</a></p>

  <hr>

  <h2>中文节目说明</h2>
  <p style="white-space: pre-wrap;">{shownotes_zh}</p>
  <p><a href="{audio_zh_url}" style="background:#c0392b;color:white;padding:8px 16px;
     border-radius:4px;text-decoration:none;">▶ 收听中文版本</a></p>

  <hr>
  <p style="color:#888;font-size:12px;">
    Generated by Aviation Digest Automation · Powered by GPT-4o-mini + edge-tts
  </p>
</body>
</html>
"""
    resend.Emails.send({
        "from": RESEND_FROM,
        "to": [RESEND_TO],
        "subject": f"✈ Aviation Digest — {date_str}",
        "html": html_body,
    })
    print(f"  Email sent to {RESEND_TO}")


# ---------------------------------------------------------------------------
# docs/ helpers
# ---------------------------------------------------------------------------

def write_episode_json(date_str: str, episode_data: dict) -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    path = EPISODES_DIR / f"{date_str}.json"
    path.write_text(json.dumps(episode_data, ensure_ascii=False, indent=2))
    print(f"  Written {path}")


def update_index_json(date_str: str) -> None:
    index_path = EPISODES_DIR / "index.json"
    if index_path.exists():
        dates: list[str] = json.loads(index_path.read_text())
    else:
        dates = []
    if date_str not in dates:
        dates.insert(0, date_str)
    index_path.write_text(json.dumps(dates, indent=2))
    print(f"  Updated index.json — {len(dates)} episode(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== generate_digest.py starting ===")
    today = date.today()
    date_str = today.isoformat()

    # 0. Skip if digest already generated today
    existing = db.get_digest_by_date(date_str)
    if existing and existing.get("shownotes_en"):
        print(f"Digest already exists for {date_str} — skipping LLM generation.")
        if not existing.get("email_sent"):
            print("Email not yet sent — sending now…")
            # rebuild article_map from top_articles ids
            articles = db.get_recent_aviation_articles(hours=24)
            article_map = {a["id"]: a for a in articles}
            send_email(
                date_str=date_str,
                shownotes_en=existing["shownotes_en"],
                shownotes_zh=existing["shownotes_zh"],
                audio_en_url=existing["audio_en_url"],
                audio_zh_url=existing["audio_zh_url"],
                top_articles=existing["top_articles"],
                article_map=article_map,
            )
            db.mark_email_sent(date_str)
        else:
            print("Email already sent — nothing to do.")
        print("=== generate_digest.py done ===")
        sys.exit(0)

    # 1. Fetch recent aviation articles
    articles = db.get_recent_aviation_articles(hours=24)
    print(f"Aviation articles in last 24h: {len(articles)}")

    article_map: dict[str, dict] = {a["id"]: a for a in articles}

    # 2. Rank top 5
    if articles:
        print("Ranking top articles via LLM…")
        top_ranking = rank_top_articles(articles)
        # Clamp to 5
        top_ranking = sorted(top_ranking, key=lambda x: x.get("rank", 99))[:5]
    else:
        top_ranking = []

    print(f"Top articles selected: {len(top_ranking)}")

    # 3. Generate show notes
    if top_ranking:
        print("Generating show notes via LLM…")
        notes = generate_show_notes(top_ranking, article_map)
    else:
        print("No aviation articles — using placeholder show notes.")
        notes = generate_no_news_notes()

    shownotes_en = notes.get("en", "")
    shownotes_zh = notes.get("zh", "")

    # 4 & 5. TTS
    en_path = f"/tmp/{date_str}-en.mp3"
    zh_path = f"/tmp/{date_str}-zh.mp3"
    print("Synthesising English audio…")
    synthesise(shownotes_en, "en-US-JennyNeural", en_path)
    print("Synthesising Chinese audio…")
    synthesise(shownotes_zh, "zh-CN-XiaoxiaoNeural", zh_path)

    # 6. Upload to Supabase Storage
    print("Uploading audio to Supabase Storage…")
    audio_en_url = upload_audio(en_path, f"{date_str}-en.mp3")
    audio_zh_url = upload_audio(zh_path, f"{date_str}-zh.mp3")
    print(f"  EN: {audio_en_url}")
    print(f"  ZH: {audio_zh_url}")

    # 7. Upsert digest record
    digest_record = {
        "date": date_str,
        "top_articles": top_ranking,
        "shownotes_en": shownotes_en,
        "shownotes_zh": shownotes_zh,
        "audio_en_url": audio_en_url,
        "audio_zh_url": audio_zh_url,
        "email_sent": False,
    }
    db.upsert_digest(digest_record)
    print("  Digest upserted to Supabase.")

    # 8. Write episode JSON for GitHub Pages
    top_articles_detail = []
    for item in top_ranking:
        art = article_map.get(item["article_id"], {})
        top_articles_detail.append({
            "rank": item.get("rank"),
            "score": item.get("score"),
            "reason": item.get("reason"),
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "summary_en": art.get("summary_en", ""),
            "summary_zh": art.get("summary_zh", ""),
        })

    episode_data = {
        "date": date_str,
        "top_articles": top_articles_detail,
        "shownotes_en": shownotes_en,
        "shownotes_zh": shownotes_zh,
        "audio_en_url": audio_en_url,
        "audio_zh_url": audio_zh_url,
    }
    write_episode_json(date_str, episode_data)

    # 9. Update index.json
    update_index_json(date_str)

    # 10. Send email
    existing = db.get_digest_by_date(date_str)
    if existing and existing.get("email_sent"):
        print("  Email already sent today — skipping.")
    else:
        print("Sending email via Resend…")
        send_email(
            date_str=date_str,
            shownotes_en=shownotes_en,
            shownotes_zh=shownotes_zh,
            audio_en_url=audio_en_url,
            audio_zh_url=audio_zh_url,
            top_articles=top_ranking,
            article_map=article_map,
        )
        db.mark_email_sent(date_str)

    print("=== generate_digest.py done ===")


if __name__ == "__main__":
    main()
