from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------- Constants --------
BASE_OLD_REDDIT: str = "https://old.reddit.com"
BASE_NEW_REDDIT: str = "https://www.reddit.com"
UA: dict[str, str] = {
    "User-Agent": "FashionTrendAnalysis/0.1 (contact: you@example.com)",
}


# -------- Utilities --------
def parse_date(date_string: str) -> int:
    """Parse a YYYY-MM-DD date string into Unix epoch seconds (UTC)."""
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _sleep_between(min_seconds: float, max_seconds: float) -> None:
    delay: float = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def _fetch_html(url: str, max_retries: int = 2, timeout: float = 20.0) -> Optional[str]:
    """Fetch a URL with basic retry and return text, or None on failure."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=UA, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"[warn] HTTP {resp.status_code} for {url}")
        except requests.RequestException as exc:
            print(f"[warn] Request error for {url}: {exc}")

        # small incremental backoff
        time.sleep(1.0 + attempt)

    return None


def _extract_posts_from_soup(
    soup: BeautifulSoup,
    subreddit: str,
    fetched_at_ts: int,
) -> list[dict]:
    """Extract post dictionaries from a listing page soup."""
    rows: list[dict] = []
    things = soup.find_all("div", class_="thing")
    for div in things:
        # Required attributes per spec (use safe access)
        post_id: Optional[str] = div.get("data-fullname")
        permalink_rel: Optional[str] = div.get("data-permalink")
        title: Optional[str] = div.get("data-title")
        score_str: Optional[str] = div.get("data-score")
        author: Optional[str] = div.get("data-author")
        ts_ms_str: Optional[str] = div.get("data-timestamp")

        if not post_id:
            # Fallback: try id attribute
            post_id = div.get("id") or ""
        if not permalink_rel:
            # Attempt to find by anchor if missing
            a_tag = div.find("a", class_="comments")
            if a_tag and a_tag.get("href"):
                # old.reddit absolute/relative mix; normalize later via urljoin
                permalink_rel = a_tag.get("href")
        if not title:
            # Try the entry title
            title_el = div.find("a", class_="title")
            title = title_el.get_text(strip=True) if title_el else ""

        # Normalize/convert
        try:
            score = int(score_str) if score_str is not None and score_str != "" else 0
        except ValueError:
            score = 0

        try:
            created_utc = (
                int(int(ts_ms_str) / 1000)
                if ts_ms_str is not None and ts_ms_str != ""
                else 0
            )
        except ValueError:
            created_utc = 0

        permalink_abs: str = urljoin(BASE_NEW_REDDIT, permalink_rel or "")

        row = {
            "id": str(post_id or ""),
            "subreddit": subreddit,
            "title": title or "",
            "selftext": "",  # listing pages do not include selftext in v1
            "score": int(score),
            "num_comments": None,  # unknown in v1 from listing-only
            "author": author or "",
            "created_utc": int(created_utc),
            "permalink": permalink_abs,
            "fetched_at": int(fetched_at_ts),
        }
        rows.append(row)
    return rows


def _find_next_link(soup: BeautifulSoup) -> Optional[str]:
    """Find the next listing URL on old.reddit."""
    next_span = soup.find("span", class_="next-button")
    if not next_span:
        return None
    a = next_span.find("a", href=True)
    if not a:
        return None
    return a["href"]


def scrape_listing(
    sub: str,
    since_ts: int,
    until_ts: Optional[int],
    max_pages: int,
    max_posts: int,
    delay_min: float,
    delay_max: float,
) -> list[dict]:
    """Scrape posts from old.reddit listing pages for a subreddit.

    Stops when:
    - max_pages reached
    - max_posts reached (if > 0)
    - pagination has passed the since boundary (oldest post on page < since_ts)
    """
    collected: list[dict] = []
    pages_fetched: int = 0

    # Sort by new to enable boundary-based stopping
    url: str = f"{BASE_OLD_REDDIT}/r/{sub}/new/"

    while url and pages_fetched < max_pages:
        print(f"[info] Fetching {sub} page {pages_fetched + 1}: {url}")
        html = _fetch_html(url)
        if html is None:
            print(f"[warn] Skipping page due to repeated failures: {url}")
            # Try to continue to next if available from current HTML (not available), so break
            break

        fetched_at_ts = int(time.time())
        soup = BeautifulSoup(html, "html.parser")
        rows = _extract_posts_from_soup(soup, sub, fetched_at_ts)

        # Filter time window and gather
        page_oldest_ts: Optional[int] = None
        for row in rows:
            created = row["created_utc"]
            if until_ts is not None and created >= until_ts:
                # too new; still above window (new listing sorted desc)
                continue
            if created < since_ts:
                # below window; record for boundary decision
                if page_oldest_ts is None or created < page_oldest_ts:
                    page_oldest_ts = created
                continue
            collected.append(row)
            if max_posts > 0 and len(collected) >= max_posts:
                print(f"[info] Hit max-posts for r/{sub}: {max_posts}")
                return collected

        pages_fetched += 1

        # Boundary check: if the oldest we saw on this page is below since_ts,
        # and we are in /new/ sorted desc, the next pages will be even older -> stop.
        if page_oldest_ts is not None and page_oldest_ts < since_ts:
            print(f"[info] Passed since boundary for r/{sub}; stopping pagination.")
            break

        # Move to next page
        next_url = _find_next_link(soup)
        if not next_url:
            print(f"[info] No next link found for r/{sub}; stopping.")
            break

        # Be polite
        _sleep_between(delay_min, delay_max)
        url = next_url

    if pages_fetched >= max_pages:
        print(f"[info] Hit max-pages for r/{sub}: {max_pages}")

    return collected


def collect_subreddit(
    subreddit: str,
    since_ts: int,
    until_ts: Optional[int],
    max_pages: int,
    max_posts: int,
    delay_min: float,
    delay_max: float,
) -> list[dict]:
    print(
        f"[start] r/{subreddit}: since={since_ts} until={until_ts if until_ts is not None else 'now'} "
        f"max_pages={max_pages} max_posts={max_posts if max_posts > 0 else '∞'}"
    )
    rows = scrape_listing(
        subreddit,
        since_ts=since_ts,
        until_ts=until_ts,
        max_pages=max_pages,
        max_posts=max_posts,
        delay_min=delay_min,
        delay_max=delay_max,
    )
    print(f"[done] r/{subreddit}: collected {len(rows)} posts")
    return rows


def _coerce_schema(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a DataFrame with the exact output schema and types."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        # build empty with schema
        df = pd.DataFrame(
            columns=[
                "id",
                "subreddit",
                "title",
                "selftext",
                "score",
                "num_comments",
                "author",
                "created_utc",
                "permalink",
                "fetched_at",
            ]
        )

    # Ensure presence of all columns
    defaults = {
        "id": "",
        "subreddit": "",
        "title": "",
        "selftext": "",
        "score": 0,
        "num_comments": None,
        "author": "",
        "created_utc": 0,
        "permalink": "",
        "fetched_at": 0,
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Order columns and enforce types
    df = df[
        [
            "id",
            "subreddit",
            "title",
            "selftext",
            "score",
            "num_comments",
            "author",
            "created_utc",
            "permalink",
            "fetched_at",
        ]
    ]

    # Coercions
    df["id"] = df["id"].astype(str)
    df["subreddit"] = df["subreddit"].astype(str)
    df["title"] = df["title"].astype(str)
    df["selftext"] = df["selftext"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    # num_comments stays as float|None -> use pandas object to preserve None
    df["num_comments"] = df["num_comments"].astype("object")
    df["author"] = df["author"].astype(str)
    df["created_utc"] = (
        pd.to_numeric(df["created_utc"], errors="coerce").fillna(0).astype(int)
    )
    df["permalink"] = df["permalink"].astype(str)
    df["fetched_at"] = (
        pd.to_numeric(df["fetched_at"], errors="coerce").fillna(0).astype(int)
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape old.reddit listing pages into Parquet snapshot."
    )
    parser.add_argument(
        "--sub", type=str, required=True, help="Comma-separated subreddits, e.g. a,b,c"
    )
    parser.add_argument(
        "--since", type=str, required=True, help="YYYY-MM-DD (inclusive)"
    )
    parser.add_argument(
        "--until", type=str, required=False, default=None, help="YYYY-MM-DD (exclusive)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=120, help="Max listing pages per subreddit"
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=0,
        help="Optional cap on posts per subreddit (0 = no cap)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=2.0, help="Min seconds between page fetches"
    )
    parser.add_argument(
        "--delay-max", type=float, default=4.0, help="Max seconds between page fetches"
    )
    parser.add_argument(
        "--outdir", type=str, default="data/raw", help="Base output directory"
    )

    args = parser.parse_args()

    subs: list[str] = [s.strip() for s in args.sub.split(",") if s.strip()]
    since_ts: int = parse_date(args.since)
    until_ts: Optional[int] = parse_date(args.until) if args.until else None

    all_rows: list[dict] = []
    for sub in subs:
        rows = collect_subreddit(
            subreddit=sub,
            since_ts=since_ts,
            until_ts=until_ts,
            max_pages=args.max_pages,
            max_posts=args.max_posts,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[info] No rows collected.")

    # Deduplicate by id
    df = _coerce_schema(all_rows)
    if not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        after = len(df)
        if after != before:
            print(f"[info] Deduplicated posts: {before} -> {after}")

    # Output path
    utc_today = datetime.utcnow().date().isoformat()
    snap_dir = os.path.join(args.outdir, utc_today)
    os.makedirs(snap_dir, exist_ok=True)
    out_path = os.path.join(snap_dir, "reddit_posts.parquet")

    # Write parquet
    # Requires pyarrow installed (declared in requirements)
    df.to_parquet(out_path, index=False)

    print(f"[result] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
