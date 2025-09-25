from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import requests

# -------- Constants --------
UA: dict[str, str] = {
    "User-Agent": "FashionTrendAnalysis/0.1 (contact: you@example.com)",
}


# -------- Utilities --------
def _sleep_between(min_seconds: float, max_seconds: float) -> None:
    """Sleep for a random duration between min and max seconds."""
    delay: float = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def _fetch_json(url: str, max_retries: int = 2) -> Optional[list[dict[str, Any]]]:
    """Fetch JSON from a Reddit permalink/.json endpoint with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=UA, timeout=20.0)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (403, 429):
                # Rate limited - back off longer
                backoff = 10.0 + (attempt * 10.0)
                print(
                    f"[warn] Rate limited ({resp.status_code}) for {url}, backing off {backoff}s"
                )
                time.sleep(backoff)
            else:
                print(f"[warn] HTTP {resp.status_code} for {url}")
                if attempt < max_retries:
                    time.sleep(2.0 + attempt * 3.0)  # Incremental backoff

        except requests.RequestException as exc:
            print(f"[warn] Request error for {url}: {exc}")
            if attempt < max_retries:
                time.sleep(2.0 + attempt * 3.0)

    return None


def _extract_post_id_from_link_id(link_id: str) -> str:
    """Extract post ID from Reddit link_id (remove t3_ prefix)."""
    if link_id.startswith("t3_"):
        return link_id[3:]
    return link_id


def _build_comment_permalink(post_permalink: str, comment_id: str) -> str:
    """Build comment permalink from post permalink and comment ID."""
    base = post_permalink.rstrip("/")
    return f"{base}/comment/{comment_id}/"


# -------- Post Selection --------
def select_posts(
    df: pd.DataFrame, per_month_top: int, limit_posts: int
) -> pd.DataFrame:
    """Select posts based on per-month top scoring and optional limit."""
    if df.empty:
        return df

    # Ensure created_utc is datetime
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    # Add month column for grouping
    df["month"] = df["created_utc"].dt.to_period("M")

    # Fill NaN scores with 0
    df["score"] = df["score"].fillna(0)

    selected_posts = []

    if per_month_top > 0:
        # Select top N posts per month
        for month, group in df.groupby("month"):
            top_posts = group.nlargest(per_month_top, "score")
            selected_posts.append(top_posts)
            print(f"[info] Month {month}: selected {len(top_posts)} posts")

    if selected_posts:
        result_df = pd.concat(selected_posts, ignore_index=True)
    else:
        result_df = df

    # Remove duplicates by ID
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    after_dedup = len(result_df)

    if before_dedup != after_dedup:
        print(f"[info] Deduplicated posts: {before_dedup} -> {after_dedup}")

    # Apply limit if specified
    if limit_posts > 0 and len(result_df) > limit_posts:
        # Sort by created_utc desc, then score desc
        result_df = result_df.sort_values(
            ["created_utc", "score"], ascending=[False, False]
        ).head(limit_posts)
        print(f"[info] Limited to top {limit_posts} posts")

    return result_df


# -------- Comment Fetching --------
def traverse_comment_tree(
    node: dict[str, Any],
    depth: int,
    max_depth: int,
    min_score: int,
    min_length: int,
    post_id: str,
    subreddit: str,
    post_permalink: str,
    fetched_at: int,
) -> list[dict[str, Any]]:
    """Recursively traverse comment tree and extract valid comments."""
    comments = []

    if node.get("kind") != "t1":
        return comments

    data = node.get("data", {})

    # Apply filters
    score = data.get("score", 0) or 0
    body = data.get("body", "") or ""

    if score < min_score or len(body) < min_length:
        # Still traverse children even if this comment doesn't qualify
        pass
    else:
        # Extract comment data
        comment_id = data.get("id", "")
        link_id = data.get("link_id", "")
        extracted_post_id = (
            _extract_post_id_from_link_id(link_id) if link_id else post_id
        )

        comment = {
            "comment_id": comment_id,
            "post_id": extracted_post_id,
            "subreddit": subreddit,
            "post_permalink": post_permalink,
            "comment_permalink": _build_comment_permalink(post_permalink, comment_id),
            "body": body,
            "score": int(score),
            "author": data.get("author", ""),
            "created_utc": int(data.get("created_utc", 0)),
            "depth": depth,
            "fetched_at": fetched_at,
        }
        comments.append(comment)

    # Traverse children if within depth limit
    if depth < max_depth:
        replies = data.get("replies")
        if replies and isinstance(replies, dict) and "data" in replies:
            children = replies["data"].get("children", [])
            for child in children:
                child_comments = traverse_comment_tree(
                    child,
                    depth + 1,
                    max_depth,
                    min_score,
                    min_length,
                    post_id,
                    subreddit,
                    post_permalink,
                    fetched_at,
                )
                comments.extend(child_comments)

    return comments


def fetch_comments_for_post(
    permalink: str,
    post_id: str,
    subreddit: str,
    max_depth: int,
    min_score: int,
    min_length: int,
    per_post_topk: int,
    delay_min: float,
    delay_max: float,
) -> list[dict[str, Any]]:
    """Fetch and process comments for a single post."""
    # Build JSON URL
    json_url = permalink.rstrip("/") + "/.json"

    print(f"[info] Fetching comments for post {post_id}: {json_url}")

    # Fetch JSON
    json_data = _fetch_json(json_url)
    if not json_data:
        print(f"[warn] Failed to fetch comments for post {post_id}")
        return []

    # Parse comments
    if len(json_data) < 2:
        print(f"[warn] Unexpected JSON structure for post {post_id}")
        return []

    comments_listing = json_data[1]
    if "data" not in comments_listing:
        print(f"[warn] No comments data for post {post_id}")
        return []

    children = comments_listing["data"].get("children", [])
    if not children:
        print(f"[info] No comments found for post {post_id}")
        return []

    # Traverse comment tree
    fetched_at = int(time.time())
    all_comments = []

    for child in children:
        comments = traverse_comment_tree(
            child,
            0,
            max_depth,
            min_score,
            min_length,
            post_id,
            subreddit,
            permalink,
            fetched_at,
        )
        all_comments.extend(comments)

    # Sort by score descending and take top K
    all_comments.sort(key=lambda x: x["score"], reverse=True)
    top_comments = all_comments[:per_post_topk]

    print(
        f"[info] Post {post_id}: found {len(all_comments)} valid comments, kept top {len(top_comments)}"
    )

    # Be polite
    _sleep_between(delay_min, delay_max)

    return top_comments


# -------- Main Processing --------
def process_posts_file(
    posts_path: str,
    per_month_top: int,
    limit_posts: int,
    per_post_topk: int,
    min_score: int,
    min_length: int,
    max_depth: int,
    delay_min: float,
    delay_max: float,
) -> list[dict[str, Any]]:
    """Process posts file and fetch comments for selected posts."""
    # Read posts file
    print(f"[info] Reading posts from {posts_path}")

    if posts_path.endswith(".parquet"):
        df = pd.read_parquet(posts_path)
    elif posts_path.endswith(".csv"):
        df = pd.read_csv(posts_path)
    else:
        raise ValueError(f"Unsupported file format: {posts_path}")

    print(f"[info] Loaded {len(df)} posts")

    # Select posts
    selected_df = select_posts(df, per_month_top, limit_posts)
    print(f"[info] Selected {len(selected_df)} posts for comment fetching")

    if selected_df.empty:
        print("[info] No posts selected")
        return []

    # Fetch comments for each selected post
    all_comments = []
    successful_posts = 0

    for _, row in selected_df.iterrows():
        try:
            comments = fetch_comments_for_post(
                permalink=row["permalink"],
                post_id=row["id"],
                subreddit=row["subreddit"],
                max_depth=max_depth,
                min_score=min_score,
                min_length=min_length,
                per_post_topk=per_post_topk,
                delay_min=delay_min,
                delay_max=delay_max,
            )
            all_comments.extend(comments)
            successful_posts += 1

        except Exception as exc:
            print(f"[warn] Error processing post {row['id']}: {exc}")
            continue

    print(f"[info] Successfully processed {successful_posts}/{len(selected_df)} posts")
    print(f"[info] Collected {len(all_comments)} total comments")

    return all_comments


def write_comments_parquet(
    comments: list[dict[str, Any]], outdir: str, outfile: Optional[str] = None
) -> str:
    """Write comments to Parquet file with proper schema."""
    if not comments:
        print("[info] No comments to write")
        return ""

    # Create DataFrame
    df = pd.DataFrame(comments)

    # Ensure proper column order and types
    schema_columns = [
        "comment_id",
        "post_id",
        "subreddit",
        "post_permalink",
        "comment_permalink",
        "body",
        "score",
        "author",
        "created_utc",
        "depth",
        "fetched_at",
    ]

    # Reorder columns
    df = df[schema_columns]

    # Ensure proper types
    df["comment_id"] = df["comment_id"].astype(str)
    df["post_id"] = df["post_id"].astype(str)
    df["subreddit"] = df["subreddit"].astype(str)
    df["post_permalink"] = df["post_permalink"].astype(str)
    df["comment_permalink"] = df["comment_permalink"].astype(str)
    df["body"] = df["body"].astype(str)
    df["score"] = df["score"].astype("int64")
    df["author"] = df["author"].astype(str)
    df["created_utc"] = df["created_utc"].astype("int64")
    df["depth"] = df["depth"].astype("int64")
    df["fetched_at"] = df["fetched_at"].astype("int64")

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    after_dedup = len(df)

    if before_dedup != after_dedup:
        print(f"[info] Deduplicated comments: {before_dedup} -> {after_dedup}")

    # Determine output path
    if outfile:
        output_path = outfile
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        utc_today = datetime.utcnow().date().isoformat()
        snap_dir = os.path.join(outdir, utc_today)
        os.makedirs(snap_dir, exist_ok=True)
        output_path = os.path.join(snap_dir, "reddit_comments.parquet")

    # Write Parquet
    df.to_parquet(output_path, index=False)

    print(f"[result] Wrote {len(df)} comments to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch top comments for selected Reddit posts"
    )
    parser.add_argument(
        "--posts", type=str, required=True, help="Path to posts Parquet or CSV file"
    )
    parser.add_argument(
        "--per-month-top",
        type=int,
        default=200,
        help="Top N posts per month by score (0 = disable)",
    )
    parser.add_argument(
        "--limit-posts",
        type=int,
        default=0,
        help="Hard cap on total selected posts (0 = no cap)",
    )
    parser.add_argument(
        "--per-post-topk", type=int, default=30, help="Top K comments per post to keep"
    )
    parser.add_argument(
        "--min-score", type=int, default=2, help="Minimum comment score to keep"
    )
    parser.add_argument(
        "--min-length", type=int, default=20, help="Minimum comment body length to keep"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum comment tree depth to traverse",
    )
    parser.add_argument(
        "--delay-min", type=float, default=2.0, help="Minimum seconds between requests"
    )
    parser.add_argument(
        "--delay-max", type=float, default=4.0, help="Maximum seconds between requests"
    )
    parser.add_argument(
        "--outdir", type=str, default="data/raw", help="Base output directory"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Exact output file path (overrides outdir)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.posts):
        print(f"[error] Posts file not found: {args.posts}")
        return

    if args.per_month_top < 0:
        print("[error] per-month-top must be >= 0")
        return

    if args.limit_posts < 0:
        print("[error] limit-posts must be >= 0")
        return

    if args.per_post_topk <= 0:
        print("[error] per-post-topk must be > 0")
        return

    if args.max_depth < 0:
        print("[error] max-depth must be >= 0")
        return

    # Process posts and fetch comments
    comments = process_posts_file(
        posts_path=args.posts,
        per_month_top=args.per_month_top,
        limit_posts=args.limit_posts,
        per_post_topk=args.per_post_topk,
        min_score=args.min_score,
        min_length=args.min_length,
        max_depth=args.max_depth,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )

    # Write results
    output_path = write_comments_parquet(
        comments=comments, outdir=args.outdir, outfile=args.outfile
    )

    if output_path:
        print(f"[success] Comments enrichment complete: {output_path}")


if __name__ == "__main__":
    main()
