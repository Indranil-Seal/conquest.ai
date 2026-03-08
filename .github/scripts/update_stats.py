#!/usr/bin/env python3
"""
conquest.ai — Repository Activity Stats Updater

Fetches repository activity from the GitHub API and writes a formatted
Markdown report to stats/activity.md. Runs daily via GitHub Actions.

Data sources:
    - Traffic Views  : GET /repos/{owner}/{repo}/traffic/views
                       Returns daily view counts + unique visitors (last 14 days).
                       GitHub does NOT expose individual viewer usernames.
    - Traffic Clones : GET /repos/{owner}/{repo}/traffic/clones
                       Returns daily clone counts + unique cloners (last 14 days).
                       GitHub does NOT expose individual cloner usernames.
    - Forks          : GET /repos/{owner}/{repo}/forks?sort=newest
                       Returns full fork metadata including owner.login (username)
                       and created_at timestamp — the only activity where GitHub
                       exposes individual user identities.
    - Stargazers     : GET /repos/{owner}/{repo}/stargazers
                       Returns the list of users who starred the repo with timestamps
                       (requires Accept: application/vnd.github.star+json header).
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

OWNER = os.environ["REPO_OWNER"]
REPO = os.environ["REPO_NAME"]
TOKEN = os.environ["GITHUB_TOKEN"]
STATS_FILE = Path("stats/activity.md")

FORKS_PER_PAGE = 50    # how many recent forks to display
STARS_PER_PAGE = 50    # how many recent stargazers to display


def gh_get(path: str, extra_headers: dict | None = None) -> dict | list:
    """
    Make an authenticated GET request to the GitHub REST API.

    Args:
        path: API path relative to /repos/{owner}/{repo}/ (e.g. "traffic/views")
              or a full URL starting with "https://".
        extra_headers: Additional headers to include in the request.

    Returns:
        Parsed JSON response (dict or list).

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
    """
    if path.startswith("https://"):
        url = path
    else:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/{path}"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "conquest-ai-stats-updater/1.0")
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fmt_dt(iso_str: str) -> str:
    """Convert an ISO 8601 UTC string to a human-readable format."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def fmt_date(iso_str: str) -> str:
    """Return just the date portion of an ISO 8601 string."""
    return iso_str[:10]


def build_views_section(views_data: dict) -> str:
    """
    Format the traffic views data into a Markdown section.

    GitHub returns total views, unique visitors, and a per-day breakdown
    for the past 14 days. Individual usernames are NOT available.
    """
    total = views_data.get("count", 0)
    unique = views_data.get("uniques", 0)

    lines = [
        "## 👁️ Page Views (last 14 days)",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total views | **{total:,}** |",
        f"| Unique visitors | **{unique:,}** |",
        "",
        "> ℹ️ GitHub does not expose individual visitor usernames — "
        "only aggregate counts are available through the Traffic API.",
        "",
        "### Daily Breakdown",
        "",
        "| Date | Views | Unique Visitors |",
        "|------|-------|-----------------|",
    ]

    for day in views_data.get("views", []):
        lines.append(
            f"| {fmt_date(day['timestamp'])} | {day['count']:,} | {day['uniques']:,} |"
        )

    if not views_data.get("views"):
        lines.append("| — | No data yet | — |")

    return "\n".join(lines)


def build_clones_section(clones_data: dict) -> str:
    """
    Format the traffic clones data into a Markdown section.

    GitHub returns total clone count, unique cloners, and a per-day breakdown
    for the past 14 days. Individual usernames are NOT available.
    """
    total = clones_data.get("count", 0)
    unique = clones_data.get("uniques", 0)

    lines = [
        "## 📥 Repository Clones (last 14 days)",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total clones | **{total:,}** |",
        f"| Unique cloners | **{unique:,}** |",
        "",
        "> ℹ️ GitHub does not expose individual cloner usernames — "
        "only aggregate counts are available through the Traffic API.",
        "",
        "### Daily Breakdown",
        "",
        "| Date | Clones | Unique Cloners |",
        "|------|--------|----------------|",
    ]

    for day in clones_data.get("clones", []):
        lines.append(
            f"| {fmt_date(day['timestamp'])} | {day['count']:,} | {day['uniques']:,} |"
        )

    if not clones_data.get("clones"):
        lines.append("| — | No data yet | — |")

    return "\n".join(lines)


def build_forks_section(forks_data: list) -> str:
    """
    Format fork data into a Markdown section with usernames and timestamps.

    Fork information IS publicly available via the GitHub API — this is the
    only activity type where individual user identities can be tracked.
    """
    lines = [
        "## 🍴 Recent Forks",
        "",
        f"Showing the {len(forks_data)} most recent fork(s). "
        "Usernames and timestamps are available because forks are public GitHub events.",
        "",
        "| # | Username | GitHub Profile | Forked At |",
        "|---|----------|----------------|-----------|",
    ]

    if not forks_data:
        lines.append("| — | No forks yet | — | — |")
    else:
        for i, fork in enumerate(forks_data, 1):
            login = fork["owner"]["login"]
            profile_url = fork["owner"]["html_url"]
            forked_at = fmt_dt(fork["created_at"])
            lines.append(
                f"| {i} | `{login}` | [@{login}]({profile_url}) | {forked_at} |"
            )

    return "\n".join(lines)


def build_stars_section(stars_data: list) -> str:
    """
    Format stargazer data into a Markdown section.

    The starred_at timestamp requires the special Accept header
    'application/vnd.github.star+json' on the stargazers endpoint.
    """
    lines = [
        "## ⭐ Recent Stargazers",
        "",
        f"Showing the {len(stars_data)} most recent star(s).",
        "",
        "| # | Username | GitHub Profile | Starred At |",
        "|---|----------|----------------|------------|",
    ]

    if not stars_data:
        lines.append("| — | No stars yet | — | — |")
    else:
        for i, entry in enumerate(stars_data, 1):
            user = entry.get("user", {})
            login = user.get("login", "unknown")
            profile_url = user.get("html_url", "#")
            starred_at = fmt_dt(entry.get("starred_at", "")) if entry.get("starred_at") else "—"
            lines.append(
                f"| {i} | `{login}` | [@{login}]({profile_url}) | {starred_at} |"
            )

    return "\n".join(lines)


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo_url = f"https://github.com/{OWNER}/{REPO}"

    print(f"Fetching activity stats for {OWNER}/{REPO}...")

    # Fetch all data
    views_data = gh_get("traffic/views")
    print(f"  Views: {views_data.get('count', 0)} total, {views_data.get('uniques', 0)} unique")

    clones_data = gh_get("traffic/clones")
    print(f"  Clones: {clones_data.get('count', 0)} total, {clones_data.get('uniques', 0)} unique")

    forks_data = gh_get(f"forks?sort=newest&per_page={FORKS_PER_PAGE}")
    print(f"  Forks: {len(forks_data)} recent")

    # Starred_at timestamps require a special Accept header
    stars_data = gh_get(
        f"stargazers?per_page={STARS_PER_PAGE}",
        extra_headers={"Accept": "application/vnd.github.star+json"},
    )
    print(f"  Stargazers: {len(stars_data)} recent")

    # Assemble the report
    sections = [
        f"# conquest.ai — Repository Activity Dashboard",
        f"",
        f"> **Auto-updated daily** by GitHub Actions · Last update: {now}",
        f"> Source: [github.com/{OWNER}/{REPO}]({repo_url})",
        f"",
        f"---",
        f"",
        build_views_section(views_data),
        f"",
        f"---",
        f"",
        build_clones_section(clones_data),
        f"",
        f"---",
        f"",
        build_forks_section(forks_data),
        f"",
        f"---",
        f"",
        build_stars_section(stars_data),
        f"",
        f"---",
        f"",
        f"## 📝 Notes on Tracking Limitations",
        f"",
        f"| Activity | Username Tracking | What's Available |",
        f"|----------|-------------------|------------------|",
        f"| Opening the GitHub page | ❌ Not possible | Aggregate view counts (14-day window) |",
        f"| Clicking/viewing code files | ❌ Not possible | Not tracked by GitHub at all |",
        f"| Cloning the repository | ❌ Not possible | Aggregate clone counts (14-day window) |",
        f"| Forking the repository | ✅ Available | Usernames + timestamps via public Forks API |",
        f"| Starring the repository | ✅ Available | Usernames + timestamps via Stargazers API |",
        f"",
        f"GitHub's privacy architecture (Camo image proxy, anonymous clone protocol) prevents",
        f"tracking individual usernames for page views and clones. This is by design.",
        f"",
    ]

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text("\n".join(sections), encoding="utf-8")
    print(f"Written to {STATS_FILE}")


if __name__ == "__main__":
    main()
