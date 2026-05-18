"""
Live web research for multi-agent debates (stdlib + optional duckduckgo-search).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; MultiAgentDialogueSimulator/1.0; +research/educational)"
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.I,
)


@dataclass
class ResearchHit:
    title: str
    snippet: str
    url: str
    source: str  # web | news | wikipedia | article


@dataclass
class ResearchBrief:
    topic: str
    fetched_at: str
    source_url: str = ""
    article_hit: Optional[ResearchHit] = None
    web_hits: List[ResearchHit] = field(default_factory=list)
    news_hits: List[ResearchHit] = field(default_factory=list)
    wiki_hits: List[ResearchHit] = field(default_factory=list)

    def to_context_block(self, max_chars: int = 2200) -> str:
        lines = [
            f"LIVE RESEARCH BRIEF — topic: {self.topic}",
            f"Fetched: {self.fetched_at}",
        ]
        if self.source_url:
            lines.append(f"Primary source: {self.source_url}")
        lines.extend(
            [
                "Agents may cite these facts naturally (do not read URLs aloud).",
                "",
            ]
        )
        if self.article_hit:
            h = self.article_hit
            lines.append("PRIMARY ARTICLE (read this first):")
            lines.append(f"- {h.title}: {h.snippet[:600]}")
            lines.append("")
        if self.news_hits:
            lines.append("RELATED HEADLINES:")
            for h in self.news_hits[:5]:
                lines.append(f"- {h.title}: {h.snippet[:200]}")
            lines.append("")
        if self.web_hits:
            lines.append("WEB FINDINGS:")
            for h in self.web_hits[:5]:
                lines.append(f"- {h.title}: {h.snippet[:200]}")
            lines.append("")
        if self.wiki_hits:
            lines.append("BACKGROUND:")
            for h in self.wiki_hits[:3]:
                lines.append(f"- {h.title}: {h.snippet[:250]}")
        text = "\n".join(lines)
        return text[:max_chars]

    def sources_markdown(self) -> str:
        rows = []
        if self.article_hit:
            h = self.article_hit
            rows.append(f"- **Article** [{h.title}]({h.url})")
        for group, label in (
            (self.news_hits, "News"),
            (self.web_hits, "Web"),
            (self.wiki_hits, "Wikipedia"),
        ):
            for h in group:
                rows.append(f"- **{label}** [{h.title}]({h.url})")
        return "\n".join(rows) if rows else "_No sources._"

    def to_dict(self) -> Dict[str, Any]:
        def _hits(xs: List[ResearchHit]) -> List[Dict[str, str]]:
            return [{"title": x.title, "snippet": x.snippet, "url": x.url, "source": x.source} for x in xs]

        return {
            "topic": self.topic,
            "fetched_at": self.fetched_at,
            "source_url": self.source_url,
            "article_hit": (
                {
                    "title": self.article_hit.title,
                    "snippet": self.article_hit.snippet,
                    "url": self.article_hit.url,
                    "source": self.article_hit.source,
                }
                if self.article_hit
                else None
            ),
            "web_hits": _hits(self.web_hits),
            "news_hits": _hits(self.news_hits),
            "wiki_hits": _hits(self.wiki_hits),
        }


def _coerce_hit(obj: Any) -> Optional[ResearchHit]:
    if obj is None:
        return None
    if isinstance(obj, ResearchHit):
        return obj
    if isinstance(obj, dict):
        return ResearchHit(
            title=str(obj.get("title", "")),
            snippet=str(obj.get("snippet", "")),
            url=str(obj.get("url", "")),
            source=str(obj.get("source", "web")),
        )
    return None


def _coerce_hits(items: Any) -> List[ResearchHit]:
    if not items:
        return []
    out: List[ResearchHit] = []
    for item in items:
        h = _coerce_hit(item)
        if h and h.title:
            out.append(h)
    return out


def normalize_research_brief(obj: Any) -> Optional[ResearchBrief]:
    """Upgrade stale session-state briefs (missing new fields) to current ResearchBrief."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        ah = obj.get("article_hit")
        return ResearchBrief(
            topic=str(obj.get("topic", "Research")),
            fetched_at=str(obj.get("fetched_at", datetime.now().isoformat(timespec="seconds"))),
            source_url=str(obj.get("source_url", "")),
            article_hit=_coerce_hit(ah),
            web_hits=_coerce_hits(obj.get("web_hits")),
            news_hits=_coerce_hits(obj.get("news_hits")),
            wiki_hits=_coerce_hits(obj.get("wiki_hits")),
        )
    if isinstance(obj, ResearchBrief):
        if hasattr(obj, "source_url"):
            return obj
        return ResearchBrief(
            topic=getattr(obj, "topic", "Research"),
            fetched_at=getattr(obj, "fetched_at", datetime.now().isoformat(timespec="seconds")),
            source_url=getattr(obj, "source_url", ""),
            article_hit=getattr(obj, "article_hit", None),
            web_hits=list(getattr(obj, "web_hits", []) or []),
            news_hits=list(getattr(obj, "news_hits", []) or []),
            wiki_hits=list(getattr(obj, "wiki_hits", []) or []),
        )
    return None


RESEARCH_COMMAND = re.compile(
    r"(?:^|\b)(?:research|look\s*up|investigate|find\s+out\s+about|search\s+for)"
    r"(?:\s+on|\s+about|\s+into)?\s+(.+)$",
    re.I,
)


def parse_research_command(text: str) -> Optional[str]:
    """Extract topic from 'research war', 'look up climate change', etc."""
    t = (text or "").strip()
    if not t:
        return None
    m = RESEARCH_COMMAND.search(t)
    if m:
        topic = m.group(1).strip().strip("?.!")
        if len(topic) >= 2:
            return topic
    return None


def is_url(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("http://") or t.startswith("https://")


def extract_url(text: str) -> Optional[str]:
    m = URL_PATTERN.search(text or "")
    if m:
        return m.group(0).rstrip(".,);]")
    t = (text or "").strip()
    if is_url(t):
        return t
    return None


def _meta_content(html: str, prop: str) -> str:
    for pattern in (
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(prop)}["\']',
        rf'<meta[^>]+name=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
    ):
        m = re.search(pattern, html, re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _json_ld_description(html: str) -> str:
    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.I | re.S,
    ):
        try:
            data = json.loads(block)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description") or item.get("abstract")
                if isinstance(desc, str) and len(desc) > 40:
                    return desc[:800]
        except json.JSONDecodeError:
            continue
    return ""


def fetch_article_from_url(url: str) -> ResearchHit:
    """Fetch title + summary from a news/article URL."""
    html = _fetch_text(url, timeout=20)
    if not html:
        raise ValueError(f"Could not fetch URL: {url}")

    title = (
        _meta_content(html, "og:title")
        or _meta_content(html, "twitter:title")
        or ""
    )
    if not title:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
        title = m.group(1).strip() if m else url

    title = re.sub(r"\s*[-|]\s*BBC News.*$", "", title, flags=re.I).strip()
    title = re.sub(r"\s+", " ", title)

    snippet = (
        _meta_content(html, "og:description")
        or _meta_content(html, "description")
        or _meta_content(html, "twitter:description")
        or _json_ld_description(html)
    )
    if not snippet:
        # First paragraph heuristic
        m = re.search(r"<p[^>]*>([^<]{80,500})</p>", html, re.I)
        if m:
            snippet = re.sub(r"<[^>]+>", "", m.group(1))
    snippet = re.sub(r"\s+", " ", (snippet or "")).strip()[:800]

    if not snippet:
        snippet = f"Article at {urllib.parse.urlparse(url).netloc} — open link for full text."

    return ResearchHit(title=title, snippet=snippet, url=url, source="article")


def parse_research_input(raw: str) -> Tuple[str, str, Optional[str], Optional[ResearchHit]]:
    """
    Returns (display_topic, search_query, optional_source_url, optional_article_hit).
    URLs are fetched for title/summary; search uses the article title, not the URL string.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Research topic cannot be empty")

    url = extract_url(raw)
    if url:
        article = fetch_article_from_url(url)
        display = article.title
        words = re.findall(r"[A-Za-z0-9]{3,}", display)
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was", "has",
            "bbc", "news", "says", "after", "over", "into", "about",
        }
        query_words = [w for w in words if w.lower() not in stop][:8]
        search_q = " ".join(query_words) if query_words else display[:80]
        return display, search_q, url, article

    return raw, raw, None, None


def _fetch_json(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        logger.warning("fetch_json failed %s: %s", url[:80], e)
        return None


def _fetch_text(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("fetch_text failed: %s", e)
        return None


def search_wikipedia(query: str, limit: int = 3) -> List[ResearchHit]:
    if is_url(query) or len(query) < 3:
        return []
    q = urllib.parse.quote(query)
    url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={q}&format=json&srlimit={limit}"
    )
    data = _fetch_json(url)
    if not data:
        return []
    hits = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        hits.append(ResearchHit(title=title, snippet=snippet, url=page_url, source="wikipedia"))
    return hits


def search_news_rss(query: str, limit: int = 6) -> List[ResearchHit]:
    if is_url(query) or len(query) < 3:
        return []
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    xml_text = _fetch_text(url)
    if not xml_text:
        return []
    hits: List[ResearchHit] = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item")[:limit]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            desc = re.sub(r"<[^>]+>", "", desc)
            if title:
                hits.append(ResearchHit(title=title, snippet=desc[:300], url=link or "#", source="news"))
    except ET.ParseError as e:
        logger.warning("RSS parse error: %s", e)
    return hits


def search_duckduckgo(query: str, limit: int = 6) -> List[ResearchHit]:
    if is_url(query) or len(query) < 3:
        return []
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return []
    hits: List[ResearchHit] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                hits.append(
                    ResearchHit(
                        title=r.get("title", "")[:200],
                        snippet=r.get("body", "")[:400],
                        url=r.get("href", ""),
                        source="web",
                    )
                )
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
    return hits


def run_research(topic: str, include_news: bool = True, include_web: bool = True) -> ResearchBrief:
    """Aggregate live web + news + Wikipedia for a debate topic (or article URL)."""
    display_topic, search_query, source_url, article = parse_research_input(topic)

    brief = ResearchBrief(
        topic=display_topic,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
        source_url=source_url or "",
        article_hit=article,
    )
    if article:
        brief.topic = article.title or display_topic

    if include_news:
        brief.news_hits = search_news_rss(search_query, limit=6)
    if include_web:
        brief.web_hits = search_duckduckgo(search_query, limit=6)
    brief.wiki_hits = search_wikipedia(search_query, limit=3)

    if not brief.web_hits and not brief.news_hits and not brief.wiki_hits and not brief.article_hit:
        brief.wiki_hits = search_wikipedia(search_query.split()[0] if search_query else "news", limit=3)

    return brief


def research_snippet_for_message(message: str, max_results: int = 3) -> str:
    """Quick lookup based on the last spoken line (for 'agent digs deeper')."""
    words = re.findall(r"[A-Za-z]{4,}", message or "")
    if not words:
        return ""
    stop = {"that", "this", "with", "from", "have", "what", "when", "your", "they", "about"}
    words = [w for w in sorted(set(words), key=len, reverse=True) if w.lower() not in stop][:4]
    query = " ".join(words[:3])
    if not query:
        return ""
    brief = run_research(query, include_news=True, include_web=False)
    return brief.to_context_block(max_chars=900)
