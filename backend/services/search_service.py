"""
Search Service — web search and knowledge extraction abstraction layer.
Supports multiple search backends with unified interface.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SearchService:
    """Unified web search service.

    Supports:
    - DuckDuckGo (no API key needed)
    - SerpAPI (with API key)
    - Custom search backend
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        # DuckDuckGo endpoint (no auth required)
        self._ddg_url = "https://api.duckduckgo.com/"
        self._serpapi_key = None  # Set via settings if available

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15.0)
        return self._http

    async def search(
        self,
        query: str,
        num_results: int = 10,
        source: str = "duckduckgo",
    ) -> list[dict[str, Any]]:
        """Execute a web search.

        Args:
            query: Search query string.
            num_results: Number of results to return.
            source: Search backend ("duckduckgo", "serpapi").

        Returns:
            List of result dicts with title, url, snippet keys.
        """
        if source == "serpapi":
            return await self._serpapi_search(query, num_results)
        return await self._duckduckgo_search(query, num_results)

    async def _duckduckgo_search(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API."""
        client = await self._client()
        results: list[dict[str, Any]] = []

        try:
            # DDG instant answers
            params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
            response = await client.get(self._ddg_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Abstract/Answer
            abstract = data.get("AbstractText", "")
            if abstract:
                results.append({
                    "title": data.get("AbstractTitle", "DuckDuckGo Answer"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": abstract,
                })

            # Related topics
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })
                if "Topics" in topic:
                    for subtopic in topic["Topics"][:3]:
                        results.append({
                            "title": subtopic.get("Text", "").split(" - ")[0] if " - " in subtopic.get("Text", "") else subtopic.get("Text", ""),
                            "url": subtopic.get("FirstURL", ""),
                            "snippet": subtopic.get("Text", ""),
                        })

            # Fallback: use HTML search if not enough results
            if len(results) < min(num_results, 5):
                html_results = await self._duckduckgo_html_search(query, num_results - len(results))
                results.extend(html_results)

        except Exception as exc:
            logger.warning("DuckDuckGo search failed", extra={"error": str(exc), "query": query})
            # Try HTML fallback
            try:
                results = await self._duckduckgo_html_search(query, num_results)
            except Exception as exc2:
                logger.error("DuckDuckGo HTML search also failed", extra={"error": str(exc2)})

        # Deduplicate
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        return unique[:num_results]

    async def _duckduckgo_html_search(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Fallback: scrape DuckDuckGo HTML search results."""
        client = await self._client()
        results: list[dict[str, Any]] = []

        response = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; JARVIS/1.0)",
            },
        )
        response.raise_for_status()

        # Simple HTML parsing for organic results
        from html.parser import HTMLParser

        class ResultParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.results: list[dict[str, Any]] = []
                self._in_result = False
                self._in_link = False
                self._current: dict[str, Any] = {}
                self._tag_stack: list[str] = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                self._tag_stack.append(tag)
                if tag == "a" and any(cls_val for name, cls_val in attrs if name == "class" and cls_val and "result__a" in cls_val):
                    self._in_result = True
                    self._in_link = True
                    for name, val in attrs:
                        if name == "href" and val:
                            self._current["url"] = val
                if tag == "a" and any(cls_val for name, cls_val in attrs if name == "class" and cls_val and "result__snippet" in cls_val):
                    pass  # Will capture text

            def handle_data(self, data: str) -> None:
                if not self._tag_stack:
                    return
                current_tag = self._tag_stack[-1]
                if self._in_result and self._in_link and current_tag == "a":
                    if "title" not in self._current:
                        self._current["title"] = data.strip()
                if current_tag == "a" and self._tag_stack and self._tag_stack[-1] == "a":
                    pass

            def handle_endtag(self, tag: str) -> None:
                if self._tag_stack:
                    self._tag_stack.pop()
                if tag == "a" and self._in_result:
                    self._in_link = False

        parser = ResultParser()
        parser.feed(response.text)

        # Simplified: extract from HTML text directly
        import re
        # Find result links
        result_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        urls = result_pattern.findall(response.text)
        snippets = snippet_pattern.findall(response.text)

        import html
        for i, (url, title) in enumerate(urls[:num_results]):
            snippet = ""
            if i < len(snippets):
                snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i]).strip())
            results.append({
                "title": html.unescape(re.sub(r"<[^>]+>", "", title).strip()),
                "url": url,
                "snippet": snippet,
            })

        return results[:num_results]

    async def _serpapi_search(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Search using SerpAPI (requires API key)."""
        if not self._serpapi_key:
            logger.warning("SerpAPI key not configured, falling back to DuckDuckGo")
            return await self._duckduckgo_search(query, num_results)

        client = await self._client()
        results: list[dict[str, Any]] = []

        response = await client.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": self._serpapi_key,
                "num": num_results,
                "engine": "google",
            },
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return results[:num_results]

    async def extract_content(self, url: str) -> str | None:
        """Extract readable content from a URL."""
        client = await self._client()
        try:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JARVIS/1.0)"},
                timeout=15.0,
                follow_redirects=True,
            )
            response.raise_for_status()

            # Try to extract main content
            content = response.text
            import re

            # Remove scripts and styles
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)

            # Extract text from HTML
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self.text_parts: list[str] = []
                    self._skip_tags = {"script", "style", "nav", "footer", "header"}

                def handle_data(self, data: str) -> None:
                    stripped = data.strip()
                    if stripped:
                        self.text_parts.append(stripped)

            extractor = TextExtractor()
            extractor.feed(content)

            text = "\n".join(extractor.text_parts)
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text[:10000]  # Limit to 10K chars

        except Exception as exc:
            logger.warning("Content extraction failed", extra={"url": url, "error": str(exc)})
            return None

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()


# Global singleton
search_service = SearchService()
