"""Unified FTS5 search service."""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger

logger = get_logger(__name__)

FTS_QUERIES: dict[str, str] = {
    "tasks": """
        SELECT t.id, 'task' AS type, t.title, t.description,
               rank
        FROM tasks_fts
        JOIN tasks t ON t.id = tasks_fts.rowid
        WHERE tasks_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """,
    "notes": """
        SELECT n.id, 'note' AS type, n.title, n.content,
               rank
        FROM notes_fts
        JOIN notes n ON n.id = notes_fts.rowid
        WHERE notes_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """,
    "memory": """
        SELECT m.id, 'memory' AS type, m.fact AS title, m.fact AS snippet,
               rank
        FROM memory_fts
        JOIN memory_entries m ON m.id = memory_fts.rowid
        WHERE memory_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """,
}


def _format_fts_query(raw: str) -> str:
    """Convert user query to FTS5 query syntax."""
    terms = raw.strip().split()
    if not terms:
        return ""
    if len(terms) == 1:
        return f'"{terms[0]}"*'
    return " AND ".join(f'"{t}"*' for t in terms)


class SearchService:
    async def search(
        self,
        db: AsyncSession,
        query_str: str,
        limit: int = 20,
        source_type: str | None = None,
    ) -> list[dict]:
        if not query_str.strip():
            return []

        fts_query = _format_fts_query(query_str)
        sources = [source_type] if source_type else list(FTS_QUERIES)
        results: list[dict] = []

        for source in sources:
            sql = FTS_QUERIES.get(source)
            if not sql:
                continue
            try:
                result = await db.execute(text(sql), {"query": fts_query, "limit": limit})
                rows = result.mappings().all()
            except Exception:
                logger.warning("fts_search_failed", source=source, query=query_str)
                continue

            for row in rows:
                snippet = row.get("description") or row.get("content") or row.get("snippet", "")
                results.append({
                    "id": row["id"],
                    "type": row["type"],
                    "title": row["title"],
                    "snippet": snippet[:200] if snippet else "",
                    "score": float(row.get("rank", 0)),
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]


search_service = SearchService()
