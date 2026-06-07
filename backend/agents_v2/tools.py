"""
Agent Tools
===========
Shared tools that agents can use for common operations:
web search, math, code execution, formatting, etc.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class AgentTools:
    """
    Collection of utility functions available to all agents.
    Each method is a self-contained tool with input/output documentation.
    """

    # ── Math & Calculation ────────────────────────────────────

    @staticmethod
    def calculate(expression: str) -> str:
        """
        Safely evaluate a mathematical expression.
        Only supports basic arithmetic: + - * / ( ) and numbers.
        """
        # Whitelist-safe characters only
        safe_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.\,]+$")
        if not safe_pattern.match(expression):
            return "Error: Expression contains disallowed characters"

        try:
            # Use Python's safe eval with restricted globals
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def extract_code_blocks(text: str) -> List[Dict[str, str]]:
        """
        Extract ```language ... ``` code blocks from text.
        Returns list of {"language": str, "code": str}.
        """
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [
            {"language": lang.strip() or "unknown", "code": code.strip()}
            for lang, code in matches
        ]

    @staticmethod
    def format_json(data: Any, indent: int = 2) -> str:
        """Format data as pretty-printed JSON string."""
        import json
        return json.dumps(data, indent=indent, default=str, ensure_ascii=False)

    @staticmethod
    def truncate(text: str, max_length: int = 2000) -> str:
        """Truncate text to max_length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text."""
        pattern = r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        return re.findall(pattern, text)

    @staticmethod
    def current_timestamp() -> str:
        """Get current UTC timestamp as ISO string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def split_markdown_sections(text: str) -> Dict[str, str]:
        """
        Split markdown text into sections by headings.
        Returns {heading_text: content}.
        """
        sections = {}
        current_heading = "preamble"
        current_content: List[str] = []

        for line in text.split("\n"):
            if line.startswith("#"):
                if current_content:
                    sections[current_heading] = "\n".join(current_content).strip()
                current_heading = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_heading] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token count estimate (4 chars per token)."""
        return len(text) // 4
