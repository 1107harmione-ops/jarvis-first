# Production-Grade Research Agent for JARVIS

**Date**: 2026-06-06
**Status**: Draft
**Author**: JARVIS AI

## Overview

Build a production-grade Research Agent system for JARVIS with internet research, deep research, source verification, fact checking, report generation, knowledge extraction, and web summarization.

**Models**:
- **Minimax M2.1** (`llm/minimax.py`) — Research model: search query generation, web search synthesis, long-context understanding
- **DeepSeek V4 Flash** (`llm/deepseek.py`) — Verification + report generation: fact-checking, structured report composition, comparative analysis

**Pattern**: Service-layer separation (matches existing `SearchService`, `VoiceService`)
**Scope**: In-place replacement of existing `agents_v2/research_agent.py` (505 lines)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer (api/research.py)          │
│  POST /search  POST /deep  POST /report  GET /history   │
│  POST /verify  GET /reports/{id}  GET /sources          │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│               Agent Layer (agents_v2/research_agent.py)  │
│           Thin LangGraph node — delegates to service     │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│            Service Layer (services/research_service.py)  │
│  Plan → Search → Collect → Rank → Verify → Summarize    │
│  → Report → Respond                                     │
│  SourceScorer, FactChecker, ReportGenerator modules      │
└─────────────┬────────────────────┬──────────────────────┘
              │                    │
     ┌────────▼──────┐    ┌───────▼────────┐
     │ SearchService │    │   MongoDB      │
     │ (existing)    │    │ research_*     │
     └───────────────┘    └────────────────┘
```

### Components

| Component | File | Role |
|-----------|------|------|
| `ResearchService` | `services/research_service.py` | Pipeline logic: plan, search, collect, rank, verify, summarize, report |
| `ResearchAgent` | `agents_v2/research_agent.py` | LangGraph agent node — type detection, delegation, state management |
| `ResearchAPI` | `api/research.py` | FastAPI router at `/api/v2/research/` |
| MongoDB collections | `research_reports`, `research_sources`, `research_cache` | Persistence layer |

---

## Research Pipeline (8 Stages)

### 1. `plan_research(message, context)` — Plan
- Detect research type from message (7 types)
- Generate optimized search queries via Minimax
- Determine depth (quick/moderate/deep/comprehensive)
- Estimate sources needed

### 2. `execute_search(plan)` — Search
- Execute queries via `SearchService.search()`
- Deduplicate by URL
- Merge results from all queries

### 3. `collect_sources(raw_results)` — Collect
- Fetch full page content via `SearchService.extract_content()`
- Extract metadata (publish date, author, domain)
- Filter out broken/non-responsive URLs
- Truncate content to 10K chars per source

### 4. `rank_sources(collected)` — Rank
Score each source on 5 weighted dimensions:

| Dimension | Weight | Measurement |
|-----------|--------|------------|
| Authority | 30% | Domain reputation (`.edu`/`.gov` +0.2, known tech domains +0.1), author credentials |
| Freshness | 20% | Recency: `max(0, 1 - days_since_publish/365)` |
| Accuracy | 25% | Factual correctness signals, citation quality |
| Relevance | 15% | Keyword overlap with query, topical match |
| Popularity | 10% | Engagement signals, references from other sources |

### 5. `verify_facts(sources, message)` — Verify [DeepSeek V4 Flash]
- Extract claims from all source content
- Cross-reference claims across sources
- Tag each claim: `verified` (2+ sources agree), `contradicted` (sources disagree), `unverifiable` (single source only)
- Output: `{verified_claims: [...], contradicted_claims: [...], unverifiable_claims: [...], overall_confidence: float}`

### 6. `synthesize(ranked_sources, verification)` — Summarize [Minimax M2.1]
- Condense top-k sources (max 15) into coherent synthesis
- Preserve source citations [1], [2], etc.
- Note contradictions where they exist

### 7. `generate_report(synthesis, verification, research_type)` — Report [DeepSeek V4 Flash]
Generate structured report based on research type:

| Type | Sections |
|------|----------|
| **Quick** | Concise answer + 2-3 sources |
| **Deep** | Executive summary + key findings + detailed analysis + conclusions + references |
| **Comparative** | Side-by-side comparison + pros/cons per option + recommendation + sources |
| **Technical** | Architecture overview + specifications + trade-offs + implementation notes |
| **Market** | Trends + competitive landscape + market size + key players + outlook |
| **Product** | Features + pricing tiers + user reviews + ratings + pros/cons + verdict |
| **Architecture** | System design + components + data flow + tech stack + alternatives |

### 8. Response
- Compose final response from report
- Store report in `research_reports` collection
- Store knowledge in `knowledge` collection for cross-agent reuse
- Cache sources in `research_cache` (24h TTL)

---

## Service Layer — ResearchService

**File**: `services/research_service.py` (~400 lines)

```python
class ResearchService:
    """Core research pipeline. Singleton pattern."""

    # Pipeline methods
    async def plan_research(self, message: str, context: dict | None = None) -> ResearchPlan
    async def execute_search(self, plan: ResearchPlan) -> list[dict]
    async def collect_sources(self, raw_results: list[dict]) -> list[dict]
    async def rank_sources(self, collected: list[dict]) -> list[dict]
    async def verify_facts(self, sources: list[dict], message: str) -> FactCheckResult
    async def synthesize(self, sources: list[dict], verification: FactCheckResult) -> str
    async def generate_report(self, synthesis: str, verification: FactCheckResult, research_type: str, depth: str) -> dict

    # Composite methods (for direct API use)
    async def quick_search(self, query: str, max_sources: int = 10) -> dict
    async def deep_research(self, query: str, research_type: str, depth: str, max_sources: int) -> dict
    async def verify_content(self, content: str, context: str | None = None) -> dict

    # Internal helpers
    def _detect_research_type(self, message: str) -> str  # 7 types
    def _score_source(self, source: dict, query: str) -> dict  # 5-dimension scoring
    def _extract_domain(self, url: str) -> str
    def _compute_freshness(self, published_date: datetime | None) -> float
    def _compute_authority(self, domain: str, author: str | None) -> float
    def _build_cache_key(self, query: str, depth: str, research_type: str) -> str
```

**Error handling**: Each pipeline stage catches exceptions and returns partial results with an `error` field. Downstream stages check for errors before processing.

---

## Agent Layer — ResearchAgent

**File**: `agents_v2/research_agent.py` (~500 lines, replaces existing)

### Research Type Detection (7 types)

Scaled keyword scoring across the user message. Each type has ~10-15 keywords/phrases:

| Type | Keywords |
|------|----------|
| quick | "what is", "who is", "define", "explain briefly" |
| deep | "deep research", "comprehensive", "in-depth", "investigate" |
| comparative | "compare", "vs", "versus", "differences", "pros and cons" |
| technical | "technical", "architecture", "specification", "implementation" |
| market | "market", "trends", "competitive", "industry" |
| product | "product", "features", "pricing", "reviews", "best" |
| architecture | "system design", "architecture", "design pattern", "flow" |

### LangGraph Integration

Follows the `BaseAgent` pattern exactly:

```python
class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="research", model_name="minimax", ...)
        get_agent_registry().register(self)

    async def process(self, state: AgentState) -> AgentState:
        # 1. Read message + context from state
        # 2. Load memory context
        # 3. Detect research type
        # 4. Call ResearchService pipeline
        # 5. Store results in state
        # 6. Persist to knowledge collection
        # 7. Log execution
        return state
```

---

## API Layer

**File**: `api/research.py` — `APIRouter(prefix="/api/v2/research", tags=["research"])`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v2/research/search` | Quick web search with synthesis |
| `POST` | `/api/v2/research/deep` | Full 8-stage deep research |
| `POST` | `/api/v2/research/report` | Generate report from provided sources |
| `POST` | `/api/v2/research/verify` | Fact-check content against web |
| `GET` | `/api/v2/research/history` | Paginated research history |
| `GET` | `/api/v2/research/reports/{report_id}` | Get specific report |
| `GET` | `/api/v2/research/sources` | Search/filter cached sources |

### Request/Response Models

```python
class ResearchType(str, Enum):
    QUICK = "quick"
    DEEP = "deep"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"
    MARKET = "market"
    PRODUCT = "product"
    ARCHITECTURE = "architecture"

class ResearchDepth(str, Enum):
    QUICK = "quick"
    MODERATE = "moderate"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"

class SourceScore(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    overall_score: float
    authority_score: float
    freshness_score: float
    relevance_score: float

class FactCheckResult(BaseModel):
    verified_claims: list[str] = []
    contradicted_claims: list[str] = []
    unverifiable_claims: list[str] = []
    overall_confidence: float = 0.0

class ResearchSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    depth: ResearchDepth = ResearchDepth.QUICK
    max_sources: int = Field(default=10, ge=3, le=50)

class ResearchSearchResponse(BaseModel):
    query: str
    research_type: ResearchType
    answer: str
    sources: list[SourceScore]
    source_count: int
    processing_time_ms: float

class DeepResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    research_type: ResearchType = ResearchType.DEEP
    depth: ResearchDepth = ResearchDepth.MODERATE
    max_sources: int = Field(default=20, ge=5, le=100)

class DeepResearchResponse(BaseModel):
    report_id: str
    title: str
    research_type: str
    depth: str
    executive_summary: str
    key_findings: list[str]
    detailed_analysis: str | None
    pros: list[str] | None
    cons: list[str] | None
    recommendations: list[str] | None
    conclusions: str | None
    sources: list[SourceScore]
    source_count: int
    fact_check: FactCheckResult | None
    processing_time_ms: float
    created_at: str

class VerifyContentRequest(BaseModel):
    content: str = Field(min_length=10, max_length=50000)
    context: str | None = Field(default=None, max_length=5000)

class VerifyContentResponse(BaseModel):
    verified_claims: list[str]
    contradicted_claims: list[str]
    unverifiable_claims: list[str]
    overall_confidence: float
    analysis: str
```

---

## Data Layer — MongoDB

### Collections

#### `research_reports`
```python
{
    "_id": ObjectId,
    "user_id": str,
    "session_id": str,
    "title": str,
    "query": str,
    "research_type": str,
    "depth": str,
    "executive_summary": str,
    "key_findings": [str],
    "detailed_analysis": str | None,
    "pros": [str] | None,
    "cons": [str] | None,
    "recommendations": [str] | None,
    "conclusions": str | None,
    "sources": [{"title": str, "url": str, "snippet": str, "domain": str,
                  "overall_score": float, "authority_score": float,
                  "freshness_score": float, "relevance_score": float}],
    "source_count": int,
    "fact_check": {"verified_claims": [str], "contradicted_claims": [str],
                    "unverifiable_claims": [str], "overall_confidence": float} | None,
    "metadata": dict,
    "tags": [str],
    "created_at": datetime,
    "updated_at": datetime,
}
```

#### `research_sources`
```python
{
    "_id": ObjectId,
    "url": str,
    "title": str,
    "snippet": str,
    "content": str,
    "domain": str,
    "published_date": datetime | None,
    "author": str | None,
    "authority_score": float,
    "freshness_score": float,
    "accuracy_score": float,
    "relevance_score": float,
    "popularity_score": float,
    "overall_score": float,
    "query": str,
    "tags": [str],
    "access_count": int,
    "created_at": datetime,
}
```

#### `research_cache`
```python
{
    "_id": ObjectId,
    "cache_key": str,
    "query": str,
    "research_type": str,
    "report_id": str | None,
    "synthesis": str | None,
    "source_urls": [str],
    "ttl": datetime,
    "created_at": datetime,
}
```

### Schema Builders (database/schemas.py)

- `new_research_report_doc(...)` → dict
- `new_research_source_doc(...)` → dict
- `new_research_cache_doc(...)` → dict

### Indexes

```python
"research_reports": [
    ("user_id", 1),
    ("created_at", -1),
    ("research_type", 1),
    [("user_id", 1), ("created_at", -1)],
    [("tags", 1)],
]
"research_sources": [
    ("url", 1),
    ("domain", 1),
    ("overall_score", -1),
    ("created_at", -1),
]
"research_cache": [
    ("cache_key", 1),
    ("ttl", 1),
]
```

### MongoDB Manager Accessors (database/mongodb.py)

```python
@property
def research_reports(self) -> AsyncIOMotorCollection: ...
@property
def research_sources(self) -> AsyncIOMotorCollection: ...
@property
def research_cache(self) -> AsyncIOMotorCollection: ...
```

---

## Configuration (config/settings.py)

```python
# ── Research ───────────────────────────────────────────
RESEARCH_CACHE_TTL_HOURS: int = Field(default=24, ge=1, description="Research cache TTL")
RESEARCH_MAX_SOURCES: int = Field(default=50, ge=5, le=200, description="Max sources per research")
RESEARCH_DEFAULT_DEPTH: str = "moderate"
```

(No separate `DEEPSEEK_V4_MODEL` — the existing `DEEPSEEK_MODEL = "deepseek-chat"` serves as V4 Flash.)

---

## Error Handling & Monitoring

### Stage-Level Error Isolation
Each pipeline stage catches its own exceptions and returns results with an `error` field. Downstream stages skip processing if upstream results contain errors:

- **SEARCH fails** → Return "No relevant sources found"
- **COLLECT fails** → Continue with search snippets only (no full content)
- **VERIFY fails** → Return report without fact-check (tagged `fact_check: null`)
- **REPORT fails** → Fall back to synthesis text as the response

### Timeouts
- Each SearchService call: 15s
- Full deep research pipeline: 120s
- Quick search: 30s
- Verification: 60s

### Observability
- Uses existing `AgentMonitor` for session tracking
- `_store_agent_log()` for per-action logging with stage-level granularity
- Each stage logs: `{agent: "research", action: "research_verify", status: "success|failed", duration_ms: N}`
- Pipeline failures include the failing stage name in error messages

### Fallback Chain
1. DeepSeek (verification + reports) → Minimax (fallback) → "Verification unavailable"
2. Minimax (synthesis) → No fallback (Minimax is the research model)
3. Database storage failure → Silently absorbed (non-critical)

---

## Files to Create/Modify

| File | Action | Est. Lines | Description |
|------|--------|-----------|-------------|
| `services/research_service.py` | **Create** | ~400 | Core research pipeline |
| `agents_v2/research_agent.py` | **Replace** | ~500 | Enhanced LangGraph agent |
| `api/research.py` | **Create** | ~250 | REST endpoints |
| `database/schemas.py` | **Extend** | +40 | 3 schema builders |
| `database/models.py` | **Extend** | +150 | Research Pydantic models |
| `database/mongodb.py` | **Extend** | +30 | 3 collections + indexes |
| `config/settings.py` | **Extend** | +6 | 3 research settings |

---

## Future Considerations

- **Streaming research results**: Could emit intermediate results via WebSocket as each stage completes
- **Source embedding + vector search**: Add embedding to `research_sources` for semantic source retrieval
- **Multi-language research**: Extend search to non-English queries
- **Scheduled research**: Automatically re-run research queries on a cron schedule and flag changes
- **Research memory consolidation**: Long-running research topics aggregated into knowledge graphs
