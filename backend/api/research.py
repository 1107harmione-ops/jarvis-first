"""
Research API Routes
===================
FastAPI endpoints for the Research Agent system.

Provides:
- POST /api/v2/research/search     - Quick web search with synthesis
- POST /api/v2/research/deep       - Full deep research pipeline
- POST /api/v2/research/verify     - Fact-check content
- GET  /api/v2/research/history    - Paginated research history
- GET  /api/v2/research/reports/{report_id} - Get specific report
- GET  /api/v2/research/sources    - Search/filter sources
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.auth import get_current_user
from backend.database.models import (
    UserResponse,
    ResearchSearchRequest,
    ResearchSearchResponse,
    DeepResearchRequest,
    DeepResearchResponse,
    VerifyContentRequest,
    VerifyContentResponse,
    SourceScore,
    FactCheckResult,
    ResearchReportSummary,
)
from backend.database.schemas import serialize_doc
from backend.database.mongodb import mongodb
from backend.services.research_service import research_service

router = APIRouter(prefix="/api/v2/research", tags=["research"])


@router.post("/search", response_model=ResearchSearchResponse)
async def research_search(
    request: ResearchSearchRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Quick web search with AI-powered synthesis.

    Runs the first stages of the research pipeline:
    Plan -> Search -> Collect -> Rank -> Synthesize
    """
    result = await research_service.quick_search(
        query=request.query,
        max_sources=request.max_sources,
    )

    sources = [SourceScore(**s) for s in result.get("sources", [])]

    return ResearchSearchResponse(
        query=result["query"],
        research_type=result.get("research_type", "quick"),
        answer=result.get("answer", ""),
        sources=sources,
        source_count=result.get("source_count", 0),
        processing_time_ms=result.get("processing_time_ms", 0.0),
    )


@router.post("/deep", response_model=DeepResearchResponse)
async def research_deep(
    request: DeepResearchRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Full deep research with the complete 8-stage pipeline.

    Plan -> Search -> Collect -> Rank -> Verify -> Synthesize -> Report -> Respond
    """
    result = await research_service.deep_research(
        query=request.query,
        research_type=request.research_type.value,
        depth=request.depth.value,
        max_sources=request.max_sources,
    )

    sources = [SourceScore(**s) for s in result.get("sources", [])]
    fact_check_data = result.get("fact_check")
    fact_check = FactCheckResult(**fact_check_data) if fact_check_data else None

    return DeepResearchResponse(
        report_id=result.get("report_id", ""),
        title=result.get("title", ""),
        research_type=result.get("research_type", request.research_type.value),
        depth=result.get("depth", request.depth.value),
        executive_summary=result.get("executive_summary", ""),
        key_findings=result.get("key_findings", []),
        detailed_analysis=result.get("detailed_analysis"),
        pros=result.get("pros"),
        cons=result.get("cons"),
        recommendations=result.get("recommendations"),
        conclusions=result.get("conclusions"),
        sources=sources,
        source_count=result.get("source_count", 0),
        fact_check=fact_check,
        processing_time_ms=result.get("processing_time_ms", 0.0),
        created_at=result.get("created_at", ""),
    )


@router.post("/verify", response_model=VerifyContentResponse)
async def research_verify(
    request: VerifyContentRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Fact-check content against web sources.

    Extracts claims from the provided content, cross-references against
    web search results, and classifies each claim as verified, contradicted,
    or unverifiable.
    """
    # Search for relevant sources first
    plan = await research_service.plan_research(request.content)
    raw_results = await research_service.execute_search({
        **plan, "sources_needed": 10,
    })
    collected = await research_service.collect_sources(raw_results)

    verification = await research_service.verify_facts(
        collected, request.content,
    )

    return VerifyContentResponse(
        verified_claims=verification.get("verified_claims", []),
        contradicted_claims=verification.get("contradicted_claims", []),
        unverifiable_claims=verification.get("unverifiable_claims", []),
        overall_confidence=verification.get("overall_confidence", 0.0),
        analysis=f"Found {len(verification.get('verified_claims', []))} verified, "
                 f"{len(verification.get('contradicted_claims', []))} contradicted, "
                 f"and {len(verification.get('unverifiable_claims', []))} unverifiable claims.",
    )


@router.get("/history")
async def research_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    research_type: str | None = Query(None),
    current_user: UserResponse = Depends(get_current_user),
):
    """Get paginated research history for the current user."""
    query_filter: dict[str, Any] = {"user_id": current_user.id}
    if research_type:
        query_filter["research_type"] = research_type

    cursor = (
        mongodb.research_reports
        .find(query_filter)
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    total = await mongodb.research_reports.count_documents(query_filter)

    items = []
    for doc in docs:
        serialized = serialize_doc(doc)
        items.append(ResearchReportSummary(
            id=serialized["id"],
            query=serialized.get("query", ""),
            research_type=serialized.get("research_type", ""),
            depth=serialized.get("depth", ""),
            executive_summary=serialized.get("executive_summary", "")[:200],
            source_count=serialized.get("source_count", 0),
            created_at=serialized.get("created_at", ""),
        ))

    return {
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/reports/{report_id}")
async def get_research_report(
    report_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Get a specific research report by ID."""
    from bson import ObjectId

    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid report ID format")

    doc = await mongodb.research_reports.find_one({
        "_id": oid,
        "user_id": current_user.id,
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return serialize_doc(doc)


@router.get("/sources")
async def search_sources(
    domain: str | None = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    current_user: UserResponse = Depends(get_current_user),
):
    """Search and filter cached research sources."""
    query_filter: dict[str, Any] = {}
    if domain:
        query_filter["domain"] = {"$regex": domain, "$options": "i"}
    if min_score > 0:
        query_filter["overall_score"] = {"$gte": min_score}

    cursor = (
        mongodb.research_sources
        .find(query_filter)
        .sort("overall_score", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)

    return [serialize_doc(doc) for doc in docs]
