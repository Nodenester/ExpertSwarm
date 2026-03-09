from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class SourceReference(BaseModel):
    """Where a fact came from."""

    url: str | None = None
    title: str | None = None
    selector: str | None = None
    snippet: str = Field(default="", description="Text excerpt supporting this fact")


class Fact(BaseModel):
    """A single extracted fact with provenance."""

    claim: str = Field(..., description="The factual claim")
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    sources: list[SourceReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, description="Topic tags for clustering")
    contradicts: list[str] = Field(default_factory=list, description="IDs of facts this contradicts")


class FactBundle(BaseModel):
    """Collection of facts from a single agent's work."""

    agent_id: str
    query: str = ""
    facts: list[Fact] = Field(default_factory=list)
    summary: str = Field(default="", description="One-paragraph summary of findings")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrossReference(BaseModel):
    """A relationship found between facts from different agents."""

    fact_a: str
    fact_b: str
    relationship: str = Field(..., description="How these facts relate: supports, contradicts, extends, qualifies")
    strength: ConfidenceLevel = ConfidenceLevel.MEDIUM


class AggregatedFacts(BaseModel):
    """Output of an aggregator agent — deduplicated, cross-referenced facts."""

    facts: list[Fact] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list, description="Major themes identified")
    gaps: list[str] = Field(default_factory=list, description="Knowledge gaps found")
    summary: str = ""
