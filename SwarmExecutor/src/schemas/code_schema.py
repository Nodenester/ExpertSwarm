from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"


class ParameterInfo(BaseModel):
    name: str
    type_hint: str = ""
    default: str | None = None


class FunctionSignature(BaseModel):
    """Token-efficient function representation — signature only, no body."""

    name: str
    params: list[ParameterInfo] = Field(default_factory=list)
    return_type: str = ""
    decorators: list[str] = Field(default_factory=list)
    docstring: str = Field(default="", description="First line of docstring only")
    is_async: bool = False
    line_number: int | None = None


class ClassOutline(BaseModel):
    """Token-efficient class representation — structure without implementation."""

    name: str
    bases: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    docstring: str = ""
    methods: list[FunctionSignature] = Field(default_factory=list)
    class_vars: list[str] = Field(default_factory=list, description="Class-level variable names")
    line_number: int | None = None


class FileStructure(BaseModel):
    """Token-efficient file representation for code analysis agents."""

    path: str
    language: str = ""
    imports: list[str] = Field(default_factory=list)
    classes: list[ClassOutline] = Field(default_factory=list)
    functions: list[FunctionSignature] = Field(default_factory=list)
    top_level_vars: list[str] = Field(default_factory=list)
    line_count: int = 0


class CodeSearchResult(BaseModel):
    """Result from a code search / grep operation."""

    file_path: str
    line_number: int
    line_content: str
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class DirectoryTree(BaseModel):
    """Token-efficient directory tree representation."""

    path: str
    entries: list[TreeEntry] = Field(default_factory=list)
    total_files: int = 0
    total_dirs: int = 0


class TreeEntry(BaseModel):
    name: str
    is_dir: bool = False
    size: int | None = None
    children: list[TreeEntry] = Field(default_factory=list)


# Rebuild to resolve forward refs
DirectoryTree.model_rebuild()
TreeEntry.model_rebuild()


class CodeAnalysisBundle(BaseModel):
    """Complete output from a code analysis agent."""

    agent_id: str
    query: str = ""
    files: list[FileStructure] = Field(default_factory=list)
    search_results: list[CodeSearchResult] = Field(default_factory=list)
    directory_tree: DirectoryTree | None = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
