from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from src.schemas.code_schema import (
    ClassOutline,
    FileStructure,
    FunctionSignature,
    ParameterInfo,
)
from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)


def _parse_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSignature:
    """Extract a token-efficient function signature from an AST node."""
    params = []
    for arg in node.args.args:
        type_hint = ""
        if arg.annotation:
            type_hint = ast.unparse(arg.annotation)
        params.append(ParameterInfo(name=arg.arg, type_hint=type_hint))

    # Handle *args, **kwargs
    if node.args.vararg:
        params.append(ParameterInfo(name=f"*{node.args.vararg.arg}"))
    if node.args.kwarg:
        params.append(ParameterInfo(name=f"**{node.args.kwarg.arg}"))

    return_type = ""
    if node.returns:
        return_type = ast.unparse(node.returns)

    decorators = [ast.unparse(d) for d in node.decorator_list]

    # First line of docstring only
    docstring = ""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant,))
        and isinstance(node.body[0].value.value, str)
    ):
        first_line = node.body[0].value.value.strip().split("\n")[0]
        docstring = first_line

    return FunctionSignature(
        name=node.name,
        params=params,
        return_type=return_type,
        decorators=decorators,
        docstring=docstring,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        line_number=node.lineno,
    )


def _parse_class(node: ast.ClassDef) -> ClassOutline:
    """Extract a token-efficient class outline from an AST node."""
    bases = [ast.unparse(b) for b in node.bases]
    decorators = [ast.unparse(d) for d in node.decorator_list]

    docstring = ""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant,))
        and isinstance(node.body[0].value.value, str)
    ):
        first_line = node.body[0].value.value.strip().split("\n")[0]
        docstring = first_line

    methods = []
    class_vars = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_parse_function(item))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    class_vars.append(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            class_vars.append(item.target.id)

    return ClassOutline(
        name=node.name,
        bases=bases,
        decorators=decorators,
        docstring=docstring,
        methods=methods,
        class_vars=class_vars,
        line_number=node.lineno,
    )


def parse_python_file(source: str, path: str = "") -> FileStructure:
    """Parse a Python file into a token-efficient structure representation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return FileStructure(path=path, language="python", line_count=source.count("\n") + 1)

    imports = []
    classes = []
    functions = []
    top_level_vars = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(_parse_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_parse_function(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_vars.append(target.id)

    return FileStructure(
        path=path,
        language="python",
        imports=imports,
        classes=classes,
        functions=functions,
        top_level_vars=top_level_vars,
        line_count=source.count("\n") + 1,
    )


@tool_registry.register(
    "analyze_python_file",
    description="Parse a Python file and return its structure: classes, functions, imports. Token-efficient — signatures only, no bodies.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the Python file"},
        },
        "required": ["path"],
    },
)
async def analyze_python_file(args: dict[str, Any]) -> dict[str, Any]:
    filepath = args["path"]
    p = Path(filepath)
    if not p.is_file():
        return {"error": f"Not a file: {filepath}"}

    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": str(exc)}

    structure = parse_python_file(source, filepath)
    return structure.model_dump()


@tool_registry.register(
    "extract_signatures",
    description="Extract only function/method signatures from a Python file. Most token-efficient view.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the Python file"},
        },
        "required": ["path"],
    },
)
async def extract_signatures(args: dict[str, Any]) -> dict[str, Any]:
    filepath = args["path"]
    p = Path(filepath)
    if not p.is_file():
        return {"error": f"Not a file: {filepath}"}

    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": str(exc)}

    structure = parse_python_file(source, filepath)

    signatures = []
    for fn in structure.functions:
        sig = f"{'async ' if fn.is_async else ''}def {fn.name}({', '.join(p.name + (': ' + p.type_hint if p.type_hint else '') for p in fn.params)})"
        if fn.return_type:
            sig += f" -> {fn.return_type}"
        signatures.append(sig)

    for cls in structure.classes:
        for method in cls.methods:
            sig = f"{'async ' if method.is_async else ''}def {cls.name}.{method.name}({', '.join(p.name + (': ' + p.type_hint if p.type_hint else '') for p in method.params)})"
            if method.return_type:
                sig += f" -> {method.return_type}"
            signatures.append(sig)

    return {"path": filepath, "signatures": signatures}
