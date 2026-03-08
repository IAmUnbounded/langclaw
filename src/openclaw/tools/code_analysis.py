"""Code analysis tools — AST-based code understanding.

Provides structural understanding of source code files:
function/class outlines, signatures, and hierarchies.
"""

from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.tools import tool


@tool
def code_outline_tool(path: str) -> str:
    """Get a structural outline of a Python source file.

    Parses the file using AST and returns the hierarchy of
    classes, functions, and their signatures.

    Args:
        path: Path to the Python file to analyze.

    Returns:
        Formatted outline with classes, methods, functions, and line numbers.
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"[error] File not found: {path}"
    if not file_path.is_file():
        return f"[error] Not a file: {path}"

    # Only support Python for AST analysis
    if file_path.suffix != ".py":
        return _simple_outline(file_path)

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"[error] Syntax error in {path}: {e}"
    except UnicodeDecodeError:
        return f"[error] Cannot read binary file: {path}"

    lines = [f"📋 **Outline:** {file_path.name}\n"]
    lines.append(f"Total lines: {len(source.split(chr(10)))}\n")

    # Top-level imports
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)

    if imports:
        lines.append(f"**Imports:** {', '.join(imports[:15])}")
        if len(imports) > 15:
            lines.append(f"  ... and {len(imports) - 15} more")
        lines.append("")

    # Classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Class
            bases = [_get_name(b) for b in node.bases]
            bases_str = f"({', '.join(bases)})" if bases else ""
            end_line = _get_end_line(node)
            lines.append(f"📦 **class {node.name}{bases_str}** (L{node.lineno}-{end_line})")

            # Docstring
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split("\n")[0]
                lines.append(f"   {first_line}")

            # Methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_async = isinstance(item, ast.AsyncFunctionDef)
                    prefix = "async " if is_async else ""
                    args_str = _format_args(item.args)
                    end_l = _get_end_line(item)
                    decorator = ""
                    for d in item.decorator_list:
                        dname = _get_name(d)
                        if dname in ("property", "staticmethod", "classmethod"):
                            decorator = f"@{dname} "
                            break
                    lines.append(
                        f"   ├─ {decorator}{prefix}def {item.name}({args_str}) (L{item.lineno}-{end_l})"
                    )
            lines.append("")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level function
            is_async = isinstance(node, ast.AsyncFunctionDef)
            prefix = "async " if is_async else ""
            args_str = _format_args(node.args)
            end_line = _get_end_line(node)
            lines.append(f"🔹 **{prefix}def {node.name}({args_str})** (L{node.lineno}-{end_line})")

            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split("\n")[0]
                lines.append(f"   {first_line}")
            lines.append("")

        elif isinstance(node, ast.Assign):
            # Top-level constants
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lines.append(f"📌 {target.id} = ... (L{node.lineno})")

    return "\n".join(lines)


def _get_name(node) -> str:
    """Extract name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return _get_name(node.func)
    return "?"


def _get_end_line(node) -> int:
    """Get the end line of an AST node."""
    return getattr(node, "end_lineno", node.lineno)


def _format_args(args: ast.arguments) -> str:
    """Format function arguments to a readable string."""
    parts = []
    for arg in args.args:
        name = arg.arg
        if name == "self" or name == "cls":
            parts.append(name)
        elif arg.annotation:
            parts.append(f"{name}: {_get_name(arg.annotation)}")
        else:
            parts.append(name)

    # Limit length
    result = ", ".join(parts)
    if len(result) > 60:
        result = result[:57] + "..."
    return result


def _simple_outline(path: Path) -> str:
    """Simple line-count based outline for non-Python files."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        total = len(lines)

        result = [f"📋 **Outline:** {path.name}", f"Total lines: {total}\n"]

        # For JS/TS files, find function/class definitions
        if path.suffix in (".js", ".ts", ".jsx", ".tsx"):
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in ("function ", "class ", "export function ", "export class ", "export default function ", "export default class ")):
                    result.append(f"  L{i}: {stripped[:80]}")
                elif "=> {" in stripped and ("const " in stripped or "let " in stripped):
                    result.append(f"  L{i}: {stripped[:80]}")

        return "\n".join(result) if len(result) > 2 else f"📋 {path.name}: {total} lines (no structural analysis available for {path.suffix} files)"

    except Exception as e:
        return f"[error] {e}"
