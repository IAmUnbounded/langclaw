"""Apply patch tool — diff-based file editing.

Mirrors openclaw's apply-patch.ts:
- Parses a custom patch format with Begin/End Patch markers
- Supports Add File, Delete File, Update File, Move to operations
- Context-aware chunk matching for precise edits
- Operates relative to cwd with workspace-only restriction

Patch format:
    *** Begin Patch
    *** Add File: path/to/new-file.txt
    <file content>
    *** End of File
    *** Update File: path/to/existing.py
    @@ context line
    -old line
    +new line
    *** Delete File: path/to/remove.txt
    *** End Patch
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Patch format markers
BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"
ADD_FILE = "*** Add File: "
DELETE_FILE = "*** Delete File: "
UPDATE_FILE = "*** Update File: "
MOVE_TO = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT = "@@ "
EMPTY_CHANGE_CONTEXT = "@@"


@dataclass
class AddFileHunk:
    kind: str = "add"
    path: str = ""
    contents: str = ""


@dataclass
class DeleteFileHunk:
    kind: str = "delete"
    path: str = ""


@dataclass
class UpdateChunk:
    change_context: str | None = None
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    is_end_of_file: bool = False


@dataclass
class UpdateFileHunk:
    kind: str = "update"
    path: str = ""
    move_path: str | None = None
    chunks: list[UpdateChunk] = field(default_factory=list)


Hunk = Union[AddFileHunk, DeleteFileHunk, UpdateFileHunk]


@dataclass
class ApplyPatchSummary:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)


class PatchParseError(Exception):
    pass


class PatchApplyError(Exception):
    pass


def _parse_patch(patch_text: str) -> list[Hunk]:
    """Parse the patch text into a list of hunks."""
    lines = patch_text.splitlines()

    # Find begin/end markers
    try:
        begin_idx = next(i for i, l in enumerate(lines) if l.strip() == BEGIN_PATCH)
    except StopIteration:
        raise PatchParseError(f"Missing '{BEGIN_PATCH}' marker")

    try:
        end_idx = next(i for i, l in enumerate(lines) if l.strip() == END_PATCH)
    except StopIteration:
        raise PatchParseError(f"Missing '{END_PATCH}' marker")

    body = lines[begin_idx + 1:end_idx]
    hunks: list[Hunk] = []
    i = 0

    while i < len(body):
        line = body[i]

        if line.startswith(ADD_FILE):
            file_path = line[len(ADD_FILE):].strip()
            i += 1
            content_lines: list[str] = []
            while i < len(body) and body[i].strip() != EOF_MARKER:
                content_lines.append(body[i])
                i += 1
            if i < len(body):
                i += 1  # Skip EOF_MARKER
            hunks.append(AddFileHunk(path=file_path, contents="\n".join(content_lines)))

        elif line.startswith(DELETE_FILE):
            file_path = line[len(DELETE_FILE):].strip()
            hunks.append(DeleteFileHunk(path=file_path))
            i += 1

        elif line.startswith(UPDATE_FILE):
            file_path = line[len(UPDATE_FILE):].strip()
            move_path = None
            i += 1

            # Check for Move to
            if i < len(body) and body[i].startswith(MOVE_TO):
                move_path = body[i][len(MOVE_TO):].strip()
                i += 1

            # Parse chunks
            chunks: list[UpdateChunk] = []
            current_chunk: UpdateChunk | None = None

            while i < len(body) and not any(
                body[i].startswith(marker)
                for marker in (ADD_FILE, DELETE_FILE, UPDATE_FILE, END_PATCH)
            ):
                chunk_line = body[i]

                if chunk_line.startswith(CHANGE_CONTEXT) or chunk_line == EMPTY_CHANGE_CONTEXT:
                    if current_chunk is not None:
                        chunks.append(current_chunk)
                    ctx = chunk_line[len(CHANGE_CONTEXT):].strip() if chunk_line.startswith(CHANGE_CONTEXT) else ""
                    current_chunk = UpdateChunk(change_context=ctx)

                elif chunk_line.strip() == EOF_MARKER:
                    if current_chunk is not None:
                        current_chunk.is_end_of_file = True
                        chunks.append(current_chunk)
                        current_chunk = None

                elif chunk_line.startswith("-") and current_chunk is not None:
                    current_chunk.old_lines.append(chunk_line[1:])

                elif chunk_line.startswith("+") and current_chunk is not None:
                    current_chunk.new_lines.append(chunk_line[1:])

                elif current_chunk is not None:
                    # Context line (unchanged) — part of both old and new
                    ctx_line = chunk_line[1:] if chunk_line.startswith(" ") else chunk_line
                    current_chunk.old_lines.append(ctx_line)
                    current_chunk.new_lines.append(ctx_line)

                i += 1

            if current_chunk is not None:
                chunks.append(current_chunk)

            hunks.append(UpdateFileHunk(path=file_path, move_path=move_path, chunks=chunks))

        else:
            i += 1

    return hunks


def _apply_update_chunk(file_lines: list[str], chunk: UpdateChunk) -> list[str]:
    """Apply a single update chunk to file lines."""
    if chunk.is_end_of_file:
        # Find old_lines at the end of the file
        old = chunk.old_lines
        if old:
            if file_lines[-len(old):] == old:
                return file_lines[:-len(old)] + chunk.new_lines
            raise PatchApplyError(
                f"End-of-file context mismatch. Expected:\n"
                f"{chr(10).join(old)}\n\nGot:\n{chr(10).join(file_lines[-len(old):])}"
            )
        return file_lines + chunk.new_lines

    # Search for context or old_lines
    old = chunk.old_lines
    if not old:
        # Pure insert — find context line and insert after
        ctx = chunk.change_context
        if ctx:
            for i, line in enumerate(file_lines):
                if line.strip() == ctx.strip():
                    return file_lines[:i + 1] + chunk.new_lines + file_lines[i + 1:]
        return file_lines + chunk.new_lines

    # Find old_lines block in file
    for i in range(len(file_lines) - len(old) + 1):
        if file_lines[i:i + len(old)] == old:
            return file_lines[:i] + chunk.new_lines + file_lines[i + len(old):]

    # Fuzzy match: strip trailing whitespace
    stripped_old = [l.rstrip() for l in old]
    for i in range(len(file_lines) - len(old) + 1):
        window = [l.rstrip() for l in file_lines[i:i + len(old)]]
        if window == stripped_old:
            return file_lines[:i] + chunk.new_lines + file_lines[i + len(old):]

    raise PatchApplyError(
        f"Could not find context to apply patch chunk.\n"
        f"Looking for:\n{chr(10).join(old[:5])}"
    )


def _resolve_path(file_path: str, cwd: Path) -> Path:
    """Resolve a patch file path against cwd, enforcing workspace boundary."""
    p = (cwd / file_path).resolve()
    try:
        p.relative_to(cwd.resolve())
    except ValueError:
        raise PatchApplyError(f"Path escapes workspace root: {file_path}")
    return p


def apply_patch_to_files(patch_text: str, cwd: Path) -> tuple[ApplyPatchSummary, str]:
    """Parse and apply a patch, returning a summary."""
    hunks = _parse_patch(patch_text)
    summary = ApplyPatchSummary()
    messages: list[str] = []

    for hunk in hunks:
        if isinstance(hunk, AddFileHunk):
            target = _resolve_path(hunk.path, cwd)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(hunk.contents, encoding="utf-8")
            summary.added.append(hunk.path)
            messages.append(f"Added: {hunk.path}")

        elif isinstance(hunk, DeleteFileHunk):
            target = _resolve_path(hunk.path, cwd)
            if target.exists():
                target.unlink()
                summary.deleted.append(hunk.path)
                messages.append(f"Deleted: {hunk.path}")
            else:
                messages.append(f"Warning: {hunk.path} not found, skipping delete")

        elif isinstance(hunk, UpdateFileHunk):
            target = _resolve_path(hunk.path, cwd)
            if not target.exists():
                raise PatchApplyError(f"File to update not found: {hunk.path}")

            file_lines = target.read_text(encoding="utf-8").splitlines()

            for chunk in hunk.chunks:
                file_lines = _apply_update_chunk(file_lines, chunk)

            new_content = "\n".join(file_lines)
            # Preserve trailing newline
            if target.read_text(encoding="utf-8").endswith("\n"):
                new_content += "\n"

            if hunk.move_path:
                dest = _resolve_path(hunk.move_path, cwd)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_content, encoding="utf-8")
                target.unlink()
                summary.modified.append(hunk.move_path)
                messages.append(f"Moved + updated: {hunk.path} → {hunk.move_path}")
            else:
                target.write_text(new_content, encoding="utf-8")
                summary.modified.append(hunk.path)
                messages.append(f"Updated: {hunk.path}")

    total = len(summary.added) + len(summary.modified) + len(summary.deleted)
    text = f"Patch applied: {total} file(s) changed\n" + "\n".join(messages)
    return summary, text


@tool
def apply_patch_tool(patch_input: str) -> str:
    """Apply a patch to one or more files using the apply_patch format.

    The patch uses *** Begin Patch / *** End Patch markers and supports:
    - Adding new files (*** Add File: path)
    - Deleting files (*** Delete File: path)
    - Updating files with context diffs (*** Update File: path)
    - Moving files (*** Move to: new_path)

    Example patch:
        *** Begin Patch
        *** Add File: src/new_module.py
        def hello():
            return "Hello!"
        *** End of File
        *** Update File: src/existing.py
        @@ some context
        -old_code = True
        +new_code = True
        *** End Patch

    Args:
        patch_input: Full patch content including Begin/End markers.

    Returns:
        Summary of changes applied (added, modified, deleted files).
    """
    import os

    cwd = Path(os.getcwd())

    try:
        summary, text = apply_patch_to_files(patch_input, cwd)
        return json.dumps({
            "status": "ok",
            "summary": {
                "added": summary.added,
                "modified": summary.modified,
                "deleted": summary.deleted,
            },
            "text": text,
        }, indent=2)

    except PatchParseError as e:
        return json.dumps({
            "status": "error",
            "error": f"Patch parse error: {e}",
        })
    except PatchApplyError as e:
        return json.dumps({
            "status": "error",
            "error": f"Patch apply error: {e}",
        })
    except Exception as e:
        logger.error(f"apply_patch failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        })


import json  # noqa: E402 — needed for the @tool functions above
