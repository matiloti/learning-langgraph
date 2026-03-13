"""
File operation tools for the coding assistant.
All paths are sandboxed to WORKSPACE for safety.

Context engineering note: tool descriptions are deliberately verbose
with examples — small models need explicit guidance on argument formats.
"""

import os
import glob as glob_module
from langchain_core.tools import tool

# Workspace root — set via env var or default to current directory
WORKSPACE = os.environ.get(
    "CODING_ASSISTANT_WORKSPACE",
    os.path.realpath(os.getcwd())
)


def safe_path(filepath: str) -> str:
    """Resolve a path and ensure it stays inside WORKSPACE."""
    if os.path.isabs(filepath):
        full = os.path.realpath(filepath)
    else:
        full = os.path.realpath(os.path.join(WORKSPACE, filepath))
    if not full.startswith(WORKSPACE):
        raise ValueError(f"Access denied: {filepath} is outside workspace ({WORKSPACE})")
    return full


# ===========================================================================
# READ TOOLS
# ===========================================================================

@tool
def list_files(directory: str = ".") -> str:
    """List all files and directories in a directory. Use '.' for the workspace root.

    Args:
        directory: Relative path to the directory. Example: '.' or 'src' or 'src/utils'
    """
    path = safe_path(directory)
    if not os.path.isdir(path):
        return f"Error: '{directory}' is not a directory."
    entries = sorted(os.listdir(path))
    if not entries:
        return f"Directory '{directory}' is empty."
    result = []
    for e in entries:
        full = os.path.join(path, e)
        prefix = "[DIR] " if os.path.isdir(full) else "      "
        result.append(f"{prefix}{e}")
    return "\n".join(result)


@tool
def read_file(filepath: str) -> str:
    """Read the full contents of a file and return it with line numbers.

    Args:
        filepath: Relative path to the file. Example: 'main.py' or 'src/app.js'
    """
    path = safe_path(filepath)
    if not os.path.isfile(path):
        return f"Error: File '{filepath}' does not exist."
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if len(lines) == 0:
        return f"File '{filepath}' is empty."
    # Return with line numbers for easier referencing
    numbered = [f"{i+1:4d} | {line.rstrip()}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


@tool
def find_in_file(filepath: str, search_string: str) -> str:
    """Search for a string in a file and return matching lines with line numbers.

    Args:
        filepath: Relative path to the file. Example: 'main.py'
        search_string: The text to search for. Example: 'def main' or 'import os'
    """
    path = safe_path(filepath)
    if not os.path.isfile(path):
        return f"Error: File '{filepath}' does not exist."
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    matches = []
    for i, line in enumerate(lines, 1):
        if search_string in line:
            matches.append(f"  Line {i}: {line.rstrip()}")
    if not matches:
        return f"No occurrences of '{search_string}' found in {filepath}"
    return f"Found {len(matches)} match(es) in {filepath}:\n" + "\n".join(matches)


@tool
def search_files(pattern: str, search_string: str = "") -> str:
    """Find files matching a glob pattern, optionally filtering by content.

    Args:
        pattern: Glob pattern. Example: '**/*.py' or 'src/**/*.js'
        search_string: Optional text that must appear in the file. Example: 'class User'
    """
    root = WORKSPACE
    matches = glob_module.glob(os.path.join(root, pattern), recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]

    if search_string:
        filtered = []
        for m in matches:
            try:
                with open(m, "r", encoding="utf-8", errors="replace") as f:
                    if search_string in f.read():
                        filtered.append(m)
            except Exception:
                pass
        matches = filtered

    if not matches:
        return f"No files found matching '{pattern}'" + (f" containing '{search_string}'" if search_string else "")

    rel = [os.path.relpath(m, root) for m in sorted(matches)]
    return "\n".join(rel[:50])  # cap at 50 results


# ===========================================================================
# WRITE TOOLS
# ===========================================================================

@tool
def write_file(filepath: str, content: str) -> str:
    """Create or overwrite a file with the given content. Use for NEW files.

    Args:
        filepath: Relative path to the file. Example: 'src/new_module.py'
        content: The complete file content to write.
    """
    path = safe_path(filepath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created/overwrote {filepath} ({len(content)} bytes)"


@tool
def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file. Use for MODIFYING existing files.
    The old_string must match exactly (including whitespace and indentation).

    Args:
        filepath: Relative path to the file. Example: 'main.py'
        old_string: The exact text to find. Must appear exactly once in the file.
        new_string: The replacement text.
    """
    path = safe_path(filepath)
    if not os.path.isfile(path):
        return f"Error: File '{filepath}' does not exist."
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_string not in content:
        return f"Error: old_string not found in {filepath}. Ensure it matches exactly including whitespace."
    count = content.count(old_string)
    if count > 1:
        return f"Error: old_string appears {count} times. Provide more context to make it unique."
    new_content = content.replace(old_string, new_string, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return f"Edited {filepath} successfully."


@tool
def delete_file(filepath: str) -> str:
    """Delete a file from the workspace.

    Args:
        filepath: Relative path to the file. Example: 'old_module.py'
    """
    path = safe_path(filepath)
    if not os.path.isfile(path):
        return f"Error: File '{filepath}' does not exist."
    os.remove(path)
    return f"Deleted {filepath}."


# ===========================================================================
# Tool collections
# ===========================================================================

read_tools = [list_files, read_file, find_in_file, search_files]
read_tools_by_name = {t.name: t for t in read_tools}

write_tools = [write_file, edit_file, delete_file]
write_tools_by_name = {t.name: t for t in write_tools}

read_write_tools = read_tools + write_tools
read_write_tools_by_name = {t.name: t for t in read_write_tools}
