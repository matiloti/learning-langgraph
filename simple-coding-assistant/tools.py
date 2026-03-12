from langchain.tools import tool
import os

WORKSPACE = os.path.realpath("/Users/matias/Projects/learning-langgraph/test-folder")

def safe_path(filepath: str) -> str:
    """Resolve a path and ensure it's inside WORKSPACE. Raises ValueError if not."""
    full = os.path.realpath(os.path.join(WORKSPACE, filepath))
    if not full.startswith(WORKSPACE):
        raise ValueError(f"Access denied: {filepath} is outside workspace")
    return full

@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file.

    Args:
        filepath: Path to the file to read
    """
    path = safe_path(filepath)
    with open(path, "r") as f:
        return f.read()

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file.

    Args:
        filepath: Path to the file to write
        content: The full content to write
    """
    path = safe_path(filepath)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"

@tool
def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact string match with new content. Use this instead of write_file when you only need to change part of a file.

    Args:
        filepath: Path to the file to edit
        old_string: The exact text to find and replace (must match exactly, including whitespace and indentation)
        new_string: The text to replace it with
    """
    path = safe_path(filepath)
    with open(path, "r") as f:
        content = f.read()
    if old_string not in content:
        return f"Error: old_string not found in {filepath}. Make sure it matches exactly, including whitespace."
    if content.count(old_string) > 1:
        return f"Error: old_string appears {content.count(old_string)} times in {filepath}. Provide a larger snippet to make it unique."
    new_content = content.replace(old_string, new_string, 1)
    with open(path, "w") as f:
        f.write(new_content)
    return f"Successfully edited {filepath}"

@tool
def insert_after_line(filepath: str, line_number: int, content: str) -> str:
    """Insert content after a specific line number in a file.

    Args:
        filepath: Path to the file to edit
        line_number: The line number after which to insert content (1-based)
        content: The text to insert
    """
    path = safe_path(filepath)
    with open(path, "r") as f:
        lines = f.readlines()
    if line_number < 1 or line_number > len(lines):
        return f"Error: line_number {line_number} is out of range. File has {len(lines)} lines."
    lines.insert(line_number, content if content.endswith("\n") else content + "\n")
    with open(path, "w") as f:
        f.writelines(lines)
    return f"Successfully inserted content after line {line_number} in {filepath}"

@tool
def find_in_file(filepath: str, search_string: str) -> str:
    """Find all occurrences of a string in a file and return their line numbers.

    Args:
        filepath: Path to the file to search
        search_string: The text to search for
    """
    path = safe_path(filepath)
    with open(path, "r") as f:
        lines = f.readlines()
    matches = []
    for i, line in enumerate(lines, 1):
        if search_string in line:
            matches.append(f"  Line {i}: {line.rstrip()}")
    if not matches:
        return f"No occurrences of '{search_string}' found in {filepath}"
    return f"Found {len(matches)} match(es) in {filepath}:\n" + "\n".join(matches)

@tool
def list_files(directory: str) -> str:
    """List files in a directory.

    Args:
        directory: Path to the directory
    """
    path = safe_path(directory)
    return "\n".join(os.listdir(path))

# Tool collections
read_tools = [read_file, find_in_file, list_files]
read_tools_by_name = {tool.name: tool for tool in read_tools}

write_tools = [write_file, edit_file, insert_after_line]
write_tools_by_name = {tool.name: tool for tool in write_tools}

read_write_tools = read_tools + write_tools
read_write_tools_by_name = {tool.name: tool for tool in read_write_tools}
