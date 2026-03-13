"""
Pretty terminal UI using Rich library.

Provides colorful, well-formatted output that makes the coding assistant
feel polished and professional.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich import box

# ---------------------------------------------------------------------------
# Theme and console
# ---------------------------------------------------------------------------

custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "task.pending": "dim",
    "task.active": "yellow bold",
    "task.done": "green",
    "task.skipped": "dim strike",
    "header": "bold magenta",
    "tool": "blue",
    "file": "cyan underline",
})

console = Console(theme=custom_theme)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
[bold magenta]  ██████╗ ██████╗ ██████╗ ███████╗    █████╗ ███████╗███████╗██╗███████╗████████╗[/]
[bold magenta] ██╔════╝██╔═══██╗██╔══██╗██╔════╝   ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝[/]
[bold magenta] ██║     ██║   ██║██║  ██║█████╗     ███████║███████╗███████╗██║███████╗   ██║   [/]
[bold magenta] ██║     ██║   ██║██║  ██║██╔══╝     ██╔══██║╚════██║╚════██║██║╚════██║   ██║   [/]
[bold magenta] ╚██████╗╚██████╔╝██████╔╝███████╗   ██║  ██║███████║███████║██║███████║   ██║   [/]
[bold magenta]  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   [/]
[dim]  LangGraph Coding Assistant • Optimized for Local Models[/dim]
"""


def print_banner():
    """Print the startup banner."""
    console.print(BANNER)
    console.print()


# ---------------------------------------------------------------------------
# Status displays
# ---------------------------------------------------------------------------

def print_config(model: str, base_url: str, workspace: str):
    """Print configuration info at startup."""
    table = Table(box=box.ROUNDED, show_header=False, border_style="dim")
    table.add_column("Key", style="dim", width=12)
    table.add_column("Value", style="info")
    table.add_row("Model", model)
    table.add_row("Endpoint", base_url)
    table.add_row("Workspace", workspace)
    console.print(Panel(table, title="[header]Configuration[/]", border_style="dim"))
    console.print()


def print_welcome():
    """Print welcome message with commands."""
    help_text = (
        "[dim]Commands:[/dim]\n"
        "  [bold]quit[/bold] / [bold]exit[/bold]  — Exit the assistant\n"
        "  [bold]tasks[/bold]         — Show current task list\n"
        "  [bold]clear[/bold]         — Clear conversation history\n"
        "  [bold]help[/bold]          — Show this message\n"
    )
    console.print(Panel(help_text, title="[header]Getting Started[/]", border_style="dim"))
    console.print()


# ---------------------------------------------------------------------------
# Message display
# ---------------------------------------------------------------------------

def print_user_message(text: str):
    """Display user message."""
    console.print(f"\n[bold green]You >[/] {text}")


def print_assistant_message(text: str):
    """Display assistant response with markdown rendering."""
    if not text or not text.strip():
        return
    # Clean up TASK COMPLETE markers for display
    display_text = text.replace("TASK COMPLETE", "").strip()
    if not display_text:
        return
    try:
        md = Markdown(display_text)
        console.print(Panel(md, title="[bold magenta]Assistant[/]",
                           border_style="magenta", padding=(1, 2)))
    except Exception:
        console.print(Panel(display_text, title="[bold magenta]Assistant[/]",
                           border_style="magenta", padding=(1, 2)))


def print_tool_call(name: str, args: dict):
    """Display a tool call."""
    args_str = ", ".join(f"[dim]{k}[/]={v!r}" for k, v in args.items()
                         if k != "content")  # don't show full file content
    if any(k == "content" for k in args):
        args_str += ", [dim]content[/]=<file content>"
    console.print(f"  [tool]⚡ {name}[/]({args_str})")


def print_tool_result(content: str, max_lines: int = 8):
    """Display tool result (truncated)."""
    lines = str(content).split("\n")
    if len(lines) > max_lines:
        display = "\n".join(lines[:max_lines]) + f"\n[dim]... ({len(lines) - max_lines} more lines)[/]"
    else:
        display = str(content)
    console.print(f"  [dim]{display}[/]")


# ---------------------------------------------------------------------------
# Task display
# ---------------------------------------------------------------------------

def print_tasks(tasks: list[dict], current_task_id: int = 0):
    """Display the task list in a nice table."""
    if not tasks:
        console.print("[dim]No tasks.[/]")
        return

    table = Table(box=box.SIMPLE_HEAVY, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", width=3)
    table.add_column("Task", ratio=1)
    table.add_column("Type", width=8)

    for t in tasks:
        status = t.get("status", "pending")
        icon = {
            "pending": "[task.pending]○[/]",
            "in_progress": "[task.active]◉[/]",
            "done": "[task.done]✓[/]",
            "skipped": "[task.skipped]–[/]",
        }.get(status, "○")

        style = {
            "pending": "task.pending",
            "in_progress": "task.active",
            "done": "task.done",
            "skipped": "task.skipped",
        }.get(status, "")

        table.add_row(
            str(t.get("id", "?")),
            icon,
            f"[{style}]{t.get('description', '')}[/]",
            t.get("type", ""),
        )

    console.print(Panel(table, title="[header]Tasks[/]", border_style="dim"))


def print_plan_created(tasks: list[dict]):
    """Announce that a plan was created."""
    console.print(f"\n[success]📋 Plan created with {len(tasks)} task(s)[/]")
    print_tasks(tasks)
    console.print()


def print_task_started(task: dict):
    """Announce task start."""
    console.print(f"\n[task.active]▶ Starting task {task.get('id', '?')}: {task.get('description', '')}[/]")


def print_task_completed(task: dict):
    """Announce task completion."""
    console.print(f"[task.done]✓ Completed task {task.get('id', '?')}: {task.get('description', '')}[/]")


def print_all_done():
    """Announce all tasks are done."""
    console.print(Panel("[bold green]All tasks completed![/]",
                       border_style="green", padding=(0, 2)))


# ---------------------------------------------------------------------------
# Thinking indicator
# ---------------------------------------------------------------------------

def get_spinner(text: str = "Thinking...") -> Live:
    """Get a live spinner for async operations."""
    return Live(
        Spinner("dots", text=f"[dim]{text}[/]", style="magenta"),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------

def print_error(text: str):
    """Display an error."""
    console.print(f"[error]✗ Error:[/] {text}")


def print_warning(text: str):
    """Display a warning."""
    console.print(f"[warning]⚠ {text}[/]")


def print_info(text: str):
    """Display info."""
    console.print(f"[info]ℹ {text}[/]")


# ---------------------------------------------------------------------------
# Input prompt
# ---------------------------------------------------------------------------

def get_input() -> str:
    """Get user input with a styled prompt."""
    try:
        return console.input("\n[bold green]You >[/] ")
    except (EOFError, KeyboardInterrupt):
        return "quit"
