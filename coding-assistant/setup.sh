#!/usr/bin/env bash
#
# setup.sh — Code Assist Setup Script
# Creates a Python virtual environment, installs dependencies,
# and adds a shell alias to your .zshrc for easy access.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
ALIAS_NAME="codeassist"

echo ""
echo -e "${MAGENTA}${BOLD}  ██████╗ ██████╗ ██████╗ ███████╗    █████╗ ███████╗███████╗██╗███████╗████████╗${NC}"
echo -e "${MAGENTA}${BOLD} ██╔════╝██╔═══██╗██╔══██╗██╔════╝   ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝${NC}"
echo -e "${MAGENTA}${BOLD} ██║     ██║   ██║██║  ██║█████╗     ███████║███████╗███████╗██║███████╗   ██║   ${NC}"
echo -e "${MAGENTA}${BOLD} ██║     ██║   ██║██║  ██║██╔══╝     ██╔══██║╚════██║╚════██║██║╚════██║   ██║   ${NC}"
echo -e "${MAGENTA}${BOLD} ╚██████╗╚██████╔╝██████╔╝███████╗   ██║  ██║███████║███████║██║███████║   ██║   ${NC}"
echo -e "${MAGENTA}${BOLD}  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   ${NC}"
echo -e "${DIM}  Setup Script${NC}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check Python
# ---------------------------------------------------------------------------
echo -e "${CYAN}[1/4]${NC} Checking Python installation..."

if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
echo -e "  ${GREEN}✓${NC} Found Python ${PYTHON_VERSION}"

# Check minimum version (3.10)
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ required, found ${PYTHON_VERSION}${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Create virtual environment
# ---------------------------------------------------------------------------
echo -e "${CYAN}[2/4]${NC} Setting up virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}⚠${NC} Virtual environment already exists at ${VENV_DIR}"
    echo -e "  ${DIM}Reinstalling dependencies...${NC}"
else
    $PYTHON -m venv "$VENV_DIR"
    echo -e "  ${GREEN}✓${NC} Created virtual environment at ${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Step 3: Install dependencies
# ---------------------------------------------------------------------------
echo -e "${CYAN}[3/4]${NC} Installing dependencies..."

pip install --upgrade pip -q 2>/dev/null
pip install -r "${SCRIPT_DIR}/requirements.txt" -q 2>/dev/null

echo -e "  ${GREEN}✓${NC} All dependencies installed"

# ---------------------------------------------------------------------------
# Step 4: Create shell alias
# ---------------------------------------------------------------------------
echo -e "${CYAN}[4/4]${NC} Setting up shell alias..."

# Build the alias command
ALIAS_CMD="alias ${ALIAS_NAME}='${VENV_DIR}/bin/python ${SCRIPT_DIR}/main.py'"

# Determine shell config file
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    # Remove old alias if it exists
    if grep -q "alias ${ALIAS_NAME}=" "$SHELL_RC" 2>/dev/null; then
        # Use a temp file approach for portability
        grep -v "alias ${ALIAS_NAME}=" "$SHELL_RC" > "${SHELL_RC}.tmp"
        mv "${SHELL_RC}.tmp" "$SHELL_RC"
        echo -e "  ${DIM}Removed old alias${NC}"
    fi

    # Add the new alias
    echo "" >> "$SHELL_RC"
    echo "# Code Assist — LangGraph Coding Assistant" >> "$SHELL_RC"
    echo "${ALIAS_CMD}" >> "$SHELL_RC"
    echo -e "  ${GREEN}✓${NC} Added alias '${BOLD}${ALIAS_NAME}${NC}' to ${SHELL_RC}"
else
    echo -e "  ${YELLOW}⚠${NC} Could not find .zshrc or .bashrc"
    echo -e "  ${DIM}Add this manually to your shell config:${NC}"
    echo -e "  ${ALIAS_CMD}"
fi

# ---------------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo ""
echo -e "  ${DIM}# Reload your shell config${NC}"
echo -e "  source ${SHELL_RC:-~/.zshrc}"
echo ""
echo -e "  ${DIM}# Make sure Ollama is running with a model${NC}"
echo -e "  ollama pull qwen2.5-coder:14b"
echo -e "  ollama serve"
echo ""
echo -e "  ${DIM}# Run the assistant in any project directory${NC}"
echo -e "  cd /path/to/your/project"
echo -e "  ${BOLD}${ALIAS_NAME}${NC}"
echo ""
echo -e "  ${DIM}# Or specify a workspace explicitly${NC}"
echo -e "  ${BOLD}${ALIAS_NAME}${NC} --workspace /path/to/project"
echo ""
echo -e "  ${DIM}# Use a different model${NC}"
echo -e "  ${BOLD}${ALIAS_NAME}${NC} --model codellama:13b"
echo ""
echo -e "  ${DIM}# Use LM Studio instead of Ollama${NC}"
echo -e "  ${BOLD}${ALIAS_NAME}${NC} --base-url http://localhost:1234/v1"
echo ""
