#!/usr/bin/env bash
# QuantAgent — startup script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "           "
echo "      "
echo "              "
echo "           "
echo "          "
echo "              "
echo "         AGENT — Live Market Intelligence"
echo ""

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.10+"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
echo "Python: $($PYTHON --version)"

# Check .env
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy .env.example and fill in your API keys."
    exit 1
fi

# Install deps if needed
if ! $PYTHON -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt -q
fi

# Parse args
MODE="live"
if [[ "$1" == "--mock" ]]; then
    MODE="mock"
    export QUANT_MOCK_MODE=1
    echo "Running in MOCK MODE (no live API calls)"
fi

echo "Mode: $MODE"
echo "Dashboard: http://localhost:$(grep DASHBOARD_PORT .env | cut -d= -f2 || echo 8765)"
echo ""

# Run
exec $PYTHON main.py
