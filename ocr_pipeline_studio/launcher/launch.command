#!/bin/sh
# Starts OCR Pipeline Studio, setting itself up on first run. Drag this into
# the Dock, or right-click it and choose Make Alias to put one on the Desktop.
#
# The macOS mirror of launch.cmd, and simpler: a .command file already opens a
# visible Terminal, so setup runs here rather than in a window of its own. No
# "activate" step -- calling the venv's own python is what activating does.
#
# Lives in launcher/, so it works one level up.
#
# The venv lives on this Mac, not in the project folder: the folder may be a
# network share opened by several machines, and a venv only runs on the one
# that built it. Code and workdir/ stay shared; each Mac sets up once.
set -e
cd "$(dirname "$0")/.."

LOCAL="$HOME/Library/Application Support/OCR Pipeline Studio"
VENV="$LOCAL/.venv"
STAMP="$VENV/.setup-complete"
# Goes into the stamp; a Mac whose stamp lacks it re-runs setup, which reuses
# the venv and only replaces paddle. Change it with the pin in requirements.txt.
ENGINE="engine=paddle-3.2.2-cpu"
mkdir -p "$LOCAL"

# Bytecode caches stay on this Mac too, rather than in the shared folder.
export PYTHONPYCACHEPREFIX="$LOCAL/pycache"

# The stamp is written last, so a venv without one is a setup that died part
# way -- indistinguishable from a finished one until the app fails on a
# missing package.
if [ ! -x "$VENV/bin/python" ] || [ ! -f "$STAMP" ] || ! grep -qx "$ENGINE" "$STAMP"; then
    echo "============================================================"
    echo " OCR Pipeline Studio - first-time setup"
    echo "============================================================"
    echo
    echo "This downloads a few hundred megabytes and takes a few minutes."
    echo "You only have to do this once."
    echo
    if [ -f "$STAMP" ]; then
        echo "This Mac was set up before, with an older OCR engine. Only the"
        echo "engine is replaced."
        echo
    fi

    # 3.11 specifically: paddleocr publishes no wheels for 3.12 or newer.
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "Python 3.11 is not installed. The app needs that exact version:"
        echo "the OCR engine publishes no packages for 3.12 or newer."
        echo
        echo "    https://www.python.org/downloads/release/python-3119/"
        echo
        echo "Install it, then run this again."
        exit 1
    fi

    [ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"

    # On macOS requirements.txt carries paddle itself, via its
    # sys_platform == "darwin" marker, so there is no second install step.
    "$VENV/bin/python" -m pip install --disable-pip-version-check -r requirements.txt

    printf 'Setup completed. Delete this file to make setup run again.\n%s\n' "$ENGINE" > "$STAMP"
    echo
    echo "Setup finished."

    if ! command -v ollama >/dev/null 2>&1; then
        echo
        echo "The vision-model step also needs Ollama: https://ollama.com"
        echo "The app runs without it; the OCR/VLM comparison does not."
    elif ! ollama list 2>/dev/null | grep -qi 'qwen2.5vl:3b'; then
        echo
        printf 'Pull the vision model qwen2.5vl:3b now (about 3 GB)? [y/N] '
        # `|| answer=` so an EOF on stdin does not trip `set -e` and stop
        # the app from starting.
        read -r answer || answer=""
        case "$answer" in [Yy]*) ollama pull qwen2.5vl:3b ;; esac
    fi
    echo
fi

exec "$VENV/bin/python" app.py
