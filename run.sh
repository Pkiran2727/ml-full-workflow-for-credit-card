#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "Yogyank Credit Scorer Linux Setup & Runner"
echo "=========================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.11 or 3.12."
    echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv"
    echo "CentOS/RHEL:   sudo dnf install -y python3 python3-pip"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

# Activate virtual env
source .venv/bin/activate

# Install requirements if not installed or requirements.txt changed
if [ ! -f ".venv/installed" ] || [ requirements.txt -nt .venv/installed ]; then
    echo "Installing/updating dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch .venv/installed
fi

# Run pipeline orchestrator forwarding all arguments
python pipelines/run_pipeline.py "$@"
