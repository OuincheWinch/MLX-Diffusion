#!/bin/zsh
cd "$(dirname "$0")/backend" && caffeinate -s ../venv/bin/uvicorn main:app --reload --port 8001
