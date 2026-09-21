#!/usr/bin/env bash
set -euo pipefail

# Resolve the project root from this script's own location, so run.sh works
# from any current directory and any install path (resolves symlinks too).
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# Locate a usable Python venv (prefer repo venv, then venv-sdxl isn't one for the API).
VENV_PY=""
for cand in "$PROJECT_DIR/venv/bin/uvicorn" "$PROJECT_DIR/backend/venv/bin/uvicorn"; do
    if [ -x "$cand" ]; then
        VENV_PY="$cand"
        break
    fi
done
if [ -z "$VENV_PY" ]; then
    echo "❌ No venv found (looked for $PROJECT_DIR/venv/bin/uvicorn). Run setup first."
    exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "⚠️  Frontend deps not installed — run: cd \"$FRONTEND_DIR\" && npm install"
fi

echo "================================================="
echo "        🚀 Starting MLX-DIFFUSION Studio         "
echo "================================================="
echo "  project: $PROJECT_DIR"
echo "  backend: $BACKEND_DIR"
echo "  frontend: $FRONTEND_DIR"
echo "  python : $VENV_PY"

# 1. Clean any stuck processes on ports 8001 or 5174
if lsof -ti :8001 >/dev/null 2>&1; then
    echo "⚠️  Port 8001 is busy, freeing..."
    lsof -ti :8001 | xargs kill -9 2>/dev/null || true
fi
if lsof -ti :5174 >/dev/null 2>&1; then
    echo "⚠️  Port 5174 is busy, freeing..."
    lsof -ti :5174 | xargs kill -9 2>/dev/null || true
fi

# Function to clean up children on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down MLX-DIFFUSION..."
    trap - SIGINT SIGTERM EXIT
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    pkill -f sdxl_engine.py 2>/dev/null || true
    lsof -ti :8001,5174 | xargs kill -9 2>/dev/null || true
    echo "✨ All processes stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 2. Launch Backend
echo "📦 Starting FastAPI backend on http://127.0.0.1:8001..."
cd "$BACKEND_DIR"
caffeinate -s "$VENV_PY" main:app --reload --port 8001 &
BACKEND_PID=$!

# Wait for backend to be ready
echo -n "⏳ Waiting for backend to initialize..."
for _ in {1..30}; do
    if curl -s http://127.0.0.1:8001/api/models >/dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    sleep 0.5
    echo -n "."
done

# 3. Launch Frontend
echo "💻 Starting Vite frontend on http://localhost:5174..."
cd "$FRONTEND_DIR"
npm run dev >/dev/null 2>&1 &
FRONTEND_PID=$!

sleep 1.5

echo ""
echo "================================================="
echo "  🎨 MLX-DIFFUSION is LIVE!                      "
echo "  👉 Web UI:        http://localhost:5174         "
echo "  👉 API Docs:      http://localhost:8001/docs    "
echo "  (Press Ctrl+C to stop all services)            "
echo "================================================="
echo ""

# Automatically open UI in default browser on macOS
if command -v open >/dev/null 2>&1; then
    open http://localhost:5174 2>/dev/null || true
fi

# Keep script running and wait for background processes
wait "$BACKEND_PID" "$FRONTEND_PID"