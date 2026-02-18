#!/usr/bin/env bash
# ------------------------------------------------------------------
# run.sh
# Creates a virtual environment, installs dependencies, and runs
# text_to_kg.py.
#
# Usage:
#   ./run.sh              # create venv + run
#   ./run.sh --clean      # delete venv & DB, recreate everything
# ------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
DB_DIR="${SCRIPT_DIR}/cog_home"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
MAIN_SCRIPT="${SCRIPT_DIR}/text_to_kg.py"

# ── Handle --clean flag ──────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "[clean] Removing virtual environment and database ..."
    rm -rf "${VENV_DIR}" "${DB_DIR}"
    echo "  done"
fi

echo "========================================"
echo " text-to-kg runner"
echo "========================================"

# ── 1. Check for .env / API key ──────────────────────────────────
if [[ ! -f "${SCRIPT_DIR}/.env" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo ""
    echo "WARNING: No .env file found and OPENAI_API_KEY is not set."
    echo "  Create one with:  echo 'OPENAI_API_KEY=sk-...' > .env"
    echo ""
fi

# ── 2. Create virtual environment if it doesn't exist ────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    echo -e "\n[1/3] Creating virtual environment ..."
    python3 -m venv "${VENV_DIR}"
    echo "  ✓ venv created at ${VENV_DIR}"
else
    echo -e "\n[1/3] Virtual environment already exists"
fi

# ── 3. Install / update dependencies ────────────────────────────
echo -e "\n[2/3] Installing dependencies ..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip --quiet
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}" --quiet
echo "  ✓ dependencies installed"

# Show installed versions
COGDB_VER=$("${VENV_DIR}/bin/python" -c "from importlib.metadata import version; print(version('cogdb'))")
OPENAI_VER=$("${VENV_DIR}/bin/python" -c "from importlib.metadata import version; print(version('openai'))")
echo "  cogdb=${COGDB_VER}  openai=${OPENAI_VER}"

# ── 4. Run the script ───────────────────────────────────────────
echo -e "\n[3/3] Running text_to_kg.py ..."
echo "----------------------------------------"
"${VENV_DIR}/bin/python" "${MAIN_SCRIPT}"
EXIT_CODE=$?
echo "----------------------------------------"

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "\n✓ Finished successfully"
else
    echo -e "\n✗ Failed (exit ${EXIT_CODE})"
fi

exit ${EXIT_CODE}
