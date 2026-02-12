#!/usr/bin/env bash
# ------------------------------------------------------------------
# test_with_local_build.sh
# Installs the latest cogdb wheel from a local dev server, cleans
# any stale DB files, and runs text_to_kg.py.
#
# Usage:
#   ./test_with_local_build.sh                  # defaults
#   ./test_with_local_build.sh 9999 3.7.0       # custom port & version
# ------------------------------------------------------------------
set -euo pipefail

SERVER_PORT="${1:-8888}"
VERSION="${2:-3.6.6}"
WHEEL="cogdb-${VERSION}-py3-none-any.whl"
WHEEL_URL="http://localhost:${SERVER_PORT}/${WHEEL}"
VENV_PYTHON="$(dirname "$0")/.venv/bin/python"
DB_DIR="/tmp/cog_home"

echo "========================================"
echo " CogDB local build → text_to_kg tester"
echo "========================================"

# 1. Check the local wheel server is reachable
echo -e "\n[1/5] Checking wheel server at localhost:${SERVER_PORT} ..."
if ! curl -sf --head "${WHEEL_URL}" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach ${WHEEL_URL}"
    echo "       Start the server first:  python scripts/local_wheel_server.py"
    exit 1
fi
echo "  ✓ ${WHEEL} is available"

# 2. Install the wheel (force-reinstall to pick up new builds)
echo -e "\n[2/5] Installing ${WHEEL} ..."
"${VENV_PYTHON}" -m pip install --force-reinstall --quiet "${WHEEL_URL}"
INSTALLED=$("${VENV_PYTHON}" -c "from importlib.metadata import version; print(version('cogdb'))")
echo "  ✓ cogdb ${INSTALLED} installed"

# 3. Clean stale database to avoid marshal errors from old builds
echo -e "\n[3/5] Cleaning stale DB at ${DB_DIR} ..."
rm -rf "${DB_DIR}"
echo "  ✓ cleaned"

# 4. Run the demo
echo -e "\n[4/5] Running text_to_kg.py ..."
echo "----------------------------------------"
"${VENV_PYTHON}" "$(dirname "$0")/text_to_kg.py"
EXIT_CODE=$?
echo "----------------------------------------"

# 5. Report result
echo -e "\n[5/5] Result"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  ✓ text_to_kg.py passed (exit 0)"
else
    echo "  ✗ text_to_kg.py FAILED (exit ${EXIT_CODE})"
fi

exit ${EXIT_CODE}
