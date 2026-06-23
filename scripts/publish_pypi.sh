#!/usr/bin/env bash
# Build and publish loopllm to PyPI. Requires:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...   # API token from pypi.org
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')"

if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "error: TWINE_PASSWORD not set (PyPI API token)." >&2
  echo "  export TWINE_USERNAME=__token__" >&2
  echo "  export TWINE_PASSWORD=pypi-..." >&2
  exit 1
fi

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -q build twine
rm -rf dist/
.venv/bin/python -m build
.venv/bin/twine check "dist/loopllm-${VERSION}"*

echo "Uploading loopllm ${VERSION} to PyPI..."
.venv/bin/twine upload "dist/loopllm-${VERSION}"*

echo "Verifying install from PyPI..."
rm -rf /tmp/loopllm-pypi-verify
python3 -m venv /tmp/loopllm-pypi-verify
/tmp/loopllm-pypi-verify/bin/pip install -q "loopllm[mcp]==${VERSION}"
/tmp/loopllm-pypi-verify/bin/python -c "import loopllm; assert loopllm.__version__ == '${VERSION}'"
/tmp/loopllm-pypi-verify/bin/loopllm --help | head -1
echo "OK: pip install loopllm[mcp]==${VERSION} works."
