# AGENTS.md

## Cursor Cloud specific instructions

`loopllm` (PromptLoop) is a pure-Python package: a CLI, an MCP stdio server (the
primary use), and an optional REST scoring server. `vscode-loopllm/` is a secondary
TypeScript VS Code extension. There is no web UI — verify via terminal/logs.

### Python environment (main product)
- A virtualenv at `.venv/` is the canonical interpreter. Ubuntu is PEP 668
  externally-managed, so a bare `pip install` into the system Python fails; always use
  the venv (e.g. `.venv/bin/pytest`, `.venv/bin/loopllm`, `.venv/bin/ruff`).
- The `python3.12-venv` system package is required to create the venv (already present
  in the cloud snapshot). The update script refreshes the editable install.
- Standard dev commands live in `CONTRIBUTING.md` / `pyproject.toml`. Run them via the
  venv: `ruff check src/ tests/`, `mypy --strict src/loopllm/`, `pytest tests/`.
- `pytest` skips 4 `integration`-marked tests by default (they need a live LLM API).

### Running the services
- MCP server: `.venv/bin/loopllm mcp-server --provider agent`. It speaks JSON-RPC over
  stdio and produces NO output until a client sends a request — silence on startup is
  normal, not a hang. IDEs (Cursor/VS Code) launch it via `.cursor/mcp.json` /
  `.vscode/mcp.json`; you rarely run it by hand.
- CLI quick check (offline, no MCP): `.venv/bin/loopllm score "<prompt>"`.
- REST scoring server (optional): `.venv/bin/loopllm serve --port 8765`.
- Persistent state (learned weights, episodes, plans) lives in `~/.loopllm/store.db`;
  override with `LOOPLLM_DB`.

### VS Code extension (`vscode-loopllm/`)
- Build with `npm run compile` (runs `tsc -p ./` → `out/`).
- `npm run lint` currently fails: no ESLint config is committed in the repo. This is a
  pre-existing repo limitation, not an environment problem — use `npm run compile` to
  validate.
