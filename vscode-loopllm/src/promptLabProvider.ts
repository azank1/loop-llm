/**
 * Prompt Lab — live quality scratchpad.
 *
 * A sidebar webview where developers draft prompts and get instant
 * per-dimension quality feedback before sending to Copilot or any agent.
 *
 * Communication protocol:
 *   webview → host: { type: "score", text: string }
 *   host → webview: { type: "result", data: ScorePayload }
 *                   { type: "error", message: string }
 */

import * as vscode from "vscode";
import * as cp from "child_process";

export interface ScorePayload {
  quality_score: number;
  grade: string;
  gauge: string;
  task_type: string;
  route: string;
  dimensions: Record<string, number>;
  suggestions: string[];
  issues: string[];
}

export class PromptLabProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "loopllm.promptLab";

  private _view?: vscode.WebviewView;
  private _dbPath: string;
  private _pending?: ReturnType<typeof setTimeout>;

  constructor(dbPath: string) {
    this._dbPath = dbPath;
  }

  resolveWebviewView(
    view: vscode.WebviewView,
    _ctx: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this._html();

    view.webview.onDidReceiveMessage((msg: { type: string; text?: string }) => {
      if (msg.type === "score" && msg.text) {
        // Debounce: cancel any pending execution
        if (this._pending) { clearTimeout(this._pending); }
        this._pending = setTimeout(() => {
          this._runScore(msg.text!);
        }, 0);
      } else if (msg.type === "insertIntoCopilot" && msg.text) {
        // Copy to clipboard and open chat
        vscode.env.clipboard.writeText(msg.text).then(() => {
          vscode.commands.executeCommand("workbench.action.chat.open");
        });
      }
    });
  }

  /** Called from extension when StatusWatcher fires (passthrough updates). */
  pushUpdate(data: ScorePayload): void {
    this._post({ type: "result", data });
  }

  private _runScore(text: string): void {
    const trimmed = text.trim();
    if (trimmed.length < 3) {
      this._post({ type: "clear" });
      return;
    }

    // VS Code child processes get a minimal PATH that often excludes the
    // Python env bin dir. Augment with common locations so loopllm is found.
    const extraPaths = [
      "/home/codespace/.python/current/bin",
      "/usr/local/bin",
      "/usr/bin",
      "/opt/homebrew/bin",
    ].join(":");
    const env = {
      ...process.env,
      PATH: `${extraPaths}:${process.env.PATH ?? ""}`,
    };

    const args = ["--db", this._dbPath, "score", "--json", trimmed.slice(0, 3000)];

    cp.execFile("loopllm", args, { timeout: 8000, env }, (err, stdout) => {
      if (!err) { this._parseAndPost(stdout); return; }

      // Fallback 1: python3 -m loopllm
      cp.execFile("python3", ["-m", "loopllm", ...args], { timeout: 8000, env }, (err2, stdout2) => {
        if (!err2) { this._parseAndPost(stdout2); return; }

        // Fallback 2: absolute path via common env layout
        cp.execFile("/home/codespace/.python/current/bin/loopllm", args, { timeout: 8000 }, (err3, stdout3) => {
          if (!err3) { this._parseAndPost(stdout3); return; }
          this._post({ type: "error", message: "loopllm not found — run: pip install -e .[mcp] in the repo root" });
        });
      });
    });
  }

  private _parseAndPost(stdout: string): void {
    try {
      const data: ScorePayload = JSON.parse(stdout.trim());
      this._post({ type: "result", data });
    } catch {
      this._post({ type: "error", message: "Failed to parse score output" });
    }
  }

  private _post(msg: unknown): void {
    this._view?.webview.postMessage(msg);
  }

  private _html(): string {
    return /* html */`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-sideBar-background, var(--vscode-editor-background));
    padding: 10px;
  }
  textarea {
    width: 100%;
    min-height: 100px;
    max-height: 280px;
    resize: vertical;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, #555);
    border-radius: 4px;
    padding: 8px;
    font-family: inherit;
    font-size: 12px;
    line-height: 1.5;
    outline: none;
    transition: border-color 0.15s;
  }
  textarea:focus { border-color: var(--vscode-focusBorder); }
  #placeholder-hint {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-top: 5px;
    line-height: 1.4;
  }
  #score-panel { margin-top: 10px; display: none; }
  #header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  #grade {
    font-size: 22px;
    font-weight: 700;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    flex-shrink: 0;
  }
  #score-pct { font-size: 18px; font-weight: 600; }
  #task-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  #gauge-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--vscode-progressBar-background, #333);
    margin-bottom: 10px;
    overflow: hidden;
  }
  #gauge-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.25s ease, background 0.25s ease;
  }
  .section-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--vscode-descriptionForeground);
    margin-bottom: 5px;
  }
  .dim-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }
  .dim-name {
    width: 68px;
    font-size: 11px;
    flex-shrink: 0;
    color: var(--vscode-descriptionForeground);
  }
  .dim-track {
    flex: 1;
    height: 5px;
    border-radius: 3px;
    background: var(--vscode-progressBar-background, #333);
    overflow: hidden;
  }
  .dim-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.25s ease;
  }
  .dim-val {
    width: 28px;
    font-size: 10px;
    text-align: right;
    color: var(--vscode-descriptionForeground);
  }
  .tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
  .tag {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
    line-height: 1.4;
  }
  .tag-issue { background: rgba(255,80,80,0.15); color: #ff8080; }
  .tag-suggest { background: rgba(80,160,255,0.15); color: #74b9ff; }
  #actions { display: flex; gap: 6px; margin-top: 10px; }
  button {
    flex: 1;
    padding: 5px 10px;
    border: none;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    font-family: inherit;
  }
  #btn-copy {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
  }
  #btn-copy:hover { background: var(--vscode-button-hoverBackground); }
  #btn-chat {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
  }
  #btn-chat:hover { background: var(--vscode-button-secondaryHoverBackground); }
  #btn-clear {
    flex: 0;
    padding: 5px 10px;
    background: transparent;
    color: var(--vscode-descriptionForeground);
    border: 1px solid var(--vscode-input-border, #555);
  }
  #btn-clear:hover { color: var(--vscode-foreground); }
  #error-msg {
    font-size: 11px;
    color: #ff8080;
    margin-top: 8px;
    display: none;
  }
  #spinner {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-top: 6px;
    display: none;
  }
  .divider { height: 1px; background: var(--vscode-widget-border, #333); margin: 8px 0; }
</style>
</head>
<body>
<textarea id="prompt-input" placeholder="Draft your prompt here…&#10;Quality scores update as you type."></textarea>
<p id="placeholder-hint">As you type, each dimension is scored. Use the grade to decide if your prompt is ready to send.</p>
<div id="spinner">Scoring…</div>
<div id="error-msg"></div>

<div id="score-panel">
  <div id="header-row">
    <div id="grade">—</div>
    <div id="score-pct">—%</div>
    <div id="task-badge">general</div>
  </div>
  <div id="gauge-bar"><div id="gauge-fill" style="width:0%"></div></div>

  <div class="section-label">Dimensions</div>
  <div id="dims"></div>

  <div class="divider"></div>
  <div id="issues-section" style="display:none">
    <div class="section-label">Issues</div>
    <div id="issues" class="tags"></div>
  </div>
  <div id="suggestions-section" style="display:none">
    <div class="section-label">Suggestions</div>
    <div id="suggestions" class="tags"></div>
  </div>

  <div id="actions">
    <button id="btn-copy" title="Copy prompt to clipboard">Copy</button>
    <button id="btn-chat" title="Copy &amp; open Copilot chat">Send to Chat</button>
    <button id="btn-clear">✕</button>
  </div>
</div>

<script>
const vscode = acquireVsCodeApi();
const input = document.getElementById('prompt-input');
const panel = document.getElementById('score-panel');
const gradeEl = document.getElementById('grade');
const pctEl = document.getElementById('score-pct');
const taskEl = document.getElementById('task-badge');
const gaugeFill = document.getElementById('gauge-fill');
const dimsEl = document.getElementById('dims');
const issuesEl = document.getElementById('issues');
const issuesSec = document.getElementById('issues-section');
const suggestEl = document.getElementById('suggestions');
const suggestSec = document.getElementById('suggestions-section');
const spinnerEl = document.getElementById('spinner');
const errorEl = document.getElementById('error-msg');
const hint = document.getElementById('placeholder-hint');

const DIM_LABELS = {
  specificity: 'Specific',
  constraint_clarity: 'Constraints',
  context_completeness: 'Context',
  ambiguity: 'Ambiguity',
  format_spec: 'Format',
};

let debounceTimer = null;
let lastText = '';

input.addEventListener('input', () => {
  const text = input.value;
  if (text === lastText) return;
  lastText = text;

  if (debounceTimer) clearTimeout(debounceTimer);
  if (text.trim().length < 3) {
    panel.style.display = 'none';
    errorEl.style.display = 'none';
    spinnerEl.style.display = 'none';
    return;
  }

  spinnerEl.style.display = 'block';
  debounceTimer = setTimeout(() => {
    vscode.postMessage({ type: 'score', text });
  }, 350);
});

document.getElementById('btn-copy').addEventListener('click', () => {
  navigator.clipboard.writeText(input.value);
  document.getElementById('btn-copy').textContent = 'Copied!';
  setTimeout(() => { document.getElementById('btn-copy').textContent = 'Copy'; }, 1500);
});

document.getElementById('btn-chat').addEventListener('click', () => {
  vscode.postMessage({ type: 'insertIntoCopilot', text: input.value });
});

document.getElementById('btn-clear').addEventListener('click', () => {
  input.value = '';
  lastText = '';
  panel.style.display = 'none';
  errorEl.style.display = 'none';
  spinnerEl.style.display = 'none';
  hint.style.display = 'block';
});

window.addEventListener('message', (event) => {
  const msg = event.data;
  if (msg.type === 'result') {
    spinnerEl.style.display = 'none';
    errorEl.style.display = 'none';
    hint.style.display = 'none';
    renderResult(msg.data);
  } else if (msg.type === 'error') {
    spinnerEl.style.display = 'none';
    panel.style.display = 'none';
    errorEl.textContent = msg.message;
    errorEl.style.display = 'block';
  } else if (msg.type === 'clear') {
    panel.style.display = 'none';
    spinnerEl.style.display = 'none';
  }
});

function gradeColor(grade) {
  const map = { A: '#4caf50', B: '#8bc34a', C: '#ff9800', D: '#ff5722', F: '#f44336' };
  return map[grade] || '#888';
}

function scoreColor(score) {
  if (score >= 0.7) return '#4caf50';
  if (score >= 0.5) return '#ff9800';
  return '#f44336';
}

function renderResult(d) {
  const pct = Math.round(d.quality_score * 100);
  const color = gradeColor(d.grade);

  gradeEl.textContent = d.grade;
  gradeEl.style.background = color + '22';
  gradeEl.style.color = color;
  gradeEl.style.border = '1px solid ' + color + '55';

  pctEl.textContent = pct + '%';
  pctEl.style.color = color;

  taskEl.textContent = (d.task_type || 'general').replace(/_/g, ' ');

  const fillColor = scoreColor(d.quality_score);
  gaugeFill.style.width = pct + '%';
  gaugeFill.style.background = fillColor;

  // Dimensions
  dimsEl.innerHTML = '';
  const dims = d.dimensions || {};
  for (const [key, raw] of Object.entries(dims)) {
    const val = Number(raw);
    // ambiguity is inverted: lower raw = better
    const displayVal = key === 'ambiguity' ? 1 - val : val;
    const pctD = Math.round(displayVal * 100);
    const dc = scoreColor(displayVal);
    dimsEl.insertAdjacentHTML('beforeend', \`
      <div class="dim-row">
        <span class="dim-name">\${DIM_LABELS[key] || key}</span>
        <div class="dim-track"><div class="dim-fill" style="width:\${pctD}%;background:\${dc}"></div></div>
        <span class="dim-val">\${pctD}%</span>
      </div>
    \`);
  }

  // Issues
  const issues = d.issues || [];
  if (issues.length > 0) {
    issuesEl.innerHTML = issues.map(i => \`<span class="tag tag-issue">\${i}</span>\`).join('');
    issuesSec.style.display = 'block';
  } else {
    issuesSec.style.display = 'none';
  }

  // Suggestions
  const sugs = d.suggestions || [];
  if (sugs.length > 0) {
    suggestEl.innerHTML = sugs.map(s => \`<span class="tag tag-suggest">\${s}</span>\`).join('');
    suggestSec.style.display = 'block';
  } else {
    suggestSec.style.display = 'none';
  }

  panel.style.display = 'block';
}
</script>
</body>
</html>`;
  }
}
