/**
 * Loop Monitor — sidebar panel for PromptLoop v0.8
 *
 * Shows:
 *   • Active agent-loop sessions (from ~/.loopllm/active_runs/*.json)
 *   • Recently completed episodes (from ~/.loopllm/episodes_feed.json)
 *
 * All data is read directly from the filesystem — no MCP connection needed.
 * The panel polls every 2 s via a setInterval driven from extension.ts.
 */

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ActiveRun {
  run_id: string;
  run_type: string;
  state: {
    session_id: string;
    goal: string;
    task_type: string;
    model_id: string;
    quality_threshold: number;
    suggested_budget: number;
    scores: number[];
    last_decision?: string;
    last_reason?: string;
    converged?: boolean | null;
    closed?: boolean;
    started_at?: number;
    last_step_at?: number;
    step_outputs?: string[];
  };
}

interface EpisodeFeedEntry {
  episode_type: string;
  goal: string;
  task_type: string;
  model_id?: string;
  score_final?: number | null;
  steps_used?: number | null;
  stop_reason?: string | null;
  recorded_at: string;
}

// ─── Provider ────────────────────────────────────────────────────────────────

export class LoopMonitorProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "loopllm.loopMonitor";

  private _view?: vscode.WebviewView;
  private readonly _loopllmDir: string;

  constructor(dbPath: string) {
    this._loopllmDir = path.dirname(dbPath);
  }

  resolveWebviewView(
    view: vscode.WebviewView,
    _ctx: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this._view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this._html();
    // Initial paint
    this.refresh();
  }

  /** Called by the poll timer in extension.ts */
  refresh(): void {
    if (!this._view?.webview) { return; }
    const runs = this._readActiveRuns();
    const episodes = this._readEpisodesFeed();
    this._view.webview.postMessage({ type: "update", runs, episodes });
  }

  // ─── Data readers ──────────────────────────────────────────────────────────

  private _readActiveRuns(): ActiveRun[] {
    const dir = path.join(this._loopllmDir, "active_runs");
    try {
      if (!fs.existsSync(dir)) { return []; }
      return fs.readdirSync(dir)
        .filter((f) => f.endsWith(".json"))
        .map((f) => {
          try {
            return JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8")) as ActiveRun;
          } catch { return null; }
        })
        .filter((r): r is ActiveRun => r !== null);
    } catch { return []; }
  }

  private _readEpisodesFeed(): EpisodeFeedEntry[] {
    const feedPath = path.join(this._loopllmDir, "episodes_feed.json");
    try {
      if (!fs.existsSync(feedPath)) { return []; }
      const data = JSON.parse(fs.readFileSync(feedPath, "utf-8"));
      return Array.isArray(data) ? (data as EpisodeFeedEntry[]).slice().reverse().slice(0, 10) : [];
    } catch { return []; }
  }

  // ─── HTML ──────────────────────────────────────────────────────────────────

  private _html(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loop Monitor</title>
<style>
  :root {
    --bg:      var(--vscode-editor-background);
    --fg:      var(--vscode-editor-foreground);
    --fg-dim:  var(--vscode-descriptionForeground);
    --border:  var(--vscode-panel-border);
    --accent:  var(--vscode-focusBorder);
    --input-bg: var(--vscode-input-background);
    --badge-bg: var(--vscode-badge-background);
    --badge-fg: var(--vscode-badge-foreground);
    --success: #4caf50;
    --warn:    #ff9800;
    --danger:  #f44336;
    --info:    #2196f3;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--vscode-font-family, system-ui);
    font-size: var(--vscode-font-size, 13px);
    color: var(--fg);
    background: var(--bg);
    padding: 8px;
    user-select: none;
  }

  h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--fg-dim);
    margin: 12px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  h3:first-of-type { margin-top: 0; }

  .empty {
    font-size: 12px;
    color: var(--fg-dim);
    padding: 6px 0;
    font-style: italic;
  }

  /* Active loop cards */
  .loop-card {
    background: var(--input-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .loop-card.stopped { border-left-color: var(--success); opacity: 0.75; }

  .loop-goal {
    font-weight: 600;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
  }
  .loop-meta {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 6px;
  }
  .badge {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--badge-bg);
    color: var(--badge-fg);
    white-space: nowrap;
  }
  .badge.green  { background: var(--success); color: #fff; }
  .badge.orange { background: var(--warn);    color: #000; }
  .badge.red    { background: var(--danger);  color: #fff; }
  .badge.blue   { background: var(--info);    color: #fff; }

  /* Score bar */
  .score-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }
  .score-label { font-size: 11px; color: var(--fg-dim); width: 58px; flex-shrink: 0; }
  .score-bar-wrap {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .score-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
  }
  .score-num { font-size: 11px; width: 32px; text-align: right; flex-shrink: 0; }

  .loop-reason {
    font-size: 11px;
    color: var(--fg-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 4px;
  }

  /* Episode rows */
  .ep-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 4px 8px;
    align-items: start;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .ep-row:last-child { border-bottom: none; }
  .ep-goal {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 12px;
  }
  .ep-score { font-weight: 600; font-size: 11px; text-align: right; }
  .ep-meta  { font-size: 10px; color: var(--fg-dim); text-align: right; white-space: nowrap; }
  .ep-stop  { grid-column: 1 / -1; font-size: 10px; color: var(--fg-dim); }

  .pulse {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 1.5s ease-in-out infinite;
    margin-right: 5px;
    vertical-align: middle;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }
</style>
</head>
<body>

<h3>Active Loops</h3>
<div id="active-loops"><p class="empty">No active loops</p></div>

<h3>Recent Episodes</h3>
<div id="episodes"><p class="empty">No episodes recorded yet</p></div>

<script>
const vscode = acquireVsCodeApi();

function scoreColor(s) {
  if (s == null) return '#888';
  if (s >= 0.8) return '#4caf50';
  if (s >= 0.6) return '#ff9800';
  return '#f44336';
}

function scorePct(s) {
  return s == null ? 0 : Math.round(s * 100);
}

function fmtScore(s) {
  return s == null ? '—' : s.toFixed(2);
}

function decisionBadge(d) {
  if (!d) return '';
  const cls = d === 'stop' ? 'green' : 'blue';
  return '<span class="badge ' + cls + '">' + d.toUpperCase() + '</span>';
}

function timeSince(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr.endsWith('Z') ? isoStr : isoStr + 'Z');
  const secs = Math.floor((Date.now() - d.getTime()) / 1000);
  if (secs < 60)  return secs + 's ago';
  if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
  return Math.floor(secs / 3600) + 'h ago';
}

function renderActiveLoops(runs) {
  const el = document.getElementById('active-loops');
  if (!runs || runs.length === 0) {
    el.innerHTML = '<p class="empty">No active loops</p>';
    return;
  }
  el.innerHTML = runs.map(run => {
    const s = run.state || {};
    const scores = s.scores || [];
    const lastScore = scores.length ? scores[scores.length - 1] : null;
    const step = scores.length;
    const budget = s.suggested_budget || '?';
    const stopped = s.closed || s.last_decision === 'stop';
    const cardCls = stopped ? 'loop-card stopped' : 'loop-card';
    const goal = (s.goal || run.run_id).slice(0, 80);
    const taskType = s.task_type || 'general';
    const pct = scorePct(lastScore);
    const col = scoreColor(lastScore);
    const reason = (s.last_reason || '').slice(0, 90);

    return \`<div class="\${cardCls}">
      <div class="loop-goal">
        \${stopped ? '' : '<span class="pulse"></span>'}
        \${goal}
      </div>
      <div class="loop-meta">
        <span class="badge">\${taskType}</span>
        <span class="badge \${stopped ? 'green' : 'orange'}">Step \${step}/\${budget}</span>
        \${decisionBadge(s.last_decision)}
      </div>
      <div class="score-row">
        <span class="score-label">Last score</span>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:\${pct}%;background:\${col}"></div>
        </div>
        <span class="score-num">\${fmtScore(lastScore)}</span>
      </div>
      <div class="score-row">
        <span class="score-label">Threshold</span>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:\${scorePct(s.quality_threshold)}%;background:#888"></div>
        </div>
        <span class="score-num">\${fmtScore(s.quality_threshold)}</span>
      </div>
      \${reason ? '<div class="loop-reason">' + reason + '</div>' : ''}
    </div>\`;
  }).join('');
}

function renderEpisodes(eps) {
  const el = document.getElementById('episodes');
  if (!eps || eps.length === 0) {
    el.innerHTML = '<p class="empty">No episodes recorded yet</p>';
    return;
  }
  el.innerHTML = eps.map(ep => {
    const score = ep.score_final;
    const col = scoreColor(score);
    const steps = ep.steps_used != null ? ep.steps_used + ' steps' : '';
    const stopLabel = ep.stop_reason ? ep.stop_reason.replace(/_/g, ' ') : '';
    const ago = timeSince(ep.recorded_at);
    const goal = (ep.goal || '(unknown)').slice(0, 70);
    return \`<div class="ep-row">
      <span class="ep-goal" title="\${ep.goal}">\${goal}</span>
      <span class="ep-score" style="color:\${col}">\${fmtScore(score)}</span>
      <span class="ep-meta">\${ep.task_type || ''}</span>
      <span class="ep-stop">\${steps}\${steps && stopLabel ? ' · ' : ''}\${stopLabel}\${ago ? ' · ' + ago : ''}</span>
    </div>\`;
  }).join('');
}

window.addEventListener('message', ev => {
  const msg = ev.data;
  if (msg.type === 'update') {
    renderActiveLoops(msg.runs);
    renderEpisodes(msg.episodes);
  }
});
</script>
</body>
</html>`;
  }
}
