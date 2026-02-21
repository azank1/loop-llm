/**
 * Sidebar webview provider — renders the prompt quality dashboard
 * with learning curve chart, dimension radar, suggestions, and history.
 *
 * Uses Chart.js loaded from CDN for visualizations.
 */

import * as vscode from "vscode";
import { DataProvider, type PromptStats } from "./dataProvider";
import type { StatusData } from "./statusWatcher";

export class DashboardViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private extensionUri: vscode.Uri;
  private dataProvider: DataProvider;
  private lastStats: PromptStats | null = null;
  private lastStatus: StatusData | null = null;

  constructor(extensionUri: vscode.Uri, dataProvider: DataProvider) {
    this.extensionUri = extensionUri;
    this.dataProvider = dataProvider;
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
    };

    webviewView.webview.html = this.getHtml();

    // Send cached data if available
    if (this.lastStats) {
      this.postMessage({ type: "stats", data: this.lastStats });
    }
    if (this.lastStatus) {
      this.postMessage({ type: "status", data: this.lastStatus });
    }
  }

  pushUpdate(status: StatusData): void {
    this.lastStatus = status;
    this.postMessage({ type: "status", data: status });
  }

  pushStats(stats: PromptStats): void {
    this.lastStats = stats;
    this.postMessage({ type: "stats", data: stats });
  }

  private postMessage(message: unknown): void {
    if (this.view?.webview) {
      this.view.webview.postMessage(message);
    }
  }

  private getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loop LLM Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: var(--vscode-editor-background);
      --fg: var(--vscode-editor-foreground);
      --border: var(--vscode-panel-border);
      --accent: var(--vscode-focusBorder);
      --success: #4caf50;
      --warning: #ff9800;
      --error: #f44336;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--fg);
      background: var(--bg);
      padding: 12px;
    }
    .section {
      margin-bottom: 16px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
    }
    .section h3 {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      opacity: 0.7;
      margin-bottom: 8px;
    }
    .gauge-container {
      text-align: center;
      padding: 8px 0;
    }
    .gauge-score {
      font-size: 36px;
      font-weight: bold;
    }
    .gauge-grade {
      font-size: 14px;
      opacity: 0.8;
      margin-top: 4px;
    }
    .gauge-bar {
      width: 100%;
      height: 8px;
      background: var(--border);
      border-radius: 4px;
      margin-top: 8px;
      overflow: hidden;
    }
    .gauge-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.5s ease, background 0.5s ease;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      font-size: 12px;
    }
    .stat-label { opacity: 0.7; }
    .stat-value { font-weight: bold; }
    .trend-up { color: var(--success); }
    .trend-down { color: var(--error); }
    .trend-stable { color: var(--warning); }
    canvas {
      width: 100% !important;
      max-height: 180px;
    }
    .suggestions {
      list-style: none;
      padding: 0;
    }
    .suggestions li {
      padding: 4px 0 4px 16px;
      position: relative;
      font-size: 12px;
    }
    .suggestions li::before {
      content: "💡";
      position: absolute;
      left: 0;
    }
    .issues li::before {
      content: "⚠️";
    }
    .no-data {
      text-align: center;
      padding: 20px;
      opacity: 0.5;
      font-style: italic;
    }
  </style>
</head>
<body>
  <!-- Quality Gauge -->
  <div class="section" id="gauge-section">
    <h3>Prompt Quality</h3>
    <div class="gauge-container" id="gauge">
      <div class="gauge-score" id="score-display">--</div>
      <div class="gauge-grade" id="grade-display">No data yet</div>
      <div class="gauge-bar">
        <div class="gauge-fill" id="gauge-fill" style="width: 0%"></div>
      </div>
    </div>
  </div>

  <!-- Stats Overview -->
  <div class="section" id="stats-section">
    <h3>Overview</h3>
    <div class="stat-row">
      <span class="stat-label">Total Prompts</span>
      <span class="stat-value" id="total-prompts">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Average Quality</span>
      <span class="stat-value" id="avg-quality">--</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Trend</span>
      <span class="stat-value" id="trend">--</span>
    </div>
  </div>

  <!-- Learning Curve Chart -->
  <div class="section">
    <h3>Learning Curve</h3>
    <div id="chart-container">
      <canvas id="learningChart"></canvas>
    </div>
  </div>

  <!-- Dimension Radar -->
  <div class="section">
    <h3>Prompt Dimensions</h3>
    <canvas id="radarChart"></canvas>
  </div>

  <!-- Suggestions -->
  <div class="section" id="suggestions-section" style="display:none">
    <h3>Suggestions</h3>
    <ul class="suggestions" id="suggestions-list"></ul>
  </div>

  <!-- Issues -->
  <div class="section" id="issues-section" style="display:none">
    <h3>Issues Found</h3>
    <ul class="suggestions issues" id="issues-list"></ul>
  </div>

  <script>
    const vscode = acquireVsCodeApi();

    let learningChart = null;
    let radarChart = null;

    function getColor(score) {
      if (score >= 0.7) return '#4caf50';
      if (score >= 0.5) return '#ff9800';
      return '#f44336';
    }

    function updateGauge(score, grade) {
      const pct = Math.round(score * 100);
      document.getElementById('score-display').textContent = pct + '%';
      document.getElementById('grade-display').textContent = 'Grade: ' + grade;
      const fill = document.getElementById('gauge-fill');
      fill.style.width = pct + '%';
      fill.style.background = getColor(score);
    }

    function updateStats(stats) {
      document.getElementById('total-prompts').textContent = stats.total_prompts || 0;
      document.getElementById('avg-quality').textContent =
        stats.avg_quality ? Math.round(stats.avg_quality * 100) + '%' : '--';

      const trendEl = document.getElementById('trend');
      const trend = stats.trend || 'no_data';
      const trendMap = {
        improving: ['↑ Improving', 'trend-up'],
        declining: ['↓ Declining', 'trend-down'],
        stable: ['→ Stable', 'trend-stable'],
        no_data: ['--', ''],
      };
      const [text, cls] = trendMap[trend] || ['--', ''];
      trendEl.textContent = text;
      trendEl.className = 'stat-value ' + cls;
    }

    function updateLearningCurve(data) {
      const ctx = document.getElementById('learningChart').getContext('2d');
      const labels = data.map((_, i) => i + 1);

      if (learningChart) {
        learningChart.data.labels = labels;
        learningChart.data.datasets[0].data = data;
        learningChart.update();
        return;
      }

      const fg = getComputedStyle(document.body).color;
      learningChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Quality Score',
            data: data,
            borderColor: '#42a5f5',
            backgroundColor: 'rgba(66, 165, 245, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 0, max: 1,
              ticks: { color: fg, stepSize: 0.2 },
              grid: { color: 'rgba(128,128,128,0.2)' },
            },
            x: {
              ticks: { color: fg, maxTicksLimit: 8 },
              grid: { display: false },
            },
          },
          plugins: {
            legend: { display: false },
          },
        },
      });
    }

    function updateRadar(dimensions) {
      const ctx = document.getElementById('radarChart').getContext('2d');
      const labels = Object.keys(dimensions).map(k =>
        k.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase())
      );
      const values = Object.entries(dimensions).map(([k, v]) =>
        k === 'ambiguity' ? 1 - v : v
      );

      if (radarChart) {
        radarChart.data.labels = labels;
        radarChart.data.datasets[0].data = values;
        radarChart.update();
        return;
      }

      const fg = getComputedStyle(document.body).color;
      radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
          labels,
          datasets: [{
            label: 'Score',
            data: values,
            backgroundColor: 'rgba(66, 165, 245, 0.2)',
            borderColor: '#42a5f5',
            pointBackgroundColor: '#42a5f5',
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          scales: {
            r: {
              min: 0, max: 1,
              ticks: { display: false },
              grid: { color: 'rgba(128,128,128,0.2)' },
              pointLabels: { color: fg, font: { size: 10 } },
            },
          },
          plugins: {
            legend: { display: false },
          },
        },
      });
    }

    function updateSuggestions(suggestions, issues) {
      const sugSection = document.getElementById('suggestions-section');
      const sugList = document.getElementById('suggestions-list');
      const issSection = document.getElementById('issues-section');
      const issList = document.getElementById('issues-list');

      if (suggestions && suggestions.length > 0) {
        sugSection.style.display = '';
        sugList.innerHTML = suggestions.map(s => '<li>' + s + '</li>').join('');
      } else {
        sugSection.style.display = 'none';
      }

      if (issues && issues.length > 0) {
        issSection.style.display = '';
        issList.innerHTML = issues.map(s => '<li>' + s + '</li>').join('');
      } else {
        issSection.style.display = 'none';
      }
    }

    // Handle messages from extension
    window.addEventListener('message', event => {
      const msg = event.data;

      if (msg.type === 'stats') {
        const stats = msg.data;
        updateStats(stats);

        if (stats.avg_quality > 0) {
          updateGauge(stats.avg_quality, stats.trend || '');
        }
        if (stats.learning_curve && stats.learning_curve.length > 0) {
          updateLearningCurve(stats.learning_curve);
        }
        if (stats.dimensions) {
          updateRadar(stats.dimensions);
        }
        if (stats.weak_areas && stats.weak_areas.length > 0) {
          updateSuggestions(
            stats.weak_areas.map(a => 'Improve: ' + a.replace(/_/g, ' ')),
            []
          );
        }
      }

      if (msg.type === 'status') {
        const status = msg.data;
        if (status.data) {
          const d = status.data;
          if (d.quality_score !== undefined && d.grade) {
            updateGauge(d.quality_score, d.grade);
          }
          if (d.suggestions) {
            updateSuggestions(d.suggestions, d.issues || []);
          }
        }
      }
    });
  </script>
</body>
</html>`;
  }
}
