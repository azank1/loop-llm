/**
 * Status bar gauge that shows prompt quality score with color coding.
 *
 * Colors:
 *   🟢 Green  — score >= 0.7 (good prompts)
 *   🟡 Yellow — 0.5 <= score < 0.7 (needs improvement)
 *   🔴 Red    — score < 0.5 (poor prompts)
 */

import * as vscode from "vscode";

export class StatusBarGauge {
  private item: vscode.StatusBarItem;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = "loopllm.showDashboard";
    this.item.tooltip = "Loop LLM — Prompt Quality (click for dashboard)";
    this.item.text = "$(pulse) LLM: --";
    this.item.show();
  }

  update(
    score: number,
    gradeOrTrend: string,
    gauge: string,
    tool: string
  ): void {
    const pct = Math.round(score * 100);

    // Color coding
    if (score >= 0.7) {
      this.item.backgroundColor = undefined; // default (no highlight)
      this.item.color = new vscode.ThemeColor(
        "statusBarItem.foreground"
      );
    } else if (score >= 0.5) {
      this.item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
      this.item.color = undefined;
    } else if (score > 0) {
      this.item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground"
      );
      this.item.color = undefined;
    }

    // Icon selection
    let icon: string;
    if (score >= 0.7) {
      icon = "$(pass-filled)";
    } else if (score >= 0.5) {
      icon = "$(warning)";
    } else if (score > 0) {
      icon = "$(error)";
    } else {
      icon = "$(pulse)";
    }

    // Trend arrow for poll updates
    let trendArrow = "";
    if (tool === "poll") {
      if (gradeOrTrend === "improving") {
        trendArrow = " ↑";
      } else if (gradeOrTrend === "declining") {
        trendArrow = " ↓";
      } else if (gradeOrTrend === "stable") {
        trendArrow = " →";
      }
      this.item.text = `${icon} LLM: ${pct}%${trendArrow}`;
    } else {
      // Real-time update from intercept
      const grade = gradeOrTrend || "";
      this.item.text = `${icon} LLM: ${pct}% [${grade}]`;
    }

    this.item.tooltip = gauge
      ? `Loop LLM — ${gauge}\nClick for dashboard`
      : `Loop LLM — Prompt Quality: ${pct}%\nClick for dashboard`;
  }

  getStatusBarItem(): vscode.StatusBarItem {
    return this.item;
  }

  dispose(): void {
    this.item.dispose();
  }
}
