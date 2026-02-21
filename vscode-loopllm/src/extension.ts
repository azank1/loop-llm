/**
 * Loop LLM — Prompt Quality Gauge
 *
 * VS Code extension that displays real-time prompt quality scoring from
 * the loop-llm MCP server. Shows a status bar gauge, sidebar dashboard
 * with learning curve chart, dimension radar, and improvement suggestions.
 */

import * as vscode from "vscode";
import { StatusBarGauge } from "./statusBar";
import { StatusWatcher } from "./statusWatcher";
import { DataProvider } from "./dataProvider";
import { DashboardViewProvider } from "./dashboardProvider";

let statusBar: StatusBarGauge;
let watcher: StatusWatcher;
let dataProvider: DataProvider;
let dashboardProvider: DashboardViewProvider;
let pollTimer: ReturnType<typeof setInterval> | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("loopllm");

  // Resolve paths
  const home = process.env.HOME || process.env.USERPROFILE || "~";
  const dbPath =
    config.get<string>("dbPath") || `${home}/.loopllm/store.db`;
  const statusFilePath =
    config.get<string>("statusFilePath") || `${home}/.loopllm/status.json`;
  const pollInterval = config.get<number>("pollIntervalMs") || 3000;

  // Initialize components
  statusBar = new StatusBarGauge();
  dataProvider = new DataProvider(dbPath);
  watcher = new StatusWatcher(statusFilePath);
  dashboardProvider = new DashboardViewProvider(
    context.extensionUri,
    dataProvider
  );

  // Register sidebar webview
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      "loopllm.dashboard",
      dashboardProvider
    )
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("loopllm.refreshStats", async () => {
      await refreshData();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("loopllm.showDashboard", () => {
      vscode.commands.executeCommand("loopllm.dashboard.focus");
    })
  );

  // Watch status.json for near-real-time updates
  watcher.onStatusChange((status) => {
    if (status.data) {
      const score = Number(status.data.quality_score ?? status.data.avg_quality ?? 0);
      const grade = String(status.data.grade ?? "");
      const gauge = String(status.data.gauge ?? "");
      statusBar.update(score, grade, gauge, status.tool);
      dashboardProvider.pushUpdate(status);
    }
  });
  watcher.start();

  // Poll database periodically for aggregate stats
  pollTimer = setInterval(async () => {
    await refreshData();
  }, pollInterval);

  // Initial load
  void refreshData();

  // Cleanup
  context.subscriptions.push({
    dispose: () => {
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      watcher.stop();
      statusBar.dispose();
      dataProvider.close();
    },
  });

  context.subscriptions.push(statusBar.getStatusBarItem());

  console.log("Loop LLM Prompt Gauge activated");
}

async function refreshData(): Promise<void> {
  try {
    const stats = await dataProvider.getPromptStats();
    if (stats) {
      statusBar.update(
        stats.avg_quality ?? 0,
        stats.trend ?? "no_data",
        "",
        "poll"
      );
      dashboardProvider.pushStats(stats);
    }
  } catch {
    // Database may not exist yet — that's OK
  }
}

export function deactivate(): void {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
}
