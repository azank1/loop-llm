/**
 * Watches ~/.loopllm/status.json for changes written by the MCP server.
 * Provides near-real-time updates without database polling.
 */

import * as fs from "fs";
import * as path from "path";
import { EventEmitter } from "events";

export interface StatusData {
  timestamp: number;
  tool: string;
  data: Record<string, unknown>;
}

export class StatusWatcher {
  private filePath: string;
  private watcher: fs.FSWatcher | null = null;
  private emitter = new EventEmitter();
  private lastTimestamp = 0;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  onStatusChange(callback: (status: StatusData) => void): void {
    this.emitter.on("change", callback);
  }

  start(): void {
    const dir = path.dirname(this.filePath);

    // Ensure directory exists
    try {
      fs.mkdirSync(dir, { recursive: true });
    } catch {
      // ignore
    }

    // Create empty file if it doesn't exist
    if (!fs.existsSync(this.filePath)) {
      try {
        fs.writeFileSync(this.filePath, "{}");
      } catch {
        // ignore
      }
    }

    try {
      this.watcher = fs.watch(this.filePath, () => {
        // Debounce rapid writes
        if (this.debounceTimer) {
          clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
          this.readAndEmit();
        }, 100);
      });
    } catch {
      // File watching not available — fall back to polling
      console.warn("Loop LLM: fs.watch not available, status updates disabled");
    }
  }

  private readAndEmit(): void {
    try {
      const content = fs.readFileSync(this.filePath, "utf-8");
      if (!content.trim()) {
        return;
      }
      const status: StatusData = JSON.parse(content);

      // Only emit if this is a new update
      if (status.timestamp && status.timestamp > this.lastTimestamp) {
        this.lastTimestamp = status.timestamp;
        this.emitter.emit("change", status);
      }
    } catch {
      // Ignore parse errors — file may be mid-write
    }
  }

  stop(): void {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.emitter.removeAllListeners();
  }
}
