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
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private watchAvailable = false;

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
        if (this.debounceTimer) {
          clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
          this.readAndEmit();
        }, 100);
      });
      this.watchAvailable = true;
    } catch {
      // fs.watch not available — use polling fallback (common in Linux containers)
    }

    // Always run a polling fallback at 1 s so the gauge updates reliably
    // even when inotify / fs.watch is not available in the dev container.
    this.pollTimer = setInterval(() => {
      this.readAndEmit();
    }, 1000);
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
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.emitter.removeAllListeners();
  }
}
