/**
 * Reads prompt history from ~/.loopllm/prompt_history.json
 * (written by the MCP server on every intercept call).
 *
 * Zero external dependencies — uses only Node.js built-ins.
 */

import * as fs from "fs";

export interface PromptStats {
  total_prompts: number;
  avg_quality: number;
  trend: string;
  grade_distribution: Record<string, number>;
  learning_curve: number[];
  dimensions: Record<string, number>;
  weak_areas: string[];
  strong_areas: string[];
}

export interface PromptRecord {
  id: number;
  timestamp: string;
  prompt_text: string;
  quality_score: number;
  grade: string;
  task_type: string;
  specificity: number;
  constraint_clarity: number;
  context_completeness: number;
  ambiguity: number;
  format_spec: number;
}

export class DataProvider {
  private historyPath: string;

  constructor(dbPath: string) {
    // Derive history JSON path from the DB path directory
    const dir = dbPath.replace(/\/[^/]+$/, "");
    this.historyPath = `${dir}/prompt_history.json`;
  }

  private readHistory(): PromptRecord[] {
    try {
      if (!fs.existsSync(this.historyPath)) {
        return [];
      }
      const raw = fs.readFileSync(this.historyPath, "utf-8");
      const data = JSON.parse(raw);
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  }

  async getPromptStats(windowSize = 50): Promise<PromptStats | null> {
    const allRecords = this.readHistory();
    if (allRecords.length === 0) {
      return null;
    }

    // Take the most recent `windowSize` records (newest first)
    const records = allRecords.slice(-windowSize).reverse();
    const total = records.length;

    // Average quality
    const scores = records.map((r) => Number(r.quality_score) || 0);
    const avg = scores.reduce((a, b) => a + b, 0) / total;

    // Trend (compare first half vs second half)
    let trend = "stable";
    if (total >= 4) {
      const mid = Math.floor(total / 2);
      const older = scores.slice(mid);
      const newer = scores.slice(0, mid);
      const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
      const newerAvg = newer.reduce((a, b) => a + b, 0) / newer.length;
      if (newerAvg - olderAvg > 0.05) {
        trend = "improving";
      } else if (olderAvg - newerAvg > 0.05) {
        trend = "declining";
      }
    }

    // Grade distribution
    const gradeDistribution: Record<string, number> = {};
    for (const rec of records) {
      const grade = String(rec.grade || "?");
      gradeDistribution[grade] = (gradeDistribution[grade] || 0) + 1;
    }

    // Dimension averages
    const dimNames = [
      "specificity",
      "constraint_clarity",
      "context_completeness",
      "ambiguity",
      "format_spec",
    ] as const;
    const dimensions: Record<string, number> = {};
    for (const dim of dimNames) {
      const vals = records.map((r) => Number(r[dim]) || 0);
      dimensions[dim] = vals.reduce((a, b) => a + b, 0) / total;
    }

    // Weak/strong areas
    const weakAreas: string[] = [];
    const strongAreas: string[] = [];
    for (const [dim, val] of Object.entries(dimensions)) {
      if (dim === "ambiguity") {
        if (val > 0.5) {
          weakAreas.push(dim);
        } else if (val < 0.2) {
          strongAreas.push(dim);
        }
      } else {
        if (val < 0.4) {
          weakAreas.push(dim);
        } else if (val >= 0.7) {
          strongAreas.push(dim);
        }
      }
    }

    // Learning curve (chronological order for chart)
    const learningCurve = [...scores].reverse();

    return {
      total_prompts: total,
      avg_quality: Math.round(avg * 1000) / 1000,
      trend,
      grade_distribution: gradeDistribution,
      learning_curve: learningCurve,
      dimensions,
      weak_areas: weakAreas,
      strong_areas: strongAreas,
    };
  }

  async getRecentPrompts(limit = 10): Promise<PromptRecord[]> {
    const records = this.readHistory();
    return records.slice(-limit).reverse();
  }

  close(): void {
    // No resources to clean up for JSON file reading
  }
}
