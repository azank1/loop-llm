/**
 * Reads prompt_history from the loop-llm SQLite database using sql.js (WASM).
 * Provides aggregate statistics and learning curve data for the dashboard.
 */

import * as fs from "fs";
import initSqlJs, { type Database, type SqlValue } from "sql.js";

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
  private dbPath: string;
  private db: Database | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(dbPath: string) {
    this.dbPath = dbPath;
  }

  private async ensureDb(): Promise<Database | null> {
    if (this.db) {
      return this.db;
    }

    if (this.initPromise) {
      await this.initPromise;
      return this.db;
    }

    this.initPromise = this.initDb();
    await this.initPromise;
    return this.db;
  }

  private async initDb(): Promise<void> {
    try {
      if (!fs.existsSync(this.dbPath)) {
        return;
      }

      const SQL = await initSqlJs();
      const buffer = fs.readFileSync(this.dbPath);
      this.db = new SQL.Database(buffer);
    } catch (err) {
      console.error("Loop LLM: Failed to open database:", err);
      this.db = null;
    }
  }

  /**
   * Reload the database from disk (since sql.js works on a snapshot).
   */
  async reload(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    this.initPromise = null;
    await this.ensureDb();
  }

  async getPromptStats(window = 50): Promise<PromptStats | null> {
    // Re-read from disk on each poll
    await this.reload();

    const db = await this.ensureDb();
    if (!db) {
      return null;
    }

    try {
      // Check if prompt_history table exists
      const tableCheck = db.exec(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_history'"
      );
      if (tableCheck.length === 0 || tableCheck[0].values.length === 0) {
        return null;
      }

      // Get recent records
      const result = db.exec(
        `SELECT quality_score, grade, specificity, constraint_clarity,
                context_completeness, ambiguity, format_spec
         FROM prompt_history
         ORDER BY timestamp DESC
         LIMIT ${window}`
      );

      if (result.length === 0 || result[0].values.length === 0) {
        return {
          total_prompts: 0,
          avg_quality: 0,
          trend: "no_data",
          grade_distribution: {},
          learning_curve: [],
          dimensions: {},
          weak_areas: [],
          strong_areas: [],
        };
      }

      const rows = result[0].values;
      const total = rows.length;

      // Average quality
      const scores = rows.map((r: SqlValue[]) => Number(r[0]) || 0);
      const avg = scores.reduce((a: number, b: number) => a + b, 0) / total;

      // Trend (compare first half vs second half)
      let trend = "stable";
      if (total >= 4) {
        const mid = Math.floor(total / 2);
        const older = scores.slice(mid);
        const newer = scores.slice(0, mid);
        const olderAvg = older.reduce((a: number, b: number) => a + b, 0) / older.length;
        const newerAvg = newer.reduce((a: number, b: number) => a + b, 0) / newer.length;
        if (newerAvg - olderAvg > 0.05) {
          trend = "improving";
        } else if (olderAvg - newerAvg > 0.05) {
          trend = "declining";
        }
      }

      // Grade distribution
      const gradeDistribution: Record<string, number> = {};
      for (const row of rows as SqlValue[][]) {
        const grade = String(row[1] || "?");
        gradeDistribution[grade] = (gradeDistribution[grade] || 0) + 1;
      }

      // Dimension averages
      const dimNames = [
        "specificity",
        "constraint_clarity",
        "context_completeness",
        "ambiguity",
        "format_spec",
      ];
      const dimensions: Record<string, number> = {};
      for (let i = 0; i < dimNames.length; i++) {
        const vals = rows.map((r: SqlValue[]) => Number(r[i + 2]) || 0);
        dimensions[dimNames[i]] = vals.reduce((a: number, b: number) => a + b, 0) / total;
      }

      // Weak/strong areas
      const weakAreas: string[] = [];
      const strongAreas: string[] = [];
      for (const [dim, val] of Object.entries(dimensions)) {
        if (dim === "ambiguity") {
          // Lower ambiguity is better
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
    } catch (err) {
      console.error("Loop LLM: Query error:", err);
      return null;
    }
  }

  async getRecentPrompts(limit = 10): Promise<PromptRecord[]> {
    const db = await this.ensureDb();
    if (!db) {
      return [];
    }

    try {
      const result = db.exec(
        `SELECT id, timestamp, prompt_text, quality_score, grade, task_type,
                specificity, constraint_clarity, context_completeness,
                ambiguity, format_spec
         FROM prompt_history
         ORDER BY timestamp DESC
         LIMIT ${limit}`
      );

      if (result.length === 0) {
        return [];
      }

      return result[0].values.map((row: SqlValue[]) => ({
        id: Number(row[0]),
        timestamp: String(row[1]),
        prompt_text: String(row[2]),
        quality_score: Number(row[3]),
        grade: String(row[4]),
        task_type: String(row[5]),
        specificity: Number(row[6]),
        constraint_clarity: Number(row[7]),
        context_completeness: Number(row[8]),
        ambiguity: Number(row[9]),
        format_spec: Number(row[10]),
      }));
    } catch {
      return [];
    }
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}
