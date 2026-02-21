/**
 * Type declarations for sql.js (WASM-based SQLite).
 */
declare module "sql.js" {
  interface SqlJsStatic {
    Database: new (data?: ArrayLike<number> | Buffer | null) => Database;
  }

  interface QueryExecResult {
    columns: string[];
    values: SqlValue[][];
  }

  type SqlValue = string | number | Uint8Array | null;

  interface Database {
    exec(sql: string, params?: SqlValue[]): QueryExecResult[];
    run(sql: string, params?: SqlValue[]): void;
    close(): void;
  }

  interface SqlJsInitOptions {
    locateFile?: (file: string) => string;
  }

  export default function initSqlJs(
    options?: SqlJsInitOptions
  ): Promise<SqlJsStatic>;

  export { Database, QueryExecResult, SqlValue, SqlJsStatic };
}
