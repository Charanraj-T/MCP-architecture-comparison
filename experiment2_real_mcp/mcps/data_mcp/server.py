#!/usr/bin/env python3
import sqlite3
import os
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP("Data MCP")
_conn = None
_db_path = None

def _get_conn():
    global _conn, _db_path
    if _conn is None:
        _db_path = os.environ.get("DATA_MCP_DB", "")
        if not _db_path:
            _db_path = str(Path(tempfile.mkdtemp()) / "experiment2.db")
            _conn = sqlite3.connect(_db_path)
            _conn.execute("CREATE TABLE IF NOT EXISTS incidents (id INTEGER PRIMARY KEY, title TEXT, description TEXT, severity TEXT, status TEXT, created_at TEXT)")
            _conn.execute("CREATE TABLE IF NOT EXISTS deployments (id INTEGER PRIMARY KEY, service TEXT, version TEXT, status TEXT, timestamp TEXT)")
            _conn.execute("CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, service TEXT, metric TEXT, value REAL, unit TEXT, timestamp TEXT)")
            _conn.execute("INSERT OR IGNORE INTO incidents VALUES (1, 'payment-api timeout', 'Payment API timeout after 30s', 'critical', 'investigating', '2025-05-21T10:23:00Z')")
            _conn.execute("INSERT OR IGNORE INTO incidents VALUES (2, 'DB pool exhaustion', 'Database connection pool at 95%', 'high', 'monitoring', '2025-05-21T09:15:00Z')")
            _conn.execute("INSERT OR IGNORE INTO deployments VALUES (1, 'payment-api', 'v2.4.1', 'failed', '2025-05-21T09:00:00Z')")
            _conn.execute("INSERT OR IGNORE INTO deployments VALUES (2, 'payment-api', 'v2.4.0', 'success', '2025-05-20T14:00:00Z')")
            _conn.execute("INSERT OR IGNORE INTO metrics VALUES (1, 'payment-api', 'cpu', 92.0, 'percent', '2025-05-21T10:20:00Z')")
            _conn.execute("INSERT OR IGNORE INTO metrics VALUES (2, 'payment-api', 'cpu', 45.0, 'percent', '2025-05-21T10:10:00Z')")
            _conn.execute("INSERT OR IGNORE INTO metrics VALUES (3, 'payment-api', 'memory', 1024, 'MB', '2025-05-21T10:20:00Z')")
            _conn.execute("INSERT OR IGNORE INTO metrics VALUES (4, 'payment-api', 'memory', 512, 'MB', '2025-05-21T10:10:00Z')")
            _conn.commit()
        else:
            _conn = sqlite3.connect(_db_path)
    return _conn

@server.tool()
def query(sql: str) -> str:
    try:
        conn = _get_conn()
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if not rows:
            return "No results"
        lines = [" | ".join(cols)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"

@server.tool()
def list_tables() -> str:
    try:
        conn = _get_conn()
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        return "\n".join(tables) if tables else "No tables"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
def describe_table(table: str) -> str:
    try:
        conn = _get_conn()
        cur = conn.execute(f"PRAGMA table_info('{table}')")
        rows = cur.fetchall()
        if not rows:
            return f"Table not found: {table}"
        cols = ["cid", "name", "type", "notnull", "default", "pk"]
        lines = [" | ".join(cols)]
        lines.append("-" * len(lines[0]))
        for r in rows:
            lines.append(" | ".join(str(v) for v in r))
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

@server.tool()
def seed_sample_data() -> str:
    conn = _get_conn()
    conn.execute("DROP TABLE IF EXISTS incidents")
    conn.execute("DROP TABLE IF EXISTS deployments")
    conn.execute("DROP TABLE IF EXISTS metrics")
    conn.execute("CREATE TABLE incidents (id INTEGER PRIMARY KEY, title TEXT, description TEXT, severity TEXT, status TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE deployments (id INTEGER PRIMARY KEY, service TEXT, version TEXT, status TEXT, timestamp TEXT)")
    conn.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, service TEXT, metric TEXT, value REAL, unit TEXT, timestamp TEXT)")
    conn.executemany("INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?)", [
        (1, "payment-api timeout", "Payment API timeout after 30s", "critical", "investigating", "2025-05-21T10:23:00Z"),
        (2, "DB pool exhaustion", "Database connection pool at 95%", "high", "monitoring", "2025-05-21T09:15:00Z"),
        (3, "deploy failure v2.4.1", "Rollback of payment-api v2.4.1", "high", "resolved", "2025-05-21T09:30:00Z"),
    ])
    conn.executemany("INSERT INTO deployments VALUES (?, ?, ?, ?, ?)", [
        (1, "payment-api", "v2.4.1", "failed", "2025-05-21T09:00:00Z"),
        (2, "payment-api", "v2.4.0", "success", "2025-05-20T14:00:00Z"),
        (3, "payment-api", "v2.3.2", "success", "2025-05-19T11:00:00Z"),
        (4, "payment-api", "v2.3.1", "rollback", "2025-05-18T09:00:00Z"),
    ])
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)", [
        (1, "payment-api", "cpu", 92.0, "percent", "2025-05-21T10:20:00Z"),
        (2, "payment-api", "cpu", 45.0, "percent", "2025-05-21T10:10:00Z"),
        (3, "payment-api", "cpu", 32.0, "percent", "2025-05-21T10:05:00Z"),
        (4, "payment-api", "cpu", 28.0, "percent", "2025-05-21T10:00:00Z"),
        (5, "payment-api", "memory", 1024.0, "MB", "2025-05-21T10:20:00Z"),
        (6, "payment-api", "memory", 768.0, "MB", "2025-05-21T10:10:00Z"),
        (7, "payment-api", "memory", 512.0, "MB", "2025-05-21T10:00:00Z"),
        (8, "payment-api", "latency_p95", 1200.0, "ms", "2025-05-21T10:20:00Z"),
        (9, "payment-api", "latency_p95", 450.0, "ms", "2025-05-21T10:00:00Z"),
    ])
    conn.commit()
    return "Sample data seeded (3 incidents, 4 deployments, 9 metrics)"

if __name__ == "__main__":
    server.run(transport="stdio")
