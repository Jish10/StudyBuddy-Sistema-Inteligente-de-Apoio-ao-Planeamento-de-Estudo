from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional

SCHEMA = '''
CREATE TABLE IF NOT EXISTS subjects(
  name TEXT PRIMARY KEY,
  last_score REAL NOT NULL,
  difficulty INTEGER NOT NULL,
  priority INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS progress(
  ts INTEGER NOT NULL,
  subject TEXT NOT NULL,
  hours REAL NOT NULL,
  predicted REAL NOT NULL,
  FOREIGN KEY(subject) REFERENCES subjects(name)
);
'''

def connect(db_path: str = "studybuddy.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)
    return conn

def upsert_subject(conn, name: str, last_score: float, difficulty: int, priority: int):
    conn.execute('''
      INSERT INTO subjects(name,last_score,difficulty,priority) VALUES(?,?,?,?)
      ON CONFLICT(name) DO UPDATE SET last_score=excluded.last_score,
                                     difficulty=excluded.difficulty,
                                     priority=excluded.priority
    ''', (name, float(last_score), int(difficulty), int(priority)))
    conn.commit()

def list_subjects(conn) -> List[Tuple[str,float,int,int]]:
    cur = conn.execute("SELECT name,last_score,difficulty,priority FROM subjects ORDER BY name")
    return list(cur.fetchall())

def add_progress(conn, ts: int, subject: str, hours: float, predicted: float):
    conn.execute("INSERT INTO progress(ts,subject,hours,predicted) VALUES(?,?,?,?)",
                 (int(ts), subject, float(hours), float(predicted)))
    conn.commit()

def get_progress(conn, subject: Optional[str]=None, limit: int = 200):
    if subject:
        cur = conn.execute("SELECT ts,subject,hours,predicted FROM progress WHERE subject=? ORDER BY ts DESC LIMIT ?",
                           (subject, int(limit)))
    else:
        cur = conn.execute("SELECT ts,subject,hours,predicted FROM progress ORDER BY ts DESC LIMIT ?",
                           (int(limit),))
    return list(cur.fetchall())
