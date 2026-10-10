"""SQLite attempt history for progress tracking."""
import os, sqlite3, json, datetime
from paths import DB_PATH, ensure_dirs

ensure_dirs()
PATH = DB_PATH
def _c():
    c = sqlite3.connect(PATH); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, rid TEXT, singer TEXT, song TEXT, mode TEXT,
                 section TEXT, created TEXT, overall REAL, scores TEXT)""")
    return c
def add(rid, singer, song, mode, section, scores, created=None):
    with _c() as c:
        c.execute("INSERT INTO attempts(rid,singer,song,mode,section,created,overall,scores) VALUES(?,?,?,?,?,?,?,?)",
                  (rid, singer or "Guest", song, mode, json.dumps(section) if section else None,
                   created or datetime.datetime.now().isoformat(timespec="seconds"), scores["overall"], json.dumps(scores)))
def singers():
    with _c() as c: return [r[0] for r in c.execute("SELECT DISTINCT singer FROM attempts ORDER BY singer")]
def songs(singer=None):
    with _c() as c:
        q = "SELECT DISTINCT song FROM attempts" + (" WHERE singer=?" if singer else "") + " ORDER BY song"
        return [r[0] for r in c.execute(q, (singer,) if singer else ())]
def attempts(singer=None, song=None):
    q, a = "SELECT * FROM attempts WHERE 1=1", []
    if singer: q += " AND singer=?"; a.append(singer)
    if song: q += " AND song=?"; a.append(song)
    with _c() as c:
        rows = [dict(r) for r in c.execute(q + " ORDER BY created", a)]
    for r in rows: r["scores"] = json.loads(r["scores"]); r["section"] = json.loads(r["section"]) if r["section"] else None
    return rows
