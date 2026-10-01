"""
WP-P2: SQLite-backed long-term-memory persistence.

Design constraints straight from the architecture doc's gate G2:
  - "SIGKILL mid-write leaves an openable DB"      -> WAL mode + one
    commit per node, never a multi-row transaction a crash could split.
  - "victim selection matches brute-force reference exactly" -> this
    module only stores/loads; selection itself lives in memory.py as a
    pure function so it can be tested against a brute-force reference
    without touching sqlite at all.
  - Frozen contract note on Signature.rgb/depth ("kept only while in
    WM"): this store NEVER persists those two fields. A node that goes
    to LTM permanently loses its raw imagery; only what retrieval and
    re-verification need (kp, kp3d, desc, valid, word_ids, global_desc)
    survives the round trip. This is a deliberate, permanent choice,
    not a shortcut -- see WP_P2_Findings.md.
"""
from __future__ import annotations
import pickle
import sqlite3
from dataclasses import replace
from typing import Optional

from pyslam.core.types import Node, Signature

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    weight INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    t REAL NOT NULL,
    blob BLOB NOT NULL
);
"""


def _strip_for_ltm(node: Node) -> Node:
    """Drop the fields the frozen contract says LTM never keeps."""
    light_sig = replace(node.sig, rgb=None, depth=None)
    return replace(node, sig=light_sig)


class LtmStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None)  # autocommit;
            # each statement is its own transaction -- see module docstring
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(_SCHEMA)

    def put(self, node: Node) -> None:
        light = _strip_for_ltm(node)
        blob = pickle.dumps(light, protocol=pickle.HIGHEST_PROTOCOL)
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (id, weight, session_id, t, blob) VALUES (?,?,?,?,?)",
            (light.id, light.weight, light.session_id, float(light.sig.t), blob),
        )

    def get(self, node_id: int) -> Optional[Node]:
        row = self.conn.execute("SELECT blob FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])

    def delete(self, node_id: int) -> None:
        self.conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    def contains(self, node_id: int) -> bool:
        row = self.conn.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone()
        return row is not None

    def all_ids(self) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT id FROM nodes")]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
