"""Read batch counters from a running designer benchmark."""
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


def progress(path):
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as db:
            complete, active = db.execute(
                "SELECT count(report), min(CASE WHEN report IS NULL THEN n END) FROM batches"
            ).fetchone()
            result = {"complete": complete}
            if active is not None:
                done, total = db.execute(
                    "SELECT count(metrics), count(*) FROM trials WHERE batch=?", (active,)
                ).fetchone()
                result.update(active=active, done=done, total=total)
            return result
    except sqlite3.OperationalError:
        return None


def main(root):
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    result = {market: {arm: progress(root/market/arm/"design.sqlite")
                       for arm in ("shared", *protocol["arms"])}
              for market in protocol["market_order"]}
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
