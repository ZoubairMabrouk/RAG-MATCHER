#!/usr/bin/env python3
"""
export_schema_to_json.py — snapshot a live database's introspected schema
to a JSON file, so downstream scripts can run with --schema-file instead
of needing a live DB connection every time.

Run this ONCE while the DB container is actually reachable:

    python scripts/export_schema_to_json.py --db-url "$DATABASE_URL" \
        --out data/schema_snapshot.json

Then in any other script, pass --schema-file data/schema_snapshot.json
instead of --db-url to skip the live connection entirely.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.di_container import DIContainer


def _default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def main():
    p = argparse.ArgumentParser(description="Export live DB schema to JSON")
    p.add_argument("--db-url", default=None, help="Database URL (overrides $DATABASE_URL)")
    p.add_argument("--dialect", default="postgresql")
    p.add_argument("--out", default="data/schema_snapshot.json")
    args = p.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: no --db-url and $DATABASE_URL not set", file=sys.stderr)
        return 2

    container = DIContainer()
    container.configure(db_url, args.dialect)
    inspector = container.get_inspector()
    schema = inspector.introspect_schema()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(schema), f, indent=2, default=_default, ensure_ascii=False)

    print(f"Exported {len(schema.tables)} table(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())