"""Transforme les JSON bruts en un tableau (une ligne = une partie à une minute).

    python -m src.build_dataset --raw-dir data/raw/sample --out data/dataset.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import DEFAULT_MINUTES, build_rows


def build_dataset(raw_dir: Path, minutes=DEFAULT_MINUTES) -> pd.DataFrame:
    rows: list[dict] = []
    for match_path in sorted((raw_dir / "matches").glob("*.json")):
        tl_path = raw_dir / "timelines" / match_path.name
        if not tl_path.exists():
            continue
        match = json.loads(match_path.read_text(encoding="utf-8"))
        timeline = json.loads(tl_path.read_text(encoding="utf-8"))
        rows.extend(build_rows(match, timeline, minutes))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/sample")
    p.add_argument("--out", default="data/dataset.csv")
    p.add_argument("--all-minutes", action="store_true",
                   help="une ligne par minute (2 à 40) au lieu de 5/10/15/20/25")
    args = p.parse_args()

    minutes = range(2, 41) if args.all_minutes else DEFAULT_MINUTES
    df = build_dataset(Path(args.raw_dir), minutes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n = df["match_id"].nunique() if not df.empty else 0
    print(f"{len(df)} lignes ({n} parties) -> {out}")


if __name__ == "__main__":
    main()
