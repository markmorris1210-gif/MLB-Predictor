"""
backfill.py — bootstrap the training set from games already played this season,
so the model isn't starting from zero in June.

    python backfill.py 2026-03-26 2026-06-11

Features for each historical game are built using only data available
*before* that game (the 14-day offense windows and last-5 starts are
date-bounded), so there's no peeking at the future. This makes many API
calls — expect roughly 1-2 minutes per slate of games. Run it once.
"""

import sys
from datetime import date, timedelta

import config
from pipeline import fetch_data as fd
from pipeline import features as feat
from pipeline import model as mdl


def backfill(start: date, end: date):
    rows = []
    d = start
    while d <= end:
        print(f"== {d} ==")
        try:
            games = fd.get_schedule(d)
        except Exception as e:
            print(f"  schedule failed: {e}")
            d += timedelta(days=1)
            continue
        for g in games:
            if g["status"] not in ("Final", "Game Over", "Completed Early"):
                continue
            if g["home_score"] is None or not g["home_pitcher_id"] or not g["away_pitcher_id"]:
                continue
            try:
                f, _ = feat.build_game_features(g, d)
            except Exception as e:
                print(f"  skipped {g['away_abbr']}@{g['home_abbr']}: {e}")
                continue
            rows.append({"game_pk": g["game_pk"], "date": d.isoformat(), **f,
                         "home_won": 1 if g["home_score"] > g["away_score"] else 0})
            print(f"  + {g['away_abbr']}@{g['home_abbr']}")
        if rows:
            mdl.append_training_rows(rows)
            rows = []
        d += timedelta(days=1)

    df = mdl.load_training_data()
    _, metrics = mdl.train(df)
    print(f"Backfill complete: {metrics}")


if __name__ == "__main__":
    s = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(config.SEASON, 3, 26)
    e = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today() - timedelta(days=1)
    backfill(s, e)
