"""
run_daily.py — the one command to run each morning (or via cron).

    python run_daily.py

What it does, in order:
  1. GRADE   — pulls yesterday's final scores, marks each prediction right or
               wrong, and appends the graded games (features + outcome) to
               data/training_data.csv.
  2. RETRAIN — refits the model on every graded game so far.
  3. PREDICT — builds features for today's slate and writes
               site/data/predictions.json (+ history.json, metrics.json).

The website is static: it just reads those three JSON files. Re-run this
script daily and the site updates itself.
"""

import os
import json
from datetime import date, timedelta

import config
from pipeline import fetch_data as fd
from pipeline import features as feat
from pipeline import model as mdl

PENDING_DIR = os.path.join(config.DATA_DIR, "pending")


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------ grading

def grade_day(day: date):
    """Match final scores against the stored predictions for `day`."""
    pending_path = os.path.join(PENDING_DIR, f"{day.isoformat()}.json")
    pending = _load_json(pending_path, {})
    if not pending:
        print(f"  no stored predictions for {day}")
        return

    finals = {str(g["game_pk"]): g for g in fd.get_schedule(day)
              if g["status"] in ("Final", "Game Over", "Completed Early")
              and g["home_score"] is not None}

    rows, graded = [], []
    for pk, entry in pending.items():
        g = finals.get(pk)
        if not g:
            continue
        home_won = 1 if g["home_score"] > g["away_score"] else 0
        rows.append({"game_pk": int(pk), "date": day.isoformat(),
                     **entry["features"], "home_won": home_won})
        predicted_home = entry["prob_home"] >= 0.5
        graded.append({
            "date": day.isoformat(),
            "matchup": f"{g['away_name']} @ {g['home_name']}",
            "prob_home": entry["prob_home"],
            "predicted": g["home_name"] if predicted_home else g["away_name"],
            "winner": g["home_name"] if home_won else g["away_name"],
            "score": f"{g['away_score']}–{g['home_score']}",
            "correct": bool(predicted_home == bool(home_won)),
        })

    if rows:
        mdl.append_training_rows(rows)
        history = _load_json(config.HISTORY_FILE, [])
        history.extend(graded)
        _save_json(config.HISTORY_FILE, history)
        hits = sum(g["correct"] for g in graded)
        print(f"  graded {len(graded)} games — {hits}/{len(graded)} correct")
    else:
        print("  finals not posted yet; will pick them up next run")


# --------------------------------------------------------------- predicting

def predict_day(day: date, n_games_trained: int):
    games = fd.get_schedule(day)
    games = [g for g in games if g["status"] not in ("Final", "Game Over")]
    predictions, pending = [], {}

    for g in games:
        print(f"  building features: {g['away_name']} @ {g['home_name']}")
        try:
            f, detail = feat.build_game_features(g, day)
        except Exception as e:
            print(f"    skipped ({e})")
            continue
        prob, ml_weight = mdl.predict_home_win_prob(f, n_games_trained)
        reasoning = mdl.build_reasoning(g, f, detail, prob, ml_weight, n_games_trained)
        pick = g["home_name"] if prob >= 0.5 else g["away_name"]
        confidence = max(prob, 1 - prob)

        predictions.append({
            "game_pk": g["game_pk"], "date": day.isoformat(),
            "game_time": g["game_time"], "venue": g["venue"],
            "home": g["home_name"], "home_abbr": g["home_abbr"],
            "away": g["away_name"], "away_abbr": g["away_abbr"],
            "home_pitcher": g["home_pitcher"], "away_pitcher": g["away_pitcher"],
            "prob_home": prob, "pick": pick,
            "confidence": round(confidence, 4),
            "ml_weight": ml_weight,
            "features": f, "detail_reasoning": reasoning,
        })
        pending[str(g["game_pk"])] = {"features": f, "prob_home": prob}

    _save_json(os.path.join(PENDING_DIR, f"{day.isoformat()}.json"), pending)
    _save_json(config.PREDICTIONS_FILE, {
        "generated": date.today().isoformat(),
        "slate_date": day.isoformat(),
        "games": sorted(predictions, key=lambda x: -x["confidence"]),
    })
    print(f"  wrote {len(predictions)} predictions -> {config.PREDICTIONS_FILE}")


# ------------------------------------------------------------------- main

def main():
    today = date.today()

    print("[1/3] Grading every pending day …")
    if os.path.isdir(PENDING_DIR):
        for fname in sorted(os.listdir(PENDING_DIR)):
            fpath = os.path.join(PENDING_DIR, fname)
            if not os.path.isfile(fpath) or not fname.endswith(".json"):
                continue
            try:
                d = date.fromisoformat(fname.replace(".json", ""))
            except ValueError:
                continue
            if d < today:
                print(f"  {d}:")
                grade_day(d)
                done = os.path.join(PENDING_DIR, "graded")
                os.makedirs(done, exist_ok=True)
                os.replace(fpath, os.path.join(done, fname))
    else:
        print("  nothing pending yet")

    print("[2/3] Retraining …")
    df = mdl.load_training_data()
    _, metrics = mdl.train(df)

    history = _load_json(config.HISTORY_FILE, [])
    if history:
        correct = sum(h["correct"] for h in history)
        metrics["season_accuracy"] = round(correct / len(history), 4)
        metrics["season_record"] = f"{correct}-{len(history) - correct}"
        last50 = history[-50:]
        metrics["last50_accuracy"] = round(sum(h["correct"] for h in last50) / len(last50), 4)
    metrics["last_updated"] = today.isoformat()
    _save_json(config.METRICS_FILE, metrics)
    print(f"  {metrics}")

    print(f"[3/3] Predicting {today} …")
    predict_day(today, metrics.get("n_games", 0))

    print("[extra] Yesterday's scores + league leaders …")
    try:
        scores = fd.get_final_scores(today - timedelta(days=1))
        _save_json(os.path.join(config.SITE_DATA_DIR, "scores.json"),
                   {"date": (today - timedelta(days=1)).isoformat(), "games": scores})
        print(f"  {len(scores)} final scores")
    except Exception as e:
        print(f"  scores skipped ({e})")
    try:
        leaders = fd.get_leaders()
        _save_json(os.path.join(config.SITE_DATA_DIR, "leaders.json"), leaders)
        print(f"  leaders: {len(leaders.get('teams', []))} teams")
    except Exception as e:
        print(f"  leaders skipped ({e})")

    print("Done. Open site/index.html to view.")


if __name__ == "__main__":
    main()
