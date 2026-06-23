# DUGOUT BRAIN — Self-Improving MLB Prediction Engine

A machine-learning model that predicts every MLB game, grades itself against
the final scores each night, retrains on the growing season dataset, and
publishes everything to an animated website with a full reasoning tab.

## The 13 variables (exactly as specced)

| # | Variable | Where it comes from |
|---|----------|---------------------|
| 1 | Starting Pitcher FIP | Computed from MLB Stats API counting stats (HR, BB, HBP, K, IP) |
| 2 | Starting Pitcher WHIP | MLB Stats API season line |
| 3 | Starting Pitcher K% | K / batters faced |
| 4 | ERA/FIP last 5 starts | Pitcher game log, 50/50 ERA-FIP blend |
| 5–8 | Bullpen quality, 3-day workload, closer available, setup available | Active roster + last-3-days boxscores |
| 9 | Team OPS last 14 days | `byDateRange` team hitting stats |
| 10 | Team OPS vs starter handedness | `statSplits` (vs LHP / vs RHP) matched to tonight's opposing starter |
| 11 | Runs/game last 14 days | Same 14-day window |
| 12 | Home field | Schedule |
| 13 | Ballpark factor | Baseball Savant 3-yr park factors (table in `config.py`) |

The improvement you suggested is built in: variables 5–8 are folded into a single
**Bullpen Availability Score (0–100)** (weights in `config.py`), so the model
actually trains on **10 inputs**, with each pitching/offense input expressed as a
home-minus-away differential.

## How it gets smarter every day

`run_daily.py` does three things every morning:

1. **Grade** — pulls yesterday's final scores, marks every prediction hit/miss,
   and appends each game (features + outcome) to `data/training_data.csv`.
2. **Retrain** — refits the model on *all* graded games so far.
   - < 40 games: predictions run on a calibrated heuristic + home-field prior
   - 40–150 games: calibrated logistic regression
   - 150+ games: **XGBoost** (400 trees, depth 3, heavy regularization for a
     ~1,000-row dataset) wrapped in isotonic calibration, with normalized
     feature importances published to the site. Falls back to sklearn
     gradient boosting automatically if xgboost isn't installed.
   - The published probability blends ML output with the prior, and the ML
     weight ramps to 100% as the training set grows — so April predictions
     stay calm and September predictions are fully learned.
   - Grading sweeps **every** pending day, so if your machine is off for a
     weekend, the next run catches up on all missed results automatically.
3. **Predict** — builds features for today's slate and writes
   `site/data/predictions.json`, `history.json`, and `metrics.json`,
   which the website reads.

## Setup

```bash
pip install -r requirements.txt

# One-time: bootstrap the training set from games already played this season
# (makes a lot of API calls — let it run; rough guide is 1–2 min per day of games)
python backfill.py 2026-03-26 2026-06-11

# Then, every day:
python run_daily.py
```

Open `site/index.html` in a browser (or serve it: `cd site && python -m http.server`,
then visit http://localhost:8000 — serving it is recommended so the JSON loads).
Until the pipeline has run once, the site shows clearly-labeled **DEMO DATA**.

### Make it update by itself (pick one)

**A. Windows Task Scheduler (easiest):** open Task Scheduler → Create Basic
Task → Daily at 9:00 AM → Start a program → browse to `run_daily.bat`.
Done — it grades, retrains, and refreshes the site every morning. You can also
just double-click `run_daily.bat` any time.

**B. Mac/Linux cron:**

```
0 9 * * * cd /path/to/mlb-predictor && /usr/bin/python3 run_daily.py >> daily.log 2>&1
```

**C. Fully serverless (computer can be off):** push this folder to a GitHub
repo and enable GitHub Pages. The included `.github/workflows/daily.yml`
runs the whole pipeline on GitHub's servers at 9 AM ET every day and commits
the fresh JSON — your hosted site updates itself forever, for free. There's
also a manual "Run workflow" button for on-demand refreshes.

## Data sources

- **MLB Stats API** (`statsapi.mlb.com`) — official, free, no key required.
  This is the source of truth behind MLB.com (and what ESPN/StatMuse ultimately
  reflect): schedules, probable pitchers, results, boxscores, game logs, splits.
- **Baseball Savant park factors** — baked into `config.py`; refresh once a year.

## Knowing it's up to date

- The header stamp shows **MODEL LAST RUN <date>** — if it isn't today, the feed is stale.
- The Reasoning tab cites the actual numbers used (FIP, WHIP, K%, last-5 form,
  bullpen scores, OPS splits, park factor) so you can sanity-check any pick
  against MLB.com or StatMuse.
- The Performance tab tracks season record, accuracy, last-50 accuracy
  (is it improving?), log loss, and a rolling-accuracy sparkline.

## Honest expectations

MLB games are close to coin flips — the best public models hit roughly 58–61%.
If this model settles around 55–58% by late season, it's working. Anything
claiming 70%+ on baseball is broken or lying.

## Project layout

```
mlb-predictor/
├── config.py              # park factors, feature list, weights, paths
├── run_daily.py           # the one daily command
├── backfill.py            # one-time season bootstrap
├── requirements.txt
├── pipeline/
│   ├── fetch_data.py      # all MLB Stats API calls
│   ├── features.py        # 13 variables -> 10 model inputs (+ bullpen score)
│   └── model.py           # training, prediction, reasoning generator
├── data/                  # training_data.csv, model.joblib, pending picks
└── site/
    ├── index.html         # the animated website (3 tabs)
    └── data/              # predictions.json, history.json, metrics.json
```
