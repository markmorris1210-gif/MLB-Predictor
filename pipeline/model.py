"""
model.py — the learning core.

An XGBoost classifier (isotonic-calibrated) retrained from scratch every day
on all graded games so far. Daily retraining on the growing dataset is what
makes the model "smarter as the season goes on": every final score becomes
a new labeled row the next morning. If xgboost isn't installed, it falls
back to scikit-learn's GradientBoostingClassifier automatically.

Cold start: before ~100 graded games exist, predictions blend the ML output
with a sensible prior (home team wins ~53% of MLB games) so early-season
probabilities stay calm instead of overreacting to a tiny sample.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

HOME_PRIOR = 0.535          # historical MLB home win rate
MIN_GAMES_FULL_TRUST = 300  # ML weight ramps from 0 -> 1 as rows accumulate
MIN_GAMES_TO_TRAIN = 40


def load_training_data():
    if os.path.exists(config.TRAINING_DATA):
        return pd.read_csv(config.TRAINING_DATA)
    return pd.DataFrame(columns=["game_pk", "date"] + config.FEATURES + ["home_won"])


def append_training_rows(rows):
    df = load_training_data()
    new = pd.DataFrame(rows)
    df = pd.concat([df, new], ignore_index=True)
    df = df.drop_duplicates(subset=["game_pk"], keep="last")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.TRAINING_DATA, index=False)
    return df


def train(df=None):
    """Retrain on every graded game and persist the model + metrics."""
    df = df if df is not None else load_training_data()
    df = df.dropna(subset=["home_won"])
    n = len(df)
    if n < MIN_GAMES_TO_TRAIN:
        return None, {"n_games": n, "status": "collecting",
                      "note": f"Need {MIN_GAMES_TO_TRAIN} graded games to train; have {n}."}

    X = df[config.FEATURES].astype(float).values
    y = df["home_won"].astype(int).values

    if n < 150:
        base = LogisticRegression(max_iter=2000, C=0.5)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        kind = "logistic regression (small sample)"
    elif HAS_XGB:
        base = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=8, reg_lambda=2.0, gamma=0.2,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, verbosity=0,
        )
        model = CalibratedClassifierCV(base, method="isotonic", cv=4)
        kind = "XGBoost + isotonic calibration"
    else:
        base = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.85)
        model = CalibratedClassifierCV(base, method="isotonic", cv=4)
        kind = "gradient boosting + isotonic calibration (xgboost not installed)"

    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    metrics = {
        "n_games": int(n),
        "model_type": kind,
        "train_accuracy": round(float(accuracy_score(y, p > 0.5)), 4),
        "train_log_loss": round(float(log_loss(y, p)), 4),
        "train_brier": round(float(brier_score_loss(y, p)), 4),
        "status": "trained",
    }

    importances = feature_importances(model)
    if importances:
        metrics["feature_importance"] = importances

    joblib.dump({"model": model, "features": config.FEATURES}, config.MODEL_FILE)
    return model, metrics


def feature_importances(calibrated_model):
    """
    Average gain-based importances across the calibrated ensemble's
    fitted boosters, normalized to percentages. Returns [] for the
    logistic tier (coefficients aren't comparable the same way).
    """
    mats = []
    for cc in getattr(calibrated_model, "calibrated_classifiers_", []):
        est = cc.estimator
        imp = getattr(est, "feature_importances_", None)
        if imp is not None:
            mats.append(np.asarray(imp, dtype=float))
    if not mats:
        return []
    avg = np.mean(mats, axis=0)
    total = avg.sum() or 1.0
    pairs = sorted(zip(config.FEATURES, avg / total), key=lambda x: -x[1])
    return [{"feature": f, "weight": round(float(w), 4)} for f, w in pairs]


def load_model():
    if os.path.exists(config.MODEL_FILE):
        return joblib.load(config.MODEL_FILE)["model"]
    return None


def predict_home_win_prob(features: dict, n_trained_on: int):
    """Blend ML probability with the home-field prior while the sample is small."""
    model = load_model()
    x = np.array([[features[f] for f in config.FEATURES]], dtype=float)

    if model is None:
        ml_prob, ml_weight = HOME_PRIOR, 0.0
    else:
        ml_prob = float(model.predict_proba(x)[0, 1])
        ml_weight = min(1.0, n_trained_on / MIN_GAMES_FULL_TRUST)

    # Lightweight heuristic prior from the feature diffs themselves, so even
    # day-1 predictions reflect the matchup rather than pure home-field.
    z = (-0.30 * features["sp_fip_diff"]
         - 0.55 * features["sp_whip_diff"]
         + 2.2 * features["sp_k_rate_diff"]
         - 0.18 * features["sp_last5_fip_diff"]
         + 0.006 * features["bullpen_score_diff"]
         + 1.4 * features["ops_14d_diff"]
         + 0.9 * features["ops_vs_hand_diff"]
         + 0.05 * features["rpg_14d_diff"]
         + 0.14)  # home field
    heuristic = 1 / (1 + np.exp(-z))

    prob = ml_weight * ml_prob + (1 - ml_weight) * float(heuristic)
    return round(max(0.02, min(0.98, prob)), 4), round(ml_weight, 2)


# ---------------------------------------------------------------- reasoning

def _edge_word(x, scale):
    a = abs(x) / scale
    if a < 0.35:
        return "a slight"
    if a < 1.0:
        return "a clear"
    return "a major"


def build_reasoning(game, features, detail, prob, ml_weight, n_games):
    """Plain-language explanation of why the model leans the way it does."""
    home, away = game["home_name"], game["away_name"]
    fav = home if prob >= 0.5 else away
    lines = []

    f = features["sp_fip_diff"]
    better = away if f > 0 else home
    sp = detail["away_sp"] if f > 0 else detail["home_sp"]
    lines.append({
        "factor": "Starting pitching",
        "lean": better,
        "text": (f"{sp['name']} holds {_edge_word(f, 0.6)} FIP edge "
                 f"({detail['home_sp']['fip']:.2f} vs {detail['away_sp']['fip']:.2f}), "
                 f"with WHIPs of {detail['home_sp']['whip']:.2f} / {detail['away_sp']['whip']:.2f} "
                 f"and K rates of {detail['home_sp']['k_rate']:.1%} / {detail['away_sp']['k_rate']:.1%}.")
    })

    l5 = features["sp_last5_fip_diff"]
    hot = away if l5 > 0 else home
    lines.append({
        "factor": "Recent form (last 5 starts)",
        "lean": hot,
        "text": (f"Over their last five starts the blended ERA/FIP marks are "
                 f"{detail['home_sp']['last5']:.2f} (home) vs {detail['away_sp']['last5']:.2f} (away) — "
                 f"{_edge_word(l5, 0.8)} edge in current form for the {hot}.")
    })

    b = features["bullpen_score_diff"]
    pen = home if b > 0 else away
    hp, ap = detail["home_bullpen"], detail["away_bullpen"]
    lines.append({
        "factor": "Bullpen availability score",
        "lean": pen,
        "text": (f"Bullpen Availability Scores: {hp['score']:.0f} vs {ap['score']:.0f} "
                 f"(quality FIP {hp['bullpen_fip']:.2f}/{ap['bullpen_fip']:.2f}, "
                 f"relief IP last 3 days {hp['relief_ip_last3']}/{ap['relief_ip_last3']}, "
                 f"closer available {bool(hp['closer_available'])}/{bool(ap['closer_available'])}).")
    })

    o = features["ops_vs_hand_diff"]
    bats = home if o > 0 else away
    lines.append({
        "factor": "Offense vs. starter handedness",
        "lean": bats,
        "text": (f"Against a {'lefty' if detail['away_sp']['hand']=='L' else 'righty'} starter, "
                 f"the {home} OPS split vs the {away} split shows "
                 f"{_edge_word(o, 0.05)} edge for the {bats}; "
                 f"14-game OPS runs {detail['home_offense']['ops_14d'] or 0:.3f} vs "
                 f"{detail['away_offense']['ops_14d'] or 0:.3f}.")
    })

    pf = features["park_factor"]
    park_txt = "hitter-friendly" if pf > 102 else "pitcher-friendly" if pf < 98 else "roughly neutral"
    lines.append({
        "factor": "Context",
        "lean": home,
        "text": (f"{home} get home field at {game['venue']} "
                 f"(park factor {pf}, {park_txt}). Home teams win about 53.5% of MLB games.")
    })

    summary = (f"The model gives the {fav} a {max(prob, 1-prob):.0%} win probability. "
               f"This forecast is {ml_weight:.0%} machine-learned (trained on {n_games} graded "
               f"games this season) and {1-ml_weight:.0%} matchup heuristics — the ML share "
               f"grows automatically as daily results are added.")
    return {"summary": summary, "factors": lines}
