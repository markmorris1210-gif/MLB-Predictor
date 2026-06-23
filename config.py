"""
Central configuration for the MLB prediction system.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SITE_DATA_DIR = os.path.join(BASE_DIR, "site", "data")

TRAINING_DATA = os.path.join(DATA_DIR, "training_data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.joblib")
PREDICTIONS_FILE = os.path.join(SITE_DATA_DIR, "predictions.json")
HISTORY_FILE = os.path.join(SITE_DATA_DIR, "history.json")
METRICS_FILE = os.path.join(SITE_DATA_DIR, "metrics.json")

MLB_API = "https://statsapi.mlb.com/api/v1"

# League-average FIP constant (recomputed loosely each year; 3.10-3.20 is typical)
FIP_CONSTANT = 3.15

# Statcast-style 3-year park factors (100 = neutral, >100 hitter-friendly).
# Keyed by home team abbreviation. Update once per season from
# baseballsavant.mlb.com/leaderboard/statcast-park-factors
PARK_FACTORS = {
    "COL": 112, "BOS": 107, "CIN": 106, "KC": 104, "ARI": 103,
    "TEX": 102, "MIN": 102, "PHI": 102, "ATL": 101, "LAA": 101,
    "TOR": 101, "CHC": 100, "WSH": 100, "BAL": 100, "HOU": 100,
    "PIT": 99,  "MIL": 99,  "NYY": 99,  "STL": 99,  "CWS": 99,
    "LAD": 98,  "DET": 98,  "AZ": 103,  "NYM": 97,  "CLE": 97,
    "TB": 96,   "MIA": 96,  "SF": 95,   "OAK": 96, "ATH": 96,
    "SD": 95,   "SEA": 92,
}

# Feature order used by the model (10-variable mode with the
# combined Bullpen Availability Score 0-100).
FEATURES = [
    # Starting pitcher (per team, expressed as home minus away differentials)
    "sp_fip_diff",          # 1. Starting Pitcher FIP
    "sp_whip_diff",         # 2. Starting Pitcher WHIP
    "sp_k_rate_diff",       # 3. Starting Pitcher K%
    "sp_last5_fip_diff",    # 4. ERA/FIP over last 5 starts
    # Bullpen
    "bullpen_score_diff",   # 5-8 combined: Bullpen Availability Score (0-100)
    # Offense
    "ops_14d_diff",         # 9. Team OPS last 14 games
    "ops_vs_hand_diff",     # 10. Team OPS vs starter handedness
    "rpg_14d_diff",         # 11. Runs per game last 14 games
    # Context
    "home_field",           # 12. Home field indicator (always 1: row = home team view)
    "park_factor",          # 13. Ballpark factor of the venue
]

# Weights inside the Bullpen Availability Score
BULLPEN_SCORE_WEIGHTS = {
    "quality": 0.45,        # season bullpen FIP-based quality
    "workload": 0.25,       # innings thrown last 3 days (fatigue)
    "closer": 0.18,         # closer available
    "setup": 0.12,          # top setup man available
}

# A reliever who pitched on BOTH of the last two days, or threw 2+ innings
# yesterday, is treated as unavailable today.
RELIEVER_UNAVAILABLE_RULES = {"back_to_back_days": 2, "long_outing_ip": 2.0}

SEASON = 2026
REQUEST_TIMEOUT = 20
REQUEST_SLEEP = 0.4  # be polite to the free API
