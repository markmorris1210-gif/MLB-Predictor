"""
features.py — turns raw API data into the model's feature vector.

Implements the exact 13-variable spec, then folds the four bullpen
variables into one Bullpen Availability Score (0-100), giving the
model 10 inputs. Each row is written from the HOME team's point of
view, so every pitching/offense feature is (home value - away value).
"""

from datetime import date

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline import fetch_data as fd

# League-average fallbacks used early in the season when a sample is missing
LEAGUE_AVG = {
    "fip": 4.10, "whip": 1.28, "k_rate": 0.222, "era": 4.10,
    "ops": 0.715, "rpg": 4.4, "bullpen_fip": 4.10,
}


def bullpen_availability_score(state):
    """
    Combine bullpen quality, workload, closer and setup availability
    into a single 0-100 score. Higher = fresher + better bullpen.
    """
    w = config.BULLPEN_SCORE_WEIGHTS

    # Quality: map FIP 3.00 (elite) -> 100, 5.00 (bad) -> 0
    q = max(0.0, min(1.0, (5.0 - state["bullpen_fip"]) / 2.0))

    # Workload: 0 relief IP over 3 days -> fresh (1.0); 12+ IP -> gassed (0.0)
    wl = max(0.0, min(1.0, 1.0 - state["relief_ip_last3"] / 12.0))

    score = 100 * (w["quality"] * q + w["workload"] * wl +
                   w["closer"] * state["closer_available"] +
                   w["setup"] * state["setup_available"])
    return round(score, 1)


def _pitcher_block(pid):
    """Season + last-5 numbers for one probable starter, with fallbacks."""
    season = fd.get_pitcher_season(pid) or {}
    last5 = fd.get_pitcher_last5(pid) or {}

    def val(v, key):
        return v if v is not None else LEAGUE_AVG[key]

    # Blend ERA and FIP for the recent-form variable (#4 in the spec)
    l5_fip = last5.get("fip")
    l5_era = last5.get("era")
    if l5_fip is not None and l5_era is not None:
        recent = round(0.5 * l5_fip + 0.5 * l5_era, 2)
    else:
        recent = val(l5_fip if l5_fip is not None else l5_era, "fip")

    return {
        "name": season.get("name", "TBD"),
        "hand": season.get("hand", "R"),
        "fip": val(season.get("fip"), "fip"),
        "whip": val(season.get("whip"), "whip"),
        "k_rate": val(season.get("k_rate"), "k_rate"),
        "last5": recent,
        "last5_detail": last5,
        "season_ip": season.get("ip", 0.0),
    }


def build_game_features(game, day: date):
    """
    Returns (features_dict, detail_dict). features_dict matches config.FEATURES;
    detail_dict carries everything the website's Reasoning tab shows.
    """
    home_sp = _pitcher_block(game["home_pitcher_id"])
    away_sp = _pitcher_block(game["away_pitcher_id"])

    home_pen = fd.get_bullpen_state(game["home_id"], day)
    away_pen = fd.get_bullpen_state(game["away_id"], day)
    home_pen_score = bullpen_availability_score(home_pen)
    away_pen_score = bullpen_availability_score(away_pen)

    home_off = fd.get_team_offense(game["home_id"], day)
    away_off = fd.get_team_offense(game["away_id"], day)

    def ops_vs(offense, opposing_hand):
        v = offense["ops_vs_hand"].get(opposing_hand)
        return v if v is not None else LEAGUE_AVG["ops"]

    def safe(v, key):
        return v if v is not None else LEAGUE_AVG[key]

    park = config.PARK_FACTORS.get(game["home_abbr"], 100)

    features = {
        "sp_fip_diff": home_sp["fip"] - away_sp["fip"],
        "sp_whip_diff": home_sp["whip"] - away_sp["whip"],
        "sp_k_rate_diff": home_sp["k_rate"] - away_sp["k_rate"],
        "sp_last5_fip_diff": home_sp["last5"] - away_sp["last5"],
        "bullpen_score_diff": home_pen_score - away_pen_score,
        "ops_14d_diff": safe(home_off["ops_14d"], "ops") - safe(away_off["ops_14d"], "ops"),
        "ops_vs_hand_diff": ops_vs(home_off, away_sp["hand"]) - ops_vs(away_off, home_sp["hand"]),
        "rpg_14d_diff": safe(home_off["rpg_14d"], "rpg") - safe(away_off["rpg_14d"], "rpg"),
        "home_field": 1,
        "park_factor": park,
    }

    detail = {
        "home_sp": home_sp, "away_sp": away_sp,
        "home_bullpen": {**home_pen, "score": home_pen_score},
        "away_bullpen": {**away_pen, "score": away_pen_score},
        "home_offense": home_off, "away_offense": away_off,
        "park_factor": park,
    }
    return features, detail
