"""
fetch_data.py — all calls to the official MLB Stats API (statsapi.mlb.com).

The MLB Stats API is free, official, and is the same feed MLB.com and ESPN
ultimately rely on for schedules, probable pitchers, and results. FanGraphs-style
rate stats (FIP, K%) are computed here from the raw counting stats the API returns.
"""

import time
import json
import hashlib
import requests
from datetime import date, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---- on-disk cache: identical API calls are fetched once, then reused ----
_CACHE_DIR = os.path.join(config.DATA_DIR, "api_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Per-day responses (schedules, boxscores) are stable once a day is final, so
# they're cached forever. "Live" season-to-date stat lines change daily, so we
# only reuse them within the same calendar day they were fetched.
_STABLE_HINTS = ("/game/", "/boxscore")


def _cache_path(path, params):
    key = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params or {}))
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, h + ".json")


def api_get(path, params=None):
    """
    GET a path on the MLB Stats API, with:
      - a disk cache so we never download the same thing twice, and
      - automatic retries so one network hiccup can't crash a long backfill.
    """
    params = params or {}
    cpath = _cache_path(path, params)
    stable = any(hint in path for hint in _STABLE_HINTS)

    if os.path.exists(cpath):
        fresh_enough = stable or (time.time() - os.path.getmtime(cpath) < 86400)
        if fresh_enough:
            try:
                with open(cpath) as f:
                    return json.load(f)
            except Exception:
                pass  # corrupt cache entry -> just refetch

    url = f"{config.MLB_API}{path}"
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            with open(cpath, "w") as f:
                json.dump(data, f)
            time.sleep(config.REQUEST_SLEEP)
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))  # back off, then retry
    raise last_err


# ---------------------------------------------------------------- schedule

def get_schedule(day: date):
    """All games on a date, with probable pitchers and final scores if played."""
    data = api_get("/schedule", {
        "sportId": 1,
        "date": day.isoformat(),
        "hydrate": "probablePitcher,team,linescore,decisions",
    })
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            games.append({
                "game_pk": g["gamePk"],
                "date": day.isoformat(),
                "status": g["status"]["detailedState"],
                "venue": g.get("venue", {}).get("name", ""),
                "home_id": home["team"]["id"],
                "home_name": home["team"]["name"],
                "home_abbr": home["team"].get("abbreviation", ""),
                "away_id": away["team"]["id"],
                "away_name": away["team"]["name"],
                "away_abbr": away["team"].get("abbreviation", ""),
                "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                "home_pitcher": (home.get("probablePitcher") or {}).get("fullName", "TBD"),
                "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "away_pitcher": (away.get("probablePitcher") or {}).get("fullName", "TBD"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "game_time": g.get("gameDate", ""),
            })
    return games


def get_team_abbreviations():
    data = api_get("/teams", {"sportId": 1, "season": config.SEASON})
    return {t["id"]: t.get("abbreviation", "") for t in data.get("teams", [])}


# ---------------------------------------------------------------- pitchers

def _fip(hr, bb, hbp, k, ip):
    if ip <= 0:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + config.FIP_CONSTANT, 2)


def _ip_to_float(ip_str):
    """MLB API reports IP like '123.2' meaning 123 and 2/3."""
    if ip_str in (None, ""):
        return 0.0
    s = str(ip_str)
    whole, _, frac = s.partition(".")
    return float(whole) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)


def get_pitcher_season(pid: int):
    """Season FIP, WHIP, K%, plus throwing hand."""
    if not pid:
        return None
    data = api_get(f"/people/{pid}", {
        "hydrate": f"stats(group=pitching,type=season,season={config.SEASON})"
    })
    people = data.get("people", [])
    if not people:
        return None
    p = people[0]
    hand = (p.get("pitchHand") or {}).get("code", "R")
    out = {"name": p.get("fullName"), "hand": hand,
           "fip": None, "whip": None, "k_rate": None, "era": None, "ip": 0.0}
    for grp in p.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            ip = _ip_to_float(s.get("inningsPitched"))
            bf = s.get("battersFaced") or 0
            k = s.get("strikeOuts") or 0
            out.update({
                "ip": ip,
                "era": float(s["era"]) if s.get("era") not in (None, "-.--") else None,
                "whip": float(s["whip"]) if s.get("whip") not in (None, "-.--") else None,
                "k_rate": round(k / bf, 3) if bf else None,
                "fip": _fip(s.get("homeRuns") or 0, s.get("baseOnBalls") or 0,
                            s.get("hitByPitch") or 0, k, ip),
            })
    return out


def get_pitcher_last5(pid: int):
    """FIP and ERA over the pitcher's last 5 starts (game log)."""
    if not pid:
        return None
    data = api_get(f"/people/{pid}/stats", {
        "stats": "gameLog", "group": "pitching", "season": config.SEASON,
    })
    starts = []
    for grp in data.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            if (s.get("gamesStarted") or 0) >= 1:
                starts.append(s)
    starts = starts[-5:]
    if not starts:
        return None
    ip = sum(_ip_to_float(s.get("inningsPitched")) for s in starts)
    if ip <= 0:
        return None
    er = sum(s.get("earnedRuns") or 0 for s in starts)
    fip = _fip(sum(s.get("homeRuns") or 0 for s in starts),
               sum(s.get("baseOnBalls") or 0 for s in starts),
               sum(s.get("hitByPitch") or 0 for s in starts),
               sum(s.get("strikeOuts") or 0 for s in starts), ip)
    return {"era": round(9 * er / ip, 2), "fip": fip, "starts": len(starts), "ip": round(ip, 1)}


# ---------------------------------------------------------------- bullpen

def get_bullpen_state(team_id: int, day: date):
    """
    Bullpen quality (season FIP of relievers), workload last 3 days,
    and closer / top setup availability, derived from recent boxscores.
    """
    # Season reliever quality: team pitching minus starters is approximated by
    # pulling each active reliever's season line.
    roster = api_get(f"/teams/{team_id}/roster", {"rosterType": "active"})
    relievers = []
    for r in roster.get("roster", []):
        if r.get("position", {}).get("code") == "1":
            relievers.append(r["person"]["id"])

    quality_fips, saves_by_pid, holds_by_pid = [], {}, {}
    for pid in relievers:
        try:
            data = api_get(f"/people/{pid}/stats", {
                "stats": "season", "group": "pitching", "season": config.SEASON})
        except Exception:
            continue
        for grp in data.get("stats", []):
            for split in grp.get("splits", []):
                s = split.get("stat", {})
                gs, g = s.get("gamesStarted") or 0, s.get("gamesPlayed") or 0
                if g == 0 or gs / max(g, 1) > 0.5:
                    continue  # starters / unused arms
                ip = _ip_to_float(s.get("inningsPitched"))
                f = _fip(s.get("homeRuns") or 0, s.get("baseOnBalls") or 0,
                         s.get("hitByPitch") or 0, s.get("strikeOuts") or 0, ip)
                if f is not None and ip >= 3:
                    quality_fips.append((f, ip))
                saves_by_pid[pid] = s.get("saves") or 0
                holds_by_pid[pid] = s.get("holds") or 0

    if quality_fips:
        total_ip = sum(ip for _, ip in quality_fips)
        bullpen_fip = round(sum(f * ip for f, ip in quality_fips) / total_ip, 2)
    else:
        bullpen_fip = 4.20  # league-ish neutral fallback

    closer_id = max(saves_by_pid, key=saves_by_pid.get) if saves_by_pid else None
    setup_id = max(holds_by_pid, key=holds_by_pid.get) if holds_by_pid else None

    # Usage over last 3 days from boxscores
    usage = {}      # pid -> list of (days_ago, ip)
    relief_ip_3d = 0.0
    for days_ago in (1, 2, 3):
        d = day - timedelta(days=days_ago)
        for g in get_schedule(d):
            if g["home_id"] != team_id and g["away_id"] != team_id:
                continue
            if g["status"] not in ("Final", "Game Over", "Completed Early"):
                continue
            side = "home" if g["home_id"] == team_id else "away"
            box = api_get(f"/game/{g['game_pk']}/boxscore")
            players = box["teams"][side]["players"]
            for pdata in players.values():
                st = pdata.get("stats", {}).get("pitching", {})
                if not st:
                    continue
                ip = _ip_to_float(st.get("inningsPitched"))
                pid = pdata["person"]["id"]
                if (st.get("gamesStarted") or 0) == 0 and ip > 0:
                    relief_ip_3d += ip
                    usage.setdefault(pid, []).append((days_ago, ip))

    def available(pid):
        if pid is None:
            return 1
        u = usage.get(pid, [])
        days = {d for d, _ in u}
        if {1, 2}.issubset(days):           # pitched back-to-back days
            return 0
        if any(d == 1 and ip >= config.RELIEVER_UNAVAILABLE_RULES["long_outing_ip"]
               for d, ip in u):             # long outing yesterday
            return 0
        return 1

    return {
        "bullpen_fip": bullpen_fip,
        "relief_ip_last3": round(relief_ip_3d, 1),
        "closer_available": available(closer_id),
        "setup_available": available(setup_id),
    }


# ---------------------------------------------------------------- offense

def get_team_offense(team_id: int, day: date):
    """OPS + runs/game over the trailing 14 days, and season OPS vs LHP / RHP."""
    start = (day - timedelta(days=14)).isoformat()
    end = (day - timedelta(days=1)).isoformat()

    data = api_get(f"/teams/{team_id}/stats", {
        "stats": "byDateRange", "group": "hitting",
        "startDate": start, "endDate": end, "season": config.SEASON,
    })
    ops14, rpg14 = None, None
    for grp in data.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            if s.get("ops"):
                ops14 = float(s["ops"])
            g = s.get("gamesPlayed") or 0
            if g:
                rpg14 = round((s.get("runs") or 0) / g, 2)

    splits_data = api_get(f"/teams/{team_id}/stats", {
        "stats": "statSplits", "group": "hitting",
        "sitCodes": "vl,vr", "season": config.SEASON,
    })
    ops_vs = {"L": None, "R": None}
    for grp in splits_data.get("stats", []):
        for split in grp.get("splits", []):
            code = (split.get("split") or {}).get("code", "")
            s = split.get("stat", {})
            if s.get("ops"):
                if code == "vl":
                    ops_vs["L"] = float(s["ops"])
                elif code == "vr":
                    ops_vs["R"] = float(s["ops"])

    return {"ops_14d": ops14, "rpg_14d": rpg14, "ops_vs_hand": ops_vs}


# ---------------------------------------------------------------- scores / leaders

def get_final_scores(day):
    """Compact list of yesterday's finals for the scoreboard strip."""
    out = []
    for g in get_schedule(day):
        if g["status"] in ("Final", "Game Over", "Completed Early") and g["home_score"] is not None:
            out.append({
                "away": g["away_name"], "away_abbr": g["away_abbr"], "away_score": g["away_score"],
                "home": g["home_name"], "home_abbr": g["home_abbr"], "home_score": g["home_score"],
                "winner": g["home_abbr"] if g["home_score"] > g["away_score"] else g["away_abbr"],
            })
    return out


def get_leaders():
    """League stat leaders (hitting + pitching) for the Players & Teams tab."""
    def hit_leaders(cat, n=5):
        try:
            d = api_get("/stats/leaders", {
                "leaderCategories": cat, "season": config.SEASON,
                "sportId": 1, "statGroup": "hitting", "limit": n})
        except Exception:
            return []
        rows = []
        for lc in d.get("leagueLeaders", []):
            for r in lc.get("leaders", [])[:n]:
                rows.append({"name": r["person"]["fullName"],
                             "team": (r.get("team") or {}).get("abbreviation", ""),
                             "value": r["value"]})
        return rows

    def pitch_leaders(cat, n=5, order_asc=False):
        try:
            d = api_get("/stats/leaders", {
                "leaderCategories": cat, "season": config.SEASON,
                "sportId": 1, "statGroup": "pitching", "limit": n})
        except Exception:
            return []
        rows = []
        for lc in d.get("leagueLeaders", []):
            for r in lc.get("leaders", [])[:n]:
                rows.append({"name": r["person"]["fullName"],
                             "team": (r.get("team") or {}).get("abbreviation", ""),
                             "value": r["value"]})
        return rows

    # Team standings (W-L, run diff) for the team table
    teams = []
    try:
        st = api_get("/standings", {"leagueId": "103,104", "season": config.SEASON,
                                    "standingsTypes": "regularSeason"})
        for rec in st.get("records", []):
            for t in rec.get("teamRecords", []):
                teams.append({
                    "team": t["team"]["name"],
                    "abbr": t["team"].get("abbreviation", ""),
                    "w": t["wins"], "l": t["losses"],
                    "pct": t.get("winningPercentage", ""),
                    "rs": t.get("runsScored", 0), "ra": t.get("runsAllowed", 0),
                    "diff": (t.get("runsScored", 0) or 0) - (t.get("runsAllowed", 0) or 0),
                    "streak": (t.get("streak") or {}).get("streakCode", ""),
                })
    except Exception:
        pass
    teams.sort(key=lambda x: -x["diff"])

    return {
        "hitting": {
            "AVG": hit_leaders("battingAverage"),
            "HR": hit_leaders("homeRuns"),
            "RBI": hit_leaders("runsBattedIn"),
            "OPS": hit_leaders("onBasePlusSlugging"),
        },
        "pitching": {
            "ERA": pitch_leaders("earnedRunAverage"),
            "Strikeouts": pitch_leaders("strikeouts"),
            "Wins": pitch_leaders("wins"),
            "WHIP": pitch_leaders("walksAndHitsPerInningPitched"),
        },
        "teams": teams,
    }
