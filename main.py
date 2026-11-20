"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (0.05/0.25/0.5/0.75/0.95). Standard library only, deterministic,
no network, no clock reads at run time: every number derives from the case
file alone, so two runs are byte-identical.

Model: count-based decomposition. A trophy's rate is earners / players
tracked. Earners only grow (players never un-earn a trophy) while the
tracked player base grows fast after launch, so the two move in opposite
directions on the rate. We reconstruct earners per trophy per read
(rate x players tracked), fit log(earners) and log(players) separately
against ln(1 + hours since launch), and take the rate's log-slope as the
difference of the two fits: progression minus dilution. Trophy-type priors (story vs grind vs platinum, grounded in GTA V's
long-run trophy shape: first story trophy ~42% of players, final mission
~8%, platinum ~0.1%, 100%-completion ~0.2% on PS5) cap slopes and ceilings
so a noisy fit cannot send a grind trophy to 40% or the platinum anywhere.
A refresh-lag shrink pulls the forecast slightly back toward the last read,
because the page at the target shows Exophase's last refresh before it, not
the live state. Spreads combine a random-walk term, the fit residual, and a
refresh-jump term, because Exophase moves rates in jumps between refreshes.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
LO, HI = 0.01, 99.99

# --- spread calibration (tuned on the simulated backtest, see work/) ---
# The grid optimum in-sim is narrower still, but the sim cannot contain
# real-world regime shocks (a weekend player surge, a growth-rate break),
# so these sit slightly above the grid optimum as insurance.
RW_SIGMA = 0.22        # logit random-walk sigma per sqrt(day)
JUMP_SIGMA = 0.20      # refresh-jump term (Exophase moves rates in jumps)
RESID_WEIGHT = 0.5     # weight on the earners-fit residual std
FLAT_EXTRA_SIGMA = 0.70  # extra spread when history is perfectly flat
LAG_HOURS = 4.0        # page lag behind the live state, in hours


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparseable."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def log_hours(moment):
    """ln(1 + hours from launch to `moment`); 0 before launch."""
    if moment is None:
        return 0.0
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_rate(pct):
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(100.0, max(0.0, value))


def logit_of(pct):
    p = min(HI, max(LO, pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(logit):
    if logit >= 0:
        return 100.0 / (1.0 + math.exp(-logit))
    top = math.exp(logit)
    return 100.0 * top / (1.0 + top)


def norm_ppf(prob):
    """Standard normal quantile (Acklam's rational approximation)."""
    if prob <= 0.0:
        return -8.0
    if prob >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    low, high = 0.02425, 1.0 - 0.02425
    if prob < low:
        q = math.sqrt(-2.0 * math.log(prob))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if prob > high:
        q = math.sqrt(-2.0 * math.log(1.0 - prob))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = prob - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def round4(value):
    """Round to 4 decimals, half away from zero; ints stay ints in JSON."""
    scaled = math.floor(abs(value) * 10000.0 + 0.5) / 10000.0
    if value < 0:
        scaled = -scaled
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def least_squares(xs, ys):
    """Ordinary least squares; returns (slope, intercept, resid_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n) if n else 0.0
        return 0.0, mean_y, resid
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, intercept, resid


def classify(trophy):
    """Trophy-type prior: (ceiling percent, log-rate slope cap).

    The slope cap bounds the RATE's log-slope per log-hour (earners slope
    minus players slope), not the raw earners slope: earners growth already
    includes player-base growth, so capping it directly would force dilution
    whenever the player base grows faster than the cap. Grounded in GTA V's
    long-run PS5 shape (Sony rates: first story trophy ~42%, mid-story
    ~11-17%, final mission ~8%, platinum 0.1%, 100% completion 0.2%,
    gold-medal grind 0.2%, explore-all 0.5%), scaled down for a day-7
    snapshot: week-one ceilings sit well below long-run levels for
    late/grind trophies, near them only for the earliest story beats.
    """
    tier = str(trophy.get("tier", "")).lower()
    text = (str(trophy.get("name", "")) + " " + str(trophy.get("description", ""))).lower()
    if tier == "platinum":
        return 1.0, 0.05
    if "100% completion" in text or "100% game completion" in text:
        return 1.5, 0.15
    grind_marks = ("250 stunts", "100 locations", "50 challenges", "gold medal",
                   "collectible card", "every spot", "every location",
                   "all of leonida", "100 vehicles", "10 vehicles",
                   "all garage", "earn $", "spend $", "20 different races",
                   "8 mission strands", "30 platinum", "70 gold")
    if any(mark in text for mark in grind_marks):
        return 4.0, 0.4
    if "chapter 1" in text or "chapter 2" in text or "gone fishing" in text:
        return 45.0, 1.6
    if "chapter 3" in text or "chapter 4" in text:
        return 25.0, 1.3
    if "final mission" in text and tier == "gold":
        return 10.0, 1.0
    if "final mission for" in text:
        return 12.0, 1.0
    if text.startswith("complete ") or "complete " in text:
        return 20.0, 1.1
    return 15.0, 1.0


def forecast_trophy(trophy, index, reads, horizon_days, target_x):
    """One trophy's five quantiles. Never raises: falls back to persistence."""
    try:
        return _forecast_trophy(trophy, index, reads, horizon_days, target_x)
    except Exception:
        last_rate = 0.0
        for read in reversed(reads):
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                last_rate = clamp_rate(rates[index])
                break
        return safe_row(trophy, last_rate, horizon_days)


def safe_row(trophy, last_rate, horizon_days):
    """Persistence median with a wide spread; always valid output."""
    centre = logit_of(last_rate)
    sigma = 0.5 * math.sqrt(max(horizon_days, 0.0)) + 0.6
    return {"id": trophy.get("id"), "q": quantiles(centre, sigma)}


def quantiles(centre_logit, sigma):
    out = []
    prev = -1.0
    for level in LEVELS:
        value = inv_logit(centre_logit + norm_ppf(level) * sigma)
        value = min(100.0, max(0.0, value))
        value = round4(value)
        if value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def _forecast_trophy(trophy, index, reads, horizon_days, target_x):
    ceiling, slope_cap = classify(trophy)
    series = []  # (x, players, rate, moment) with usable data, oldest first
    last_rate = None
    for read in reads:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        players = read.get("players_tracked")
        moment = parse_time(read.get("at"))
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        rate = clamp_rate(rates[index])
        series.append((log_hours(moment), float(players), rate, moment))
        last_rate = rate
    if last_rate is None:
        return {"id": trophy.get("id"), "q": quantiles(logit_of(0.5), 1.5)}

    # A trophy everybody holds stays held: new players clear the first
    # mission fast, and the launch-night refresh left the 100% trophy at 100%.
    if last_rate >= 99.5:
        return {"id": trophy.get("id"), "q": quantiles(logit_of(99.5), 0.15)}

    horizon_hours = max(horizon_days * 24.0, 0.0)
    flat = len(series) >= 2 and all(s[1] == series[0][1] for s in series) \
        and all(s[2] == series[0][2] for s in series)

    if flat and horizon_hours > 0:
        # No refresh has been seen yet: the page is stale and the next
        # refresh will dilute rates (launch night: players x6.06, earned
        # x~4.5, every rate but the 100% one down). New players earn early
        # trophies fast but late/grind trophies slowly, so dilution is
        # type-dependent: early-story rates hold up best, platinum and
        # grinds dilute hardest. Expect a smaller jump than launch night's,
        # with a wide band around it: the first refresh is genuinely
        # uncertain, and the house's 1.51 practice loss is what a narrow
        # band costs when it lands.
        if slope_cap >= 1.6:
            flat_mult = 0.85
        elif slope_cap <= 0.15:
            flat_mult = 0.45
        elif slope_cap <= 0.4:
            flat_mult = 0.50
        else:
            flat_mult = 0.67
        median_rate = min(last_rate * flat_mult, ceiling)
        centre = logit_of(median_rate)
        sigma = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                          + JUMP_SIGMA ** 2 + FLAT_EXTRA_SIGMA ** 2)
        return {"id": trophy.get("id"), "q": quantiles(centre, sigma)}

    if len(series) < 3 or horizon_hours <= 0:
        sigma = math.sqrt((RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2
                          + JUMP_SIGMA ** 2) if horizon_hours > 0 else 0.05
        return {"id": trophy.get("id"), "q": quantiles(logit_of(last_rate), sigma)}

    # Last-24h window for both fits, by clock time from the last read.
    last_moment = series[-1][3]
    window = series
    if last_moment is not None:
        timed = [s for s in series
                 if s[3] is not None and (last_moment - s[3]).total_seconds() <= 24 * 3600]
        if len(timed) >= 3:
            window = timed
    if len(window) < 3:
        window = series[-3:] if len(series) >= 3 else series

    xs = [s[0] for s in window]
    last_x = series[-1][0]
    earners = [max(s[2] / 100.0 * s[1], 0.0) for s in window]
    players = [s[1] for s in window]
    log_e = [math.log(e + 0.5) for e in earners]
    log_p = [math.log(p) for p in players]
    e_slope, _, e_resid = least_squares(xs, log_e)
    p_slope, _, p_resid = least_squares(xs, log_p)
    p_slope = max(0.0, min(p_slope, 2.0))

    # The rate's log-slope is the DIFFERENCE of the two fits: progression
    # pushes earners up, player growth dilutes the rate down. Shrink the
    # net slope (not the earners slope alone: shrinking one side only would
    # bias the rate down whenever the player base grows), cap it by trophy
    # type, and never let it imply falling earners (net >= -players slope).
    net_slope = e_slope - p_slope
    shrink = 1.0 / (1.0 + 4.0 * (e_resid * e_resid + p_resid * p_resid))
    net_slope = net_slope * shrink
    net_slope = max(-p_slope, min(net_slope, slope_cap))
    net_slope = max(net_slope, -0.6)

    median_rate = last_rate * math.exp(net_slope * (target_x - last_x))
    median_rate = min(median_rate, ceiling)
    median_rate = max(median_rate, 0.0)

    # Refresh lag: the target page shows the last refresh before the target,
    # so shrink the forecast a little back toward the last read.
    lag_weight = min(0.30, LAG_HOURS / horizon_hours) if horizon_hours > 0 else 0.0
    centre = (1.0 - lag_weight) * logit_of(median_rate) + lag_weight * logit_of(last_rate)

    sigma = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2
                      + (RESID_WEIGHT * e_resid) ** 2 + JUMP_SIGMA ** 2)
    return {"id": trophy.get("id"), "q": quantiles(centre, sigma)}


def forecast_case(case):
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    horizon_days = max(float(case.get("horizon_minutes", 0) or 0) / 1440.0, 0.0)
    target_x = log_hours(parse_time(case.get("target")))
    if target_x <= 0.0:
        # No usable target: fall back to the checkpoint plus the horizon.
        checkpoint = parse_time(case.get("at"))
        if checkpoint is not None:
            from datetime import timedelta
            target_x = log_hours(checkpoint + timedelta(minutes=float(case.get("horizon_minutes", 0) or 0)))
    rows = []
    for index, trophy in enumerate(trophies):
        rows.append(forecast_trophy(trophy, index, reads, horizon_days, target_x))
    return {"trophies": rows}


def fallback_case(case):
    """Last-read persistence for every trophy; used if forecasting fails."""
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    horizon_days = max(float(case.get("horizon_minutes", 0) or 0) / 1440.0, 0.0)
    rows = []
    for index, trophy in enumerate(trophies):
        last_rate = 0.0
        for read in reversed(reads):
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                last_rate = clamp_rate(rates[index])
                break
        rows.append(safe_row(trophy, last_rate, horizon_days))
    return {"trophies": rows}


def validate(payload, trophy_count):
    """True only when the payload matches the required output shape."""
    if not isinstance(payload, dict) or set(payload.keys()) != {"trophies"}:
        return False
    rows = payload["trophies"]
    if not isinstance(rows, list) or len(rows) != trophy_count:
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row.keys()) != {"id", "q"}:
            return False
        if row["id"] in seen:
            return False
        seen.add(row["id"])
        qs = row["q"]
        if not isinstance(qs, list) or len(qs) != 5:
            return False
        prev = -1.0
        for value in qs:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            if not math.isfinite(value) or value < 0 or value > 100:
                return False
            if value < prev:
                return False
            prev = value
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(case, dict):
            continue
        try:
            payload = forecast_case(case)
            if not validate(payload, len(case.get("trophies") or [])):
                payload = fallback_case(case)
        except Exception:
            try:
                payload = fallback_case(case)
            except Exception:
                continue
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            continue
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
