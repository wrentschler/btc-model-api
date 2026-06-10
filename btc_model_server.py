"""
Bitcoin Liquidity Model — Backend API
======================================
Pulls the underlying data, runs the OLS fair-value regression, and serves a
single JSON endpoint the front-end fetches:  GET /api/btc-model

Model (matches the published methodology):
    log(BTC) ~ Fed_Net_Liquidity($T) + GlobalM2_YoY(%) + Stablecoin_Supply($B)
               + Stablecoin_30d_delta(%) + halving_cycle_pos + time_trend(weeks)

Fair value = exp(fitted log price).
Bands: +/-1 sigma (likely) and +/-2 sigma (extreme) of the log residual.
Z-score (Cheap/Dear) = standardized current log residual.

------------------------------------------------------------------------------
DATA SOURCES (all overridable via environment variables)
  - BTC price ............ CoinGecko (no key) or any OHLC source
  - Fed Net Liquidity .... FRED series WALCL, WTREGEN, RRPONTSYD  (FRED_API_KEY)
  - Global M2 ............ FRED M2SL as a US proxy, or supply your own composite
  - Stablecoin supply .... STABLE_SUPPLY_URL (JSON) or static fallback

If a source is unavailable, the server falls back to the last good cache, then
to a bundled representative snapshot, so the endpoint always returns something
usable and clearly flags data freshness in the payload.

------------------------------------------------------------------------------
RUN
  pip install fastapi uvicorn numpy pandas requests apscheduler python-dotenv
  export FRED_API_KEY=...          # https://fred.stlouisfed.org/docs/api/api_key.html
  uvicorn btc_model_server:app --host 0.0.0.0 --port 8002

ENV VARS
  FRED_API_KEY        required for live Fed liquidity + M2 (free key)
  STABLE_SUPPLY_URL   optional JSON endpoint returning {"supply_usd_billions": <num>}
  ALLOWED_ORIGINS     comma-separated CORS origins (defaults to localhost + *.netlify.app)
  REFRESH_HOURS       how often to rebuild the model (default 12)
  CACHE_PATH          where to persist the last good build (default ./btc_model_cache.json)
"""

import os
import json
import math
import time
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── CONFIG ──────────────────────────────────────────────────────────────────
FRED_API_KEY     = os.environ.get("FRED_API_KEY", "")
STABLE_SUPPLY_URL = os.environ.get("STABLE_SUPPLY_URL", "")
REFRESH_HOURS    = float(os.environ.get("REFRESH_HOURS", "12"))
CACHE_PATH       = os.environ.get("CACHE_PATH", "btc_model_cache.json")
START_DATE       = "2015-01-01"
HTTP_TIMEOUT     = 20

_default_origins = "http://localhost:8000,http://127.0.0.1:8000,https://*.netlify.app"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app = FastAPI(title="Bitcoin Liquidity Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_methods=["GET"],
    allow_headers=["*"],
)

_STATE = {"payload": None, "built_at": None, "warnings": []}


# ── DATA FETCHERS ────────────────────────────────────────────────────────────
def _fred_series(series_id: str) -> Optional[pd.Series]:
    """Fetch a FRED series as a daily-indexed pandas Series (forward-filled)."""
    if not FRED_API_KEY:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
    }
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    idx, val = [], []
    for o in obs:
        if o["value"] in (".", "", None):
            continue
        idx.append(pd.Timestamp(o["date"]))
        val.append(float(o["value"]))
    if not idx:
        return None
    return pd.Series(val, index=idx).sort_index()


def _btc_from_coingecko() -> Optional[pd.Series]:
    """
    CoinGecko market_chart. The free (keyless) tier no longer allows
    days=max / interval=daily (returns 401). We:
      - send a Demo API key if COINGECKO_API_KEY is set (header), and
      - request the largest window the tier allows (365d keyless),
        or 'max' when a key is present.
    """
    key = os.environ.get("COINGECKO_API_KEY", "").strip()
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    headers = {"accept": "application/json"}
    if key:
        # Demo keys use this header; Pro keys use x-cg-pro-api-key.
        headers["x-cg-demo-api-key"] = key
    # With a key we can ask for max; without, the free tier caps daily history.
    attempts = ["max", "365"] if key else ["365"]
    for days in attempts:
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                continue
            prices = r.json().get("prices", [])
            if not prices:
                continue
            s = pd.Series(
                [p[1] for p in prices],
                index=[pd.Timestamp(p[0], unit="ms") for p in prices],
            ).sort_index()
            return s
        except Exception:
            continue
    return None


def _btc_from_coinbase() -> Optional[pd.Series]:
    """
    Keyless fallback: Coinbase Exchange daily candles (granularity=86400).
    Coinbase caps each request to 300 candles, so we page backwards in
    ~300-day chunks to assemble multi-year weekly history.
    """
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    headers = {"User-Agent": "btc-liquidity-model"}
    end = dt.datetime.utcnow()
    start_floor = dt.datetime(2015, 1, 1)
    rows = {}
    # page back up to ~16 chunks (~13 years)
    for _ in range(16):
        start = end - dt.timedelta(days=300)
        if end <= start_floor:
            break
        params = {
            "granularity": 86400,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            # candle = [time, low, high, open, close, volume]
            for c in data:
                rows[int(c[0])] = float(c[4])
        except Exception:
            break
        end = start
        time.sleep(0.25)  # be polite to the public endpoint
    if not rows:
        return None
    idx = [pd.Timestamp(t, unit="s") for t in rows.keys()]
    s = pd.Series(list(rows.values()), index=idx).sort_index()
    return s


def fetch_btc_weekly() -> Optional[pd.Series]:
    """Weekly BTC close (USD). Tries CoinGecko, then Coinbase, then None."""
    s = _btc_from_coingecko()
    if s is None or len(s) < 60:
        s2 = _btc_from_coinbase()
        if s2 is not None and (s is None or len(s2) > len(s)):
            s = s2
    if s is None:
        return None
    s = s[s.index >= pd.Timestamp(START_DATE)]
    return s.resample("W").last().dropna()


def fetch_fed_net_liquidity() -> Optional[pd.Series]:
    """WALCL - TGA - RRP, in $ trillions, weekly."""
    walcl = _fred_series("WALCL")          # Fed assets, $M
    tga   = _fred_series("WTREGEN")        # Treasury General Account, $B
    rrp   = _fred_series("RRPONTSYD")      # Reverse repo, $B
    if walcl is None:
        return None
    df = pd.DataFrame({"walcl": walcl})
    if tga is not None:
        df["tga"] = tga
    if rrp is not None:
        df["rrp"] = rrp
    df = df.resample("W").last().ffill()
    # WALCL is $millions; TGA & RRP are $billions -> convert all to $trillions
    net = df["walcl"] / 1e6
    if "tga" in df:
        net = net - df["tga"] / 1e3
    if "rrp" in df:
        net = net - df["rrp"] / 1e3
    return net.dropna()


def fetch_global_m2_yoy() -> Optional[pd.Series]:
    """
    Global M2 YoY %. True 'global' M2 needs multi-country aggregation + FX.
    Here we use US M2 (M2SL) YoY as a transparent proxy; override by supplying
    your own composite series file via GLOBAL_M2_CSV (date,value columns).
    """
    csv_path = os.environ.get("GLOBAL_M2_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        s = df.iloc[:, 0].resample("W").last().ffill()
        return s.dropna()
    m2 = _fred_series("M2SL")  # US M2, monthly, $B
    if m2 is None:
        return None
    m2w = m2.resample("W").last().ffill()
    yoy = (m2w / m2w.shift(52) - 1.0) * 100.0
    return yoy.dropna()


def fetch_stablecoin_supply() -> Optional[pd.Series]:
    """
    Stablecoin supply ($B), weekly. A clean historical series usually needs a
    paid/aggregator source; if STABLE_SUPPLY_URL is set we read the latest point
    and hold it flat over recent weeks. Otherwise returns None (fallback used).
    """
    if not STABLE_SUPPLY_URL:
        return None
    try:
        r = requests.get(STABLE_SUPPLY_URL, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        latest = float(r.json().get("supply_usd_billions"))
        # Build a short flat tail just so the regression has a current value.
        idx = pd.date_range(end=pd.Timestamp.today(), periods=8, freq="W")
        return pd.Series([latest] * len(idx), index=idx)
    except Exception:
        return None


# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_model() -> dict:
    """Assemble weekly panel, run OLS in log space, compute fair value/bands/z."""
    warnings = []

    btc = fetch_btc_weekly()
    if btc is None or len(btc) < 60:
        raise RuntimeError("BTC price unavailable")

    fnl = fetch_fed_net_liquidity()
    m2  = fetch_global_m2_yoy()
    scs = fetch_stablecoin_supply()

    if fnl is None: warnings.append("Fed Net Liquidity unavailable (need FRED_API_KEY) — using trend fallback.")
    if m2  is None: warnings.append("Global M2 unavailable (need FRED_API_KEY) — using trend fallback.")
    if scs is None: warnings.append("Stablecoin supply source not configured — using trend fallback.")

    panel = pd.DataFrame({"btc": btc})
    panel["fnl"] = fnl.reindex(panel.index).ffill() if fnl is not None else np.nan
    panel["m2"]  = m2.reindex(panel.index).ffill()  if m2  is not None else np.nan
    panel["scs"] = scs.reindex(panel.index).ffill() if scs is not None else np.nan

    # Fallbacks for any missing driver so the model still fits (flagged above).
    n = len(panel)
    if panel["fnl"].isna().all():
        panel["fnl"] = np.linspace(4.0, 5.9, n)
    if panel["m2"].isna().all():
        panel["m2"] = 6.0 + 4.0 * np.sin(np.linspace(0, 6 * math.pi, n))
    if panel["scs"].isna().all():
        panel["scs"] = np.clip(np.linspace(-50, 266, n), 0, None)
    panel = panel.ffill().bfill()

    # Engineered factors
    weeks = np.arange(n, dtype=float)
    panel["scs_30d"] = panel["scs"].diff(4).fillna(0.0) / panel["scs"].shift(4).replace(0, np.nan)
    panel["scs_30d"] = (panel["scs_30d"].fillna(0.0) * 100.0)
    # halving cycle position: BTC halvings ~ every 210k blocks ≈ 4 years
    halvings = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")]
    def cycle_pos(ts):
        last = max([h for h in halvings if h <= ts], default=halvings[0])
        return min((ts - last).days / (4 * 365.25), 1.0)
    panel["cycle"] = [cycle_pos(ts) for ts in panel.index]
    panel["trend"] = weeks

    X = panel[["fnl", "m2", "scs", "scs_30d", "cycle", "trend"]].values
    X = np.column_stack([np.ones(n), X])           # intercept
    y = np.log(panel["btc"].values)

    # OLS via least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    sigma = float(np.std(resid, ddof=X.shape[1]))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    fair = np.exp(fitted)
    up1, lo1 = np.exp(fitted + sigma), np.exp(fitted - sigma)
    up2, lo2 = np.exp(fitted + 2 * sigma), np.exp(fitted - 2 * sigma)
    z = (resid - resid.mean()) / (resid.std() or 1.0)

    btc_now   = float(panel["btc"].iloc[-1])
    fair_now  = float(fair[-1])
    z_now     = float(z[-1])
    vs_fair   = (btc_now / fair_now - 1.0) * 100.0
    if   z_now <= -1.5: signal = "Strong Cheap"
    elif z_now <  -0.5: signal = "Cheap"
    elif z_now <=  0.5: signal = "Fair"
    elif z_now <   1.5: signal = "Dear"
    else:               signal = "Strong Dear"

    labels = [ts.strftime("%Y-%m") for ts in panel.index]
    dlabels = labels

    def rnd(a, d=0): return [round(float(v), d) for v in a]

    payload = {
        "as_of": panel.index[-1].strftime("%b %d, %Y"),
        "current": {
            "btc": round(btc_now),
            "fair": round(fair_now),
            "low1": round(float(lo1[-1])),
            "high1": round(float(up1[-1])),
            "vs_fair_pct": round(vs_fair, 2),
            "z": round(z_now, 2),
            "signal": signal,
            "fed_net_liquidity_t": round(float(panel["fnl"].iloc[-1]), 2),
            "global_m2_yoy": round(float(panel["m2"].iloc[-1]), 2),
            "stablecoin_supply_b": round(float(panel["scs"].iloc[-1]), 1),
        },
        "series": {
            "labels": labels,
            "btc": rnd(panel["btc"].values),
            "fair": rnd(fair),
            "up1": rnd(up1), "lo1": rnd(lo1),
            "up2": rnd(up2), "lo2": rnd(lo2),
            "z": rnd(z, 3),
            "dlabels": dlabels,
            "fnl": rnd(panel["fnl"].values, 2),
            "m2": rnd(panel["m2"].values, 2),
            "scs": rnd(panel["scs"].values, 1),
        },
        "model": {
            "r2": round(r2, 3),
            "observations": int(n),
            "residual_sigma": round(sigma, 3),
            "coefficients": {
                "intercept": round(float(beta[0]), 4),
                "fed_net_liquidity": round(float(beta[1]), 4),
                "global_m2_yoy": round(float(beta[2]), 4),
                "stablecoin_supply": round(float(beta[3]), 4),
                "stablecoin_30d_delta": float(f"{beta[4]:.3e}"),
                "halving_cycle_pos": round(float(beta[5]), 4),
                "time_trend": round(float(beta[6]), 4),
            },
        },
        "data_quality": {
            "live": len(warnings) == 0,
            "warnings": warnings,
            "built_at": dt.datetime.utcnow().isoformat() + "Z",
        },
    }
    return payload


def refresh(force: bool = False):
    try:
        payload = build_model()
        _STATE["payload"] = payload
        _STATE["built_at"] = time.time()
        _STATE["warnings"] = payload["data_quality"]["warnings"]
        try:
            with open(CACHE_PATH, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass
        print(f"[btc-model] rebuilt OK · live={payload['data_quality']['live']} · z={payload['current']['z']}")
    except Exception as e:
        print(f"[btc-model] rebuild failed: {e}")
        if _STATE["payload"] is None and os.path.exists(CACHE_PATH):
            try:
                _STATE["payload"] = json.load(open(CACHE_PATH))
                print("[btc-model] served cache from disk")
            except Exception:
                pass


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/api/btc-model")
def get_model():
    if _STATE["payload"] is None:
        refresh(force=True)
    if _STATE["payload"] is None:
        return {"error": "model unavailable", "data_quality": {"live": False}}
    return _STATE["payload"]


@app.get("/api/health")
def health():
    return {
        "ok": _STATE["payload"] is not None,
        "built_at": _STATE["built_at"],
        "live": _STATE["payload"]["data_quality"]["live"] if _STATE["payload"] else False,
        "warnings": _STATE["warnings"],
    }


# ── STARTUP / SCHEDULER ─────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    refresh(force=True)
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(refresh, "interval", hours=REFRESH_HOURS, id="refresh")
    sched.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8002")))





