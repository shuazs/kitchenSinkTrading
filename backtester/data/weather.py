"""US weather: GHCN-Daily station actuals and GEFS ensemble-mean forecasts.

Both come from NOAA's AWS open-data buckets, which the cloud environment can
reach (NOAA/NCEI/CPC websites, open-meteo and meteostat are blocked).

- `ghcn(station)`: daily TMAX/TMIN (deg F) and PRCP (inches) for one station.
- `actuals(cities)`: wide frames of daily mean temperature per city.
- `gefs_run(day, run)`: 2 m temperature (deg F) at every city in `CITIES`
  from one GEFS v12 ensemble-mean run (0.5 deg, every 12 h to 384 h).
- `gefs_panel(start, end, run)`: many runs, cached per month.
- `demand_index`: population-weighted HDD + CDD, the usual "gas-weighted
  degree day" proxy that natural-gas traders watch.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .cache import cache_dir

GHCN = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{}.csv"
GHCN_YEAR = "https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_year/{}.csv.gz"
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"

# name: (lat, lon, metro population in millions, GHCN station, region tags)
CITIES = {
    "NYC": (40.78, -73.97, 19.5, "USW00094728", "ne"),
    "LAX": (33.94, -118.39, 13.0, "USW00023174", "w"),
    "CHI": (41.96, -87.93, 9.4, "USW00094846", "mw"),
    "DFW": (32.90, -97.02, 7.9, "USW00003927", "tx"),
    "HOU": (29.98, -95.36, 7.3, "USW00012960", "tx"),
    "DCA": (38.85, -77.03, 6.3, "USW00013743", "ne"),
    "PHL": (39.87, -75.23, 6.2, "USW00013739", "ne"),
    "MIA": (25.79, -80.32, 6.1, "USW00012839", "se"),
    "ATL": (33.63, -84.44, 6.2, "USW00013874", "se"),
    "BOS": (42.36, -71.01, 4.9, "USW00014739", "ne"),
    "PHX": (33.43, -112.00, 5.0, "USW00023183", "w"),
    "SFO": (37.62, -122.37, 4.7, "USW00023234", "w"),
    "DTW": (42.23, -83.33, 4.4, "USW00094847", "mw"),
    "SEA": (47.44, -122.31, 4.0, "USW00024233", "w"),
    "MSP": (44.89, -93.23, 3.7, "USW00014922", "mw"),
    "TPA": (27.96, -82.54, 3.2, "USW00012842", "se"),
    "DEN": (39.85, -104.66, 3.0, "USW00003017", "w"),
    "STL": (38.75, -90.37, 2.8, "USW00013994", "mw"),
    "BAL": (39.17, -76.68, 2.8, "USW00093721", "ne"),
    "CLT": (35.22, -80.95, 2.7, "USW00013881", "se"),
    "PIT": (40.48, -80.21, 2.4, "USW00094823", "ne"),
    "SAT": (29.54, -98.48, 2.6, "USW00012921", "tx"),
    "AUS": (30.18, -97.68, 2.4, "USW00013904", "tx"),
    "PDX": (45.60, -122.61, 2.5, "USW00024229", "w"),
    "CVG": (39.04, -84.67, 2.3, "USW00093814", "mw"),
    "KC": (39.30, -94.73, 2.2, "USW00003947", "mw"),
    "CLE": (41.41, -81.86, 2.1, "USW00014820", "mw"),
    "IND": (39.71, -86.28, 2.1, "USW00093819", "mw"),
    "BNA": (36.11, -86.69, 2.0, "USW00013897", "se"),
    "MKE": (42.95, -87.90, 1.6, "USW00014839", "mw"),
    # Corn Belt (weight 0: excluded from the demand index)
    "DSM": (41.53, -93.65, 0.0, "USW00014933", "corn"),
    "SPI": (39.85, -89.68, 0.0, "USW00093822", "corn"),
    "CMI": (40.03, -88.28, 0.0, "USW00094870", "corn"),
    "PIA": (40.67, -89.68, 0.0, "USW00014842", "corn"),
    "LNK": (40.83, -96.76, 0.0, "USW00014939", "corn"),
    "OMA": (41.31, -95.90, 0.0, "USW00014942", "corn"),
    "FSD": (43.59, -96.73, 0.0, "USW00014944", "corn"),
    "FWA": (40.97, -85.21, 0.0, "USW00014827", "corn"),
    "CID": (41.88, -91.71, 0.0, "USW00014990", "corn"),
    "RST": (43.90, -92.49, 0.0, "USW00014925", "corn"),
}


# Kalshi monthly rain / snow markets: city code in the series name -> GHCN station (assumed settlement site)
PRECIP_STATIONS = {
    "NYC": "USW00094728", "NY": "USW00094728", "CHI": "USW00014819", "AUS": "USW00013904", "DAL": "USW00003927",
    "DEN": "USW00003017", "HOU": "USW00012960", "LAX": "USW00023174", "MIA": "USW00012839", "SEA": "USW00024233",
    "SFO": "USW00023234", "BOS": "USW00014739", "DC": "USW00013743", "DET": "USW00094847", "PHIL": "USW00013739",
    "SLC": "USW00024127", "JACW": "USW00024166",
}


# Non-US GEFS points (lat, lon, weight): Europe gas demand (EU), NE Asia LNG demand (ASIA), crop areas
WORLD_POINTS = {
    "LON": (51.5, -0.1, 9.0, "EU"), "PAR": (48.9, 2.4, 11.0, "EU"), "BER": (52.5, 13.4, 6.0, "EU"),
    "AMS": (52.4, 4.9, 4.0, "EU"), "MIL": (45.5, 9.2, 7.0, "EU"), "MAD": (40.4, -3.7, 6.5, "EU"),
    "WAW": (52.2, 21.0, 3.0, "EU"), "VIE": (48.2, 16.4, 2.5, "EU"), "FRA": (50.1, 8.7, 5.0, "EU"),
    "TYO": (35.7, 139.7, 37.0, "ASIA"), "SEL": (37.6, 127.0, 25.0, "ASIA"), "SHA": (31.2, 121.5, 25.0, "ASIA"),
    "PEK": (39.9, 116.4, 20.0, "ASIA"),
    "MINAS": (-21.5, -45.5, 1.0, "COFFEE"), "MATOG": (-13.0, -56.0, 1.0, "SOYBR"), "ROSAR": (-33.0, -60.7, 1.0, "SOYAR"),
    "ABIDJ": (6.8, -5.3, 1.0, "COCOA"), "KUMAS": (6.7, -1.6, 1.0, "COCOA"),
    "KRASN": (45.0, 39.0, 1.0, "WHEAT"), "KYIV": (50.4, 30.5, 1.0, "WHEAT"), "WICHI": (37.7, -97.3, 1.0, "WHEAT"),
    "REGIN": (50.4, -104.6, 1.0, "WHEAT"),
}


def ghcn(station: str, refresh: bool = False) -> pd.DataFrame:
    """Daily TMAX, TMIN (deg F), PRCP and SNOW (inches) for one GHCN station."""
    path = cache_dir("weather/ghcn") / f"{station}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = pd.read_csv(io.BytesIO(requests.get(GHCN.format(station), timeout=120).content),
                      usecols=["DATE", "ELEMENT", "DATA_VALUE", "Q_FLAG"], dtype={"Q_FLAG": str})
    raw = raw[raw["DATE"] >= 19800101]
    # by_station files stopped updating in Feb 2025; the by_year files are current
    recent = [_by_year(y) for y in range(2025, pd.Timestamp.now().year + 1)]
    recent = [r[r["ID"] == station] for r in recent]
    raw = pd.concat([raw[raw["DATE"] < 20250101]] + [r[raw.columns] for r in recent], ignore_index=True)
    raw = raw[raw["ELEMENT"].isin(["TMAX", "TMIN", "PRCP", "SNOW"]) & raw["Q_FLAG"].isna()]
    out = raw.pivot_table(index="DATE", columns="ELEMENT", values="DATA_VALUE", aggfunc="first")
    out.index = pd.to_datetime(out.index.astype(str))
    out = out.loc["1980":]
    for c in ("TMAX", "TMIN"):
        out[c] = out[c] / 10 * 9 / 5 + 32
    out["PRCP"] = out["PRCP"] / 254
    out["SNOW"] = out["SNOW"] / 25.4 if "SNOW" in out else np.nan
    out.to_parquet(path)
    return out


def _by_year(year: int) -> pd.DataFrame:
    """TMAX/TMIN/PRCP/SNOW rows of every `CITIES` + `PRECIP_STATIONS` station from one by_year file (cached)."""
    import gzip

    path = cache_dir("weather/ghcn") / f"by_year_{year}.parquet"
    fresh = path.exists() and (year < pd.Timestamp.now().year
                               or pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s") < pd.Timedelta(days=1))
    if fresh:
        return pd.read_parquet(path)
    ids = {v[3].encode() for v in CITIES.values()} | {v.encode() for v in PRECIP_STATIONS.values()}
    keep = []
    with requests.get(GHCN_YEAR.format(year), stream=True, timeout=600) as r:
        with gzip.GzipFile(fileobj=r.raw) as f:
            for line in f:
                if line[:11] in ids and line[21:25] in (b"TMAX", b"TMIN", b"PRCP", b"SNOW"):
                    keep.append(line.decode())
    cols = ["ID", "DATE", "ELEMENT", "DATA_VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]
    out = pd.read_csv(io.StringIO("".join(keep)), names=cols, dtype={"Q_FLAG": str, "M_FLAG": str, "S_FLAG": str, "OBS_TIME": str})
    out.to_parquet(path)
    return out


def actuals(cities=None, field: str = "TMEAN") -> pd.DataFrame:
    """Wide frame (date x city). field: TMEAN (avg of max/min), TMAX, TMIN, PRCP."""
    cities = cities or list(CITIES)
    cols = {}
    for c in cities:
        d = ghcn(CITIES[c][3])
        cols[c] = (d["TMAX"] + d["TMIN"]) / 2 if field == "TMEAN" else d[field]
    return pd.DataFrame(cols).sort_index()


def normals(frame: pd.DataFrame, start="1991", end="2020", smooth: int = 15) -> pd.DataFrame:
    """Day-of-year climatology per column, centred moving average of `smooth` days."""
    base = frame.loc[start:end]
    doy = base.groupby(base.index.dayofyear).mean()
    doy = doy.reindex(range(1, 367)).interpolate()
    wrap = pd.concat([doy.iloc[-smooth:], doy, doy.iloc[:smooth]])
    sm = wrap.rolling(smooth, center=True, min_periods=1).mean().iloc[smooth:-smooth]
    sm.index = range(1, 367)
    return sm


def anomaly(frame: pd.DataFrame, norm: pd.DataFrame | None = None) -> pd.DataFrame:
    norm = normals(frame) if norm is None else norm
    return frame - norm.loc[frame.index.dayofyear].to_numpy()


def degree_days(tmean, base: float = 65.0):
    """(HDD, CDD) from daily mean temperature."""
    return np.maximum(base - tmean, 0), np.maximum(tmean - base, 0)


def weights(region: str | None = None) -> pd.Series:
    w = pd.Series({k: v[2] for k, v in CITIES.items() if (region is None or v[4] == region)})
    w = w[w > 0] if region != "corn" else w + 1
    return w / w.sum()


def demand_index(tmean: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    """Population-weighted HDD, CDD and their sum for each row of a date x city frame."""
    w = weights(region)
    t = tmean[w.index]
    hdd, cdd = degree_days(t)
    ok = t.notna()
    wsum = ok.mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    h = hdd.mul(w, axis=1).sum(axis=1) / wsum
    c = cdd.mul(w, axis=1).sum(axis=1) / wsum
    return pd.DataFrame({"hdd": h, "cdd": c, "dd": h + c})


# ---------------- GEFS ----------------

def _grid_index(points=None):
    """Flat index into the 0.5 deg global grid (lat 90..-90, lon 0..359.5) per point."""
    out = {}
    for k, (lat, lon, *_rest) in (points or CITIES).items():
        i = int(round((90 - lat) / 0.5))
        j = int(round((lon % 360) / 0.5)) % 720
        out[k] = i * 720 + j
    return out


def _tmp2m(sess, url: str) -> bytes | None:
    r = sess.get(url + ".idx", timeout=60)
    if r.status_code != 200:
        return None
    lines = r.text.splitlines()
    for i, line in enumerate(lines):
        if ":TMP:2 m above ground:" in line:
            a = int(line.split(":")[1])
            b = int(lines[i + 1].split(":")[1]) - 1
            return sess.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120).content
    return None


def gefs_run(day, run: str = "12", hours=range(6, 385, 12), sess=None, workers: int = 16, points=None) -> pd.DataFrame | None:
    """City 2 m temperatures (deg F) from one ensemble-mean run.

    Returns rows (valid time UTC) x cities, or None if the run is missing.
    Default hours give 06z and 18z valid times for a 12z run (near daily min/max).
    """
    import eccodes

    day = pd.Timestamp(day)
    sess = sess or requests.Session()
    base = f"{GEFS}/gefs.{day:%Y%m%d}/{run}/atmos/pgrb2ap5/geavg.t{run}z.pgrb2a.0p50"
    with cf.ThreadPoolExecutor(workers) as ex:
        msgs = list(ex.map(lambda h: _tmp2m(sess, f"{base}.f{h:03d}"), hours))
    if sum(m is not None for m in msgs) < len(msgs) * 0.8:
        return None
    gi = _grid_index(points)
    keys, idx = list(gi), np.array(list(gi.values()))
    rows, times = [], []
    t0 = day + pd.Timedelta(hours=int(run))
    for h, m in zip(hours, msgs):
        if m is None:
            continue
        g = eccodes.codes_new_from_message(m)
        v = eccodes.codes_get_values(g)
        eccodes.codes_release(g)
        rows.append((v[idx] - 273.15) * 9 / 5 + 32)
        times.append(t0 + pd.Timedelta(hours=h))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name="valid"), columns=keys)


def gefs_panel(start, end, run: str = "12", refresh: bool = False, world: bool = False) -> pd.DataFrame:
    """Long frame (init, valid, city columns) for every run day in [start, end]; cached by month.

    world=True fetches `WORLD_POINTS` instead of the US `CITIES` (separate cache).
    """
    folder = cache_dir(f"weather/gefs{run}z" + ("_world" if world else ""))
    sess = requests.Session()
    parts = []
    for month in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        path = Path(folder) / f"{month}.parquet"
        done = month < pd.Timestamp.now().to_period("M")
        if path.exists() and not refresh:
            parts.append(pd.read_parquet(path))
            continue
        rows = []
        for day in pd.date_range(month.start_time, month.end_time, freq="D"):
            if day > pd.Timestamp.now() - pd.Timedelta(days=1):
                break
            try:
                f = gefs_run(day, run, sess=sess, points=WORLD_POINTS if world else None)
            except Exception:
                f = None
            if f is not None:
                f = f.reset_index()
                f.insert(0, "init", day + pd.Timedelta(hours=int(run)))
                rows.append(f)
        if rows:
            p = pd.concat(rows, ignore_index=True)
            if done:
                p.to_parquet(path)
            parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    return out[(out["init"] >= pd.Timestamp(start)) & (out["init"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]


def gefs_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily mean (avg of the 06z and 18z valid values) per init and US local date.

    Valid times are mapped to a local calendar day by subtracting 6 h (06z -> early
    morning of the same date, 18z -> early afternoon). Adds `lead` in days.
    """
    p = panel.copy()
    p["date"] = (p["valid"] - pd.Timedelta(hours=6)).dt.normalize()
    cities = [c for c in p.columns if c not in ("init", "valid", "date")]
    d = p.groupby(["init", "date"])[cities].mean().reset_index()
    d["lead"] = (d["date"] - d["init"].dt.normalize()).dt.days
    return d


# ---------------- hourly observations and hurricanes ----------------

ISD = "https://noaa-isd-pds.s3.amazonaws.com/data/{year}/{station}-{year}.gz"
# Kalshi daily-high settlement stations: city -> (ISD USAF-WBAN, UTC offset of local standard time)
KALSHI_STATIONS = {
    "NY": ("725053-94728", -5), "CHI": ("725340-14819", -6), "MIA": ("722020-12839", -5),
    "AUS": ("722540-13904", -6), "PHIL": ("724080-13739", -5), "DEN": ("725650-03017", -7),
    "LAX": ("722950-23174", -8),
}


def parse_isd(lines) -> pd.DataFrame:
    """METAR (FM-15) and SPECI (FM-16) air temperatures from ISD fixed-width lines: ts (UTC), temp_f."""
    rows = []
    for line in lines:
        if line[41:46] not in ("FM-15", "FM-16"):
            continue
        t = line[87:92]
        if t == "+9999" or line[92] not in "0145":
            continue
        rows.append((line[15:27], int(t) / 10 * 9 / 5 + 32))
    d = pd.DataFrame(rows, columns=["ts", "temp_f"])
    d["ts"] = pd.to_datetime(d["ts"], format="%Y%m%d%H%M", utc=True)
    return d.sort_values("ts").reset_index(drop=True)


def isd_hourly(station: str, years) -> pd.DataFrame:
    """Hourly + special observations for one ISD station (the S3 copy lags a few months)."""
    import gzip

    parts = []
    for y in years:
        path = cache_dir("weather/isd") / f"{station}-{y}.parquet"
        if path.exists() and y < pd.Timestamp.now().year - 1:
            parts.append(pd.read_parquet(path))
            continue
        r = requests.get(ISD.format(year=y, station=station), timeout=120)
        if r.status_code != 200 or r.content[:2] != b"\x1f\x8b":
            continue
        d = parse_isd(gzip.decompress(r.content).decode("latin-1").splitlines())
        d.to_parquet(path)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "temp_f"])


IBTRACS = "https://huggingface.co/datasets/glavvrach79/hurricanes/resolve/main/ibtracs.since1980.list.v04r01.csv"


def us_landfalls(path: str = IBTRACS, since: str = "2000-01-01") -> pd.DataFrame:
    """First US landfall time of each North Atlantic hurricane (IBTrACS, 1980 - Oct 2025)."""
    ib = pd.read_csv(path, skiprows=[1], low_memory=False, keep_default_na=False,
                     usecols=["SID", "BASIN", "NAME", "ISO_TIME", "LAT", "LON", "DIST2LAND", "USA_SSHS"])
    ib = ib[ib["BASIN"] == "NA"].copy()
    for c in ["LAT", "LON", "DIST2LAND", "USA_SSHS"]:
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    ib["t"] = pd.to_datetime(ib["ISO_TIME"])
    us = ib["LAT"].between(24, 45) & ib["LON"].between(-98, -66)
    hit = ib[us & (ib["DIST2LAND"] == 0) & (ib["USA_SSHS"] >= 1)]
    out = hit.groupby("SID").agg(name=("NAME", "first"), t=("t", "min"), cat=("USA_SSHS", "max"))
    return out[out["t"] >= since].sort_values("t")
