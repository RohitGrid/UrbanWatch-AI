import pandas as pd
import numpy as np
import ast
import json

INPUT_FILE = "police.csv"
OUTPUT_FILE = "junction_csi_clean.csv"

MIN_VIOLATIONS = 5
LOG_TRANSFORM_METRICS = {"violation_count"}
CLIP_PERCENTILES = {
    "violation_count": (0, 100),  
    "repeat_rate": (0, 100),      
}
DEFAULT_CLIP_PERCENTILES = (5, 95)

WEIGHTS = {
    "density": 0.40,
    "repeat": 0.25,
    "peak": 0.20,
    "road": 0.15,
}

PEAK_HOUR_RANGES = [(8, 11), (17, 21)]
SEVERITY_WEIGHTS = {
    "WRONG PARKING": 6,
    "PARKING NEAR ROAD CROSSING": 10,
    "PARKING IN A MAIN ROAD": 10,
    "DOUBLE PARKING": 9,
    "PARKING ON FOOTPATH": 8,
    "NO PARKING": 5,
}
DEFAULT_SEVERITY = 6
NAME_MAX_LEN = 60
EPS = 1e-9

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

df = pd.read_csv(INPUT_FILE)
df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
df = df.dropna(subset=["created_datetime"])
df["hour"] = df["created_datetime"].dt.hour

def parse_violations(val):
    if pd.isna(val): return []
    s = str(val).strip()
    if not s: return []
    try:
        result = json.loads(s)
        if isinstance(result, list): return result
    except (json.JSONDecodeError, TypeError): pass
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list): return result
    except (ValueError, SyntaxError): pass
    return []

df["violation_list"] = df["violation_type"].apply(parse_violations)

if "validation_status" in df.columns:
    df = df[df["validation_status"] == "approved"].copy()

def get_name(row):
    j = str(row.get("junction_name", "")).strip()
    if j and j.lower() not in {"no junction", "nan", "", "none"}:
        return j[:NAME_MAX_LEN]
    for col in ["address", "location", "street"]:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            return val.split(",")[0].strip()[:NAME_MAX_LEN]
    return "Unknown"

df["junction_clean"] = df.apply(get_name, axis=1)
df = df[df["junction_clean"] != "Unknown"]

density = df.groupby("junction_clean")["id"].count().rename("violation_count")

repeat = df.groupby("junction_clean").agg(
    violations=("id", "count"),
    unique_vehicles=("vehicle_number", "nunique"),
)
repeat["repeat_rate"] = repeat["violations"] / (repeat["unique_vehicles"] + EPS)

peak_mask = df["hour"].apply(lambda h: any(lo <= h <= hi for lo, hi in PEAK_HOUR_RANGES))
peak = df.assign(is_peak=peak_mask).groupby("junction_clean")["is_peak"].mean() * 100
peak = peak.rename("peak_rate")

def avg_severity(series):
    vals = []
    for lst in series:
        if isinstance(lst, list):
            for v in lst:
                vals.append(SEVERITY_WEIGHTS.get(str(v).strip().upper(), DEFAULT_SEVERITY))
    return np.mean(vals) if vals else np.nan

road = df.groupby("junction_clean")["violation_list"].apply(avg_severity).rename("road_weight")

out = pd.concat([density, repeat["repeat_rate"], peak, road], axis=1).reset_index()
out = out[out["violation_count"] >= MIN_VIOLATIONS].copy()
out["road_weight"] = out["road_weight"].fillna(out["road_weight"].mean())

def robust_minmax(series, metric_name, log_transform=False):
    work = np.log1p(series) if log_transform else series.astype(float)
    pct_bounds = CLIP_PERCENTILES.get(metric_name, DEFAULT_CLIP_PERCENTILES)
    lo = np.percentile(work, pct_bounds[0])
    hi = np.percentile(work, pct_bounds[1])
    if abs(hi - lo) < EPS:
        return pd.Series(50.0, index=series.index)
    clipped = work.clip(lower=lo, upper=hi)
    return ((clipped - lo) / (hi - lo)) * 100

out["density_score"] = robust_minmax(out["violation_count"], "violation_count", "violation_count" in LOG_TRANSFORM_METRICS).round(1)
out["repeat_score"] = robust_minmax(out["repeat_rate"], "repeat_rate", "repeat_rate" in LOG_TRANSFORM_METRICS).round(1)
out["peak_score"] = robust_minmax(out["peak_rate"], "peak_rate", "peak_rate" in LOG_TRANSFORM_METRICS).round(1)
out["road_score"] = robust_minmax(out["road_weight"], "road_weight", "road_weight" in LOG_TRANSFORM_METRICS).round(1)

out["CSI"] = (
    WEIGHTS["density"] * out["density_score"]
    + WEIGHTS["repeat"] * out["repeat_score"]
    + WEIGHTS["peak"] * out["peak_score"]
    + WEIGHTS["road"] * out["road_score"]
).round(1)

out = out.sort_values("CSI", ascending=False).reset_index(drop=True)
out.index += 1
out.index.name = "rank"
out = out.reset_index()

cols = [
    "rank", "junction_clean", "violation_count", "repeat_rate", "peak_rate", "road_weight",
    "density_score", "repeat_score", "peak_score", "road_score", "CSI",
]
out = out[cols]
out.to_csv(OUTPUT_FILE, index=False)

print(out.head(20).to_string(index=False))
