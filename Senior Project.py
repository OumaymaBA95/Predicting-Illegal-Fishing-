import pandas as pd
import numpy as np
import os
import glob
import re
import math
import csv
import json
import sys
import zipfile
import io
# ML libraries imported for future use (not used in current data processing step)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# Project directory (used for relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow locally vendored dependencies (e.g., ./vendor/pycountry)
VENDOR_DIR = os.path.join(SCRIPT_DIR, "vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

# Optional: country code mapping if available
try:
    import pycountry  # type: ignore
except Exception:
    pycountry = None

def is_in_target_regions(df):
    """Filter for Gulf of Mexico OR Mediterranean Sea"""
    gulf = (df['cell_ll_lat'].between(20, 30)) & (df['cell_ll_lon'].between(-98, -80))
    mediterranean = (df['cell_ll_lat'].between(30, 46)) & (df['cell_ll_lon'].between(-6, 36))
    return gulf | mediterranean

def gulf_mask(df):
    return (df['cell_ll_lat'].between(20, 30)) & (df['cell_ll_lon'].between(-98, -80))

def mediterranean_mask(df):
    return (df['cell_ll_lat'].between(30, 46)) & (df['cell_ll_lon'].between(-6, 36))

def normalize_mmsi(value) -> str | None:
    """
    Normalize MMSI to digits-only string.
    Handles values like '123456789.0', whitespace, and mixed types.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    digits = re.sub(r"\D", "", s)
    return digits if digits else None

def normalize_mmsi_series(series: pd.Series) -> pd.Series:
    """
    Vectorized MMSI normalization for pandas Series (digits-only).
    Returns a Series of strings or <NA>.
    """
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower().eq("nan"))
    s = s.str.replace(r"\D", "", regex=True)
    s = s.replace("", pd.NA)
    return s

def normalize_imo(value) -> str | None:
    """
    Normalize IMO to digits-only string (usually 7 digits).
    Note: GFW daily effort files do NOT include IMO, so IMO matching requires
    a separate vessel registry mapping MMSI -> IMO.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    digits = re.sub(r"\D", "", s)
    if not digits or digits == "0":
        return None
    return digits

def normalize_flag(value) -> str | None:
    """
    Normalize country/flag code to uppercase ISO3 when possible.
    Handles values like "USA", "unknown-CHN", or "flag-usa".
    Returns None for blanks or NaN-like values.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper()
    if "-" in s:
        parts = [p for p in s.split("-") if p]
        if parts and len(parts[-1]) in (2, 3):
            s = parts[-1]
    if len(s) == 2 and pycountry is not None:
        try:
            return pycountry.countries.get(alpha_2=s).alpha_3  # type: ignore[union-attr]
        except Exception:
            return s
    return s

def normalize_geartype(value) -> str | None:
    """
    Normalize gear type string for consistent grouping.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s.lower()

def _country_to_iso3(name: str) -> str | None:
    """
    Convert a country name to ISO3 code using pycountry when available.
    Returns None if no mapping is found.
    """
    if not name:
        return None
    if pycountry is None:
        return None
    def _try_lookup(val: str) -> str | None:
        try:
            return pycountry.countries.lookup(val).alpha_3  # type: ignore[union-attr]
        except Exception:
            return None

    variants = [
        name,
        name.replace("&", "and"),
        name.replace("St.", "Saint"),
        name.replace("St ", "Saint "),
        name.replace("Côte", "Cote"),
        name.replace("São", "Sao"),
        name.replace("’", "'"),
        name.replace(",", ""),
        name.replace(".", ""),
    ]
    for v in variants:
        code = _try_lookup(v)
        if code:
            return code
    return None

def load_iuu_risk_index(path: str, *, year: int | None = None) -> pd.DataFrame | None:
    """
    Load IUU risk index and return a normalized table with columns:
    - flag (uppercase)
    - risk_score (numeric)
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    cols_lower = {c.lower(): c for c in df.columns}
    # Case 1: already has flag + risk score
    flag_col = (
        cols_lower.get("flag")
        or cols_lower.get("flag_code")
        or cols_lower.get("country_code")
        or cols_lower.get("iso3")
        or cols_lower.get("iso")
        or cols_lower.get("iso_code")
    )
    risk_col = (
        cols_lower.get("risk_score")
        or cols_lower.get("risk")
        or cols_lower.get("score")
        or cols_lower.get("risk_index")
        or cols_lower.get("iuu_risk")
    )
    if flag_col and risk_col:
        out = df[[flag_col, risk_col]].copy()
        out["flag"] = out[flag_col].map(normalize_flag)
        out["risk_score"] = pd.to_numeric(out[risk_col], errors="coerce")
        out = out.dropna(subset=["flag", "risk_score"])
        out = out[["flag", "risk_score"]].drop_duplicates("flag")
        return out

    # Case 2: indicator score table (Country + Year + Score)
    country_col = cols_lower.get("country")
    year_col = cols_lower.get("year")
    score_col = cols_lower.get("score")
    if country_col and year_col and score_col:
        tmp = df[[country_col, year_col, score_col]].copy()
        tmp["year"] = pd.to_numeric(tmp[year_col], errors="coerce")
        tmp["score"] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp.dropna(subset=[country_col, "year", "score"])
        if tmp.empty:
            return None
        years = sorted(set(int(y) for y in tmp["year"].dropna().unique()))
        if not years:
            return None
        if year is None:
            target_year = years[-1]
        else:
            le_years = [y for y in years if y <= int(year)]
            target_year = le_years[-1] if le_years else years[-1]
        tmp = tmp[tmp["year"] == target_year]
        if tmp.empty:
            return None
        agg = tmp.groupby(country_col)["score"].mean().reset_index()
        agg["flag"] = agg[country_col].map(_country_to_iso3)
        agg["risk_score"] = agg["score"]
        out = agg.dropna(subset=["flag", "risk_score"])
        out["flag"] = out["flag"].map(normalize_flag)
        out = out[["flag", "risk_score"]].drop_duplicates("flag")
        return out

    return None

def _bin_05(value: float) -> float:
    # 0.5° binning (≈55km latitude) for coarse spatial join
    return math.floor(float(value) / 0.5) * 0.5

def _bin_key(lat, lon) -> tuple[float, float] | None:
    try:
        return (_bin_05(lat), _bin_05(lon))
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    # Great-circle distance
    r = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def load_eez_boundary_points(path: str, *, stride: int = 20) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """
    Load EEZ boundary vertices from eez_boundaries_v12.csv (MultiLineString GeoJSON per row)
    and bin them into 1° grid cells for fast approximate nearest-distance queries.
    Returns bins[(lat_deg, lon_deg)] -> [(lat, lon), ...]
    """
    csv.field_size_limit(sys.maxsize)
    bins: dict[tuple[int, int], list[tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom = row.get("geometry")
            if not geom:
                continue
            try:
                g = json.loads(geom)
                coords = g.get("coordinates", [])
            except Exception:
                continue
            # MultiLineString -> list of lines -> list of [lon, lat]
            idx = 0
            for line in coords:
                for pt in line:
                    idx += 1
                    if stride > 1 and (idx % stride) != 0:
                        continue
                    try:
                        lon, lat = float(pt[0]), float(pt[1])
                    except Exception:
                        continue
                    key = (int(math.floor(lat)), int(math.floor(lon)))
                    bins.setdefault(key, []).append((lat, lon))
    return bins

def nearest_boundary_distance_km(
    lat: float,
    lon: float,
    bins: dict[tuple[int, int], list[tuple[float, float]]],
    *,
    max_radius_deg: int = 8,
) -> float | None:
    """
    Approximate distance to nearest EEZ boundary using binned vertices.
    Searches neighboring 1° bins out to max_radius_deg.
    """
    try:
        latf = float(lat)
        lonf = float(lon)
    except Exception:
        return None
    base = (int(math.floor(latf)), int(math.floor(lonf)))
    best = None
    for r in range(0, max_radius_deg + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                pts = bins.get((base[0] + di, base[1] + dj))
                if not pts:
                    continue
                for plat, plon in pts:
                    d = haversine_km(latf, lonf, plat, plon)
                    if best is None or d < best:
                        best = d
        # Early stop: if we found something very close, no need to expand further
        if best is not None and best <= 20:
            break
    return best

def iter_fishing_vessels_chunks(
    *,
    fishing_vessels_csv_path: str | None,
    fishing_vessels_zip_path: str | None,
    fishing_vessels_zip_inner: str = "fishing-vessels-v3.csv",
    chunksize: int = 200000,
):
    """
    Yield pandas chunks from fishing-vessels-v3.csv, either as a plain CSV file
    or from inside a ZIP (without extracting).
    """
    usecols = [
        "mmsi",
        "year",
        "flag_gfw",
        "flag_registry",
        "flag_ais",
        "vessel_class_gfw",
        "self_reported_fishing_vessel",
        "length_m_gfw",
        "engine_power_kw_gfw",
        "tonnage_gt_gfw",
        "registries_listed",
        "active_hours",
        "fishing_hours",
    ]

    if fishing_vessels_csv_path and os.path.exists(fishing_vessels_csv_path):
        yield from pd.read_csv(
            fishing_vessels_csv_path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )
        return

    if fishing_vessels_zip_path and os.path.exists(fishing_vessels_zip_path):
        with zipfile.ZipFile(fishing_vessels_zip_path) as z:
            with z.open(fishing_vessels_zip_inner) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                yield from pd.read_csv(text, usecols=usecols, chunksize=chunksize, low_memory=False)
        return

    return

def aggregate_by_mmsi(df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """
    Aggregate a (possibly filtered) chunk to per-MMSI sums + counts.
    We aggregate sums so we can compute true means later.
    """
    if df.empty:
        return pd.DataFrame()
    tmp = df[["mmsi", "fishing_hours", "cell_ll_lat", "cell_ll_lon"]].copy()
    tmp["cell_ll_lat_sq"] = pd.to_numeric(df["cell_ll_lat"], errors="coerce") ** 2
    tmp["cell_ll_lon_sq"] = pd.to_numeric(df["cell_ll_lon"], errors="coerce") ** 2
    tmp["hours"] = df["hours"] if "hours" in df.columns else df["fishing_hours"]
    tmp["_n_points"] = 1
    tmp["_fishing_days"] = (df["fishing_hours"] > 0).astype(int)
    agg = tmp.groupby("mmsi", dropna=True).agg(
        {
            "fishing_hours": "sum",
            "hours": "sum",
            "cell_ll_lat": "sum",
            "cell_ll_lon": "sum",
            "cell_ll_lat_sq": "sum",
            "cell_ll_lon_sq": "sum",
            "_n_points": "sum",
            "_fishing_days": "sum",
        }
    )
    return agg.rename(
        columns={
            "fishing_hours": f"{prefix}fishing_hours_sum",
            "hours": f"{prefix}hours_sum",
            "cell_ll_lat": f"{prefix}lat_sum",
            "cell_ll_lon": f"{prefix}lon_sum",
            "cell_ll_lat_sq": f"{prefix}lat_sq_sum",
            "cell_ll_lon_sq": f"{prefix}lon_sq_sum",
            "_n_points": f"{prefix}n_points",
            "_fishing_days": f"{prefix}fishing_days",
        }
    )

# Load Combined IUU List (known bad vessels – gold labels)
# Prefer a converted `combined_iuu_list.csv` if present, otherwise fall back.
def _read_iuu_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".xls", ".xlsx"]:
            xl = pd.ExcelFile(path)
            best_sheet = None
            for sheet in xl.sheet_names:
                preview = xl.parse(sheet, nrows=5)
                if preview.empty:
                    continue
                cols_norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in preview.columns}
                if "mmsi" in cols_norm or "imo" in cols_norm:
                    best_sheet = sheet
                    break
            if best_sheet is None and xl.sheet_names:
                best_sheet = xl.sheet_names[0]
            return xl.parse(best_sheet) if best_sheet is not None else pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        print(f"Skipping IUU list (read failed): {path} ({e})", flush=True)
        return pd.DataFrame()


def _load_iuu_file(path: str) -> pd.DataFrame:
    df = _read_iuu_table(path)
    if df.empty:
        return df
    # Normalize columns if present
    cols_norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    mmsi_col = cols_norm.get("mmsi")
    imo_col = cols_norm.get("imo")
    listed_col = cols_norm.get("currentlylisted")
    if not mmsi_col and not imo_col:
        print(f"Skipping IUU list (no MMSI/IMO columns): {path}", flush=True)
        return pd.DataFrame()

    if mmsi_col:
        df["mmsi"] = df[mmsi_col].map(normalize_mmsi)
    else:
        df["mmsi"] = pd.NA

    if imo_col:
        df["imo_norm"] = df[imo_col].map(normalize_imo)
    else:
        df["imo_norm"] = pd.NA

    if listed_col:
        df["is_iuu"] = df[listed_col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        df["is_iuu"] = 1

    return df


def _split_env_paths(value: str | None) -> list[str]:
    if not value:
        return []
    parts = []
    for chunk in value.split(os.pathsep):
        parts.extend([p.strip() for p in chunk.split(",") if p.strip()])
    return [p for p in parts if p]


def _discover_iuu_sources() -> list[str]:
    env_paths = _split_env_paths(os.getenv("IUU_LIST_PATH"))
    if env_paths:
        return [p for p in env_paths if os.path.exists(p)]

    candidates: list[str] = []
    candidates += [os.path.join(SCRIPT_DIR, "combined_iuu_list.csv")]
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "IUUList-*.csv"))
    candidates += glob.glob(
        os.path.join(
            SCRIPT_DIR,
            "Combined IUU Vessel List (Global RFMO Compilation) copy",
            "IUUList-*.xls*",
        )
    )
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "Updated CSV", "**", "IUUList-*.xls*"), recursive=True)
    return [p for p in candidates if os.path.exists(p)]

iuu_sources: list[str] = _discover_iuu_sources()
iuu_frames: list[pd.DataFrame] = []
loaded_sources: list[str] = []

for src in iuu_sources:
    frame = _load_iuu_file(src)
    if frame is not None and not frame.empty:
        iuu_frames.append(frame)
        loaded_sources.append(src)

if not iuu_frames:
    raise FileNotFoundError("No IUU list found. Set IUU_LIST_PATH or add combined_iuu_list.csv.")

print(f"Using IUU list sources: {loaded_sources}", flush=True)
iuu = pd.concat(iuu_frames, ignore_index=True)
iuu = iuu.drop_duplicates(subset=["mmsi", "imo_norm", "is_iuu"])

# Filter out rows with missing/invalid MMSI for MMSI-based lookup
iuu_mmsi = iuu[iuu["mmsi"].notna() & (iuu["mmsi"] != "") & (iuu["mmsi"] != "nan")]
iuu_dict = dict(zip(iuu_mmsi["mmsi"], iuu_mmsi["is_iuu"]))
print(f"Loaded {len(iuu_dict)} IUU vessels from MMSI entries", flush=True)

iuu_imo = iuu[iuu["imo_norm"].notna()]
iuu_imo_dict = dict(zip(iuu_imo["imo_norm"], iuu_imo["is_iuu"]))
print(f"Loaded {len(iuu_imo_dict)} IUU vessels with IMO numbers", flush=True)

# ---- World EEZ boundaries (optional feature) ----
# We use the boundary line dataset (MultiLineString) to compute a proxy feature:
# approximate distance (km) from a vessel mean position to the nearest EEZ boundary.
EEZ_BOUNDARIES_PATH = os.getenv("EEZ_BOUNDARIES_PATH")
if not EEZ_BOUNDARIES_PATH:
    candidates = []
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "Updated CSV", "*World EEZ Boundaries*",
                                        "World_EEZ_v12_20231025_gpkg", "eez_boundaries_v12.csv"))
    # Some folder names include a leading space
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "Updated CSV", " World EEZ Boundaries (v12)",
                                        "World_EEZ_v12_20231025_gpkg", "eez_boundaries_v12.csv"))
    EEZ_BOUNDARIES_PATH = candidates[0] if candidates else None

eez_bins: dict[tuple[int, int], list[tuple[float, float]]] = {}
if EEZ_BOUNDARIES_PATH and os.path.exists(EEZ_BOUNDARIES_PATH):
    try:
        stride = int(os.getenv("EEZ_POINT_STRIDE", "20"))
        eez_bins = load_eez_boundary_points(EEZ_BOUNDARIES_PATH, stride=stride)
        n_pts = sum(len(v) for v in eez_bins.values())
        print(
            f"Loaded EEZ boundary bins from {EEZ_BOUNDARIES_PATH}: {len(eez_bins)} bins, ~{n_pts} points (stride={stride})",
            flush=True,
        )
    except Exception as e:
        print(f"Could not load EEZ boundaries for features: {e}", flush=True)
        eez_bins = {}
else:
    print(
        "EEZ boundary file not found; skipping EEZ features. "
        "Set EEZ_BOUNDARIES_PATH to eez_boundaries_v12.csv to enable.",
        flush=True,
    )

# ---- xView3 SAR (manual) detection bins (optional feature) ----
# This is a lightweight proxy feature: count of xView3 manual detections near a vessel's mean position.
# It does NOT do full AIS-gap ±3h matching (GFW effort data here is day-aggregated), but it gives a
# defensible SAR-based risk signal for your report/demo.
sar_bins_all: dict[tuple[float, float], int] = {}
sar_bins_target: dict[tuple[float, float], int] = {}

XVIEW3_DIR = os.path.join(SCRIPT_DIR, "xView3-SAR Ship Detection Dataset")
XVIEW3_PUBLIC = os.path.join(XVIEW3_DIR, "public.csv")
XVIEW3_VALIDATION = os.path.join(XVIEW3_DIR, "validation.csv")

if os.path.exists(XVIEW3_PUBLIC) or os.path.exists(XVIEW3_VALIDATION):
    try:
        for path in [p for p in [XVIEW3_PUBLIC, XVIEW3_VALIDATION] if os.path.exists(p)]:
            df_sar = pd.read_csv(path, usecols=["detect_lat", "detect_lon", "source", "is_vessel"])
            # Focus on manual-labeled detections (closest proxy to SAR detections)
            src = df_sar["source"].fillna("").astype(str)
            is_manual = src.str.contains("manual", case=False, na=False)
            # Keep only rows explicitly marked as vessels when available
            # (some rows may have missing is_vessel)
            is_vessel = df_sar["is_vessel"]
            keep = is_manual & (is_vessel.isna() | (is_vessel == True))
            df_sar = df_sar.loc[keep, ["detect_lat", "detect_lon"]].dropna()

            for lat, lon in zip(df_sar["detect_lat"].tolist(), df_sar["detect_lon"].tolist()):
                key = _bin_key(lat, lon)
                if key is None:
                    continue
                sar_bins_all[key] = sar_bins_all.get(key, 0) + 1
                # also track bins that fall inside your project regions
                if (20 <= lat <= 30 and -98 <= lon <= -80) or (30 <= lat <= 46 and -6 <= lon <= 36):
                    sar_bins_target[key] = sar_bins_target.get(key, 0) + 1

        print(
            f"Loaded xView3 manual detections into bins: {len(sar_bins_all)} bins (all), {len(sar_bins_target)} bins (target regions)",
            flush=True,
        )
    except Exception as e:
        print(f"Could not load xView3 SAR CSVs for features: {e}", flush=True)
else:
    print("xView3 SAR CSVs not found; skipping SAR-based features.", flush=True)

# ---- GFW daily CSV selection ----
# By default this script will read *all* year folders it finds under DATA_ROOT, e.g.:
#   ./mmsi-daily-csvs-10-v3-2021/
#   ./mmsi-daily-csvs-10-v3-2022/
#   ...
#
# If you want to limit the run to a year range (faster), you can override without editing code:
#   GFW_START_YEAR=2020 GFW_END_YEAR=2024 python -u "Senior Project.py"
_ENV_START_YEAR = os.getenv("GFW_START_YEAR")
_ENV_END_YEAR = os.getenv("GFW_END_YEAR")
START_YEAR = int(_ENV_START_YEAR) if _ENV_START_YEAR else None
END_YEAR = int(_ENV_END_YEAR) if _ENV_END_YEAR else None
if START_YEAR is not None and END_YEAR is not None and START_YEAR > END_YEAR:
    START_YEAR, END_YEAR = END_YEAR, START_YEAR

def _pick_existing_dir(candidates: list[str]) -> str | None:
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None

# Prefer a data root that actually contains year folders.
def _has_year_folders(data_root: str) -> bool:
    if not data_root or not os.path.isdir(data_root):
        return False
    # Accept slight naming variations (e.g. "mmsi-daily-csvs-10-v3-2020 copy/")
    return len(glob.glob(os.path.join(data_root, "mmsi-daily-csvs-10-v3-20*"))) > 0

def _extract_year_from_filename(path: str) -> int | None:
    # Expected: mmsi-daily-csvs-10-v3-YYYY-MM-DD.csv
    base = os.path.basename(path)
    parts = base.split("-")
    if len(parts) < 7:
        return None
    year_str = parts[5]
    try:
        return int(year_str)
    except ValueError:
        return None

# Parent folder that contains the yearly subfolders (one folder per year), e.g.:
#   <DATA_ROOT>/mmsi-daily-csvs-10-v3-2024/
#
# Default behavior: use folders INSIDE this project (relative paths).
# Override if needed:
#   GFW_DATA_ROOT="/path/to/GFW Fishing Effort" python -u "Senior Project.py"
DEFAULT_DATA_ROOT = _pick_existing_dir(
    [
        # Most convenient: year folders directly in the project folder
        SCRIPT_DIR,

        # Original download layout (note the trailing space in your folder name)
        os.path.join(SCRIPT_DIR, "Updated CSV", "GFW Fishing Effort "),  # trailing space
        os.path.join(SCRIPT_DIR, "Updated CSV", "GFW Fishing Effort"),   # no trailing space

        # Alternate layout
        os.path.join(SCRIPT_DIR, "CSV", "GFW Fishing Effort "),
        os.path.join(SCRIPT_DIR, "CSV", "GFW Fishing Effort"),
    ]
)

DATA_ROOT = os.getenv("GFW_DATA_ROOT") or DEFAULT_DATA_ROOT
if not DATA_ROOT:
    raise FileNotFoundError(
        "Could not find the GFW Fishing Effort data folder inside this project.\n"
        "Expected something like:\n"
        "  ./Updated CSV/GFW Fishing Effort /mmsi-daily-csvs-10-v3-2024/\n"
        "Set the env var GFW_DATA_ROOT to point to the folder that contains the yearly folders."
    )

if not _has_year_folders(DATA_ROOT):
    raise FileNotFoundError(
        f"DATA_ROOT exists but doesn't contain any year folders: {DATA_ROOT}\n"
        "Expected folders like:\n"
        "  mmsi-daily-csvs-10-v3-2024/\n"
        "  mmsi-daily-csvs-10-v3-2023/\n"
        "If your data is elsewhere, set GFW_DATA_ROOT to the folder that contains those year folders."
    )

print(f"Using DATA_ROOT: {DATA_ROOT}", flush=True)

year_folders = sorted(glob.glob(os.path.join(DATA_ROOT, "mmsi-daily-csvs-10-v3-20*")))

csv_files: list[str] = []
for folder in year_folders:
    # Read all daily files inside each year folder (folder name may include "copy", etc.)
    csv_files.extend(glob.glob(os.path.join(folder, "mmsi-daily-csvs-10-v3-20??-*.csv")))

csv_files = sorted(csv_files)

# Optional year range filter (if env vars were provided)
if START_YEAR is not None or END_YEAR is not None:
    min_year = START_YEAR if START_YEAR is not None else -10**9
    max_year = END_YEAR if END_YEAR is not None else 10**9
    csv_files = [
        p for p in csv_files
        if (y := _extract_year_from_filename(p)) is not None and (min_year <= y <= max_year)
    ]

if START_YEAR is not None or END_YEAR is not None:
    start_s = str(START_YEAR) if START_YEAR is not None else "min"
    end_s = str(END_YEAR) if END_YEAR is not None else "max"
    print(f"Found {len(csv_files)} daily CSVs for years {start_s}-{end_s}", flush=True)
else:
    print(f"Found {len(csv_files)} daily CSVs across {len(year_folders)} year folder(s)", flush=True)

iuu_mmsi_set = set(iuu_dict.keys())
iuu_imo_set = set(iuu_imo_dict.keys())
matched_iuu_anywhere: set[str] = set()
matched_iuu_in_region: set[str] = set()

print("Build scope: ALL vessels (with Gulf+Med features)", flush=True)

# Optional: load vessel registry mapping MMSI -> IMO to enable IMO fallback matching.
# Common file names: mmsi_imo_registry.csv or fishing-vessels-v3.csv (if it contains IMO).
def _discover_registry_sources() -> list[str]:
    env_paths = _split_env_paths(os.getenv("GFW_VESSEL_REGISTRY"))
    if env_paths:
        return [p for p in env_paths if os.path.exists(p)]

    candidates: list[str] = []
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "mmsi_imo_registry.csv"))
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "needs_imo_lookup.csv"))
    candidates += glob.glob(os.path.join(SCRIPT_DIR, "fishing-vessels-v3*.csv"))
    candidates += glob.glob(
        os.path.join(SCRIPT_DIR, "fa615300-b670-11f0-b282-dfbcfd65324c copy", "fishing-vessels-v3*.csv")
    )
    candidates += glob.glob(
        os.path.join(SCRIPT_DIR, "Updated CSV", "**", "mmsi_imo_registry.csv"), recursive=True
    )
    return [p for p in candidates if os.path.exists(p)]


def _load_registry_file(path: str) -> pd.DataFrame:
    try:
        registry = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping vessel registry (read failed): {path} ({e})", flush=True)
        return pd.DataFrame()
    mmsi_col = next((c for c in ["mmsi", "MMSI"] if c in registry.columns), None)
    imo_col = next((c for c in ["imo", "IMO", "imo_number", "imo_num"] if c in registry.columns), None)
    if not mmsi_col or not imo_col:
        print(f"Skipping vessel registry (missing MMSI/IMO columns): {path}", flush=True)
        return pd.DataFrame()

    reg_mmsi = normalize_mmsi_series(registry[mmsi_col])
    reg_imo = registry[imo_col].map(normalize_imo)
    reg = pd.DataFrame({"mmsi": reg_mmsi, "imo": reg_imo}).dropna()
    reg = reg[reg["mmsi"].notna() & reg["imo"].notna()]
    if reg.empty:
        return reg
    reg = reg.drop_duplicates(subset=["mmsi"], keep="last")
    return reg


mmsi_to_imo: dict[str, str] = {}
registry_sources = _discover_registry_sources()
registry_frames: list[pd.DataFrame] = []
loaded_registry_sources: list[str] = []

for src in registry_sources:
    frame = _load_registry_file(src)
    if frame is not None and not frame.empty:
        registry_frames.append(frame)
        loaded_registry_sources.append(src)

if registry_frames:
    reg_all = pd.concat(registry_frames, ignore_index=True)
    reg_all = reg_all.dropna().drop_duplicates(subset=["mmsi"], keep="last")
    mmsi_to_imo = dict(zip(reg_all["mmsi"], reg_all["imo"]))
    print(
        f"Loaded vessel registry MMSI->IMO mapping: {len(mmsi_to_imo)} entries "
        f"from {len(loaded_registry_sources)} source(s)",
        flush=True,
    )
    # Helpful diagnostic: how many IUU IMOs exist in the registry?
    if iuu_imo_set:
        print(
            f"IUU IMOs present in vessel registry: {len(set(mmsi_to_imo.values()) & iuu_imo_set)}",
            flush=True,
        )
else:
    print(
        "IMO fallback matching is OFF (no vessel registry file found).\n"
        "To enable it, add `mmsi_imo_registry.csv` (or another MMSI/IMO registry) to this project or set:\n"
        "  GFW_VESSEL_REGISTRY=/path/to/registry.csv",
        flush=True,
    )

agg_all_list = []
agg_gulf_list = []
agg_med_list = []
gap_agg_list = []
gap_bins_list = []
cell_unique_counts_list = []
fishing_hours_max_list = []
fishing_hours_bins_list = []
fishing_month_bins_list = []
flag_counts_list = []
geartype_counts_list = []
date_ranges_list = []
active_days_counts: dict[str, int] = {}
fishing_days_counts: dict[str, int] = {}
for file_path in csv_files:
    try:
        daily_mmsi_seen: set[str] = set()
        daily_mmsi_fishing: set[str] = set()
        for chunk in pd.read_csv(file_path, chunksize=100000):
            # Normalize MMSI once per chunk (used for matching + grouping)
            chunk_mmsi = normalize_mmsi_series(chunk["mmsi"])
            chunk = chunk.copy()
            chunk["mmsi"] = chunk_mmsi
            chunk = chunk[chunk["mmsi"].notna()]
            if chunk.empty:
                continue

            # Diagnostic: do ANY IUU MMSIs appear in the raw GFW data?
            any_mask = chunk["mmsi"].isin(iuu_mmsi_set)
            if any_mask.any():
                matched_iuu_anywhere.update(chunk.loc[any_mask, "mmsi"].dropna().unique().tolist())

            # Region masks
            gulf = gulf_mask(chunk)
            med = mediterranean_mask(chunk)
            region = gulf | med

            # Diagnostic: IUU MMSIs that appear within the target regions
            region_any_mask = region & any_mask
            if region_any_mask.any():
                matched_iuu_in_region.update(chunk.loc[region_any_mask, "mmsi"].dropna().unique().tolist())

            # Aggregate all vessels (training scope)
            agg_all = aggregate_by_mmsi(chunk, prefix="all_")
            if not agg_all.empty:
                agg_all_list.append(agg_all)

            # Track active days and fishing days per MMSI (daily file = single date)
            daily_mmsi_seen.update(chunk["mmsi"].dropna().unique().tolist())
            if "fishing_hours" in chunk.columns:
                daily_mmsi_fishing.update(
                    chunk.loc[chunk["fishing_hours"] > 0, "mmsi"].dropna().unique().tolist()
                )

            # Aggregate AIS gap behavior if available
            if "hours_gap" in chunk.columns:
                tmp_gaps = chunk[["mmsi", "hours_gap"]].copy()
                tmp_gaps["hours_gap"] = pd.to_numeric(tmp_gaps["hours_gap"], errors="coerce")
                tmp_gaps = tmp_gaps[tmp_gaps["hours_gap"].notna()]
                if not tmp_gaps.empty:
                    tmp_gaps["_gap_n"] = 1
                    tmp_gaps["_gap_days"] = (tmp_gaps["hours_gap"] > 0).astype(int)
                    tmp_gaps["_gap_over_6h"] = (tmp_gaps["hours_gap"] >= 6).astype(int)
                    tmp_gaps["_gap_over_12h"] = (tmp_gaps["hours_gap"] >= 12).astype(int)
                    tmp_gaps["_gap_over_24h"] = (tmp_gaps["hours_gap"] >= 24).astype(int)
                    agg_gaps = tmp_gaps.groupby("mmsi", dropna=True).agg(
                        {
                            "hours_gap": ["sum", "max"],
                            "_gap_n": "sum",
                            "_gap_days": "sum",
                            "_gap_over_6h": "sum",
                            "_gap_over_12h": "sum",
                            "_gap_over_24h": "sum",
                        }
                    )
                    agg_gaps.columns = [
                        "gap_hours_sum",
                        "gap_hours_max",
                        "gap_n_points",
                        "gap_days",
                        "gap_over_6h_count",
                        "gap_over_12h_count",
                        "gap_over_24h_count",
                    ]
                    gap_agg_list.append(agg_gaps)

                    # Approximate gap-hour percentiles via bins (0-168 hours)
                    tmp_bins = tmp_gaps[["mmsi", "hours_gap"]].copy()
                    tmp_bins["gap_bin"] = tmp_bins["hours_gap"].clip(lower=0, upper=168).astype(int)
                    gap_counts = tmp_bins.groupby(["mmsi", "gap_bin"]).size().rename("count")
                    if not gap_counts.empty:
                        gap_bins_list.append(gap_counts)

            # Unique cell coverage per MMSI (sum of per-file unique cells)
            tmp_cells = chunk[["mmsi", "cell_ll_lat", "cell_ll_lon"]].dropna()
            if not tmp_cells.empty:
                cell_counts = (
                    tmp_cells.groupby(["mmsi", "cell_ll_lat", "cell_ll_lon"]).size()
                    .groupby("mmsi")
                    .size()
                    .rename("cell_unique_count_sum")
                )
                cell_unique_counts_list.append(cell_counts)

            # Max daily fishing hours per MMSI
            if "fishing_hours" in chunk.columns:
                tmp_max = (
                    chunk[["mmsi", "fishing_hours"]]
                    .dropna()
                    .groupby("mmsi")["fishing_hours"]
                    .max()
                    .rename("fishing_hours_max")
                )
                if not tmp_max.empty:
                    fishing_hours_max_list.append(tmp_max)

                # Approx median via 1-hour bins (0-24)
                tmp_fh = chunk[["mmsi", "fishing_hours"]].copy()
                tmp_fh["fishing_hours"] = pd.to_numeric(tmp_fh["fishing_hours"], errors="coerce").fillna(0.0)
                tmp_fh["fh_bin"] = tmp_fh["fishing_hours"].clip(lower=0, upper=24).astype(int)
                fh_counts = (
                    tmp_fh.groupby(["mmsi", "fh_bin"]).size().rename("count")
                )
                if not fh_counts.empty:
                    fishing_hours_bins_list.append(fh_counts)

            # Active date range per MMSI (min/max YYYY-MM-DD)
            if "date" in chunk.columns:
                tmp_dates = chunk[["mmsi", "date"]].copy()
                tmp_dates = tmp_dates[tmp_dates["date"].notna()]
                if not tmp_dates.empty:
                    tmp_dates["date"] = tmp_dates["date"].astype(str).str.slice(0, 10)
                    ranges = tmp_dates.groupby("mmsi")["date"].agg(["min", "max"])
                    ranges = ranges.rename(columns={"min": "date_min", "max": "date_max"})
                    date_ranges_list.append(ranges)

            # Capture most common flag + geartype per MMSI
            if "flag" in chunk.columns:
                tmp_flags = chunk[["mmsi", "flag"]].copy()
                tmp_flags["flag"] = tmp_flags["flag"].map(normalize_flag)
                tmp_flags = tmp_flags[tmp_flags["flag"].notna()]
                if not tmp_flags.empty:
                    flag_counts = tmp_flags.groupby(["mmsi", "flag"]).size().rename("n")
                    flag_counts_list.append(flag_counts)

            if "geartype" in chunk.columns:
                tmp_gears = chunk[["mmsi", "geartype"]].copy()
                tmp_gears["geartype"] = tmp_gears["geartype"].map(normalize_geartype)
                tmp_gears = tmp_gears[tmp_gears["geartype"].notna()]
                if not tmp_gears.empty:
                    gear_counts = tmp_gears.groupby(["mmsi", "geartype"]).size().rename("n")
                    geartype_counts_list.append(gear_counts)

            # Monthly fishing-hours distribution (seasonality proxy)
            if "date" in chunk.columns and "fishing_hours" in chunk.columns:
                tmp_month = chunk[["mmsi", "date", "fishing_hours"]].copy()
                tmp_month["date"] = pd.to_datetime(tmp_month["date"], errors="coerce")
                tmp_month = tmp_month[tmp_month["date"].notna()]
                if not tmp_month.empty:
                    tmp_month["month"] = tmp_month["date"].dt.month
                    tmp_month["fishing_hours"] = (
                        pd.to_numeric(tmp_month["fishing_hours"], errors="coerce").fillna(0.0)
                    )
                    month_counts = (
                        tmp_month.groupby(["mmsi", "month"])["fishing_hours"].sum().rename("fh_sum")
                    )
                    if not month_counts.empty:
                        fishing_month_bins_list.append(month_counts)

            # Aggregate region-specific subsets for features
            if gulf.any():
                agg_gulf = aggregate_by_mmsi(chunk[gulf], prefix="gulf_")
                if not agg_gulf.empty:
                    agg_gulf_list.append(agg_gulf)
            if med.any():
                agg_med = aggregate_by_mmsi(chunk[med], prefix="med_")
                if not agg_med.empty:
                    agg_med_list.append(agg_med)

        # Update per-day counts (each file represents one day)
        for m in daily_mmsi_seen:
            active_days_counts[m] = active_days_counts.get(m, 0) + 1
        for m in daily_mmsi_fishing:
            fishing_days_counts[m] = fishing_days_counts.get(m, 0) + 1
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}", flush=True)
        continue

# Aggregate all data after processing all files
if agg_all_list:
    gfw_all = pd.concat(agg_all_list).groupby("mmsi").sum(numeric_only=True)
    gfw_gulf = (
        pd.concat(agg_gulf_list).groupby("mmsi").sum(numeric_only=True)
        if agg_gulf_list
        else pd.DataFrame()
    )
    gfw_med = (
        pd.concat(agg_med_list).groupby("mmsi").sum(numeric_only=True)
        if agg_med_list
        else pd.DataFrame()
    )

    gfw_agg = gfw_all.join(gfw_gulf, how="left").join(gfw_med, how="left").fillna(0).reset_index()

    # Join AIS gap aggregates if available
    if gap_agg_list:
        gap_agg = pd.concat(gap_agg_list).groupby("mmsi").agg(
            {
                "gap_hours_sum": "sum",
                "gap_hours_max": "max",
                "gap_n_points": "sum",
                "gap_days": "sum",
                "gap_over_6h_count": "sum",
                "gap_over_12h_count": "sum",
                "gap_over_24h_count": "sum",
            }
        )
        gfw_agg = gfw_agg.merge(gap_agg.reset_index(), on="mmsi", how="left")
    else:
        gfw_agg["gap_hours_sum"] = pd.NA
        gfw_agg["gap_hours_max"] = pd.NA
        gfw_agg["gap_n_points"] = pd.NA
        gfw_agg["gap_days"] = pd.NA
        gfw_agg["gap_over_6h_count"] = pd.NA
        gfw_agg["gap_over_12h_count"] = pd.NA
        gfw_agg["gap_over_24h_count"] = pd.NA

    # Compute means from sums
    gfw_agg["all_lat_mean"] = gfw_agg["all_lat_sum"] / gfw_agg["all_n_points"].replace(0, pd.NA)
    gfw_agg["all_lon_mean"] = gfw_agg["all_lon_sum"] / gfw_agg["all_n_points"].replace(0, pd.NA)
    gfw_agg["gulf_lat_mean"] = gfw_agg["gulf_lat_sum"] / gfw_agg["gulf_n_points"].replace(0, pd.NA)
    gfw_agg["gulf_lon_mean"] = gfw_agg["gulf_lon_sum"] / gfw_agg["gulf_n_points"].replace(0, pd.NA)
    gfw_agg["med_lat_mean"] = gfw_agg["med_lat_sum"] / gfw_agg["med_n_points"].replace(0, pd.NA)
    gfw_agg["med_lon_mean"] = gfw_agg["med_lon_sum"] / gfw_agg["med_n_points"].replace(0, pd.NA)

    # Lat/Lon dispersion (variance) from sums of squares
    gfw_agg["all_lat_var"] = (
        gfw_agg["all_lat_sq_sum"] / gfw_agg["all_n_points"].replace(0, pd.NA)
        - gfw_agg["all_lat_mean"] ** 2
    )
    gfw_agg["all_lon_var"] = (
        gfw_agg["all_lon_sq_sum"] / gfw_agg["all_n_points"].replace(0, pd.NA)
        - gfw_agg["all_lon_mean"] ** 2
    )

    # Useful boolean / ratio features
    gfw_agg["in_gulf"] = (gfw_agg["gulf_n_points"] > 0).astype(int)
    gfw_agg["in_mediterranean"] = (gfw_agg["med_n_points"] > 0).astype(int)
    gfw_agg["in_target_region"] = ((gfw_agg["in_gulf"] == 1) | (gfw_agg["in_mediterranean"] == 1)).astype(int)
    gfw_agg["region_fishing_hours_sum"] = gfw_agg["gulf_fishing_hours_sum"] + gfw_agg["med_fishing_hours_sum"]
    gfw_agg["region_hours_fraction"] = (
        gfw_agg["region_fishing_hours_sum"] / gfw_agg["all_fishing_hours_sum"].replace(0, pd.NA)
    )
    gfw_agg["fishing_hours_fraction"] = (
        gfw_agg["all_fishing_hours_sum"] / gfw_agg["all_hours_sum"].replace(0, pd.NA)
    )
    gfw_agg["fishing_days_fraction"] = (
        gfw_agg["all_fishing_days"] / gfw_agg["all_n_points"].replace(0, pd.NA)
    )
    gfw_agg["gap_hours_mean"] = (
        gfw_agg["gap_hours_sum"] / gfw_agg["gap_n_points"].replace(0, pd.NA)
    )
    gfw_agg["gap_days_fraction"] = (
        gfw_agg["gap_days"] / gfw_agg["all_n_points"].replace(0, pd.NA)
    )
    gfw_agg["gap_over_6h_fraction"] = (
        gfw_agg["gap_over_6h_count"] / gfw_agg["gap_n_points"].replace(0, pd.NA)
    )
    gfw_agg["gap_over_12h_fraction"] = (
        gfw_agg["gap_over_12h_count"] / gfw_agg["gap_n_points"].replace(0, pd.NA)
    )
    gfw_agg["gap_over_24h_fraction"] = (
        gfw_agg["gap_over_24h_count"] / gfw_agg["gap_n_points"].replace(0, pd.NA)
    )

    # Join per-MMSI active and fishing day counts (per-day, not per-grid-cell)
    if active_days_counts:
        active_df = pd.DataFrame(
            {"mmsi": list(active_days_counts.keys()), "active_days": list(active_days_counts.values())}
        )
        gfw_agg = gfw_agg.merge(active_df, on="mmsi", how="left")
    else:
        gfw_agg["active_days"] = pd.NA

    if fishing_days_counts:
        fishing_df = pd.DataFrame(
            {"mmsi": list(fishing_days_counts.keys()), "fishing_days_by_date": list(fishing_days_counts.values())}
        )
        gfw_agg = gfw_agg.merge(fishing_df, on="mmsi", how="left")
    else:
        gfw_agg["fishing_days_by_date"] = pd.NA

    gfw_agg["fishing_days_fraction_by_date"] = (
        gfw_agg["fishing_days_by_date"] / gfw_agg["active_days"].replace(0, pd.NA)
    )

    # Join unique cell coverage (sum of per-file unique cells)
    if cell_unique_counts_list:
        cell_counts = pd.concat(cell_unique_counts_list).groupby("mmsi").sum().rename("cell_unique_count_sum")
        gfw_agg = gfw_agg.merge(cell_counts.reset_index(), on="mmsi", how="left")
        gfw_agg["cell_unique_count_mean_per_day"] = (
            gfw_agg["cell_unique_count_sum"] / gfw_agg["active_days"].replace(0, pd.NA)
        )
    else:
        gfw_agg["cell_unique_count_sum"] = pd.NA
        gfw_agg["cell_unique_count_mean_per_day"] = pd.NA

    # Join max daily fishing hours
    if fishing_hours_max_list:
        fh_max = pd.concat(fishing_hours_max_list).groupby("mmsi").max().rename("fishing_hours_max")
        gfw_agg = gfw_agg.merge(fh_max.reset_index(), on="mmsi", how="left")
    else:
        gfw_agg["fishing_hours_max"] = pd.NA

    # Approximate median and p95 daily fishing hours via binned counts
    if fishing_hours_bins_list:
        fh_bins = pd.concat(fishing_hours_bins_list).groupby(["mmsi", "fh_bin"]).sum().reset_index()
        fh_bins = fh_bins.sort_values(["mmsi", "fh_bin"])
        medians = {}
        p95s = {}
        for m, group in fh_bins.groupby("mmsi"):
            counts = group["count"].to_numpy()
            bins = group["fh_bin"].to_numpy()
            total = counts.sum()
            if total <= 0:
                continue
            cumsum = counts.cumsum()
            idx = int((cumsum >= (total / 2)).argmax())
            medians[m] = float(bins[idx])
            idx_p95 = int((cumsum >= (total * 0.95)).argmax())
            p95s[m] = float(bins[idx_p95])
        if medians:
            med_df = pd.DataFrame({"mmsi": list(medians.keys()), "fishing_hours_median": list(medians.values())})
            gfw_agg = gfw_agg.merge(med_df, on="mmsi", how="left")
        if p95s:
            p95_df = pd.DataFrame({"mmsi": list(p95s.keys()), "fishing_hours_p95": list(p95s.values())})
            gfw_agg = gfw_agg.merge(p95_df, on="mmsi", how="left")
    else:
        gfw_agg["fishing_hours_median"] = pd.NA
        gfw_agg["fishing_hours_p95"] = pd.NA

    # Approximate p95 AIS gap hours via binned counts
    if gap_bins_list:
        gap_bins = pd.concat(gap_bins_list).groupby(["mmsi", "gap_bin"]).sum().reset_index()
        gap_bins = gap_bins.sort_values(["mmsi", "gap_bin"])
        gap_p95 = {}
        for m, group in gap_bins.groupby("mmsi"):
            counts = group["count"].to_numpy()
            bins = group["gap_bin"].to_numpy()
            total = counts.sum()
            if total <= 0:
                continue
            cumsum = counts.cumsum()
            idx_p95 = int((cumsum >= (total * 0.95)).argmax())
            gap_p95[m] = float(bins[idx_p95])
        if gap_p95:
            gap_p95_df = pd.DataFrame({"mmsi": list(gap_p95.keys()), "gap_hours_p95": list(gap_p95.values())})
            gfw_agg = gfw_agg.merge(gap_p95_df, on="mmsi", how="left")
    else:
        gfw_agg["gap_hours_p95"] = pd.NA

    # Monthly seasonality of fishing hours
    if fishing_month_bins_list:
        month_bins = pd.concat(fishing_month_bins_list).groupby(["mmsi", "month"]).sum().reset_index()
        month_bins = month_bins.sort_values(["mmsi", "month"])
        seasonality = {}
        for m, group in month_bins.groupby("mmsi"):
            fh = group["fh_sum"].to_numpy(dtype=float)
            total = fh.sum()
            if total <= 0:
                continue
            p = fh / total
            entropy = float(-(p * np.log(p + 1e-9)).sum())
            peak = float(p.max())
            active_months = int((p > 0).sum())
            seasonality[m] = (entropy, peak, active_months)
        if seasonality:
            season_df = pd.DataFrame(
                [
                    {"mmsi": m, "fishing_month_entropy": e, "fishing_peak_month_fraction": p, "fishing_months_active": a}
                    for m, (e, p, a) in seasonality.items()
                ]
            )
            gfw_agg = gfw_agg.merge(season_df, on="mmsi", how="left")
    gfw_agg["region_fishing_days"] = gfw_agg["gulf_fishing_days"] + gfw_agg["med_fishing_days"]
    gfw_agg["region_n_points"] = gfw_agg["gulf_n_points"] + gfw_agg["med_n_points"]
    gfw_agg["region_fishing_days_fraction"] = (
        gfw_agg["region_fishing_days"] / gfw_agg["region_n_points"].replace(0, pd.NA)
    )

    # Global behavioral features (no region bias)
    gfw_agg["fishing_hours_per_point"] = (
        gfw_agg["all_fishing_hours_sum"] / gfw_agg["all_n_points"].replace(0, pd.NA)
    )
    gfw_agg["points_per_fishing_hour"] = (
        gfw_agg["all_n_points"] / gfw_agg["all_fishing_hours_sum"].replace(0, pd.NA)
    )
    gfw_agg["log_all_fishing_hours_sum"] = np.log1p(gfw_agg["all_fishing_hours_sum"])
    gfw_agg["log_all_n_points"] = np.log1p(gfw_agg["all_n_points"])
    gfw_agg["abs_all_lat_mean"] = gfw_agg["all_lat_mean"].abs()
    gfw_agg["abs_all_lon_mean"] = gfw_agg["all_lon_mean"].abs()

    # Most common flag and geartype per MMSI (from daily GFW data)
    if flag_counts_list:
        flag_counts = pd.concat(flag_counts_list).groupby(["mmsi", "flag"]).sum().reset_index()
        flag_mode = (
            flag_counts.sort_values(["mmsi", "n"], ascending=[True, False])
            .drop_duplicates("mmsi")
            .rename(columns={"flag": "flag_mode"})
            .loc[:, ["mmsi", "flag_mode"]]
        )
        gfw_agg = gfw_agg.merge(flag_mode, on="mmsi", how="left")

    if geartype_counts_list:
        gear_counts = pd.concat(geartype_counts_list).groupby(["mmsi", "geartype"]).sum().reset_index()
        gear_mode = (
            gear_counts.sort_values(["mmsi", "n"], ascending=[True, False])
            .drop_duplicates("mmsi")
            .rename(columns={"geartype": "geartype_mode"})
            .loc[:, ["mmsi", "geartype_mode"]]
        )
        gfw_agg = gfw_agg.merge(gear_mode, on="mmsi", how="left")

    # Flag/geartype diversity (proxy for switches)
    if flag_counts_list:
        flag_counts = pd.concat(flag_counts_list).groupby(["mmsi", "flag"]).sum().reset_index()
        flag_total = flag_counts.groupby("mmsi")["n"].sum().rename("flag_total_obs")
        flag_top = flag_counts.groupby("mmsi")["n"].max().rename("flag_top_obs")
        flag_unique = flag_counts.groupby("mmsi")["flag"].nunique().rename("flag_unique_count")
        flag_entropy = (
            flag_counts.assign(p=flag_counts["n"] / flag_counts.groupby("mmsi")["n"].transform("sum"))
            .groupby("mmsi")["p"]
            .apply(lambda s: float(-(s * np.log(s + 1e-9)).sum()))
            .rename("flag_entropy")
        )
        gfw_agg = gfw_agg.merge(flag_total.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(flag_top.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(flag_unique.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(flag_entropy.reset_index(), on="mmsi", how="left")
        gfw_agg["flag_top_fraction"] = gfw_agg["flag_top_obs"] / gfw_agg["flag_total_obs"].replace(0, pd.NA)

    if geartype_counts_list:
        gear_counts = pd.concat(geartype_counts_list).groupby(["mmsi", "geartype"]).sum().reset_index()
        gear_total = gear_counts.groupby("mmsi")["n"].sum().rename("geartype_total_obs")
        gear_top = gear_counts.groupby("mmsi")["n"].max().rename("geartype_top_obs")
        gear_unique = gear_counts.groupby("mmsi")["geartype"].nunique().rename("geartype_unique_count")
        gear_entropy = (
            gear_counts.assign(p=gear_counts["n"] / gear_counts.groupby("mmsi")["n"].transform("sum"))
            .groupby("mmsi")["p"]
            .apply(lambda s: float(-(s * np.log(s + 1e-9)).sum()))
            .rename("geartype_entropy")
        )
        gfw_agg = gfw_agg.merge(gear_total.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(gear_top.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(gear_unique.reset_index(), on="mmsi", how="left")
        gfw_agg = gfw_agg.merge(gear_entropy.reset_index(), on="mmsi", how="left")
        gfw_agg["geartype_top_fraction"] = (
            gfw_agg["geartype_top_obs"] / gfw_agg["geartype_total_obs"].replace(0, pd.NA)
        )

    # Active span (days between first and last AIS day seen)
    if date_ranges_list:
        date_ranges = (
            pd.concat(date_ranges_list)
            .groupby("mmsi")
            .agg({"date_min": "min", "date_max": "max"})
            .reset_index()
        )
        gfw_agg = gfw_agg.merge(date_ranges, on="mmsi", how="left")
        dmin = pd.to_datetime(gfw_agg["date_min"], errors="coerce")
        dmax = pd.to_datetime(gfw_agg["date_max"], errors="coerce")
        gfw_agg["active_span_days"] = (dmax - dmin).dt.days + 1
        gfw_agg["hours_per_active_day"] = (
            gfw_agg["all_hours_sum"] / gfw_agg["active_span_days"].replace(0, pd.NA)
        )
        gfw_agg["fishing_hours_per_active_day"] = (
            gfw_agg["all_fishing_hours_sum"] / gfw_agg["active_span_days"].replace(0, pd.NA)
        )

    # fishing-vessels-v3 metadata features (optional)
    # This file does NOT contain IMO, but adds useful numeric vessel attributes.
    FISHING_VESSELS_CSV = os.getenv("FISHING_VESSELS_PATH")
    FISHING_VESSELS_ZIP = os.getenv("FISHING_VESSELS_ZIP")

    if not FISHING_VESSELS_CSV and not FISHING_VESSELS_ZIP:
        # Local CSV (if you extracted it)
        candidates = []
        candidates += glob.glob(os.path.join(SCRIPT_DIR, "fishing-vessels-v3.csv"))
        candidates += glob.glob(os.path.join(SCRIPT_DIR, "Updated CSV", "**", "fishing-vessels-v3.csv"), recursive=True)
        FISHING_VESSELS_CSV = candidates[0] if candidates else None

    if not FISHING_VESSELS_ZIP:
        # Bulk bundle zip (you have this in ./CSV/)
        preferred = os.path.join(SCRIPT_DIR, "CSV", "fa615300-b670-11f0-b282-dfbcfd65324c.zip")
        if os.path.exists(preferred):
            FISHING_VESSELS_ZIP = preferred

    if FISHING_VESSELS_CSV or FISHING_VESSELS_ZIP:
        try:
            mmsi_set = set(gfw_agg["mmsi"].astype(str))
            min_year = START_YEAR if START_YEAR is not None else -10**9
            max_year = END_YEAR if END_YEAR is not None else 10**9

            fv_meta: dict[str, dict] = {}
            latest_year: dict[str, int] = {}

            for chunk in iter_fishing_vessels_chunks(
                fishing_vessels_csv_path=FISHING_VESSELS_CSV,
                fishing_vessels_zip_path=FISHING_VESSELS_ZIP,
            ):
                chunk["mmsi"] = normalize_mmsi_series(chunk["mmsi"])
                chunk = chunk[chunk["mmsi"].notna()]
                chunk = chunk[chunk["mmsi"].isin(mmsi_set)]
                if chunk.empty:
                    continue

                chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
                chunk = chunk[chunk["year"].notna()]
                chunk = chunk[(chunk["year"] >= min_year) & (chunk["year"] <= max_year)]
                if chunk.empty:
                    continue

                for row in chunk.itertuples(index=False):
                    m = str(row.mmsi)
                    y = int(row.year)
                    prev = latest_year.get(m)
                    if prev is None or y > prev:
                        latest_year[m] = y
                        fv_meta[m] = {
                            "fv_year": y,
                            "fv_flag_gfw": row.flag_gfw,
                            "fv_flag_registry": getattr(row, "flag_registry", pd.NA),
                            "fv_flag_ais": getattr(row, "flag_ais", pd.NA),
                            "fv_vessel_class_gfw": row.vessel_class_gfw,
                            "fv_self_reported_fishing_vessel": row.self_reported_fishing_vessel,
                            "fv_length_m_gfw": row.length_m_gfw,
                            "fv_engine_power_kw_gfw": row.engine_power_kw_gfw,
                            "fv_tonnage_gt_gfw": row.tonnage_gt_gfw,
                            "fv_registries_listed": row.registries_listed,
                            "fv_active_hours": row.active_hours,
                            "fv_fishing_hours": row.fishing_hours,
                        }

            if fv_meta:
                fv_df = (
                    pd.DataFrame.from_dict(fv_meta, orient="index")
                    .reset_index()
                    .rename(columns={"index": "mmsi"})
                )
                gfw_agg = gfw_agg.merge(fv_df, on="mmsi", how="left")

                if "fv_self_reported_fishing_vessel" in gfw_agg.columns:
                    gfw_agg["fv_self_reported_fishing_vessel"] = (
                        gfw_agg["fv_self_reported_fishing_vessel"]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1", "yes"])
                    ).astype(int)

                print(f"Joined fishing-vessels metadata for {len(fv_meta)} MMSIs", flush=True)
            else:
                print("No fishing-vessels metadata rows matched current MMSIs/years", flush=True)
        except Exception as e:
            print(f"Could not join fishing-vessels-v3 metadata: {e}", flush=True)
    else:
        print(
            "fishing-vessels-v3 metadata not found; skipping. "
            "Set FISHING_VESSELS_PATH or FISHING_VESSELS_ZIP to enable.",
            flush=True,
        )

    # IUU Risk Index feature (optional)
    # If present, we attach a country-level risk score by flag.
    RISK_INDEX_PATH = os.getenv("IUU_RISK_INDEX_PATH")
    if not RISK_INDEX_PATH:
        candidates = []
        candidates += glob.glob(os.path.join(SCRIPT_DIR, "iuu_risk_index*.csv"))
        candidates += glob.glob(os.path.join(SCRIPT_DIR, "*risk_index*.csv"))
        candidates += glob.glob(
            os.path.join(
                SCRIPT_DIR,
                "iuu_fishing_index_2025-data_and_disclaimer",
                "iuu_fishing_index_2019-2025_indicator_scores*.csv",
            )
        )
        candidates += glob.glob(os.path.join(SCRIPT_DIR, "Updated CSV", "**", "*risk_index*.csv"), recursive=True)
        RISK_INDEX_PATH = candidates[0] if candidates else None

    risk_year = END_YEAR if END_YEAR is not None else None
    risk_df = load_iuu_risk_index(RISK_INDEX_PATH, year=risk_year) if RISK_INDEX_PATH else None
    if risk_df is not None and not risk_df.empty:
        # Prefer registry/ais flags when available, else fall back to GFW daily mode
        fallback = pd.Series([pd.NA] * len(gfw_agg), index=gfw_agg.index)
        if "fv_flag_registry" in gfw_agg.columns:
            flag_best = (
                gfw_agg["fv_flag_registry"].combine_first(gfw_agg.get("fv_flag_ais", fallback))
                .combine_first(gfw_agg.get("fv_flag_gfw", fallback))
                .combine_first(gfw_agg.get("flag_mode", fallback))
            )
        else:
            flag_best = gfw_agg.get("flag_mode", fallback)

        gfw_agg["flag_best"] = flag_best.map(normalize_flag) if flag_best is not None else pd.NA
        gfw_agg = gfw_agg.merge(risk_df, left_on="flag_best", right_on="flag", how="left")
        gfw_agg = gfw_agg.rename(columns={"risk_score": "flag_risk_score"})
        gfw_agg = gfw_agg.drop(columns=["flag"])
        joined_count = int(gfw_agg["flag_risk_score"].notna().sum())
        if pycountry is None:
            print(
                "IUU risk index loaded from indicator scores, but pycountry is not installed; "
                "country-name to ISO3 mapping may be incomplete.",
                flush=True,
            )
        print(f"Joined IUU risk index for {joined_count} vessels", flush=True)
    else:
        gfw_agg["flag_risk_score"] = pd.NA
        print(
            "IUU risk index not found or unusable; skipping flag risk feature. "
            "Set IUU_RISK_INDEX_PATH to a risk index CSV to enable.",
            flush=True,
        )

    # SAR-bin features: how many xView3 manual detections exist near the vessel's mean position.
    def _lookup_bin_counts(lat, lon, bins: dict[tuple[float, float], int]) -> int:
        key = _bin_key(lat, lon)
        return int(bins.get(key, 0)) if key is not None else 0

    lat_all = pd.to_numeric(gfw_agg["all_lat_mean"], errors="coerce").fillna(0.0)
    lon_all = pd.to_numeric(gfw_agg["all_lon_mean"], errors="coerce").fillna(0.0)
    gfw_agg["sar_manual_bin_hits_at_all_mean"] = [
        _lookup_bin_counts(lat, lon, sar_bins_all)
        for lat, lon in zip(lat_all.to_numpy(), lon_all.to_numpy())
    ]

    lat_region = (
        gfw_agg["gulf_lat_mean"]
        .combine_first(gfw_agg["med_lat_mean"])
        .combine_first(gfw_agg["all_lat_mean"])
    )
    lon_region = (
        gfw_agg["gulf_lon_mean"]
        .combine_first(gfw_agg["med_lon_mean"])
        .combine_first(gfw_agg["all_lon_mean"])
    )
    lat_region = pd.to_numeric(lat_region, errors="coerce").fillna(0.0)
    lon_region = pd.to_numeric(lon_region, errors="coerce").fillna(0.0)
    gfw_agg["sar_manual_bin_hits_at_region_mean"] = [
        _lookup_bin_counts(lat, lon, sar_bins_target)
        for lat, lon in zip(lat_region.to_numpy(), lon_region.to_numpy())
    ]

    # EEZ boundary distance features (computed only for target-region vessels to keep runtime reasonable)
    if eez_bins:
        max_deg = int(os.getenv("EEZ_MAX_RADIUS_DEG", "8"))
        mask = gfw_agg["in_target_region"].fillna(0).astype(int) == 1
        d_all = [None] * len(gfw_agg)
        d_region = [None] * len(gfw_agg)
        for idx in gfw_agg.index[mask]:
            d_all[idx] = nearest_boundary_distance_km(
                float(lat_all.iloc[idx]),
                float(lon_all.iloc[idx]),
                eez_bins,
                max_radius_deg=max_deg,
            )
            d_region[idx] = nearest_boundary_distance_km(
                float(lat_region.iloc[idx]),
                float(lon_region.iloc[idx]),
                eez_bins,
                max_radius_deg=max_deg,
            )
        gfw_agg["eez_boundary_dist_km_all_mean"] = d_all
        gfw_agg["eez_boundary_dist_km_region_mean"] = d_region
        gfw_agg["near_eez_boundary_50km"] = (
            pd.to_numeric(gfw_agg["eez_boundary_dist_km_region_mean"], errors="coerce").fillna(1e9) <= 50
        ).astype(int)
    else:
        gfw_agg["eez_boundary_dist_km_all_mean"] = pd.NA
        gfw_agg["eez_boundary_dist_km_region_mean"] = pd.NA
        gfw_agg["near_eez_boundary_50km"] = 0
    
    # Add IUU labels using dictionary lookup 
    # Attach IMO if we have a registry mapping
    if mmsi_to_imo:
        gfw_agg["imo"] = gfw_agg["mmsi"].map(mmsi_to_imo)
    else:
        gfw_agg["imo"] = pd.NA

    label_by_mmsi = gfw_agg["mmsi"].map(iuu_dict)
    label_by_imo = gfw_agg["imo"].map(iuu_imo_dict) if mmsi_to_imo else pd.Series([pd.NA] * len(gfw_agg))

    matched_mmsi = int(label_by_mmsi.notna().sum())
    matched_imo = int(label_by_imo.notna().sum()) if mmsi_to_imo else 0

    label_mmsi_int = label_by_mmsi.fillna(0).astype(int)
    label_imo_int = label_by_imo.fillna(0).astype(int) if mmsi_to_imo else 0
    gfw_agg["is_iuu"] = ((label_mmsi_int + label_imo_int) > 0).astype(int)
    
    print(f"\nTotal vessels processed: {len(gfw_agg)}", flush=True)
    print(f"IUU vessels: {gfw_agg['is_iuu'].sum()}", flush=True)
    print(f"Matched by MMSI: {matched_mmsi}", flush=True)
    print(f"Matched by IMO (requires registry): {matched_imo}", flush=True)
    print(f"IUU MMSIs seen anywhere in GFW data: {len(matched_iuu_anywhere)}", flush=True)
    print(f"IUU MMSIs seen in target regions: {len(matched_iuu_in_region)}", flush=True)
    if matched_iuu_anywhere:
        sample_any = sorted(matched_iuu_anywhere)[:10]
        print(f"Sample IUU MMSIs found anywhere: {sample_any}", flush=True)
    if matched_iuu_in_region:
        sample_region = sorted(matched_iuu_in_region)[:10]
        print(f"Sample IUU MMSIs found in regions: {sample_region}", flush=True)

    # Save the aggregated dataset for modeling
    output_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    if START_YEAR is not None or END_YEAR is not None:
        start_s = str(START_YEAR) if START_YEAR is not None else "min"
        end_s = str(END_YEAR) if END_YEAR is not None else "max"
        out_name = f"vessel_features_{start_s}-{end_s}.csv"
    else:
        out_name = "vessel_features_all_years.csv"
    out_path = os.path.join(output_dir, out_name)
    gfw_agg.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path}", flush=True)
else:
    print("No data found matching the criteria!", flush=True)



