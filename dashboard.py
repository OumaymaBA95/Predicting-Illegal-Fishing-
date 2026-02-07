import os
import glob
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")


def _sorted_by_mtime_desc(paths: Iterable[str]) -> list[str]:
    paths = [p for p in paths if os.path.exists(p)]
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_DIR)
    except Exception:
        return path


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, usecols=usecols)


@st.cache_data(show_spinner=False)
def read_text_cached(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def csv_columns(path: str) -> list[str]:
    # fast schema-only read
    df0 = pd.read_csv(path, nrows=0, low_memory=False)
    return list(df0.columns)


def _pick_cols(existing: list[str], desired: list[str]) -> list[str]:
    have = set(existing)
    return [c for c in desired if c in have]


def _to_int01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _best_lat_lon(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Prefer Gulf/Med mean locations when present; else fallback to all_*.
    Mirrors logic used in feature building.
    """
    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return _to_float(df[name])
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    lat = _col("gulf_lat_mean").combine_first(_col("med_lat_mean")).combine_first(_col("all_lat_mean"))
    lon = _col("gulf_lon_mean").combine_first(_col("med_lon_mean")).combine_first(_col("all_lon_mean"))
    return lat, lon


def top_k_table(y_true: np.ndarray, y_score: np.ndarray, ks: list[int]) -> pd.DataFrame:
    """
    Build Top-K ranking stats for binary labels.
    Precision here is "known-positive precision" when labels are sparse.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0:
        return pd.DataFrame(columns=["K", "TP", "Precision", "Recall", "Lift"])

    base_rate = float(y_true.mean()) if n else 0.0
    order = np.argsort(-y_score)
    out = []
    n_pos = int(y_true.sum())
    for k in ks:
        if k <= 0:
            continue
        k_eff = min(int(k), n)
        top_idx = order[:k_eff]
        tp = int(y_true[top_idx].sum())
        precision = tp / k_eff if k_eff else 0.0
        recall = (tp / n_pos) if n_pos else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0
        out.append({"K": k_eff, "TP": tp, "Precision": precision, "Recall": recall, "Lift": lift})
    return pd.DataFrame(out)


st.set_page_config(
    page_title="IUU Risk Dashboard",
    page_icon="🛰️",
    layout="wide",
)

st.title("IUU Risk Dashboard")
st.caption("Interactive view of `outputs/` ranked lists, explainability signals, maps, and evaluation summaries.")

if not os.path.isdir(OUTPUTS_DIR):
    st.error(f"Missing outputs folder at `{_rel(OUTPUTS_DIR)}`. Run the pipeline first.")
    st.stop()


dataset_candidates = _sorted_by_mtime_desc(glob.glob(os.path.join(OUTPUTS_DIR, "vessel_features_*.csv")))
default_dataset = dataset_candidates[0] if dataset_candidates else os.path.join(OUTPUTS_DIR, "vessel_features_all_years.csv")

rank_candidates = _sorted_by_mtime_desc(glob.glob(os.path.join(OUTPUTS_DIR, "top_risky_vessels*.csv")))
default_rank = next((p for p in rank_candidates if os.path.basename(p) == "top_risky_vessels_pu.csv"), None) or (
    rank_candidates[0] if rank_candidates else ""
)

doc_candidates = [
    os.path.join(PROJECT_DIR, "MODEL_EVALUATION_RESULTS.md"),
    os.path.join(PROJECT_DIR, "MODEL_COMPARISON_SUMMARY.md"),
    os.path.join(PROJECT_DIR, "Cross validation.md"),
    os.path.join(PROJECT_DIR, "Cross-validation.html"),
    os.path.join(OUTPUTS_DIR, "pu_scoring_report.md"),
]
doc_candidates = [p for p in doc_candidates if os.path.exists(p)]


with st.sidebar:
    st.header("Inputs")
    dataset_path = st.selectbox(
        "Feature dataset (`outputs/vessel_features_*.csv`)",
        options=dataset_candidates or [default_dataset],
        index=0,
        format_func=_rel,
    )
    rank_path = st.selectbox(
        "Ranked list (`outputs/top_risky_vessels*.csv`)",
        options=rank_candidates or ([default_rank] if default_rank else []),
        index=0 if (rank_candidates or default_rank) else None,
        format_func=_rel,
    )

    st.divider()
    st.header("Display")
    top_n = st.slider("Top‑N rows", min_value=25, max_value=5000, value=250, step=25)
    show_docs = st.checkbox("Show evaluation write-ups", value=True)


dataset_cols = csv_columns(dataset_path)
base_cols = _pick_cols(
    dataset_cols,
    [
        "mmsi",
        "imo",
        "is_iuu",
        "in_target_region",
        "in_gulf",
        "in_mediterranean",
        "all_lat_mean",
        "all_lon_mean",
        "gulf_lat_mean",
        "gulf_lon_mean",
        "med_lat_mean",
        "med_lon_mean",
        "all_fishing_hours_sum",
        "region_fishing_hours_sum",
        "region_hours_fraction",
        "flag_risk_score",
        "near_eez_boundary_50km",
        "sar_manual_bin_hits_at_region_mean",
        "gap_hours_mean",
        "gap_days_fraction",
    ],
)

df = read_csv_cached(dataset_path, usecols=base_cols if base_cols else None)

if "mmsi" in df.columns:
    df["mmsi"] = df["mmsi"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

y = _to_int01(df["is_iuu"]) if "is_iuu" in df.columns else None


tab_overview, tab_ranked, tab_map, tab_diagnostics, tab_docs = st.tabs(
    ["Overview", "Ranked list", "Map", "Diagnostics", "Evaluation docs"]
)


if rank_path:
    rank_cols = csv_columns(rank_path)
    score_col = None
    for c in ["risk_score", "ensemble_score", "anomaly_score", "score"]:
        if c in rank_cols:
            score_col = c
            break

    risk_usecols = [c for c in ["mmsi", score_col] if c] if score_col else ["mmsi"]
    df_rank = read_csv_cached(rank_path, usecols=risk_usecols)
    df_rank["mmsi"] = df_rank["mmsi"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    merged = df_rank.merge(df, on="mmsi", how="left", suffixes=("", "_feat"))
    if "is_iuu" in merged.columns:
        merged["is_iuu"] = _to_int01(merged["is_iuu"])
    if "in_target_region" in merged.columns:
        merged["in_target_region"] = _to_int01(merged["in_target_region"])
    if "near_eez_boundary_50km" in merged.columns:
        merged["near_eez_boundary_50km"] = _to_int01(merged["near_eez_boundary_50km"])
    if "sar_manual_bin_hits_at_region_mean" in merged.columns:
        merged["sar_manual_bin_hits_at_region_mean"] = _to_float(merged["sar_manual_bin_hits_at_region_mean"])
    if "flag_risk_score" in merged.columns:
        merged["flag_risk_score"] = _to_float(merged["flag_risk_score"])
    if "region_hours_fraction" in merged.columns:
        merged["region_hours_fraction"] = _to_float(merged["region_hours_fraction"])
    if "all_fishing_hours_sum" in merged.columns:
        merged["all_fishing_hours_sum"] = _to_float(merged["all_fishing_hours_sum"])
    if score_col and score_col in merged.columns:
        merged[score_col] = _to_float(merged[score_col])

    if score_col and score_col in merged.columns:
        merged = merged.sort_values(score_col, ascending=False)

    # -------- Filters (sidebar, depends on merged/score range) --------
    with st.sidebar:
        st.divider()
        st.header("Filters")

        only_target = st.checkbox("Only Gulf+Med (`in_target_region = 1`)", value=False, disabled=("in_target_region" not in merged.columns))
        only_known_iuu = st.checkbox("Only known IUU (`is_iuu = 1`)", value=False, disabled=("is_iuu" not in merged.columns))

        near_eez_only = st.checkbox("Only near EEZ boundary", value=False, disabled=("near_eez_boundary_50km" not in merged.columns))
        min_sar = st.number_input(
            "Min SAR hits (region mean bin hits)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            disabled=("sar_manual_bin_hits_at_region_mean" not in merged.columns),
        )
        min_flag_risk = st.number_input(
            "Min flag risk score",
            min_value=0.0,
            value=0.0,
            step=0.5,
            disabled=("flag_risk_score" not in merged.columns),
        )
        min_region_frac = st.slider(
            "Min region hours fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            disabled=("region_hours_fraction" not in merged.columns),
        )
        require_location = st.checkbox("Only rows with mappable location", value=True)

        if score_col and score_col in merged.columns and merged[score_col].notna().any():
            smin = float(np.nanmin(merged[score_col].to_numpy()))
            smax = float(np.nanmax(merged[score_col].to_numpy()))
            if np.isfinite(smin) and np.isfinite(smax) and smin < smax:
                score_min = st.slider(f"Min {score_col}", min_value=smin, max_value=smax, value=smin, step=(smax - smin) / 200.0)
            else:
                score_min = smin
        else:
            score_min = None

        st.caption("Tip: start with Top‑1000 and tighten filters to build a defendable shortlist.")

    filtered = merged.copy()

    if only_target and "in_target_region" in filtered.columns:
        filtered = filtered[filtered["in_target_region"].fillna(0).astype(int) == 1]
    if only_known_iuu and "is_iuu" in filtered.columns:
        filtered = filtered[filtered["is_iuu"].fillna(0).astype(int) == 1]
    if near_eez_only and "near_eez_boundary_50km" in filtered.columns:
        filtered = filtered[filtered["near_eez_boundary_50km"].fillna(0).astype(int) == 1]
    if min_sar > 0 and "sar_manual_bin_hits_at_region_mean" in filtered.columns:
        filtered = filtered[_to_float(filtered["sar_manual_bin_hits_at_region_mean"]).fillna(0.0) >= float(min_sar)]
    if min_flag_risk > 0 and "flag_risk_score" in filtered.columns:
        filtered = filtered[_to_float(filtered["flag_risk_score"]).fillna(0.0) >= float(min_flag_risk)]
    if min_region_frac > 0 and "region_hours_fraction" in filtered.columns:
        filtered = filtered[_to_float(filtered["region_hours_fraction"]).fillna(0.0) >= float(min_region_frac)]
    if score_min is not None and score_col and score_col in filtered.columns:
        filtered = filtered[_to_float(filtered[score_col]).fillna(-np.inf) >= float(score_min)]

    lat_all, lon_all = _best_lat_lon(filtered)
    filtered = filtered.copy()
    filtered["lat"] = lat_all
    filtered["lon"] = lon_all
    if require_location:
        filtered = filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"])

    # Helpful derived labels
    if "is_iuu" in filtered.columns:
        filtered["label_status"] = np.where(filtered["is_iuu"].fillna(0).astype(int) == 1, "Known IUU", "Unlabeled")
    else:
        filtered["label_status"] = "Unknown"
    if "in_target_region" in filtered.columns:
        filtered["region_status"] = np.where(filtered["in_target_region"].fillna(0).astype(int) == 1, "Gulf+Med", "Other")
    else:
        filtered["region_status"] = "Unknown"

    # Cap for display
    filtered_display = filtered.head(top_n).copy()

    # -------- Overview tab --------
    with tab_overview:
        colA, colB, colC, colD, colE = st.columns([1.1, 1.1, 1.1, 1.4, 1.2])
        colA.metric("Dataset vessels", f"{len(df):,}")
        if y is not None:
            colB.metric("Known IUU labels", f"{int(y.sum()):,}")
            colC.metric("Base rate", f"{float(y.mean() * 100):.4f}%")
        else:
            colB.metric("Known IUU labels", "N/A")
            colC.metric("Base rate", "N/A")
        colD.metric("Rank file", os.path.basename(rank_path))
        colE.metric("After filters", f"{len(filtered):,}")

        st.markdown(
            "**How to present this:** scores are used to **rank vessels for inspection**. "
            "With sparse labels, the most defensible evaluation is **Top‑K retrieval** (how many known IUU appear near the top)."
        )

        if score_col and score_col in filtered.columns and "is_iuu" in filtered.columns:
            st.subheader("Top‑K ranking (current selection)")
            ks = [10, 25, 50, 100, 250, 500, 1000]
            tbl = top_k_table(filtered["is_iuu"].to_numpy(), _to_float(filtered[score_col]).fillna(0.0).to_numpy(), ks)
            if not tbl.empty:
                tbl_fmt = tbl.copy()
                tbl_fmt["Precision"] = tbl_fmt["Precision"].map(lambda x: f"{x:.4f}")
                tbl_fmt["Recall"] = tbl_fmt["Recall"].map(lambda x: f"{x:.4f}")
                tbl_fmt["Lift"] = tbl_fmt["Lift"].map(lambda x: f"{x:.2f}")
                st.dataframe(tbl_fmt, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data to compute Top‑K metrics for this selection.")
        else:
            st.info("Top‑K metrics require both a score column and `is_iuu` labels in the joined data.")

    # -------- Ranked list tab --------
    with tab_ranked:
        st.subheader("Ranked list (joined to key features)")
        st.caption(f"Source: `{_rel(rank_path)}`. Scores are **relative rankings**, not legal determinations.")

        st.dataframe(filtered_display, use_container_width=True, height=520)

        c1, c2 = st.columns([1, 1])
        with c1:
            dl = filtered_display.copy()
            csv_bytes = dl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download displayed rows as CSV",
                data=csv_bytes,
                file_name=f"dashboard_export_{os.path.basename(rank_path).replace('.csv','')}_top{top_n}.csv",
                mime="text/csv",
            )
        with c2:
            if "mmsi" in filtered_display.columns and len(filtered_display) > 0:
                vessel_id = st.selectbox("Vessel drill‑down (MMSI)", options=filtered_display["mmsi"].astype(str).tolist())
            else:
                vessel_id = None

        if vessel_id is not None:
            st.subheader("Selected vessel: key signals")
            row = filtered[filtered["mmsi"].astype(str) == str(vessel_id)].head(1)
            if row.empty:
                st.info("No row found for this MMSI after filters.")
            else:
                # Show a tight, presentation-friendly subset
                keep = [c for c in [
                    "mmsi",
                    score_col,
                    "label_status",
                    "region_status",
                    "flag_risk_score",
                    "near_eez_boundary_50km",
                    "sar_manual_bin_hits_at_region_mean",
                    "region_hours_fraction",
                    "region_fishing_hours_sum",
                    "all_fishing_hours_sum",
                    "gap_hours_mean",
                    "gap_days_fraction",
                    "lat",
                    "lon",
                ] if c and c in row.columns]
                st.dataframe(row[keep], use_container_width=True, hide_index=True)

    # -------- Map tab --------
    with tab_map:
        st.subheader("Map (mean vessel locations)")
        st.caption("Each point is a vessel mean location (not a track). Gulf/Med means are preferred when available.")

        if filtered_display.empty:
            st.info("No mappable rows after filters.")
        else:
            map_mode = st.radio("Map mode", options=["Scatter (tiles)", "Density (tiles)", "Scatter (globe)"], horizontal=True)
            max_points = st.slider("Max points to plot", min_value=100, max_value=5000, value=min(int(top_n), 2000), step=100)
            plot_df = filtered.head(int(max_points)).copy()

            color = score_col if (score_col and score_col in plot_df.columns) else None
            hover_cols = [c for c in ["label_status", "region_status", "flag_risk_score", "near_eez_boundary_50km", "sar_manual_bin_hits_at_region_mean", "region_hours_fraction", "all_fishing_hours_sum"] if c in plot_df.columns]

            if map_mode == "Scatter (tiles)":
                fig = px.scatter_mapbox(
                    plot_df,
                    lat="lat",
                    lon="lon",
                    color=color,
                    hover_name="mmsi" if "mmsi" in plot_df.columns else None,
                    hover_data=hover_cols,
                    zoom=0.8,
                    height=600,
                )
                # Highlight known IUU vessels as an overlay (Plotly 6 removed `symbol=` for scatter_mapbox)
                if "label_status" in plot_df.columns:
                    known = plot_df[plot_df["label_status"] == "Known IUU"]
                    if not known.empty:
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=known["lat"],
                                lon=known["lon"],
                                mode="markers",
                                marker=dict(size=14, color="#dc2626", symbol="star"),
                                text=known["mmsi"] if "mmsi" in known.columns else None,
                                hovertemplate="<b>Known IUU</b><br>MMSI=%{text}<extra></extra>",
                                name="Known IUU",
                            )
                        )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

            elif map_mode == "Density (tiles)":
                radius = st.slider("Density radius", min_value=2, max_value=30, value=8, step=1)
                fig = px.density_mapbox(
                    plot_df,
                    lat="lat",
                    lon="lon",
                    z=color if color else None,
                    radius=radius,
                    hover_name="mmsi" if "mmsi" in plot_df.columns else None,
                    hover_data=hover_cols,
                    zoom=0.8,
                    height=600,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

            else:
                fig = px.scatter_geo(
                    plot_df,
                    lat="lat",
                    lon="lon",
                    color=color,
                    symbol="label_status" if "label_status" in plot_df.columns else None,
                    hover_name="mmsi" if "mmsi" in plot_df.columns else None,
                    hover_data=hover_cols,
                    projection="natural earth",
                    height=600,
                )
                fig.update_geos(showland=True, landcolor="#f3f4f6", showcountries=True, countrycolor="#d1d5db")
                st.plotly_chart(fig, use_container_width=True)

    # -------- Diagnostics tab --------
    with tab_diagnostics:
        st.subheader("Diagnostics (for analysis)")
        if score_col and score_col in filtered.columns and filtered[score_col].notna().any():
            st.caption("These plots help you explain how the score relates to interpretable proxies.")

            dd = filtered.head(min(len(filtered), 5000)).copy()
            dd = dd.replace([np.inf, -np.inf], np.nan)

            fig_h = px.histogram(dd, x=score_col, nbins=50, color="label_status" if "label_status" in dd.columns else None, height=360)
            st.plotly_chart(fig_h, use_container_width=True)

            cand = [c for c in ["region_hours_fraction", "flag_risk_score", "sar_manual_bin_hits_at_region_mean", "all_fishing_hours_sum", "gap_hours_mean"] if c in dd.columns]
            if cand:
                xcol = st.selectbox("Feature to compare vs score", options=cand, index=0)
                fig_s = px.scatter(
                    dd,
                    x=xcol,
                    y=score_col,
                    color="label_status" if "label_status" in dd.columns else None,
                    hover_name="mmsi" if "mmsi" in dd.columns else None,
                    height=420,
                    opacity=0.7,
                )
                st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("Diagnostics require a numeric score column in the selected ranked list.")

else:
    with tab_overview:
        st.warning("No ranked list CSV found in `outputs/`. Run `train_model.py` or `anomaly_model.py` to generate one.")


with tab_docs:
    if show_docs and doc_candidates:
        st.subheader("Evaluation write-ups")
        doc_path = st.selectbox("Choose a report to display", options=doc_candidates, format_func=_rel)
        if doc_path.lower().endswith(".html"):
            components.html(read_text_cached(doc_path), height=900, scrolling=True)
        else:
            st.markdown(read_text_cached(doc_path))
    else:
        st.info("No evaluation write-ups found (or display disabled).")
