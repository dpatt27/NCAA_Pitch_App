import pandas as pd
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = "data.csv"

FEATURES = [
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HorzBreak",
    "Extension",
    "RelHeight",
    "RelSide",
    "PlateLocHeight",
    "PlateLocSide",
    "pitch_type_encoded",
]

PITCH_TYPE_MAP = {
    # Fastballs
    "Fastball":        0,
    "FourSeam":        0,  # treat as same group
    "TwoSeamFastBall": 1,
    "Sinker":          2,
    "Cutter":          3,
    # Breaking balls
    "Slider":          4,
    "Sweeper":         5,
    "Slurve":          6,
    "Curveball":       7,
    "Gyro":            8,
    "Reaper":          9,   # gyro-heavy curveball variant
    # Offspeed
    "ChangeUp":        10,
    "Changeup":        10,  # normalise alternate spelling
    "Splitter":        11,
}

SWING_CALLS = {
    "StrikeSwinging",
    "FoulBallNotFieldable",
    "FoulBallFieldable",
    "FoulballNotFieldable",
    "InPlay",
}

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Trackman CSV with mixed-type columns suppressed."""
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of the raw Trackman dataframe.

    Steps
    -----
    1. Normalise pitch type spellings.
    2. Add boolean swing / whiff columns.
    3. Encode pitch type as an integer.
    4. Drop rows missing any model feature or the whiff target.
    5. Reset index.
    """
    df = df.copy()

    # 1. Normalise pitch type
    df["TaggedPitchType"] = df["TaggedPitchType"].replace({"Changeup": "ChangeUp"})

    # 2. Swing / whiff flags
    df["is_swing"] = df["PitchCall"].isin(SWING_CALLS)
    df["whiff"]    = (df["PitchCall"] == "StrikeSwinging").astype(int)

    # 3. Encode pitch type
    df["pitch_type_encoded"] = df["TaggedPitchType"].map(PITCH_TYPE_MAP)

    # 4. Keep only swings with complete feature rows
    swing_df = df[df["is_swing"]].copy()
    swing_df = swing_df.dropna(subset=FEATURES + ["whiff"])
    swing_df = swing_df[swing_df["pitch_type_encoded"].notna()]

    # 5. Reset index
    swing_df = swing_df.reset_index(drop=True)

    return swing_df


def get_pitcher_profiles(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data to one row per pitcher × pitch type.

    Columns returned
    ----------------
    Pitcher, TaggedPitchType, pitch_count, usage_pct,
    avg_velo, avg_spin, avg_ivb, avg_hb, whiff_rate
    """
    grp = (
        df_clean
        .groupby(["Pitcher", "TaggedPitchType"])
        .agg(
            pitch_count      = ("whiff", "count"),
            avg_velo         = ("RelSpeed",         "mean"),
            avg_spin         = ("SpinRate",          "mean"),
            avg_ivb          = ("InducedVertBreak",  "mean"),
            avg_hb           = ("HorzBreak",         "mean"),
            avg_extension    = ("Extension",         "mean"),
            avg_rel_height   = ("RelHeight",         "mean"),
            whiff_rate       = ("whiff",             "mean"),
        )
        .reset_index()
    )

    # Usage % within each pitcher
    totals = grp.groupby("Pitcher")["pitch_count"].transform("sum")
    grp["usage_pct"] = (grp["pitch_count"] / totals * 100).round(1)

    # Round display columns
    for col in ["avg_velo","avg_spin","avg_ivb","avg_hb","avg_extension","avg_rel_height"]:
        grp[col] = grp[col].round(1)
    grp["whiff_rate"] = (grp["whiff_rate"] * 100).round(1)  # as %

    return grp


def get_pitcher_summary(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pitcher with arsenal-level averages — used for KMeans clustering.

    Columns: Pitcher, avg_velo, avg_spin, avg_ivb, avg_hb, avg_extension,
             n_pitch_types, overall_whiff_rate
    """
    summary = (
        df_clean
        .groupby("Pitcher")
        .agg(
            avg_velo          = ("RelSpeed",         "mean"),
            avg_spin          = ("SpinRate",          "mean"),
            avg_ivb           = ("InducedVertBreak",  "mean"),
            avg_hb            = ("HorzBreak",         "mean"),
            avg_extension     = ("Extension",         "mean"),
            n_pitch_types     = ("TaggedPitchType",   "nunique"),
            overall_whiff_rate= ("whiff",             "mean"),
            total_pitches     = ("whiff",             "count"),
        )
        .reset_index()
    )
    return summary
