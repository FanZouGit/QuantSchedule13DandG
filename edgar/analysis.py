"""Data Analysis module: analyze 13D/13G holdings, ownership changes, and visualizations."""

import os
import sqlite3

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DB_PATH = "filings.db"
_FILINGS_TABLE = "filings"


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def build_dataframe(filings):
    """Build a pandas DataFrame from a list of filing metadata / parsed dicts.

    Args:
        filings: List of dicts.  Expected keys include issuer_name, filer_name,
                 form_type, file_date, percent_owned, aggregate_owned, cusip.

    Returns:
        pandas DataFrame with numeric types coerced where applicable.
    """
    df = pd.DataFrame(filings)
    if df.empty:
        return df

    if "file_date" in df.columns:
        df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")

    for col in ("percent_owned", "aggregate_owned"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

    return df


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def save_to_database(df, db_path=DB_PATH, table=_FILINGS_TABLE):
    """Append a DataFrame to a local SQLite database table.

    Args:
        df: DataFrame to save.
        db_path: Path to the SQLite file.  Created if it does not exist.
        table: Table name.
    """
    if df.empty:
        return

    df_copy = df.copy()
    # SQLite does not understand Timestamp objects natively.
    if "file_date" in df_copy.columns:
        df_copy["file_date"] = df_copy["file_date"].astype(str)

    conn = sqlite3.connect(db_path)
    try:
        df_copy.to_sql(table, conn, if_exists="append", index=False)
    finally:
        conn.close()


def load_from_database(db_path=DB_PATH, table=_FILINGS_TABLE):
    """Load all filings from the local SQLite database.

    Args:
        db_path: Path to the SQLite file.
        table: Table name.

    Returns:
        pandas DataFrame (empty if the database or table does not exist).
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        # Validate the table name against the SQLite master table to avoid injection.
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if cursor.fetchone() is None:
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql(f"SELECT * FROM {table}", conn)  # noqa: S608
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if not df.empty and "file_date" in df.columns:
        df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_ownership_changes(df):
    """Add a *change* column showing the delta in ownership % between filings.

    Filings are grouped by (filer_name, issuer_name) and sorted by date.

    Args:
        df: DataFrame containing at minimum file_date, filer_name,
            issuer_name, and percent_owned columns.

    Returns:
        Copy of the input DataFrame sorted by filer/issuer/date with an
        extra *change* column (NaN for the first filing of each group).
    """
    required = {"file_date", "filer_name", "issuer_name", "percent_owned"}
    if df.empty or not required.issubset(df.columns):
        return df

    df_sorted = (
        df.dropna(subset=["percent_owned"])
        .sort_values(["filer_name", "issuer_name", "file_date"])
        .copy()
    )
    df_sorted["change"] = df_sorted.groupby(
        ["filer_name", "issuer_name"]
    )["percent_owned"].diff()

    return df_sorted


def top_holders(df, n=10):
    """Return the top *n* holders ranked by most-recent ownership percentage.

    Args:
        df: DataFrame with filing data.
        n: Number of holders to return.

    Returns:
        DataFrame of up to *n* rows.
    """
    if df.empty or "percent_owned" not in df.columns:
        return df

    idx = df.groupby(["filer_name", "issuer_name"])["file_date"].idxmax()
    latest = df.loc[idx].dropna(subset=["percent_owned"])
    cols = [c for c in ("filer_name", "issuer_name", "percent_owned", "file_date", "form_type")
            if c in latest.columns]
    return latest.nlargest(n, "percent_owned")[cols]


def summary_stats(df):
    """Print a human-readable summary of the filings DataFrame.

    Args:
        df: DataFrame with filing data.
    """
    if df.empty:
        print("No data available.")
        return

    print(f"Total records  : {len(df)}")
    if "issuer_name" in df.columns:
        print(f"Unique issuers : {df['issuer_name'].nunique()}")
    if "filer_name" in df.columns:
        print(f"Unique filers  : {df['filer_name'].nunique()}")
    if "file_date" in df.columns:
        print(f"Date range     : {df['file_date'].min()} → {df['file_date'].max()}")
    if "percent_owned" in df.columns:
        pct = df["percent_owned"].dropna()
        if not pct.empty:
            print(
                f"Ownership %%    : min={pct.min():.2f}  "
                f"median={pct.median():.2f}  max={pct.max():.2f}"
            )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ownership_trend(df, issuer_name, output_file="ownership_trend.png"):
    """Plot ownership percentage over time for all filers of a given issuer.

    Args:
        df: DataFrame with filing data.
        issuer_name: Full or partial issuer name (case-insensitive substring match).
        output_file: File path for the PNG output.

    Returns:
        Path to the saved image.

    Raises:
        ValueError: If no matching records are found.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    mask = df["issuer_name"].str.contains(issuer_name, case=False, na=False)
    issuer_df = df[mask].dropna(subset=["percent_owned", "file_date"])

    if issuer_df.empty:
        raise ValueError(f"No data found for issuer: {issuer_name!r}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for filer, group in issuer_df.groupby("filer_name"):
        grp = group.sort_values("file_date")
        ax.plot(grp["file_date"], grp["percent_owned"], marker="o", label=filer[:50])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha="right")
    ax.set_title(f"Ownership Trend: {issuer_name}")
    ax.set_xlabel("Filing Date")
    ax.set_ylabel("Ownership (%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def plot_top_holders_bar(df, n=10, output_file="top_holders.png"):
    """Plot a horizontal bar chart of the top *n* holders by ownership percentage.

    Args:
        df: DataFrame with filing data.
        n: Number of top holders to show.
        output_file: File path for the PNG output.

    Returns:
        Path to the saved image.

    Raises:
        ValueError: If no holder data is available.
    """
    top = top_holders(df, n=n)
    if top.empty:
        raise ValueError("No holder data available.")

    labels = [
        f"{row.get('filer_name', '')[:30]}\n({row.get('issuer_name', '')[:20]})"
        for _, row in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, top["percent_owned"].values, color="steelblue")
    ax.set_xlabel("Ownership (%)")
    ax.set_title(f"Top {n} Holders by Ownership Percentage")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file
