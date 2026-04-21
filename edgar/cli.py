"""Command-Line Interface for QuantSchedule13DandG.

Usage
-----
python main.py search "Berkshire Hathaway" --max-results 5
python main.py download 1067983 0001193125-24-123456 --parse --save-db
python main.py parse filings/0001193125-24-123456.htm --save-db
python main.py analyze --top 10 --chart
"""

import argparse
import json
import os
import sys

from .retrieval import search_filings, get_filings_by_cik, download_filing
from .parser import parse_filing
from .analysis import (
    build_dataframe,
    save_to_database,
    load_from_database,
    analyze_ownership_changes,
    top_holders,
    summary_stats,
    plot_ownership_trend,
    plot_top_holders_bar,
)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_search(args):
    """Handle the *search* sub-command."""
    forms = [f.strip() for f in args.forms.split(",")] if args.forms else None
    print(f"Searching for {args.query!r} (forms: {args.forms}) …")
    results = search_filings(
        query=args.query,
        form_types=forms,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
    )
    if not results:
        print("No filings found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n{i:>3}. {r['entity_name']} ({r['form_type']})  —  {r['file_date']}")
        print(f"      Accession: {r['accession_no']}")
        if r.get("display_names"):
            print(f"      Filers   : {', '.join(r['display_names'][:3])}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def cmd_download(args):
    """Handle the *download* sub-command."""
    print(f"Downloading accession {args.accession} for CIK {args.cik} …")
    try:
        path = download_filing(args.cik, args.accession, output_dir=args.output_dir)
        print(f"Saved to: {path}")

        if args.parse or args.save_db:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            data = parse_filing(content)
            data.update({
                "cik": args.cik,
                "accession_no": args.accession,
            })
            print("\nParsed fields:")
            for k, v in data.items():
                print(f"  {k}: {v or '(not found)'}")

            if args.save_db:
                df = build_dataframe([data])
                save_to_database(df, db_path=args.db)
                print(f"\nData saved to database: {args.db}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_parse(args):
    """Handle the *parse* sub-command."""
    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    data = parse_filing(content)
    print("Parsed fields:")
    for k, v in data.items():
        print(f"  {k}: {v or '(not found)'}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nParsed data saved to {args.output}")

    if args.save_db:
        df = build_dataframe([data])
        save_to_database(df, db_path=args.db)
        print(f"Data saved to database: {args.db}")


def cmd_analyze(args):
    """Handle the *analyze* sub-command."""
    df = load_from_database(db_path=args.db)
    if df.empty:
        print(f"No data found in {args.db}.  Download and parse some filings first.")
        return

    print(f"Loaded {len(df)} record(s) from {args.db}.\n")
    summary_stats(df)

    if args.changes:
        df_chg = analyze_ownership_changes(df)
        cols = [c for c in ("filer_name", "issuer_name", "file_date", "percent_owned", "change")
                if c in df_chg.columns]
        print("\nOwnership changes (latest 20):")
        print(df_chg[cols].tail(20).to_string(index=False))

    if args.top:
        print(f"\nTop {args.top} holders:")
        print(top_holders(df, n=args.top).to_string(index=False))
        if args.chart:
            path = plot_top_holders_bar(df, n=args.top, output_file="top_holders.png")
            print(f"Chart saved to: {path}")

    if args.issuer and args.chart:
        safe_name = args.issuer.replace(" ", "_")
        path = plot_ownership_trend(df, args.issuer, output_file=f"{safe_name}_trend.png")
        print(f"Trend chart saved to: {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="edgar",
        description="Retrieve and analyse SEC EDGAR Schedule 13D/13G filings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- search ----
    search_p = subparsers.add_parser("search", help="Search EDGAR for 13D/13G filings.")
    search_p.add_argument("query", nargs="?", default="",
                          help="Search keyword (company name, ticker, investor).")
    search_p.add_argument("--forms", default="SC 13D,SC 13G",
                          help="Comma-separated form types (default: 'SC 13D,SC 13G').")
    search_p.add_argument("--start-date", dest="start_date", metavar="YYYY-MM-DD",
                          help="Only include filings on or after this date.")
    search_p.add_argument("--end-date", dest="end_date", metavar="YYYY-MM-DD",
                          help="Only include filings on or before this date.")
    search_p.add_argument("--max-results", dest="max_results", type=int, default=10,
                          help="Maximum number of results (default: 10).")
    search_p.add_argument("--output", "-o", metavar="FILE",
                          help="Save results as JSON to FILE.")
    search_p.set_defaults(func=cmd_search)

    # ---- download ----
    dl_p = subparsers.add_parser("download", help="Download a specific filing by CIK and accession.")
    dl_p.add_argument("cik", help="Company CIK number.")
    dl_p.add_argument("accession", help="Accession number (e.g. 0001234567-24-000001).")
    dl_p.add_argument("--output-dir", dest="output_dir", default="filings",
                      help="Directory to save the filing (default: filings/).")
    dl_p.add_argument("--parse", action="store_true",
                      help="Parse the downloaded filing immediately.")
    dl_p.add_argument("--save-db", dest="save_db", action="store_true",
                      help="Save parsed data to the local database.")
    dl_p.add_argument("--db", default="filings.db",
                      help="SQLite database file (default: filings.db).")
    dl_p.set_defaults(func=cmd_download)

    # ---- parse ----
    parse_p = subparsers.add_parser("parse", help="Parse a locally saved filing HTML file.")
    parse_p.add_argument("file", help="Path to the filing HTML file.")
    parse_p.add_argument("--output", "-o", metavar="FILE",
                         help="Save parsed fields as JSON to FILE.")
    parse_p.add_argument("--save-db", dest="save_db", action="store_true",
                         help="Save parsed data to the local database.")
    parse_p.add_argument("--db", default="filings.db",
                         help="SQLite database file (default: filings.db).")
    parse_p.set_defaults(func=cmd_parse)

    # ---- analyze ----
    an_p = subparsers.add_parser("analyze", help="Analyze filings stored in the local database.")
    an_p.add_argument("--db", default="filings.db",
                      help="SQLite database file (default: filings.db).")
    an_p.add_argument("--changes", action="store_true",
                      help="Show ownership changes between consecutive filings.")
    an_p.add_argument("--top", type=int, metavar="N",
                      help="Show the top N holders by ownership percentage.")
    an_p.add_argument("--issuer", metavar="NAME",
                      help="Name (or partial name) of the issuer to trend-plot.")
    an_p.add_argument("--chart", action="store_true",
                      help="Generate and save PNG charts.")
    an_p.set_defaults(func=cmd_analyze)

    return parser


def main():
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
