"""QuantSchedule13DandG – SEC EDGAR Schedule 13D/13G filing retrieval and analysis."""

__version__ = "0.1.0"

from .retrieval import search_filings, get_filings_by_cik, download_filing
from .parser import parse_filing
from .analysis import build_dataframe, analyze_ownership_changes, plot_ownership_trend

__all__ = [
    "search_filings",
    "get_filings_by_cik",
    "download_filing",
    "parse_filing",
    "build_dataframe",
    "analyze_ownership_changes",
    "plot_ownership_trend",
]
