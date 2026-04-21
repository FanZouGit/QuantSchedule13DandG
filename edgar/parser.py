"""File Parsing module: extract holdings and ownership data from 13D/13G filings."""

import re

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Compiled regex patterns for the standard 13D/13G cover-page fields
# ---------------------------------------------------------------------------

# Item 1 / Row 1 – Name of Reporting Person (the filer / investor)
_RE_FILER = re.compile(
    r"(?:names?\s+of\s+reporting\s+persons?|^1\.)[:\s]+([^\n\r<|]{2,100})",
    re.IGNORECASE | re.MULTILINE,
)

# Name of Issuer (the company whose securities are being reported)
_RE_ISSUER = re.compile(
    r"name\s+of\s+issuer[:\s]+([^\n\r<|]{2,100})",
    re.IGNORECASE,
)

# Title / Class of Securities
_RE_CLASS = re.compile(
    r"(?:title\s+of\s+class(?:\s+of\s+securities)?|class\s+of\s+securities)[:\s]+([^\n\r<|]{2,80})",
    re.IGNORECASE,
)

# CUSIP number
_RE_CUSIP = re.compile(
    r"cusip\s*(?:no\.?|number)?\s*[:\s#]*\s*([0-9A-Za-z\-]{8,12})",
    re.IGNORECASE,
)

# Item 6 / Row 6 – Citizenship or Place of Organization
_RE_CITIZENSHIP = re.compile(
    r"citizenship\s+or\s+place\s+of\s+organization[:\s]+([^\n\r<|]{2,80})",
    re.IGNORECASE,
)

# Item 11 / Row 11 – Aggregate Amount Beneficially Owned
_RE_AGGREGATE = re.compile(
    r"(?:aggregate\s+amount\s+beneficially\s+owned(?:\s+by\s+each\s+reporting\s+person)?|"
    r"row\s*\(?\s*11\s*\)?\s*[:\-])[:\s]+([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

# Item 13 / Row 13 – Percent of Class
_RE_PERCENT = re.compile(
    r"(?:percent\s+of\s+class\s+represented\s+by\s+amount\s+in\s+row\s*\(?\s*11\s*\)?|"
    r"row\s*\(?\s*13\s*\)?\s*[:\-])[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%?",
    re.IGNORECASE,
)


def _first_match(pattern, text, default=""):
    """Return the stripped value of the first capturing group, or *default*."""
    m = pattern.search(text)
    return m.group(1).strip() if m else default


def parse_filing(content):
    """Parse a 13D/13G filing HTML/text and extract key ownership data.

    The function converts the HTML to plain text and applies regex patterns
    that match the standard SEC Schedule 13D/13G cover-page layout.

    Args:
        content: HTML (or plain text) string of the filing document.

    Returns:
        Dict with the following keys:
            issuer_name, cusip, class_of_securities,
            filer_name, citizenship, aggregate_owned, percent_owned.
        Values are strings; empty string means the field was not found.
    """
    soup = BeautifulSoup(content, "lxml")
    text = soup.get_text(separator="\n")

    return {
        "issuer_name": _first_match(_RE_ISSUER, text),
        "cusip": _first_match(_RE_CUSIP, text),
        "class_of_securities": _first_match(_RE_CLASS, text),
        "filer_name": _first_match(_RE_FILER, text),
        "citizenship": _first_match(_RE_CITIZENSHIP, text),
        "aggregate_owned": _first_match(_RE_AGGREGATE, text),
        "percent_owned": _first_match(_RE_PERCENT, text),
    }


def parse_filings_batch(contents):
    """Parse a list of filing HTML strings.

    Args:
        contents: Iterable of HTML strings.

    Returns:
        List of parsed dicts (same structure as :func:`parse_filing`).
    """
    return [parse_filing(c) for c in contents]
