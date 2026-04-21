"""Data Retrieval module: interact with the SEC EDGAR API to download 13D/13G filings."""

import os
import time

import requests
from bs4 import BeautifulSoup

EDGAR_BASE_URL = "https://www.sec.gov"
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"

# SEC requires a User-Agent header identifying the requester (name + email).
DEFAULT_HEADERS = {
    "User-Agent": "QuantSchedule13DandG contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Honour SEC rate limit: no more than 10 requests per second.
REQUEST_DELAY = 0.15


def _get(url, params=None):
    """Make a rate-limited GET request to SEC EDGAR."""
    response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response


def search_filings(query="", form_types=None, start_date=None, end_date=None, max_results=40):
    """Search EDGAR for 13D/13G filings by keyword.

    Args:
        query: Search keyword (e.g. company name, ticker, or investor name).
        form_types: List of form type strings.  Defaults to ["SC 13D", "SC 13G"].
        start_date: Only return filings on or after this date (YYYY-MM-DD).
        end_date: Only return filings on or before this date (YYYY-MM-DD).
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with filing metadata.
    """
    if form_types is None:
        form_types = ["SC 13D", "SC 13G"]

    params = {
        "q": query,
        "forms": ",".join(form_types),
        "from": 0,
        "size": min(max_results, 100),
    }
    if start_date or end_date:
        params["dateRange"] = "custom"
    if start_date:
        params["startdt"] = start_date
    if end_date:
        params["enddt"] = end_date

    results = []
    while len(results) < max_results:
        resp = _get(EFTS_SEARCH_URL, params=params)
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            src = hit.get("_source", {})
            ciks = src.get("ciks", [])
            results.append({
                "accession_no": src.get("accession_no", ""),
                "cik": ciks[0] if ciks else "",
                "entity_name": src.get("entity_name", ""),
                "form_type": src.get("form_type", ""),
                "file_date": src.get("file_date", ""),
                "period_of_report": src.get("period_of_report", ""),
                "display_names": src.get("display_names", []),
            })
        params["from"] += len(hits)
        if len(hits) < params["size"]:
            break

    return results[:max_results]


def get_filings_by_cik(cik, form_types=None):
    """Retrieve all 13D/13G filings for a company identified by CIK.

    Args:
        cik: SEC Central Index Key (integer or string).
        form_types: List of form type strings.  Defaults to SC 13D/G including amendments.

    Returns:
        List of dicts with filing metadata.
    """
    if form_types is None:
        form_types = {"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}
    else:
        form_types = set(form_types)

    cik_padded = str(int(cik)).zfill(10)
    url = f"{SUBMISSIONS_BASE_URL}/CIK{cik_padded}.json"
    resp = _get(url)
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    filings = []
    for i, form in enumerate(forms):
        if form in form_types:
            filings.append({
                "cik": str(int(cik)),
                "entity_name": data.get("name", ""),
                "accession_no": accession_numbers[i] if i < len(accession_numbers) else "",
                "form_type": form,
                "file_date": filing_dates[i] if i < len(filing_dates) else "",
                "period_of_report": report_dates[i] if i < len(report_dates) else "",
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
            })
    return filings


def get_filing_index(cik, accession_no):
    """Fetch the filing index page and return the list of documents.

    Args:
        cik: Company CIK.
        accession_no: Accession number (dashes are optional).

    Returns:
        List of dicts with keys: seq, description, filename, url, type.
    """
    accession_clean = accession_no.replace("-", "")
    url = f"{EDGAR_BASE_URL}/Archives/edgar/data/{cik}/{accession_clean}/"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    documents = []
    for row in soup.select("table tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) >= 3:
            link = cells[2].find("a", href=True)
            if link:
                documents.append({
                    "seq": cells[0].get_text(strip=True),
                    "description": cells[1].get_text(strip=True),
                    "filename": link.get_text(strip=True),
                    "url": f"{EDGAR_BASE_URL}{link['href']}",
                    "type": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                })
    return documents


def download_filing(cik, accession_no, output_dir="filings"):
    """Download the primary document of a 13D/13G filing and save it locally.

    Args:
        cik: Company CIK.
        accession_no: Accession number.
        output_dir: Directory to save the filing.  Created if it does not exist.

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    accession_clean = accession_no.replace("-", "")

    documents = get_filing_index(cik, accession_no)

    # Prefer the first .htm/.html file that looks like the primary form document.
    primary_doc_url = None
    for doc in documents:
        fname = doc["filename"].lower()
        if fname.endswith((".htm", ".html")):
            primary_doc_url = doc["url"]
            break

    if not primary_doc_url and documents:
        primary_doc_url = documents[0]["url"]

    if not primary_doc_url:
        raise ValueError(f"No primary document found for accession {accession_no}")

    resp = _get(primary_doc_url)
    filename = os.path.join(output_dir, f"{accession_clean}.htm")
    with open(filename, "w", encoding="utf-8", errors="replace") as f:
        f.write(resp.text)

    return filename
