"""Tests for QuantSchedule13DandG using BlackRock, Inc as the test entity.

Covers Schedule 13D and 13G filings filed within the last 5 years (2021-2026).
All network calls are mocked so these tests run fully offline.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, call, patch

import pandas as pd

from edgar.retrieval import search_filings, get_filings_by_cik, get_filing_index, download_filing
from edgar.parser import parse_filing, parse_filings_batch
from edgar.analysis import (
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
# Constants
# ---------------------------------------------------------------------------

BLACKROCK_CIK = "1364742"
BLACKROCK_NAME = "BlackRock, Inc."

# ---------------------------------------------------------------------------
# Realistic mock data: 10 BlackRock 13D/13G filings spread across 2021-2026
# covering both original filings and amendments, and multiple issuers.
# ---------------------------------------------------------------------------

_FIVE_YEAR_HITS = [
    # 2021 filings
    {
        "_source": {
            "accession_no": "0001364742-21-000010",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Apple Inc.",
            "form_type": "SC 13G",
            "file_date": "2021-02-05",
            "period_of_report": "2020-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    {
        "_source": {
            "accession_no": "0001364742-21-000020",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Microsoft Corporation",
            "form_type": "SC 13G",
            "file_date": "2021-02-08",
            "period_of_report": "2020-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    # 2022 filings
    {
        "_source": {
            "accession_no": "0001364742-22-000010",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Apple Inc.",
            "form_type": "SC 13G/A",
            "file_date": "2022-02-04",
            "period_of_report": "2021-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    {
        "_source": {
            "accession_no": "0001364742-22-000030",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Tesla, Inc.",
            "form_type": "SC 13G",
            "file_date": "2022-02-10",
            "period_of_report": "2021-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    # 2023 filings
    {
        "_source": {
            "accession_no": "0001364742-23-000010",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Apple Inc.",
            "form_type": "SC 13G/A",
            "file_date": "2023-02-03",
            "period_of_report": "2022-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    {
        "_source": {
            "accession_no": "0001364742-23-000040",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Exxon Mobil Corporation",
            "form_type": "SC 13D",
            "file_date": "2023-05-15",
            "period_of_report": "2023-05-10",
            "display_names": [BLACKROCK_NAME],
        }
    },
    # 2024 filings
    {
        "_source": {
            "accession_no": "0001364742-24-000010",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Apple Inc.",
            "form_type": "SC 13G/A",
            "file_date": "2024-02-01",
            "period_of_report": "2023-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    {
        "_source": {
            "accession_no": "0001364742-24-000050",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "NVIDIA Corporation",
            "form_type": "SC 13G",
            "file_date": "2024-02-13",
            "period_of_report": "2023-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    # 2025 filings
    {
        "_source": {
            "accession_no": "0001364742-25-000010",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Apple Inc.",
            "form_type": "SC 13G/A",
            "file_date": "2025-02-04",
            "period_of_report": "2024-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
    {
        "_source": {
            "accession_no": "0001364742-25-000060",
            "ciks": [BLACKROCK_CIK],
            "entity_name": "Amazon.com, Inc.",
            "form_type": "SC 13G",
            "file_date": "2025-02-07",
            "period_of_report": "2024-12-31",
            "display_names": [BLACKROCK_NAME],
        }
    },
]

MOCK_EFTS_RESPONSE = {"hits": {"hits": _FIVE_YEAR_HITS}}
MOCK_EFTS_EMPTY = {"hits": {"hits": []}}

# Submissions JSON — 10 target filings plus one 10-K that must be filtered out.
MOCK_SUBMISSIONS_RESPONSE = {
    "name": BLACKROCK_NAME,
    "filings": {
        "recent": {
            "form": [h["_source"]["form_type"] for h in _FIVE_YEAR_HITS] + ["10-K"],
            "accessionNumber": [h["_source"]["accession_no"] for h in _FIVE_YEAR_HITS]
                               + ["0001364742-25-000099"],
            "filingDate": [h["_source"]["file_date"] for h in _FIVE_YEAR_HITS]
                         + ["2025-04-01"],
            "reportDate": [h["_source"]["period_of_report"] for h in _FIVE_YEAR_HITS]
                         + ["2024-12-31"],
            "primaryDocument": [f"doc{i}.htm" for i in range(len(_FIVE_YEAR_HITS))]
                               + ["10k.htm"],
        }
    },
}

# Filing index JSON (mirrors the real EDGAR index.json format).
import json as _json
MOCK_INDEX_HTML = _json.dumps({
    "directory": {
        "item": [
            {"name": "0001364742-24-000010-index.html", "type": "text.gif", "size": ""},
            {"name": "0001364742-24-000010-index-headers.html", "type": "text.gif", "size": ""},
            {"name": "0001364742-24-000010.txt", "type": "text.gif", "size": ""},
            {"name": "sc13ga.xml", "type": "text.gif", "size": "12000"},
            {"name": "sc13ga.htm", "type": "text.gif", "size": "8000"},
        ],
        "name": "/Archives/edgar/data/1364742/000136474224000010",
    }
})

# Realistic 13G/A filing HTML — BlackRock's annual amendment for Apple (2024 filing).
MOCK_FILING_HTML_APPLE_2024 = """
<html><body>
<p>SCHEDULE 13G/A</p>
<p>Names of Reporting Persons: BlackRock, Inc.</p>
<p>Name of Issuer: Apple Inc.</p>
<p>Title of Class of Securities: Common Stock</p>
<p>CUSIP No. 037833100</p>
<p>Citizenship or Place of Organization: Delaware</p>
<p>Aggregate Amount Beneficially Owned by Each Reporting Person: 1,023,456,789</p>
<p>Percent of Class Represented by Amount in Row (11): 6.87 %</p>
</body></html>
"""

# 13D filing HTML — BlackRock filing a 13D for Exxon (activist-style).
MOCK_FILING_HTML_EXXON_13D = """
<html><body>
<p>SCHEDULE 13D</p>
<p>Names of Reporting Persons: BlackRock, Inc.</p>
<p>Name of Issuer: Exxon Mobil Corporation</p>
<p>Title of Class of Securities: Common Stock</p>
<p>CUSIP No. 28823HC89</p>
<p>Citizenship or Place of Organization: Delaware</p>
<p>Aggregate Amount Beneficially Owned by Each Reporting Person: 412,000,000</p>
<p>Percent of Class Represented by Amount in Row (11): 9.82 %</p>
</body></html>
"""

# 13G filing HTML — BlackRock filing a 13G for NVIDIA (new position 2024).
MOCK_FILING_HTML_NVIDIA_2024 = """
<html><body>
<p>SCHEDULE 13G</p>
<p>1. Names of Reporting Persons: BlackRock, Inc.</p>
<p>Name of Issuer: NVIDIA Corporation</p>
<p>Title of Class of Securities: Common Stock</p>
<p>CUSIP No. 67066G104</p>
<p>Citizenship or Place of Organization: Delaware</p>
<p>Aggregate Amount Beneficially Owned by Each Reporting Person: 213,000,000</p>
<p>Percent of Class Represented by Amount in Row (11): 8.50 %</p>
</body></html>
"""


def _make_response(json_data=None, text=None, status_code=200):
    """Create a mock requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if text is not None:
        resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Helper: build a realistic multi-year DataFrame for BlackRock
# ---------------------------------------------------------------------------

def _make_five_year_df():
    """DataFrame with BlackRock 13D/13G filings across 2021-2025."""
    records = [
        # Apple – 5 annual 13G/A amendments
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Apple Inc.",
            "form_type": "SC 13G",  "file_date": "2021-02-05",
            "percent_owned": "6.72", "aggregate_owned": "1010000000",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Apple Inc.",
            "form_type": "SC 13G/A", "file_date": "2022-02-04",
            "percent_owned": "6.58", "aggregate_owned": "985000000",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Apple Inc.",
            "form_type": "SC 13G/A", "file_date": "2023-02-03",
            "percent_owned": "6.89", "aggregate_owned": "1055000000",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Apple Inc.",
            "form_type": "SC 13G/A", "file_date": "2024-02-01",
            "percent_owned": "6.87", "aggregate_owned": "1023456789",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Apple Inc.",
            "form_type": "SC 13G/A", "file_date": "2025-02-04",
            "percent_owned": "7.03", "aggregate_owned": "1060000000",
            "cusip": "037833100",
        },
        # Microsoft – 2 filings
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Microsoft Corporation",
            "form_type": "SC 13G",  "file_date": "2021-02-08",
            "percent_owned": "7.50", "aggregate_owned": "562000000",
            "cusip": "594918104",
        },
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Microsoft Corporation",
            "form_type": "SC 13G/A", "file_date": "2024-02-02",
            "percent_owned": "8.32", "aggregate_owned": "620000000",
            "cusip": "594918104",
        },
        # Tesla – 1 filing (2022)
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Tesla, Inc.",
            "form_type": "SC 13G",  "file_date": "2022-02-10",
            "percent_owned": "5.12", "aggregate_owned": "162000000",
            "cusip": "88160R101",
        },
        # Exxon – 13D (2023, activist)
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Exxon Mobil Corporation",
            "form_type": "SC 13D",  "file_date": "2023-05-15",
            "percent_owned": "9.82", "aggregate_owned": "412000000",
            "cusip": "28823HC89",
        },
        # NVIDIA – 1 filing (2024)
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "NVIDIA Corporation",
            "form_type": "SC 13G",  "file_date": "2024-02-13",
            "percent_owned": "8.50", "aggregate_owned": "213000000",
            "cusip": "67066G104",
        },
        # Amazon – 1 filing (2025)
        {
            "filer_name": BLACKROCK_NAME, "issuer_name": "Amazon.com, Inc.",
            "form_type": "SC 13G",  "file_date": "2025-02-07",
            "percent_owned": "6.15", "aggregate_owned": "630000000",
            "cusip": "023135106",
        },
    ]
    return build_dataframe(records)


# ===========================================================================
# 1. RETRIEVAL TESTS
# ===========================================================================

class TestSearchFilingsBlackRock(unittest.TestCase):
    """edgar.retrieval.search_filings — BlackRock 13D/13G last-5-year scenarios."""

    @patch("edgar.retrieval.requests.get")
    def test_returns_all_ten_filings(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=20)
        self.assertEqual(len(results), 10)

    @patch("edgar.retrieval.requests.get")
    def test_result_fields_populated(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=20)
        for r in results:
            self.assertIn("accession_no", r)
            self.assertIn("cik", r)
            self.assertIn("entity_name", r)
            self.assertIn("form_type", r)
            self.assertIn("file_date", r)
            self.assertEqual(r["cik"], BLACKROCK_CIK)

    @patch("edgar.retrieval.requests.get")
    def test_both_13d_and_13g_form_types_present(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=20)
        form_types = {r["form_type"] for r in results}
        self.assertIn("SC 13D", form_types)
        self.assertIn("SC 13G", form_types)

    @patch("edgar.retrieval.requests.get")
    def test_five_year_date_range_filter(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        search_filings(
            query="BlackRock",
            start_date="2021-01-01",
            end_date="2026-04-21",
            max_results=20,
        )
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["dateRange"], "custom")
        self.assertEqual(params["startdt"], "2021-01-01")
        self.assertEqual(params["enddt"], "2026-04-21")

    @patch("edgar.retrieval.requests.get")
    def test_max_results_respected(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=3)
        self.assertLessEqual(len(results), 3)

    @patch("edgar.retrieval.requests.get")
    def test_only_13d_form_type_filter(self, mock_get):
        # Return only the SC 13D hit
        sc13d_hits = [h for h in _FIVE_YEAR_HITS if h["_source"]["form_type"] == "SC 13D"]
        mock_get.return_value = _make_response(
            json_data={"hits": {"hits": sc13d_hits}}
        )
        results = search_filings(query="BlackRock", form_types=["SC 13D"])
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["forms"], "SC 13D")
        self.assertTrue(all(r["form_type"] == "SC 13D" for r in results))

    @patch("edgar.retrieval.requests.get")
    def test_only_13g_form_type_filter(self, mock_get):
        sc13g_hits = [h for h in _FIVE_YEAR_HITS
                      if h["_source"]["form_type"] in ("SC 13G", "SC 13G/A")]
        mock_get.return_value = _make_response(
            json_data={"hits": {"hits": sc13g_hits}}
        )
        results = search_filings(query="BlackRock", form_types=["SC 13G", "SC 13G/A"])
        self.assertTrue(all(r["form_type"] in ("SC 13G", "SC 13G/A") for r in results))

    @patch("edgar.retrieval.requests.get")
    def test_no_results_for_unknown_company(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_EMPTY)
        results = search_filings(query="NonExistentEntityXYZ")
        self.assertEqual(results, [])

    @patch("edgar.retrieval.requests.get")
    def test_display_names_contains_blackrock(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=20)
        for r in results:
            self.assertIn(BLACKROCK_NAME, r["display_names"])

    @patch("edgar.retrieval.requests.get")
    def test_period_of_report_field_present(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        results = search_filings(query="BlackRock", max_results=20)
        for r in results:
            self.assertIn("period_of_report", r)


class TestGetFilingsByCikBlackRock(unittest.TestCase):
    """edgar.retrieval.get_filings_by_cik — BlackRock CIK-based lookup."""

    @patch("edgar.retrieval.requests.get")
    def test_filters_to_13d_13g_only(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        filings = get_filings_by_cik(BLACKROCK_CIK)
        valid_types = {"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}
        for f in filings:
            self.assertIn(f["form_type"], valid_types,
                          f"Unexpected form_type {f['form_type']!r} should be excluded")

    @patch("edgar.retrieval.requests.get")
    def test_10k_excluded(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        filings = get_filings_by_cik(BLACKROCK_CIK)
        self.assertFalse(any(f["form_type"] == "10-K" for f in filings))

    @patch("edgar.retrieval.requests.get")
    def test_returns_ten_eligible_filings(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        filings = get_filings_by_cik(BLACKROCK_CIK)
        self.assertEqual(len(filings), 10)

    @patch("edgar.retrieval.requests.get")
    def test_entity_name_is_blackrock(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        filings = get_filings_by_cik(BLACKROCK_CIK)
        for f in filings:
            self.assertEqual(f["entity_name"], BLACKROCK_NAME)

    @patch("edgar.retrieval.requests.get")
    def test_cik_is_zero_padded_in_url(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        get_filings_by_cik(BLACKROCK_CIK)
        called_url = mock_get.call_args[0][0]
        self.assertIn("CIK" + BLACKROCK_CIK.zfill(10), called_url)

    @patch("edgar.retrieval.requests.get")
    def test_custom_form_types_respected(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)
        filings = get_filings_by_cik(BLACKROCK_CIK, form_types=["SC 13D"])
        self.assertTrue(all(f["form_type"] == "SC 13D" for f in filings))


class TestGetFilingIndexBlackRock(unittest.TestCase):
    """edgar.retrieval.get_filing_index — index page parsing."""

    @patch("edgar.retrieval.requests.get")
    def test_parses_single_document(self, mock_get):
        mock_get.return_value = _make_response(json_data=_json.loads(MOCK_INDEX_HTML))
        docs = get_filing_index(BLACKROCK_CIK, "0001364742-24-000010")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["filename"], "sc13ga.xml")
        expected_url = (
            "https://www.sec.gov/Archives/edgar/data/1364742/"
            "000136474224000010/sc13ga.xml"
        )
        self.assertEqual(docs[0]["url"], expected_url)

    @patch("edgar.retrieval.requests.get")
    def test_document_type_field(self, mock_get):
        mock_get.return_value = _make_response(json_data=_json.loads(MOCK_INDEX_HTML))
        docs = get_filing_index(BLACKROCK_CIK, "0001364742-24-000010")
        self.assertIn("type", docs[0])

    @patch("edgar.retrieval.requests.get")
    def test_accession_dashes_stripped_in_url(self, mock_get):
        mock_get.return_value = _make_response(json_data=_json.loads(MOCK_INDEX_HTML))
        get_filing_index(BLACKROCK_CIK, "0001364742-24-000010")
        called_url = mock_get.call_args[0][0]
        self.assertNotIn("0001364742-24-000010", called_url)  # dashes removed
        self.assertIn("000136474224000010", called_url)


class TestDownloadFilingBlackRock(unittest.TestCase):
    """edgar.retrieval.download_filing — full download flow."""

    def _run_download(self, mock_get, filing_html, accession="0001364742-24-000010"):
        mock_get.side_effect = [
            _make_response(json_data=_json.loads(MOCK_INDEX_HTML)),
            _make_response(text=filing_html),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = download_filing(BLACKROCK_CIK, accession, output_dir=tmpdir)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
        return path, content

    @patch("edgar.retrieval.requests.get")
    def test_apple_filing_saved(self, mock_get):
        _, content = self._run_download(mock_get, MOCK_FILING_HTML_APPLE_2024)
        self.assertIn("BlackRock", content)
        self.assertIn("Apple Inc.", content)

    @patch("edgar.retrieval.requests.get")
    def test_exxon_13d_filing_saved(self, mock_get):
        _, content = self._run_download(
            mock_get, MOCK_FILING_HTML_EXXON_13D, "0001364742-23-000040"
        )
        self.assertIn("Exxon Mobil Corporation", content)
        self.assertIn("13D", content)

    @patch("edgar.retrieval.requests.get")
    def test_nvidia_filing_saved(self, mock_get):
        _, content = self._run_download(
            mock_get, MOCK_FILING_HTML_NVIDIA_2024, "0001364742-24-000050"
        )
        self.assertIn("NVIDIA Corporation", content)

    @patch("edgar.retrieval.requests.get")
    def test_output_file_is_xml(self, mock_get):
        path, _ = self._run_download(mock_get, MOCK_FILING_HTML_APPLE_2024)
        self.assertTrue(path.endswith(".xml"))

    @patch("edgar.retrieval.requests.get")
    def test_makes_two_requests(self, mock_get):
        mock_get.side_effect = [
            _make_response(json_data=_json.loads(MOCK_INDEX_HTML)),
            _make_response(text=MOCK_FILING_HTML_APPLE_2024),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            download_filing(BLACKROCK_CIK, "0001364742-24-000010", output_dir=tmpdir)
        self.assertEqual(mock_get.call_count, 2)


# ===========================================================================
# 2. PARSER TESTS
# ===========================================================================

class TestParseFilingApple2024(unittest.TestCase):
    """parse_filing — BlackRock SC 13G/A for Apple (2024)."""

    def setUp(self):
        self.parsed = parse_filing(MOCK_FILING_HTML_APPLE_2024)

    def test_filer_name(self):
        self.assertIn("BlackRock", self.parsed["filer_name"])

    def test_issuer_name(self):
        self.assertIn("Apple", self.parsed["issuer_name"])

    def test_cusip(self):
        self.assertEqual(self.parsed["cusip"], "037833100")

    def test_class_of_securities(self):
        self.assertIn("Common Stock", self.parsed["class_of_securities"])

    def test_citizenship_delaware(self):
        self.assertIn("Delaware", self.parsed["citizenship"])

    def test_aggregate_owned_numeric_string(self):
        raw = self.parsed["aggregate_owned"].replace(",", "")
        self.assertTrue(raw.isdigit())

    def test_percent_owned_value(self):
        self.assertAlmostEqual(float(self.parsed["percent_owned"]), 6.87, places=1)

    def test_all_keys_present(self):
        expected = {
            "issuer_name", "cusip", "class_of_securities",
            "filer_name", "citizenship", "aggregate_owned", "percent_owned",
        }
        self.assertEqual(set(self.parsed.keys()), expected)


class TestParseFilingExxon13D(unittest.TestCase):
    """parse_filing — BlackRock SC 13D for Exxon (2023)."""

    def setUp(self):
        self.parsed = parse_filing(MOCK_FILING_HTML_EXXON_13D)

    def test_filer_name(self):
        self.assertIn("BlackRock", self.parsed["filer_name"])

    def test_issuer_name(self):
        self.assertIn("Exxon", self.parsed["issuer_name"])

    def test_cusip(self):
        self.assertEqual(self.parsed["cusip"], "28823HC89")

    def test_percent_owned(self):
        self.assertAlmostEqual(float(self.parsed["percent_owned"]), 9.82, places=1)


class TestParseFilingNvidia2024(unittest.TestCase):
    """parse_filing — BlackRock SC 13G for NVIDIA (2024)."""

    def setUp(self):
        self.parsed = parse_filing(MOCK_FILING_HTML_NVIDIA_2024)

    def test_filer_name(self):
        self.assertIn("BlackRock", self.parsed["filer_name"])

    def test_issuer_name(self):
        self.assertIn("NVIDIA", self.parsed["issuer_name"])

    def test_percent_owned(self):
        self.assertAlmostEqual(float(self.parsed["percent_owned"]), 8.50, places=1)


class TestParseFilingEdgeCases(unittest.TestCase):
    """parse_filing — edge-case inputs."""

    def test_empty_string_returns_empty_values(self):
        result = parse_filing("")
        for v in result.values():
            self.assertEqual(v, "")

    def test_plain_text_no_html_tags(self):
        plain = (
            "Names of Reporting Persons: BlackRock, Inc.\n"
            "Name of Issuer: Apple Inc.\n"
            "CUSIP No. 037833100\n"
            "Citizenship or Place of Organization: Delaware\n"
            "Percent of Class Represented by Amount in Row (11): 6.87 %\n"
        )
        r = parse_filing(plain)
        self.assertIn("BlackRock", r["filer_name"])
        self.assertEqual(r["cusip"], "037833100")
        self.assertEqual(r["percent_owned"], "6.87")

    def test_batch_all_three_filings(self):
        results = parse_filings_batch([
            MOCK_FILING_HTML_APPLE_2024,
            MOCK_FILING_HTML_EXXON_13D,
            MOCK_FILING_HTML_NVIDIA_2024,
        ])
        self.assertEqual(len(results), 3)
        issuers = [r["issuer_name"] for r in results]
        self.assertTrue(any("Apple" in i for i in issuers))
        self.assertTrue(any("Exxon" in i for i in issuers))
        self.assertTrue(any("NVIDIA" in i for i in issuers))

    def test_batch_empty_list(self):
        self.assertEqual(parse_filings_batch([]), [])


# ===========================================================================
# 3. ANALYSIS TESTS
# ===========================================================================

class TestBuildDataframeBlackRock(unittest.TestCase):
    """build_dataframe — multi-year BlackRock data."""

    def setUp(self):
        self.df = _make_five_year_df()

    def test_row_count(self):
        self.assertEqual(len(self.df), 11)

    def test_percent_owned_numeric(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df["percent_owned"]))

    def test_file_date_datetime(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["file_date"]))

    def test_date_range_spans_five_years(self):
        earliest = self.df["file_date"].min().year
        latest = self.df["file_date"].max().year
        self.assertLessEqual(earliest, 2021)
        self.assertGreaterEqual(latest, 2025)

    def test_all_filers_are_blackrock(self):
        self.assertTrue((self.df["filer_name"] == BLACKROCK_NAME).all())

    def test_multiple_unique_issuers(self):
        self.assertGreaterEqual(self.df["issuer_name"].nunique(), 5)

    def test_empty_input(self):
        self.assertTrue(build_dataframe([]).empty)


class TestDatabaseRoundTripBlackRock(unittest.TestCase):
    """save_to_database / load_from_database — five-year BlackRock data."""

    def test_full_roundtrip(self):
        df = _make_five_year_df()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(df, db_path=db_path)
            loaded = load_from_database(db_path=db_path)
            self.assertEqual(len(loaded), len(df))
            self.assertIn("filer_name", loaded.columns)
            self.assertTrue((loaded["filer_name"] == BLACKROCK_NAME).all())
        finally:
            os.unlink(db_path)

    def test_append_preserves_all_rows(self):
        df = _make_five_year_df()
        half = len(df) // 2
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(df.iloc[:half], db_path=db_path)
            save_to_database(df.iloc[half:], db_path=db_path)
            loaded = load_from_database(db_path=db_path)
            self.assertEqual(len(loaded), len(df))
        finally:
            os.unlink(db_path)

    def test_missing_db_returns_empty(self):
        result = load_from_database(db_path="/tmp/blackrock_no_such_file.db")
        self.assertTrue(result.empty)

    def test_save_empty_is_noop(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(pd.DataFrame(), db_path=db_path)
            loaded = load_from_database(db_path=db_path)
            self.assertTrue(loaded.empty)
        finally:
            os.unlink(db_path)


class TestOwnershipChangesBlackRock(unittest.TestCase):
    """analyze_ownership_changes — Apple stake trend 2021-2025."""

    def setUp(self):
        self.df = _make_five_year_df()
        self.result = analyze_ownership_changes(self.df)

    def test_change_column_present(self):
        self.assertIn("change", self.result.columns)

    def test_apple_first_filing_nan(self):
        apple = (
            self.result[self.result["issuer_name"] == "Apple Inc."]
            .sort_values("file_date")
        )
        self.assertTrue(pd.isna(apple.iloc[0]["change"]))

    def test_apple_multi_year_changes_computed(self):
        apple = (
            self.result[self.result["issuer_name"] == "Apple Inc."]
            .sort_values("file_date")
        )
        # 4 non-NaN deltas for 5 filings
        non_nan = apple["change"].dropna()
        self.assertEqual(len(non_nan), 4)

    def test_exxon_single_filing_nan(self):
        exxon = self.result[self.result["issuer_name"] == "Exxon Mobil Corporation"]
        self.assertTrue(pd.isna(exxon.iloc[0]["change"]))

    def test_empty_df_returns_empty(self):
        self.assertTrue(analyze_ownership_changes(pd.DataFrame()).empty)

    def test_missing_required_column_returns_input(self):
        df_no_pct = _make_five_year_df().drop(columns=["percent_owned"])
        result = analyze_ownership_changes(df_no_pct)
        self.assertNotIn("change", result.columns)


class TestTopHoldersBlackRock(unittest.TestCase):
    """top_holders — ranked by most-recent ownership percentage."""

    def setUp(self):
        self.df = _make_five_year_df()

    def test_top_1_is_exxon(self):
        """Exxon has the highest most-recent ownership (9.82%)."""
        result = top_holders(self.df, n=1)
        self.assertIn("Exxon", result.iloc[0]["issuer_name"])

    def test_top_n_length(self):
        for n in (1, 3, 5, 10):
            result = top_holders(self.df, n=n)
            self.assertLessEqual(len(result), n)

    def test_sorted_descending(self):
        result = top_holders(self.df, n=10)
        pcts = result["percent_owned"].tolist()
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_uses_most_recent_filing_per_issuer(self):
        """Apple's most-recent pct (7.03 from 2025) should appear, not an earlier value."""
        result = top_holders(self.df, n=10)
        apple_row = result[result["issuer_name"] == "Apple Inc."]
        self.assertFalse(apple_row.empty)
        self.assertAlmostEqual(apple_row.iloc[0]["percent_owned"], 7.03, places=1)

    def test_empty_df_returns_empty(self):
        self.assertTrue(top_holders(pd.DataFrame(), n=5).empty)


class TestSummaryStatsBlackRock(unittest.TestCase):
    """summary_stats — output sanity checks."""

    def test_correct_record_count(self):
        df = _make_five_year_df()
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary_stats(df)
        self.assertIn("11", buf.getvalue())

    def test_shows_issuer_count(self):
        df = _make_five_year_df()
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary_stats(df)
        # 6 unique issuers in our fixture
        self.assertIn("6", buf.getvalue())

    def test_empty_df_prints_no_data(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary_stats(pd.DataFrame())
        self.assertIn("No data", buf.getvalue())


class TestChartsBlackRock(unittest.TestCase):
    """plot_ownership_trend and plot_top_holders_bar — file creation checks."""

    def setUp(self):
        self.df = _make_five_year_df()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _out(self, name):
        return os.path.join(self.tmpdir, name)

    def test_apple_trend_chart_created(self):
        path = plot_ownership_trend(self.df, "Apple", output_file=self._out("apple.png"))
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_exxon_trend_chart_created(self):
        path = plot_ownership_trend(self.df, "Exxon", output_file=self._out("exxon.png"))
        self.assertTrue(os.path.isfile(path))

    def test_nvidia_trend_chart_created(self):
        path = plot_ownership_trend(self.df, "NVIDIA", output_file=self._out("nvidia.png"))
        self.assertTrue(os.path.isfile(path))

    def test_top_holders_bar_chart_created(self):
        path = plot_top_holders_bar(self.df, n=5, output_file=self._out("top5.png"))
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_trend_chart_raises_for_unknown_issuer(self):
        with self.assertRaises(ValueError):
            plot_ownership_trend(self.df, "NoSuchCorp", output_file=self._out("x.png"))

    def test_trend_chart_raises_for_empty_df(self):
        with self.assertRaises(ValueError):
            plot_ownership_trend(pd.DataFrame(), "Apple", output_file=self._out("x.png"))

    def test_bar_chart_raises_for_empty_df(self):
        with self.assertRaises(ValueError):
            plot_top_holders_bar(pd.DataFrame(), n=5, output_file=self._out("x.png"))


# ===========================================================================
# 4. CLI INTEGRATION TESTS
# ===========================================================================

class TestCliSearchBlackRock(unittest.TestCase):
    """CLI search command — BlackRock 13D/13G last-5-year scenarios."""

    @patch("edgar.retrieval.requests.get")
    def test_prints_ten_results(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        from edgar.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["search", "BlackRock", "--max-results", "20"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            args.func(args)
        output = buf.getvalue()
        # Ten numbered entries expected
        for i in range(1, 11):
            self.assertIn(str(i), output)

    @patch("edgar.retrieval.requests.get")
    def test_output_contains_form_types(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        from edgar.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["search", "BlackRock", "--max-results", "20"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            args.func(args)
        output = buf.getvalue()
        self.assertIn("SC 13G", output)
        self.assertIn("SC 13D", output)

    @patch("edgar.retrieval.requests.get")
    def test_saves_json_file_with_correct_count(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        from edgar.cli import build_parser
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out_path = f.name
        try:
            parser = build_parser()
            args = parser.parse_args(
                ["search", "BlackRock", "--max-results", "20", "--output", out_path]
            )
            args.func(args)
            with open(out_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data), 10)
        finally:
            os.unlink(out_path)

    @patch("edgar.retrieval.requests.get")
    def test_date_range_passed_to_search(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)
        from edgar.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "search", "BlackRock",
            "--start-date", "2021-01-01",
            "--end-date", "2026-04-21",
            "--max-results", "20",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            args.func(args)
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["startdt"], "2021-01-01")
        self.assertEqual(params["enddt"], "2026-04-21")

    @patch("edgar.retrieval.requests.get")
    def test_no_results_message(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_EMPTY)
        from edgar.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["search", "BlackRock"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            args.func(args)
        self.assertIn("No filings found", buf.getvalue())


class TestCliDownloadBlackRock(unittest.TestCase):
    """CLI download command — fetch and parse a BlackRock 13G/A."""

    @patch("edgar.retrieval.requests.get")
    def test_download_and_parse_apple_filing(self, mock_get):
        mock_get.side_effect = [
            _make_response(json_data=_json.loads(MOCK_INDEX_HTML)),
            _make_response(text=MOCK_FILING_HTML_APPLE_2024),
        ]
        from edgar.cli import build_parser
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = build_parser()
            args = parser.parse_args([
                "download", BLACKROCK_CIK, "0001364742-24-000010",
                "--output-dir", tmpdir,
                "--parse",
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
        output = buf.getvalue()
        self.assertIn("BlackRock", output)
        self.assertIn("Apple", output)
        self.assertIn("6.87", output)

    @patch("edgar.retrieval.requests.get")
    def test_download_parse_save_db(self, mock_get):
        mock_get.side_effect = [
            _make_response(json_data=_json.loads(MOCK_INDEX_HTML)),
            _make_response(text=MOCK_FILING_HTML_APPLE_2024),
        ]
        from edgar.cli import build_parser
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            parser = build_parser()
            args = parser.parse_args([
                "download", BLACKROCK_CIK, "0001364742-24-000010",
                "--output-dir", tmpdir,
                "--parse", "--save-db", "--db", db_path,
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            self.assertTrue(os.path.isfile(db_path))
            loaded = load_from_database(db_path=db_path)
            self.assertFalse(loaded.empty)


class TestCliAnalyzeBlackRock(unittest.TestCase):
    """CLI analyze command — five-year BlackRock data in a temp DB."""

    def _setup_db(self, tmpdir):
        db_path = os.path.join(tmpdir, "blackrock.db")
        save_to_database(_make_five_year_df(), db_path=db_path)
        return db_path

    def test_analyze_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._setup_db(tmpdir)
            from edgar.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["analyze", "--db", db_path])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            output = buf.getvalue()
            self.assertIn("11", output)

    def test_analyze_top_holders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._setup_db(tmpdir)
            from edgar.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["analyze", "--db", db_path, "--top", "3"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            output = buf.getvalue()
            self.assertIn("Exxon", output)

    def test_analyze_ownership_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._setup_db(tmpdir)
            from edgar.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["analyze", "--db", db_path, "--changes"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            output = buf.getvalue()
            self.assertIn("Apple", output)

    def test_analyze_empty_db_prints_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")
            from edgar.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["analyze", "--db", db_path])
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            self.assertIn("No data", buf.getvalue())

    def test_analyze_chart_top_holders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._setup_db(tmpdir)
            chart_path = os.path.join(tmpdir, "top_holders.png")
            from edgar.cli import build_parser
            parser = build_parser()
            args = parser.parse_args([
                "analyze", "--db", db_path,
                "--top", "5", "--chart",
            ])
            # Override output file location to stay inside tmpdir
            orig_plot = plot_top_holders_bar
            with patch("edgar.cli.plot_top_holders_bar",
                       side_effect=lambda df, n, output_file: orig_plot(
                           df, n=n, output_file=chart_path
                       )) as mock_plot:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    args.func(args)
                mock_plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
