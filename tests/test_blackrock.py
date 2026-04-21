"""Tests for QuantSchedule13DandG using BlackRock, Inc as the test entity.

All network calls are mocked so these tests run fully offline.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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
# Fixtures
# ---------------------------------------------------------------------------

BLACKROCK_CIK = "1364742"
BLACKROCK_NAME = "BlackRock, Inc."

# Minimal EFTS JSON response for a BlackRock SC 13G search.
MOCK_EFTS_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "accession_no": "0001364742-24-000001",
                    "ciks": [BLACKROCK_CIK],
                    "entity_name": "Apple Inc.",
                    "form_type": "SC 13G",
                    "file_date": "2024-02-01",
                    "period_of_report": "2023-12-31",
                    "display_names": [BLACKROCK_NAME],
                }
            },
            {
                "_source": {
                    "accession_no": "0001364742-24-000002",
                    "ciks": [BLACKROCK_CIK],
                    "entity_name": "Microsoft Corporation",
                    "form_type": "SC 13G",
                    "file_date": "2024-02-02",
                    "period_of_report": "2023-12-31",
                    "display_names": [BLACKROCK_NAME],
                }
            },
        ]
    }
}

# Minimal submissions JSON response for BlackRock's CIK.
MOCK_SUBMISSIONS_RESPONSE = {
    "name": BLACKROCK_NAME,
    "filings": {
        "recent": {
            "form": ["SC 13G", "SC 13G/A", "10-K"],
            "accessionNumber": [
                "0001364742-24-000001",
                "0001364742-24-000002",
                "0001364742-24-000099",
            ],
            "filingDate": ["2024-02-01", "2024-03-01", "2024-04-01"],
            "reportDate": ["2023-12-31", "2023-12-31", "2023-12-31"],
            "primaryDocument": ["sc13g.htm", "sc13ga.htm", "10k.htm"],
        }
    },
}

# Minimal filing index HTML (table with one document row).
MOCK_INDEX_HTML = """
<html><body>
<table>
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
  <tr>
    <td>1</td>
    <td>Schedule 13G</td>
    <td><a href="/Archives/edgar/data/1364742/000136474224000001/sc13g.htm">sc13g.htm</a></td>
    <td>SC 13G</td>
  </tr>
</table>
</body></html>
"""

# Minimal 13G filing HTML for BlackRock holding Apple shares.
MOCK_FILING_HTML = """
<html><body>
<p>Names of Reporting Persons: BlackRock, Inc.</p>
<p>Name of Issuer: Apple Inc.</p>
<p>Title of Class of Securities: Common Stock</p>
<p>CUSIP No. 037833100</p>
<p>Citizenship or Place of Organization: Delaware</p>
<p>Aggregate Amount Beneficially Owned by Each Reporting Person: 1,023,456,789</p>
<p>Percent of Class Represented by Amount in Row (11): 6.87 %</p>
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
# Retrieval tests
# ---------------------------------------------------------------------------

class TestSearchFilings(unittest.TestCase):
    """Tests for edgar.retrieval.search_filings."""

    @patch("edgar.retrieval.requests.get")
    def test_search_returns_results(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        results = search_filings(query="BlackRock", max_results=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["entity_name"], "Apple Inc.")
        self.assertEqual(results[0]["form_type"], "SC 13G")
        self.assertEqual(results[0]["cik"], BLACKROCK_CIK)
        self.assertIn(BLACKROCK_NAME, results[0]["display_names"])

    @patch("edgar.retrieval.requests.get")
    def test_search_empty_query_returns_results(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        results = search_filings(query="", max_results=10)
        self.assertIsInstance(results, list)

    @patch("edgar.retrieval.requests.get")
    def test_search_respects_max_results(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        results = search_filings(query="BlackRock", max_results=1)
        self.assertLessEqual(len(results), 1)

    @patch("edgar.retrieval.requests.get")
    def test_search_no_hits(self, mock_get):
        mock_get.return_value = _make_response(json_data={"hits": {"hits": []}})

        results = search_filings(query="NonExistentEntityXYZ")
        self.assertEqual(results, [])

    @patch("edgar.retrieval.requests.get")
    def test_search_date_filters_passed(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        search_filings(
            query="BlackRock",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        call_params = mock_get.call_args[1]["params"]
        self.assertEqual(call_params["dateRange"], "custom")
        self.assertEqual(call_params["startdt"], "2024-01-01")
        self.assertEqual(call_params["enddt"], "2024-12-31")

    @patch("edgar.retrieval.requests.get")
    def test_search_custom_form_types(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        search_filings(query="BlackRock", form_types=["SC 13D"])
        call_params = mock_get.call_args[1]["params"]
        self.assertEqual(call_params["forms"], "SC 13D")


class TestGetFilingsByCik(unittest.TestCase):
    """Tests for edgar.retrieval.get_filings_by_cik."""

    @patch("edgar.retrieval.requests.get")
    def test_returns_13g_filings_only(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)

        filings = get_filings_by_cik(BLACKROCK_CIK)

        # Only SC 13G and SC 13G/A — the 10-K should be excluded.
        self.assertEqual(len(filings), 2)
        form_types = {f["form_type"] for f in filings}
        self.assertTrue(form_types.issubset({"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}))

    @patch("edgar.retrieval.requests.get")
    def test_entity_name_populated(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)

        filings = get_filings_by_cik(BLACKROCK_CIK)
        for f in filings:
            self.assertEqual(f["entity_name"], BLACKROCK_NAME)

    @patch("edgar.retrieval.requests.get")
    def test_cik_zero_padded_in_url(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_SUBMISSIONS_RESPONSE)

        get_filings_by_cik(BLACKROCK_CIK)
        called_url = mock_get.call_args[0][0]
        self.assertIn("CIK" + BLACKROCK_CIK.zfill(10), called_url)


class TestGetFilingIndex(unittest.TestCase):
    """Tests for edgar.retrieval.get_filing_index."""

    @patch("edgar.retrieval.requests.get")
    def test_parses_documents(self, mock_get):
        mock_get.return_value = _make_response(text=MOCK_INDEX_HTML)

        docs = get_filing_index(BLACKROCK_CIK, "0001364742-24-000001")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "sc13g.htm")
        self.assertTrue(docs[0]["url"].endswith("sc13g.htm"))


class TestDownloadFiling(unittest.TestCase):
    """Tests for edgar.retrieval.download_filing."""

    @patch("edgar.retrieval.requests.get")
    def test_saves_filing_to_disk(self, mock_get):
        mock_get.side_effect = [
            _make_response(text=MOCK_INDEX_HTML),   # index page
            _make_response(text=MOCK_FILING_HTML),  # filing document
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = download_filing(
                BLACKROCK_CIK, "0001364742-24-000001", output_dir=tmpdir
            )
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("BlackRock", content)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseFiling(unittest.TestCase):
    """Tests for edgar.parser.parse_filing."""

    def setUp(self):
        self.parsed = parse_filing(MOCK_FILING_HTML)

    def test_filer_name(self):
        self.assertIn("BlackRock", self.parsed["filer_name"])

    def test_issuer_name(self):
        self.assertIn("Apple", self.parsed["issuer_name"])

    def test_cusip(self):
        self.assertEqual(self.parsed["cusip"], "037833100")

    def test_class_of_securities(self):
        self.assertIn("Common Stock", self.parsed["class_of_securities"])

    def test_citizenship(self):
        self.assertIn("Delaware", self.parsed["citizenship"])

    def test_aggregate_owned(self):
        self.assertTrue(self.parsed["aggregate_owned"].replace(",", "").isdigit() or
                        self.parsed["aggregate_owned"] != "")

    def test_percent_owned(self):
        self.assertRegex(self.parsed["percent_owned"], r"^\d+(\.\d+)?$")

    def test_returns_dict_with_all_keys(self):
        expected_keys = {
            "issuer_name", "cusip", "class_of_securities",
            "filer_name", "citizenship", "aggregate_owned", "percent_owned",
        }
        self.assertEqual(set(self.parsed.keys()), expected_keys)

    def test_empty_content_returns_empty_strings(self):
        result = parse_filing("")
        for v in result.values():
            self.assertEqual(v, "")

    def test_plain_text_content(self):
        plain = (
            "Names of Reporting Persons: BlackRock, Inc.\n"
            "Name of Issuer: Apple Inc.\n"
            "CUSIP No. 037833100\n"
            "Percent of Class Represented by Amount in Row (11): 6.87 %\n"
        )
        result = parse_filing(plain)
        self.assertIn("BlackRock", result["filer_name"])
        self.assertEqual(result["percent_owned"], "6.87")


class TestParseFilingsBatch(unittest.TestCase):
    """Tests for edgar.parser.parse_filings_batch."""

    def test_batch_returns_list(self):
        results = parse_filings_batch([MOCK_FILING_HTML, MOCK_FILING_HTML])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("BlackRock", r["filer_name"])

    def test_empty_batch(self):
        self.assertEqual(parse_filings_batch([]), [])


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

def _make_sample_df():
    """Return a small DataFrame mimicking parsed BlackRock filings."""
    records = [
        {
            "filer_name": BLACKROCK_NAME,
            "issuer_name": "Apple Inc.",
            "form_type": "SC 13G",
            "file_date": "2023-02-01",
            "percent_owned": "6.87",
            "aggregate_owned": "1023456789",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME,
            "issuer_name": "Apple Inc.",
            "form_type": "SC 13G/A",
            "file_date": "2024-02-01",
            "percent_owned": "7.01",
            "aggregate_owned": "1045000000",
            "cusip": "037833100",
        },
        {
            "filer_name": BLACKROCK_NAME,
            "issuer_name": "Microsoft Corporation",
            "form_type": "SC 13G",
            "file_date": "2024-02-02",
            "percent_owned": "8.32",
            "aggregate_owned": "620000000",
            "cusip": "594918104",
        },
    ]
    return build_dataframe(records)


class TestBuildDataframe(unittest.TestCase):
    def test_numeric_percent_owned(self):
        df = _make_sample_df()
        self.assertTrue(pd.api.types.is_numeric_dtype(df["percent_owned"]))

    def test_datetime_file_date(self):
        df = _make_sample_df()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["file_date"]))

    def test_empty_input(self):
        df = build_dataframe([])
        self.assertTrue(df.empty)

    def test_row_count(self):
        df = _make_sample_df()
        self.assertEqual(len(df), 3)


class TestDatabase(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        df = _make_sample_df()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(df, db_path=db_path)
            loaded = load_from_database(db_path=db_path)
            self.assertEqual(len(loaded), len(df))
            self.assertIn("filer_name", loaded.columns)
        finally:
            os.unlink(db_path)

    def test_load_missing_db_returns_empty(self):
        df = load_from_database(db_path="/tmp/does_not_exist_ever.db")
        self.assertTrue(df.empty)

    def test_save_empty_df_is_noop(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_to_database(pd.DataFrame(), db_path=db_path)
            loaded = load_from_database(db_path=db_path)
            self.assertTrue(loaded.empty)
        finally:
            os.unlink(db_path)


class TestAnalyzeOwnershipChanges(unittest.TestCase):
    def test_change_column_added(self):
        df = _make_sample_df()
        result = analyze_ownership_changes(df)
        self.assertIn("change", result.columns)

    def test_first_filing_has_nan_change(self):
        df = _make_sample_df()
        result = analyze_ownership_changes(df)
        apple_rows = result[result["issuer_name"] == "Apple Inc."].sort_values("file_date")
        self.assertTrue(pd.isna(apple_rows.iloc[0]["change"]))

    def test_second_filing_has_positive_change(self):
        df = _make_sample_df()
        result = analyze_ownership_changes(df)
        apple_rows = result[result["issuer_name"] == "Apple Inc."].sort_values("file_date")
        # 7.01 - 6.87 ≈ 0.14
        self.assertAlmostEqual(apple_rows.iloc[1]["change"], 0.14, places=1)

    def test_empty_df_returns_empty(self):
        result = analyze_ownership_changes(pd.DataFrame())
        self.assertTrue(result.empty)


class TestTopHolders(unittest.TestCase):
    def test_returns_top_n(self):
        df = _make_sample_df()
        result = top_holders(df, n=2)
        self.assertLessEqual(len(result), 2)

    def test_sorted_by_percent_descending(self):
        df = _make_sample_df()
        result = top_holders(df, n=10)
        pcts = result["percent_owned"].values
        self.assertTrue(all(pcts[i] >= pcts[i + 1] for i in range(len(pcts) - 1)))

    def test_empty_df_returns_empty(self):
        result = top_holders(pd.DataFrame(), n=5)
        self.assertTrue(result.empty)


class TestSummaryStats(unittest.TestCase):
    def test_runs_without_error(self):
        df = _make_sample_df()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary_stats(df)
        output = buf.getvalue()
        self.assertIn("3", output)  # 3 total records

    def test_empty_df_prints_no_data(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary_stats(pd.DataFrame())
        self.assertIn("No data", buf.getvalue())


class TestPlotOwnershipTrend(unittest.TestCase):
    def test_saves_png(self):
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "trend.png")
            result = plot_ownership_trend(df, "Apple", output_file=out)
            self.assertEqual(result, out)
            self.assertTrue(os.path.isfile(out))

    def test_raises_on_unknown_issuer(self):
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "trend.png")
            with self.assertRaises(ValueError):
                plot_ownership_trend(df, "NoSuchIssuerXYZ", output_file=out)

    def test_raises_on_empty_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "trend.png")
            with self.assertRaises(ValueError):
                plot_ownership_trend(pd.DataFrame(), "Apple", output_file=out)


class TestPlotTopHoldersBar(unittest.TestCase):
    def test_saves_png(self):
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "top.png")
            result = plot_top_holders_bar(df, n=3, output_file=out)
            self.assertEqual(result, out)
            self.assertTrue(os.path.isfile(out))

    def test_raises_on_empty_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "top.png")
            with self.assertRaises(ValueError):
                plot_top_holders_bar(pd.DataFrame(), n=3, output_file=out)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCliSearch(unittest.TestCase):
    """Test the CLI *search* command end-to-end (network mocked)."""

    @patch("edgar.retrieval.requests.get")
    def test_cli_search_prints_results(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        import io
        from contextlib import redirect_stdout
        from edgar.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["search", "BlackRock", "--max-results", "5"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            args.func(args)
        output = buf.getvalue()
        self.assertIn("Apple Inc.", output)
        self.assertIn("SC 13G", output)

    @patch("edgar.retrieval.requests.get")
    def test_cli_search_saves_json(self, mock_get):
        mock_get.return_value = _make_response(json_data=MOCK_EFTS_RESPONSE)

        from edgar.cli import build_parser

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out_path = f.name
        try:
            parser = build_parser()
            args = parser.parse_args(
                ["search", "BlackRock", "--max-results", "5", "--output", out_path]
            )
            args.func(args)
            with open(out_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data), 2)
        finally:
            os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()
