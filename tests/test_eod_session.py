"""EOD publish must match NY as_of or the process fails before gh-pages."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from volume_scanner.cli import main
from volume_scanner.eod_session import (
    PUBLIC_EOD_BASE,
    SessionDateError,
    fetch_public_text,
    is_nyse_session,
    main as check_main,
    ny_as_of,
    nyse_full_closures,
    retain_as_of_rows,
    verify_eod_outputs,
)
from volume_scanner.report_html import write

NY = ZoneInfo("America/New_York")
HKT = ZoneInfo("Asia/Hong_Kong")

# Official NYSE full-day closures (nyse.com holiday calendar, 2026–2028).
_OFFICIAL = {
    2026: {
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    },
    2027: {
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
        date(2027, 7, 5),
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),
    },
    2028: {
        date(2028, 1, 17),
        date(2028, 2, 21),
        date(2028, 4, 14),
        date(2028, 5, 29),
        date(2028, 6, 19),
        date(2028, 7, 4),
        date(2028, 9, 4),
        date(2028, 11, 23),
        date(2028, 12, 25),
    },
}


def _ny(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY)


def _row(day: str, symbol: str = "AAA") -> dict:
    return {
        "symbol": symbol,
        "date": day,
        "avg_volume": 100,
        "last_volume": 400,
        "prev_volume": 90,
        "avg_volume_10d": 95,
        "rvol": 4.0,
        "open": 10.0,
        "close": 10.5,
        "pct_change": 5.0,
        "day_range_pct": 2.0,
        "range_vs_avg": 1.2,
    }


def _write_pair(directory: Path, frame: pd.DataFrame, *, mode: str = "eod") -> tuple[Path, Path]:
    csv_path = directory / "report.csv"
    html_path = directory / "index.html"
    frame.to_csv(csv_path, index=False)
    write(frame, str(html_path), market="us", min_rvol=3, mode=mode)
    return csv_path, html_path


class NyAsOfTests(unittest.TestCase):
    def test_official_closures_2026_through_2028(self):
        for year, expected in _OFFICIAL.items():
            self.assertEqual(nyse_full_closures(year), expected)

    def test_new_years_2028_saturday_does_not_close_prior_friday(self):
        # NYSE: Jan 1 2028 falls on Saturday and is not observed.
        self.assertTrue(is_nyse_session(date(2027, 12, 31)))
        self.assertNotIn(date(2027, 12, 31), nyse_full_closures(2027))
        self.assertNotIn(date(2028, 1, 1), nyse_full_closures(2028))

    def test_scheduled_and_late_runs_use_the_completed_session(self):
        # Cron 22:00 UTC Friday = 17:00 EST. Session is that Friday.
        self.assertEqual(ny_as_of(_ny(2026, 11, 13, 17, 0)), date(2026, 11, 13))
        # Ops 06:05 HKT Saturday = 17:05 EST Friday.
        self.assertEqual(
            ny_as_of(datetime(2026, 11, 14, 6, 5, tzinfo=HKT)),
            date(2026, 11, 13),
        )
        # A run that lands 12:00 HKT Saturday is still Friday's session.
        self.assertEqual(
            ny_as_of(datetime(2026, 11, 14, 12, 0, tzinfo=HKT)),
            date(2026, 11, 13),
        )
        # Before the Monday close, the latest completed session is Friday.
        self.assertEqual(ny_as_of(_ny(2026, 11, 16, 15, 59)), date(2026, 11, 13))
        self.assertEqual(ny_as_of(_ny(2026, 11, 16, 16, 0)), date(2026, 11, 16))

    def test_holiday_and_early_close(self):
        # Thanksgiving 2026 is closed. The prior Wednesday is the session.
        self.assertEqual(ny_as_of(_ny(2026, 11, 26, 18, 0)), date(2026, 11, 25))
        # The day after Thanksgiving is an early close, and it is still that session.
        self.assertEqual(ny_as_of(_ny(2026, 11, 27, 18, 0)), date(2026, 11, 27))
        # Independence Day observed Friday 2026-07-03. Thursday is the session.
        self.assertEqual(ny_as_of(_ny(2026, 7, 3, 18, 0)), date(2026, 7, 2))

    def test_naive_datetime_is_new_york_local(self):
        self.assertEqual(ny_as_of(datetime(2026, 11, 13, 17, 0)), date(2026, 11, 13))


class VerifyOutputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.now = _ny(2026, 11, 13, 17, 30)

    def tearDown(self):
        self._tmp.cleanup()

    def test_matching_dates_pass(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row("2026-11-13"), _row("2026-11-13", "BBB")])
        )
        self.assertEqual(
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now),
            date(2026, 11, 13),
        )

    def test_stale_dates_fail(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row("2026-11-12")] * 3)
        )
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        message = str(caught.exception)
        self.assertIn("2026-11-12", message)
        self.assertIn("2026-11-13", message)
        self.assertIn("report.csv", message)
        self.assertIn("index.html", message)

    def test_mixed_dates_fail(self):
        csv_path, html_path = _write_pair(
            self.dir,
            pd.DataFrame([_row("2026-11-13", "AAA"), _row("2026-11-12", "BBB")]),
        )
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        self.assertIn("2026-11-12", str(caught.exception))
        self.assertIn("2026-11-13", str(caught.exception))

    def test_missing_dates_fail(self):
        frame = pd.DataFrame(columns=["symbol", "date", "rvol"])
        csv_path, html_path = _write_pair(self.dir, frame)
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        message = str(caught.exception)
        self.assertIn("no row dates", message)

    def test_blank_date_cell_fails(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row(""), _row("2026-11-13", "BBB")])
        )
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        self.assertIn("missing a session date", str(caught.exception))

    def test_intraday_mode_fails_even_when_dates_match(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row("2026-11-13")]), mode="intraday"
        )
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        self.assertIn("mode='intraday'", str(caught.exception))
        self.assertIn("expected 'eod'", str(caught.exception))

    def test_csv_stale_while_index_matches_fails(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row("2026-11-13")])
        )
        pd.DataFrame([_row("2026-11-12")]).to_csv(csv_path, index=False)
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        self.assertIn("report.csv", str(caught.exception))
        self.assertIn("2026-11-12", str(caught.exception))

    def test_missing_files_fail(self):
        with self.assertRaises(SessionDateError):
            verify_eod_outputs(
                str(self.dir / "report.csv"),
                str(self.dir / "index.html"),
                now=self.now,
            )

    def test_json_rows_must_match_even_if_html_was_rewritten(self):
        csv_path, html_path = _write_pair(
            self.dir, pd.DataFrame([_row("2026-11-13")])
        )
        payload_path = self.dir / "index.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["rows_html"] = payload["rows_html"].replace("2026-11-13", "2026-11-12")
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SessionDateError) as caught:
            verify_eod_outputs(str(csv_path), str(html_path), now=self.now)
        self.assertIn("index.json", str(caught.exception))
        self.assertIn("2026-11-12", str(caught.exception))


class CliGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(
        self,
        frame: pd.DataFrame,
        *,
        intraday: bool = False,
        as_of: date | None = None,
    ) -> tuple[int, str]:
        import volume_scanner.cli as cli_mod
        import volume_scanner.eod_session as session

        def _universe(*_args, **_kwargs):
            return ["AAA"]

        def _scan(*_args, **_kwargs):
            return frame

        original_universe = cli_mod.universe.build_universe
        original_scan = cli_mod.scan
        original_intraday = cli_mod.scan_intraday
        original_as_of = session.ny_as_of
        original_decision = session.public_eod_decision
        original_output = os.environ.pop("GITHUB_OUTPUT", None)
        cli_mod.universe.build_universe = _universe
        cli_mod.scan = _scan
        cli_mod.scan_intraday = _scan
        if as_of is not None:
            session.ny_as_of = lambda now=None, pinned=as_of: pinned

        def _decision(*_args, **_kwargs):
            if intraday:
                raise AssertionError("intraday run checked public EOD freshness")
            return False, "test stub: not fresh"

        session.public_eod_decision = _decision
        argv = [
            "--market", "us",
            "--html", str(self.dir / "index.html"),
            "--out", str(self.dir / "report.csv"),
        ]
        if intraday:
            argv.append("--intraday")
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = main(argv)
        finally:
            cli_mod.universe.build_universe = original_universe
            cli_mod.scan = original_scan
            cli_mod.scan_intraday = original_intraday
            session.ny_as_of = original_as_of
            session.public_eod_decision = original_decision
            if original_output is None:
                os.environ.pop("GITHUB_OUTPUT", None)
            else:
                os.environ["GITHUB_OUTPUT"] = original_output
        return code, buffer.getvalue()

    def test_cli_matching_dates_exit_zero(self):
        code, out = self._run(
            pd.DataFrame([_row("2026-11-13")]), as_of=date(2026, 11, 13)
        )
        self.assertEqual(code, 0, out)
        self.assertIn("EOD session check passed", out)
        self.assertIn("2026-11-13", out)
        self.assertNotIn("Refusing to publish", out)
        checker, checker_out = _capture_check(
            [str(self.dir / "report.csv"), str(self.dir / "index.html")],
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(checker, 0, checker_out)

    def test_cli_stale_dates_exit_nonzero_and_do_not_publish(self):
        code, out = self._run(
            pd.DataFrame([_row("2020-01-02")]), as_of=date(2026, 11, 13)
        )
        self.assertEqual(code, 1)
        self.assertIn("EOD session check failed", out)
        self.assertIn("Refusing to publish", out)
        self.assertIn("2020-01-02", out)
        # The workflow publish step is the next command. A non-zero status
        # skips it. The pre-publish checker agrees.
        checker, checker_out = _capture_check(
            [str(self.dir / "report.csv"), str(self.dir / "index.html")],
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(checker, 1)
        self.assertIn("Refusing to publish", checker_out)

    def test_cli_missing_rows_exit_nonzero(self):
        frame = pd.DataFrame(columns=list(_row("2026-11-13")))
        code, out = self._run(frame)
        self.assertEqual(code, 1)
        self.assertIn("no row dates", out)
        self.assertIn("Refusing to publish", out)

    def test_mixed_rows_are_rewritten_and_publish_check_passes(self):
        code, out = self._run(
            pd.DataFrame([
                _row("2026-11-13", "AAA"),
                _row("2026-11-12", "BBB"),
                _row("", "CCC"),
                _row("2026-11-13", "DDD"),
            ]),
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(code, 0, out)
        self.assertIn("Dropped 2 row(s)", out)
        self.assertIn("2026-11-12", out)
        self.assertIn("missing", out)
        self.assertIn("EOD session check passed", out)
        self.assertNotIn("Refusing to publish", out)
        with open(self.dir / "report.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["symbol"] for row in rows], ["AAA", "DDD"])
        self.assertTrue(all(row["date"] == "2026-11-13" for row in rows))
        payload = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "eod")
        self.assertEqual(payload["n"], len(rows))
        self.assertNotIn("BBB", payload["rows_html"])
        self.assertNotIn("CCC", payload["rows_html"])
        self.assertNotIn("<td>2026-11-12</td>", payload["rows_html"])
        html = (self.dir / "index.html").read_text(encoding="utf-8")
        self.assertIn(f'id="nflag">{payload["n"]}<', html)
        self.assertNotIn("<td>2026-11-12</td>", html)
        self.assertIn("<td>2026-11-13</td>", html)
        checker, checker_out = _capture_check(
            [str(self.dir / "report.csv"), str(self.dir / "index.html")],
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(checker, 0, checker_out)

    def test_all_old_rows_are_still_blocked(self):
        code, out = self._run(
            pd.DataFrame([
                _row("2026-11-12", "AAA"),
                _row("2026-11-10", "BBB"),
            ]),
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(code, 1, out)
        self.assertIn("EOD session check failed", out)
        self.assertIn("Refusing to publish", out)
        self.assertIn("2026-11-12", out)
        self.assertIn("2026-11-10", out)
        with open(self.dir / "report.csv", newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        checker, checker_out = _capture_check(
            [str(self.dir / "report.csv"), str(self.dir / "index.html")],
            as_of=date(2026, 11, 13),
        )
        self.assertEqual(checker, 1)
        self.assertIn("Refusing to publish", checker_out)

    def test_intraday_does_not_apply_the_eod_gate(self):
        code, out = self._run(
            pd.DataFrame([_row("2020-01-02"), _row("2026-11-13", "BBB")]),
            intraday=True,
        )
        self.assertEqual(code, 0, out)
        self.assertNotIn("Refusing to publish", out)
        self.assertNotIn("Dropped", out)
        self.assertNotIn("Skipping scan", out)
        with open(self.dir / "report.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["symbol"] for row in rows], ["AAA", "BBB"])


class WorkflowTests(unittest.TestCase):
    def test_publish_step_is_after_the_date_check(self):
        text = Path(".github/workflows/eod-scan.yml").read_text(encoding="utf-8")
        fresh = text.index("--public-fresh")
        scan = text.index("python -m volume_scanner.cli")
        verify = text.index(
            "python -m volume_scanner.eod_session site/report.csv site/index.html"
        )
        publish = text.index("uses: peaceiris/actions-gh-pages")
        self.assertLess(fresh, scan)
        self.assertLess(scan, verify)
        self.assertLess(verify, publish)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("site/report.csv", text)
        self.assertIn("site/index.html", text)
        self.assertIn("steps.fresh.outputs.fresh != 'true'", text)
        self.assertIn("steps.scan.outputs.fresh != 'true'", text)
        intraday = Path(".github/workflows/intraday-scan.yml").read_text(encoding="utf-8")
        self.assertNotIn("--public-fresh", intraday)
        self.assertNotIn("steps.fresh.outputs.fresh", intraday)


class SkipIfFreshTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._cwd = os.getcwd()
        self._output = os.environ.pop("GITHUB_OUTPUT", None)

    def tearDown(self):
        os.chdir(self._cwd)
        if self._output is None:
            os.environ.pop("GITHUB_OUTPUT", None)
        else:
            os.environ["GITHUB_OUTPUT"] = self._output
        self._tmp.cleanup()

    def _pages(self, frame: pd.DataFrame, *, mode: str = "eod") -> dict[str, str]:
        folder = self.dir / "public-snapshot"
        folder.mkdir(exist_ok=True)
        csv_path, html_path = _write_pair(folder, frame, mode=mode)
        return {
            "index.html": html_path.read_text(encoding="utf-8"),
            "index.json": (folder / "index.json").read_text(encoding="utf-8"),
            "report.csv": csv_path.read_text(encoding="utf-8"),
        }

    def _run_cli(self, pages: dict[str, str] | None, *, fetch=None):
        import volume_scanner.cli as cli_mod
        import volume_scanner.eod_session as session

        calls = {"scan": 0, "urls": []}

        def _fetch(url: str) -> str:
            calls["urls"].append(url)
            if fetch is not None:
                return fetch(url)
            assert pages is not None
            for name, body in pages.items():
                if f"/{name}?" in url:
                    return body
            raise OSError(f"unexpected url {url}")

        def _universe(*_args, **_kwargs):
            return ["AAA"]

        def _scan(*_args, **_kwargs):
            calls["scan"] += 1
            return pd.DataFrame([_row("2026-11-13")])

        original_universe = cli_mod.universe.build_universe
        original_scan = cli_mod.scan
        original_intraday = cli_mod.scan_intraday
        original_as_of = session.ny_as_of
        original_fetch = session.fetch_public_text
        cli_mod.universe.build_universe = _universe
        cli_mod.scan = _scan
        cli_mod.scan_intraday = _scan
        session.ny_as_of = lambda now=None: date(2026, 11, 13)
        session.fetch_public_text = _fetch
        argv = [
            "--market", "us",
            "--html", str(self.dir / "out" / "index.html"),
            "--out", str(self.dir / "out" / "report.csv"),
        ]
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = main(argv)
        finally:
            cli_mod.universe.build_universe = original_universe
            cli_mod.scan = original_scan
            cli_mod.scan_intraday = original_intraday
            session.ny_as_of = original_as_of
            session.fetch_public_text = original_fetch
        return code, buffer.getvalue(), calls

    def _run_public_fresh_command(self, pages: dict[str, str]) -> tuple[int, str, str]:
        import volume_scanner.eod_session as session

        output = self.dir / "github_output.txt"
        os.environ["GITHUB_OUTPUT"] = str(output)

        def _fetch(url: str) -> str:
            for name, body in pages.items():
                if f"/{name}?" in url:
                    return body
            raise OSError(f"unexpected url {url}")

        original_as_of = session.ny_as_of
        original_fetch = session.fetch_public_text
        session.ny_as_of = lambda now=None: date(2026, 11, 13)
        session.fetch_public_text = _fetch
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = check_main(["--public-fresh"])
        finally:
            session.ny_as_of = original_as_of
            session.fetch_public_text = original_fetch
        text = output.read_text(encoding="utf-8") if output.exists() else ""
        return code, buffer.getvalue(), text

    def test_public_base_is_the_live_site(self):
        self.assertEqual(
            PUBLIC_EOD_BASE.rstrip("/"),
            "https://williamyeung72.github.io/volume-scanner",
        )
        self.assertTrue(PUBLIC_EOD_BASE.startswith("https://"))

    def test_fetcher_refuses_non_https(self):
        with self.assertRaises(OSError):
            fetch_public_text("file:///tmp/index.html")
        with self.assertRaises(OSError):
            fetch_public_text(
                "http://williamyeung72.github.io/volume-scanner/index.html"
            )

    def test_fresh_public_page_skips_and_exits_zero(self):
        pages = self._pages(pd.DataFrame([
            _row("2026-11-13", "AAA"),
            _row("2026-11-13", "BBB"),
        ]))
        code, out, calls = self._run_cli(pages)
        self.assertEqual(code, 0, out)
        self.assertEqual(calls["scan"], 0)
        self.assertIn("Skipping scan and publish.", out)
        self.assertIn(PUBLIC_EOD_BASE, out)
        self.assertNotIn("Refusing to publish", out)
        self.assertFalse((self.dir / "out" / "report.csv").exists())
        for name in ("index.html", "index.json", "report.csv"):
            self.assertTrue(
                any(f"/{name}?" in url for url in calls["urls"]),
                calls["urls"],
            )
        self.assertTrue(all(url.startswith("https://") for url in calls["urls"]))

    def test_stale_public_page_does_not_skip(self):
        pages = self._pages(pd.DataFrame([_row("2026-11-12")] * 2))
        code, out, calls = self._run_cli(pages)
        self.assertEqual(calls["scan"], 1)
        self.assertEqual(code, 0, out)
        self.assertIn("Continuing with the scan", out)
        self.assertNotIn("Skipping scan", out)

    def test_mixed_public_dates_do_not_skip(self):
        pages = self._pages(pd.DataFrame([
            _row("2026-11-13", "AAA"),
            _row("2026-11-12", "BBB"),
        ]))
        code, out, calls = self._run_cli(pages)
        self.assertEqual(calls["scan"], 1, out)
        self.assertIn("Continuing with the scan", out)
        self.assertNotIn("Skipping scan", out)
        self.assertEqual(code, 0, out)

    def test_public_intraday_mode_does_not_skip(self):
        pages = self._pages(pd.DataFrame([_row("2026-11-13")]), mode="intraday")
        _code, out, calls = self._run_cli(pages)
        self.assertEqual(calls["scan"], 1, out)
        self.assertIn("Continuing with the scan", out)
        self.assertNotIn("Skipping scan", out)

    def test_unreachable_public_page_does_not_skip(self):
        def _boom(_url: str) -> str:
            raise TimeoutError("timed out")

        _code, out, calls = self._run_cli({}, fetch=_boom)
        self.assertEqual(calls["scan"], 1, out)
        self.assertIn("Continuing with the scan", out)
        self.assertNotIn("Skipping scan", out)

    def test_local_capture_does_not_count_as_fresh(self):
        site = self.dir / "site"
        capture = self.dir / "capture"
        site.mkdir()
        capture.mkdir()
        _write_pair(
            site,
            pd.DataFrame([_row("2026-11-13"), _row("2026-11-13", "BBB")]),
        )
        _write_pair(capture, pd.DataFrame([_row("2026-11-13")]))
        os.chdir(self.dir)
        pages = self._pages(pd.DataFrame([_row("2026-11-12")]))
        _code, out, calls = self._run_cli(pages)
        self.assertEqual(calls["scan"], 1, out)
        self.assertIn("Continuing with the scan", out)
        self.assertNotIn("Skipping scan", out)
        self.assertTrue(any("index.json" in url for url in calls["urls"]))

    def test_public_fresh_command_sets_output_and_exits_zero(self):
        pages = self._pages(pd.DataFrame([_row("2026-11-13")]))
        code, out, output = self._run_public_fresh_command(pages)
        self.assertEqual(code, 0, out)
        self.assertIn("Skipping scan and publish.", out)
        self.assertIn("fresh=true", output)

    def test_public_fresh_command_does_not_fail_when_the_page_is_mixed(self):
        pages = self._pages(pd.DataFrame([
            _row("2026-11-13", "AAA"),
            _row("2026-11-12", "BBB"),
        ]))
        code, out, output = self._run_public_fresh_command(pages)
        self.assertEqual(code, 0, out)
        self.assertIn("Continuing with the scan", out)
        self.assertIn("fresh=false", output)
        self.assertNotIn("fresh=true", output)


class RetainAsOfRowTests(unittest.TestCase):
    def test_keeps_only_as_of_rows_without_mutating_the_input(self):
        import volume_scanner.eod_session as session

        frame = pd.DataFrame([
            _row("2026-11-13", "AAA"),
            _row("", "BBB"),
            _row("2026-11-12", "CCC"),
            _row("2026-11-13", "DDD"),
        ])
        original = session.ny_as_of
        session.ny_as_of = lambda now=None: date(2026, 11, 13)
        try:
            kept, as_of, dropped = retain_as_of_rows(frame)
        finally:
            session.ny_as_of = original
        self.assertEqual(as_of, date(2026, 11, 13))
        self.assertEqual(list(kept["symbol"]), ["AAA", "DDD"])
        self.assertEqual(dropped, ["2026-11-12", "missing"])
        self.assertEqual(list(frame["symbol"]), ["AAA", "BBB", "CCC", "DDD"])


def _capture_check(argv: list[str], *, as_of: date | None = None) -> tuple[int, str]:
    import volume_scanner.eod_session as session

    original = session.ny_as_of
    if as_of is not None:
        session.ny_as_of = lambda now=None, pinned=as_of: pinned
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = check_main(argv)
    finally:
        session.ny_as_of = original
    return code, buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
