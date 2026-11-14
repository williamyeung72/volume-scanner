"""EOD publish rules for the completed NY session.

NY ``as_of`` is the most recent completed US regular session: a weekday that
is not a full-day NYSE holiday, once the clock in America/New_York is at or
after 16:00 on that date. Weekends and full-day holidays are not sessions.
An early close (1:00 p.m. ET) still counts as that calendar date after 16:00 ET.

Two decisions use that date, and neither one weakens the publish gate:

* Skip-if-fresh. Before an EOD scan, fetch the public Pages files
  (``index.html``, ``index.json``, ``report.csv``). When they are
  ``mode=eod`` and every row date equals NY as_of, the run exits 0 without
  scanning or publishing. A file in a local capture does not count. If
  freshness cannot be proven, the scan runs.

* As-of rows only. After a scan, rows whose date is not NY as_of are removed
  and the page is written from what remains, with ``n`` matching the kept
  rows. :func:`verify_eod_outputs` still fails closed unless ``mode=eod`` and
  every remaining row date is NY as_of. Mixed dates are not published as-is.

Intraday runs do not use either decision.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_NY = "America/New_York"
_CLOSE = time(16, 0)
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_INTRADAY_RE = re.compile(r"const\s+intraday\s*=\s*true")

_RULE = (
    "NY as_of is the most recent completed US regular session "
    "(weekday, not a full-day NYSE holiday, at or after 16:00 America/New_York). "
    "Early closes still count as that session after 16:00 ET."
)

# Live GitHub Pages site. Freshness is this site's bytes, never a local capture.
PUBLIC_EOD_BASE = "https://williamyeung72.github.io/volume-scanner/"
PUBLIC_EOD_FILES = ("index.html", "index.json", "report.csv")
_FETCH_LIMIT = 5_000_000
_FETCH_TIMEOUT = 20


class SessionDateError(Exception):
    """Built EOD artifacts are not the NY as_of session. Do not publish."""


def ny_as_of(now: datetime | None = None) -> date:
    """Return the NY as_of session date for ``now`` (default: current UTC)."""
    try:
        zone = ZoneInfo(_NY)
    except Exception as exc:  # pragma: no cover - tzdata is present on Actions
        raise SessionDateError(
            f"cannot resolve {_NY} ({exc}); refusing to publish"
        ) from exc

    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now_ny = now.replace(tzinfo=zone)
    else:
        now_ny = now.astimezone(zone)

    day = now_ny.date()
    for _ in range(14):
        if is_nyse_session(day):
            close = datetime.combine(day, _CLOSE, tzinfo=zone)
            if now_ny >= close:
                return day
        day -= timedelta(days=1)
    raise SessionDateError(
        "no completed NYSE session within 14 days; refusing to publish"
    )


def is_nyse_session(day: date) -> bool:
    """True when ``day`` is a regular NYSE session (early closes included)."""
    return day.weekday() < 5 and day not in nyse_full_closures(day.year)


def nyse_full_closures(year: int) -> set[date]:
    """Full-day NYSE closures in ``year``.

    Saturday holidays are observed the Friday before, Sunday holidays the
    Monday after, except New Year's Day on Saturday: NYSE stays open on the
    preceding Friday (no year-end closure). One-off closures (for example a
    national day of mourning) are not included; a run on such a day fails
    closed if the page is still the previous session.
    """
    closures: set[date] = set()

    def add(day: date, *, saturday_unobserved: bool = False) -> None:
        observed = _observed(day, saturday_unobserved=saturday_unobserved)
        if observed is not None and observed.year == year:
            closures.add(observed)

    add(date(year, 1, 1), saturday_unobserved=True)
    add(_nth_weekday(year, 1, 0, 3))   # Martin Luther King Jr. Day
    add(_nth_weekday(year, 2, 0, 3))   # Washington's Birthday
    add(_good_friday(year))
    add(_last_weekday(year, 5, 0))     # Memorial Day
    add(date(year, 6, 19))             # Juneteenth
    add(date(year, 7, 4))              # Independence Day
    add(_nth_weekday(year, 9, 0, 1))   # Labor Day
    add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    add(date(year, 12, 25))            # Christmas
    return closures


def verify_eod_outputs(
    csv_path: str,
    html_path: str | None = None,
    *,
    now: datetime | None = None,
) -> date:
    """Require EOD artifacts to match NY as_of.

    ``html_path`` is the built ``index.html``. Its sibling ``index.json`` must
    say ``mode=eod``, and every date cell in the index (HTML and JSON) and
    every ``date`` cell in ``report.csv`` must equal NY as_of. Missing, mixed,
    or stale dates raise :class:`SessionDateError`.

    This does not drop rows. Callers that scanned a mixed session must filter
    with :func:`retain_as_of_rows` and rewrite the files first. A page that is
    still mixed fails closed.

    When ``html_path`` is omitted, only the CSV is checked. The publish path
    always passes the index.
    """
    expected = ny_as_of(now)
    errors: list[str] = []

    if html_path:
        errors.extend(_check_index(html_path, expected))
    errors.extend(_check_csv(csv_path, expected))

    if errors:
        detail = "\n".join(errors)
        raise SessionDateError(
            f"NY as_of is {expected.isoformat()}. {_RULE}\n{detail}"
        )
    return expected


def _check_index(html_path: str, expected: date) -> list[str]:
    errors: list[str] = []
    try:
        html = Path(html_path).read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{html_path}: cannot read index ({exc})"]

    if _INTRADAY_RE.search(html):
        errors.append(f"{html_path}: page is intraday, expected mode=eod")

    tbody = re.search(r"<tbody>(.*?)</tbody>", html, re.IGNORECASE | re.DOTALL)
    if tbody is None:
        errors.append(f"{html_path}: no table body")
    else:
        errors.extend(
            _date_problems(html_path, _dates_from_rows(tbody.group(1)), expected)
        )

    json_path = _json_sibling(html_path)
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{json_path}: cannot read index snapshot ({exc})")
        return errors

    mode = payload.get("mode")
    if mode != "eod":
        errors.append(f"{json_path}: mode={mode!r}, expected 'eod'")
    rows_html = payload.get("rows_html")
    if not isinstance(rows_html, str):
        errors.append(f"{json_path}: rows_html missing")
    else:
        errors.extend(
            _date_problems(
                f"{json_path} rows", _dates_from_rows(rows_html), expected
            )
        )
    return errors


def _check_csv(csv_path: str, expected: date) -> list[str]:
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "date" not in reader.fieldnames:
                return [f"{csv_path}: missing date column"]
            dates = [_parse_session_date(record.get("date")) for record in reader]
    except OSError as exc:
        return [f"{csv_path}: cannot read report.csv ({exc})"]
    return _date_problems(csv_path, dates, expected)


def _dates_from_rows(fragment: str) -> list[date | None]:
    dates: list[date | None] = []
    for row in _ROW_RE.findall(fragment):
        cells = _CELL_RE.findall(row)
        if len(cells) < 2:
            continue
        text = " ".join(_TAG_RE.sub(" ", cells[1]).split())
        dates.append(_parse_session_date(text))
    return dates


def _date_problems(label: str, dates: list[date | None], expected: date) -> list[str]:
    if not dates:
        return [f"{label}: no row dates"]
    missing = sum(item is None for item in dates)
    found = sorted({item.isoformat() for item in dates if item is not None})
    if missing:
        readable = ", ".join(found) if found else "none"
        return [
            f"{label}: {missing} of {len(dates)} row(s) missing a session date "
            f"(readable: {readable})"
        ]
    if found != [expected.isoformat()]:
        return [
            f"{label}: row dates {', '.join(found)} ({len(dates)} rows) "
            f"!= NY as_of {expected.isoformat()}"
        ]
    return []


def _parse_session_date(value) -> date | None:
    if value is None:
        return None
    match = _ISO_DATE.search(str(value).strip())
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _json_sibling(html_path: str) -> str:
    stem, dot, ext = html_path.rpartition(".")
    if dot and ext.lower() in {"html", "htm"}:
        return stem + ".json"
    return html_path + ".json"


def _observed(day: date, *, saturday_unobserved: bool = False) -> date | None:
    if day.weekday() == 5:  # Saturday
        if saturday_unobserved:
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _good_friday(year: int) -> date:
    # Anonymous Gregorian computus; NYSE is closed the Friday before Easter.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)


def retain_as_of_rows(df, *, now: datetime | None = None):
    """Copy ``df``, keeping only rows whose session date is NY as_of.

    Returns ``(kept, as_of, dropped_labels)``. ``dropped_labels`` is a sorted
    list of the session dates that were removed; a row with no session date
    contributes ``"missing"``. The input frame is not modified. An empty frame,
    or one without a ``date`` column, is returned unchanged so the publish
    gate can still refuse it.
    """
    expected = ny_as_of(now)
    if df is None or len(df) == 0 or "date" not in df.columns:
        return df, expected, []
    keep_at: list[int] = []
    dropped: list[str] = []
    for _index, value in enumerate(df["date"].tolist()):
        parsed = _parse_session_date(value)
        if parsed == expected:
            keep_at.append(_index)
        else:
            dropped.append(parsed.isoformat() if parsed is not None else "missing")
    kept = df.iloc[keep_at].copy()
    return kept, expected, sorted(set(dropped))


def fetch_public_text(url: str, *, timeout: float = _FETCH_TIMEOUT) -> str:
    """GET one public https URL and return its body as text.

    Non-https URLs are refused so a local ``file:`` path cannot satisfy the
    freshness check. Oversized bodies raise, which the caller treats as
    "not proven fresh".
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise OSError(f"refusing non-https URL scheme {parsed.scheme!r}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "volume-scanner-eod-freshness",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Accept": "text/html,application/json,text/csv,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https":
            raise OSError(f"refusing redirect to {final.scheme!r}")
        raw = response.read(_FETCH_LIMIT + 1)
    if len(raw) > _FETCH_LIMIT:
        raise OSError(f"response larger than {_FETCH_LIMIT} bytes")
    return raw.decode("utf-8")


def public_eod_decision(
    now: datetime | None = None,
    *,
    base_url: str = PUBLIC_EOD_BASE,
    fetch: Callable[[str], str] | None = None,
) -> tuple[bool, str]:
    """Decide whether an EOD run should skip because the public page is fresh.

    Returns ``(True, message)`` only when ``index.html``, ``index.json``, and
    ``report.csv`` were fetched from ``base_url`` and :func:`verify_eod_outputs`
    accepts them for NY as_of. Any uncertainty returns ``(False, message)`` so
    the caller scans. This function does not raise.
    """
    try:
        reason = _public_mismatch(now, base_url=base_url, fetch=fetch)
    except Exception as exc:
        reason = f"freshness check error: {exc.__class__.__name__}: {exc}"
    if reason is None:
        try:
            day = ny_as_of(now).isoformat()
        except Exception:
            day = "NY as_of"
        return True, (
            f"Public EOD page at {base_url} is mode=eod and every row date is "
            f"{day}. Skipping scan and publish."
        )
    return False, (
        "Public EOD page is not proven fresh. Continuing with the scan.\n"
        + reason
    )


def write_github_output(name: str, value: str) -> None:
    """Append one ``name=value`` line to ``GITHUB_OUTPUT`` when Actions sets it."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _public_file_url(base_url: str, name: str) -> str:
    base = base_url if base_url.endswith("/") else base_url + "/"
    stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    return f"{base}{name}?t={stamp}"


def _public_mismatch(
    now: datetime | None,
    *,
    base_url: str,
    fetch: Callable[[str], str] | None,
) -> str | None:
    getter = fetch if fetch is not None else fetch_public_text
    pages: dict[str, str] = {}
    for name in PUBLIC_EOD_FILES:
        url = _public_file_url(base_url, name)
        try:
            body = getter(url)
        except Exception as exc:
            return f"could not fetch {name}: {exc.__class__.__name__}: {exc}"
        if not isinstance(body, str) or not body.strip():
            return f"{name} was empty or not text"
        pages[name] = body
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in pages.items():
                (root / name).write_text(body, encoding="utf-8")
            verify_eod_outputs(
                str(root / "report.csv"),
                str(root / "index.html"),
                now=now,
            )
    except SessionDateError as exc:
        return str(exc)
    except Exception as exc:
        return (
            f"could not read the public EOD page: {exc.__class__.__name__}: {exc}"
        )
    return None


def _public_fresh_main() -> int:
    """Always exit 0. ``fresh=true`` only when the public page proves it."""
    skip, message = public_eod_decision()
    print(message)
    write_github_output("fresh", "true" if skip else "false")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--public-fresh"]:
        return _public_fresh_main()
    if len(args) != 2:
        print(
            "usage: python -m volume_scanner.eod_session REPORT_CSV INDEX_HTML\n"
            "       python -m volume_scanner.eod_session --public-fresh",
            file=sys.stdout,
        )
        return 2
    try:
        expected = verify_eod_outputs(args[0], args[1])
    except SessionDateError as exc:
        print("EOD session check failed.")
        print(exc)
        print("Refusing to publish.")
        return 1
    print(
        f"EOD session check passed. mode=eod and every row date is "
        f"{expected.isoformat()}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
