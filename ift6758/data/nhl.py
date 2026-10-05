"""Download, cache and load raw NHL play-by-play data (Milestone 1, Task 1).

Endpoint::

    https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play

Game IDs are 10 digits, ``YYYY TT NNNN``:

* ``YYYY``: year the season *starts* (``2016`` means 2016-17)
* ``TT``: game type, ``02`` = regular season, ``03`` = playoffs
* ``NNNN``: regular season -> game number (``0001``, ``0002``, ...);
  playoffs -> ``0RMG`` = round (1-4), matchup within the round, game (1-7).
  E.g. ``2022030411`` = 2022-23 playoffs, round 4, matchup 1, game 1.

Raw JSON is cached under ``$NHL_DATA_DIR`` (default ``./data/raw``) as
``<season>/<game_id>.json``. The cache is never committed (see ``.gitignore``);
every function below checks it before hitting the network, because someone else
pays for each request.

Example::

    from ift6758.data.nhl import download_season, load_season

    download_season(2016)                  # first call downloads, later calls are free
    games = load_season(2016, "regular")   # list of raw dicts, straight from disk
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
DATA_DIR_ENV = "NHL_DATA_DIR"
DEFAULT_DATA_DIR = "data/raw"

REGULAR = "regular"
PLAYOFFS = "playoffs"
GAME_TYPE_CODE = {REGULAR: "02", PLAYOFFS: "03"}

# Number of series (matchups) in each playoff round.
MATCHUPS_PER_ROUND = {1: 8, 2: 4, 3: 2, 4: 1}
MAX_PLAYOFF_GAMES_PER_SERIES = 7
# Safety cap when counting up through a regular season (the longest has ~1,300 games).
MAX_REGULAR_SEASON_GAMES = 1500


# --------------------------------------------------------------------------- ids
def regular_season_game_id(season: int, number: int) -> str:
    """ID of regular-season game ``number`` (1-based) of the season starting in ``season``."""
    return f"{season}02{number:04d}"


def playoff_game_id(season: int, round_: int, matchup: int, game: int) -> str:
    """ID of a playoff game, e.g. ``playoff_game_id(2022, 4, 1, 1) == '2022030411'``."""
    return f"{season}030{round_}{matchup}{game}"


# ------------------------------------------------------------------------- cache
def get_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Cache root: explicit argument > ``$NHL_DATA_DIR`` > ``./data/raw``."""
    path = Path(data_dir or os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(game_id: str, data_dir: str | os.PathLike | None) -> Path:
    return get_data_dir(data_dir) / game_id[:4] / f"{game_id}.json"


# ------------------------------------------------------------------ one game
def get_game(
    game_id: str,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
    retries: int = 3,
) -> dict | None:
    """Return the raw play-by-play dict for one game.

    Reads the local cache first and only downloads on a miss (then saves the
    raw JSON). Returns ``None`` if the API says the game does not exist (404);
    that is normal, e.g. a playoff series that ended before game 7. Transient
    failures (429, 5xx, timeouts) are retried with back-off and then raised;
    any other HTTP error is raised immediately.

    Args:
        game_id: 10-digit game ID, see the module docstring.
        data_dir: cache root; defaults to ``$NHL_DATA_DIR`` or ``./data/raw``.
        session: optional ``requests.Session`` to reuse connections.
        delay: seconds to sleep after each *network* request (not on cache hits).
        retries: number of attempts for transient failures (429, 5xx, timeouts).
    """
    path = _cache_path(game_id, data_dir)
    if path.exists():
        return json.loads(path.read_text())

    http = session or requests
    url = PBP_URL.format(game_id=game_id)
    for attempt in range(1, retries + 1):
        try:
            resp = http.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
            time.sleep(delay + 2**attempt)
            continue
        time.sleep(delay)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:  # transient: back off, retry
            if attempt == retries:
                resp.raise_for_status()
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()  # any other 4xx is a real error, don't retry
        data = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"unreachable: retries must be >= 1 (got {retries})")


# ------------------------------------------------------------------- one season
def download_regular_season(
    season: int,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
    max_misses: int = 5,
) -> list[dict]:
    """Download games 1, 2, 3, ... until ``max_misses`` consecutive 404s.

    Season lengths vary (lockout/COVID/expansion), so we do not hard-code them.
    """
    games, misses = [], 0
    with tqdm(desc=f"{season} regular", unit="game") as bar:
        for number in range(1, MAX_REGULAR_SEASON_GAMES + 1):
            data = get_game(regular_season_game_id(season, number), data_dir, session, delay)
            if data is None:
                misses += 1
                if misses >= max_misses:
                    break
                continue
            misses = 0
            games.append(data)
            bar.update(1)
    return games


def download_playoffs(
    season: int,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
) -> list[dict]:
    """Download every playoff series, skipping to the next series at the first missing game."""
    games = []
    with tqdm(desc=f"{season} playoffs", unit="game") as bar:
        for round_, n_matchups in MATCHUPS_PER_ROUND.items():
            for matchup in range(1, n_matchups + 1):
                for game in range(1, MAX_PLAYOFF_GAMES_PER_SERIES + 1):
                    gid = playoff_game_id(season, round_, matchup, game)
                    data = get_game(gid, data_dir, session, delay)
                    if data is None:
                        break  # series is over
                    games.append(data)
                    bar.update(1)
    return games


def download_season(
    season: int,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
) -> dict[str, list[dict]]:
    """Download (or load from cache) a full season: ``{"regular": [...], "playoffs": [...]}``."""
    session = session or requests.Session()
    return {
        REGULAR: download_regular_season(season, data_dir, session, delay),
        PLAYOFFS: download_playoffs(season, data_dir, session, delay),
    }


def download_seasons(
    first: int = 2016,
    last: int = 2023,
    data_dir: str | os.PathLike | None = None,
    delay: float = 0.5,
) -> dict[int, dict[str, list[dict]]]:
    """Download 2016-17 ... 2023-24 by default (``first``/``last`` are start years)."""
    session = requests.Session()
    return {s: download_season(s, data_dir, session, delay) for s in range(first, last + 1)}


# ------------------------------------------------------------ offline loading
def load_season(
    season: int,
    game_type: str = REGULAR,
    data_dir: str | os.PathLike | None = None,
) -> list[dict]:
    """Load already-downloaded games from disk only (never touches the network).

    Args:
        season: start year, e.g. ``2016`` for 2016-17.
        game_type: ``"regular"`` or ``"playoffs"``.
    """
    if game_type not in GAME_TYPE_CODE:
        raise ValueError(f"game_type must be one of {sorted(GAME_TYPE_CODE)}, got {game_type!r}")
    folder = get_data_dir(data_dir) / str(season)
    files = sorted(folder.glob(f"{season}{GAME_TYPE_CODE[game_type]}*.json"))
    return [json.loads(f.read_text()) for f in files]


def main() -> None:
    """Command line entry point: ``uv run download-nhl [--first 2016] [--last 2023]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Download NHL play-by-play data into the local cache.")
    parser.add_argument("--first", type=int, default=2016, help="first season start year (default 2016)")
    parser.add_argument("--last", type=int, default=2023, help="last season start year (default 2023)")
    parser.add_argument("--data-dir", default=None, help="cache folder (default $NHL_DATA_DIR or data/raw)")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()
    download_seasons(args.first, args.last, args.data_dir, args.delay)
