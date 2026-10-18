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
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw"

REGULAR = "regular"
PLAYOFFS = "playoffs"
GAME_TYPE_CODE = {REGULAR: "02", PLAYOFFS: "03"}

# Number of series (matchups) in each playoff round.
MATCHUPS_PER_ROUND = {1: 8, 2: 4, 3: 2, 4: 1}
MAX_PLAYOFF_GAMES_PER_SERIES = 7
# Safety cap when counting up through a regular season (the longest has ~1,300 games).
MAX_REGULAR_SEASON_GAMES = 1500


# --------------------------------------------------------------------------- ids
# Returns game id for a regular season game given the season and game number
def regular_season_game_id(season: int, number: int) -> str:
    """ID of regular-season game ``number`` (1-based) of the season starting in ``season``."""
    return f"{season}02{number:04d}"

# Returns game id for a playoff game given the season, round, matchup, and game number
def playoff_game_id(season: int, round_: int, matchup: int, game: int) -> str:
    """ID of a playoff game, e.g. ``playoff_game_id(2022, 4, 1, 1) == '2022030411'``."""
    return f"{season}030{round_}{matchup}{game}"


# Returns a Path object and creates one if it does not exist.
def get_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Cache root: explicit argument > ``$NHL_DATA_DIR`` > ``./data/raw``."""
    path = Path(data_dir or os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path

# Creates a path pointer to the cache file for a given game_id and data_dir
def _cache_path(game_id: str, data_dir: str | os.PathLike | None) -> Path:
    return get_data_dir(data_dir) / game_id[:4] / f"{game_id}.json"


# ------------------------------------------------------------------ one game
# Fetches a single game from the NHL API, caches it, and returns the raw JSON data
def get_game(
    game_id: str,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
) -> dict | None:
    path = _cache_path(game_id, data_dir)
    # if path exists, load the cached JSON and return it
    if path.exists():
        return json.loads(path.read_text())

    # otherwise, request the game from the NHL API
    resp = (session or requests).get(PBP_URL.format(game_id=game_id), timeout=30)
    time.sleep(delay)  # wait a bit to avoid hammering the NHL servers
    # if the game does not exist, return None
    if resp.status_code == 404:
        return None
    # if the request failed for any other reason, raise an exception
    resp.raise_for_status()

    # if the request succeeded, parse the JSON, save it to the cache, and return it
    data = resp.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ------------------------------------------------------------------- one season
def download_regular_season(
    season: int,
    data_dir: str | os.PathLike | None = None,
    session: requests.Session | None = None,
    delay: float = 0.5,
    max_misses: int = 5,
) -> list[dict]:
    # download games 1, 2, 3, ... until ``max_misses`` consecutive 404s
    games, misses = [], 0
    # tdqm progress bar for the number of games downloaded successfully
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
    # download every playoff series, skipping to the next series at the first missing game
    games = []
    with tqdm(desc=f"{season} playoffs", unit="game") as bar:
        for round_, n_matchups in MATCHUPS_PER_ROUND.items():
            # n_matchups is 8, 4, 2, 1 for rounds 1, 2, 3, 4
            for matchup in range(1, n_matchups + 1):
                # max 7 games per series
                for game in range(1, MAX_PLAYOFF_GAMES_PER_SERIES + 1):
                    gid = playoff_game_id(season, round_, matchup, game)
                    data = get_game(gid, data_dir, session, delay)
                    if data is None:
                        break  # series is over
                    games.append(data)
                    bar.update(1)
    return games

# ----------------------------------------------------------- one season
def download_season(season: int, data_dir: str | os.PathLike | None = None) -> dict[str, list[dict]]:
    """Download (or load from cache) one full season: ``{"regular": [...], "playoffs": [...]}``."""
    session = requests.Session()
    return {
        REGULAR: download_regular_season(season, data_dir, session),
        PLAYOFFS: download_playoffs(season, data_dir, session),
    }

# ---------------------------------------------------------- multiple seasons
def download_seasons(first=2016, last=2023, data_dir=None, delay=0.5) -> None:
    """Download 2016-17 ... 2023-24 by default into the cache (``first``/``last`` are start years)."""
    session = requests.Session()
    for season in range(first, last + 1):
        download_regular_season(season, data_dir, session, delay)
        download_playoffs(season, data_dir, session)

# ------------------------------------------------------------ offline loading
# Useful for task 3
def load_season(
    season: int,
    game_type: str = REGULAR,
    data_dir: str | os.PathLike | None = None,
) -> list[dict]:
    # Load already-downloaded games from disk only (never touches the network)
    if game_type not in GAME_TYPE_CODE:
        raise ValueError(f"game_type must be one of {sorted(GAME_TYPE_CODE)}, got {game_type!r}")
    folder = get_data_dir(data_dir) / str(season)
    files = sorted(folder.glob(f"{season}{GAME_TYPE_CODE[game_type]}*.json"))
    return [json.loads(f.read_text()) for f in files]


def main():
    # Command line entry point: ``uv run download-nhl [--first 2016] [--last 2023]``.
    import argparse

    parser = argparse.ArgumentParser(description="Download NHL play-by-play data into the local cache.")
    parser.add_argument("--first", type=int, default=2016, help="first season start year (default 2016)")
    parser.add_argument("--last", type=int, default=2023, help="last season start year (default 2023)")
    parser.add_argument("--data-dir", default=None, help="cache folder (default $NHL_DATA_DIR or data/raw)")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()
    download_seasons(args.first, args.last, args.data_dir, args.delay)


# # step 1: fetch one game and look at it
# import requests

# url = "https://api-web.nhle.com/v1/gamecenter/2022030311/play-by-play"
# resp = requests.get(url, timeout=30)
# print(resp.status_code)
# data = resp.json()
# print(data.keys())

# # step 2: save it to disk
# import json
# from pathlib import Path

# path = Path("data/raw/2022030311.json")
# path.parent.mkdir(parents=True, exist_ok=True)
# path.write_text(json.dumps(data))


# # step 3: add the cache check, which is the heart of this task
# def get_game(game_id):
#     path = Path(f"data/raw/{game_id}.json")
#     if path.exists():
#         return json.loads(path.read_text())
#     url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
#     resp = requests.get(url, timeout=30)
#     if resp.status_code == 404:
#         return None
#     resp.raise_for_status()
#     data = resp.json()
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(data))
#     return data