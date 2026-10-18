"""Data acquisition helpers.

Importing here lets you write ``from ift6758.data import get_game`` instead of
``from ift6758.data.nhl import get_game`` (both work).
"""
from .nhl import (
    download_playoffs,
    download_regular_season,
    download_season,
    download_seasons,
    get_game,
    load_season,
    playoff_game_id,
    regular_season_game_id,
)

from .tidy_data import game_to_pandas
from .question_1 import get_player_stats

__all__ = [
    "download_playoffs",
    "download_regular_season",
    "download_season",
    "download_seasons",
    "get_game",
    "get_player_stats",
    "load_season",
    "playoff_game_id",
    "regular_season_game_id",
]
