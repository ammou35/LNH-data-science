"""Construction rapide (et mise en cache) du dataframe de la partie 2 (Feature Engineering I).

Pourquoi ce fichier ?
    L'ancienne version traitait chaque match séparément (json_normalize, convert_dtypes,
    infer_attack_direction... x ~9000 matchs) et recommençait tout à chaque exécution :
    4 à 8 minutes. Ici :
      1. on extrait seulement les champs utiles des tirs, en Python pur (pas de json_normalize) ;
      2. on fait tous les calculs (direction, distance, angle, filet désert) UNE fois par saison ;
      3. on sauvegarde chaque saison en Parquet : les appels suivants prennent ~1 seconde.

Utilisation :
    from ift6758.features.feat_eng_1_fast import load_feat_eng_1_df

    df = load_feat_eng_1_df(2016, 2022)            # 1re fois : construit + met en cache
    df = load_feat_eng_1_df(2016, 2022)            # ensuite : lit le cache (~1 s)
    df = load_feat_eng_1_df(2016, 2022, rebuild=True)  # forcer la reconstruction (code modifié)

    X, y = df[["distance"]], df["is_goal"]

Colonnes retournées :
    season, game_id, event_id, period, team_id, is_home_team, event_type,
    x, y, distance, angle, is_goal, empty_net
    - is_goal   : bool
    - empty_net : booléen nullable (pd.NA si le situationCode ne permet pas de le déterminer)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ift6758.data.nhl import GAME_TYPE_CODE, REGULAR, get_data_dir
from ift6758.data.tidy_data import infer_attack_direction

SHOT_TYPES = ("shot-on-goal", "goal")
NET_X = 89  # les filets sont à x = ±89 pi

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "feat_eng_1"

# Incrémenter quand la logique change : les anciens caches sont alors ignorés automatiquement.
CACHE_VERSION = 1


# --------------------------------------------------------------------------------------
# 1. Extraction : JSON brut -> lignes (dictionnaires), sans pandas
# --------------------------------------------------------------------------------------
def _shot_rows(game: dict) -> list[dict]:
    """Extrait les tirs et buts d'un match sous forme de liste de dictionnaires."""
    home_id = game["homeTeam"]["id"]
    rows = []
    for play in game.get("plays", []):
        if play.get("typeDescKey") not in SHOT_TYPES:
            continue
        details = play.get("details") or {}
        team_id = details.get("eventOwnerTeamId")
        rows.append({
            "game_id": game["id"],
            "event_id": play.get("eventId"),
            "period": (play.get("periodDescriptor") or {}).get("number"),
            "team_id": team_id,
            "is_home_team": None if team_id is None else team_id == home_id,
            "home_team_defending_side": play.get("homeTeamDefendingSide"),
            "event_type": play["typeDescKey"],
            "x": details.get("xCoord"),
            "y": details.get("yCoord"),
            "zone": details.get("zoneCode"),
            "situation": play.get("situationCode"),
        })
    return rows


def _iter_game_files(season: int, game_type: str, data_dir):
    folder = get_data_dir(data_dir) / str(season)
    return sorted(folder.glob(f"{season}{GAME_TYPE_CODE[game_type]}*.json"))


# --------------------------------------------------------------------------------------
# 2. Calculs vectorisés sur toute la saison
# --------------------------------------------------------------------------------------
def _empty_net_from_situation(situation: pd.Series, is_home_team: pd.Series) -> pd.Series:
    """Filet désert de l'équipe qui défend, à partir du situationCode.

    situationCode = 4 chiffres : [gardien visiteur][patineurs visiteurs][patineurs locaux][gardien local].
    Si l'équipe locale tire, le filet qui compte est celui du visiteur (1er chiffre), et inversement.
    Valeur manquante (pd.NA) si le code est absent/mal formé ou si l'équipe du tireur est inconnue.
    """
    code = situation.astype("string")
    valid = code.str.fullmatch(r"\d{4}").fillna(False) & is_home_team.notna()

    defending_goalie = pd.Series(pd.NA, index=situation.index, dtype="string")
    home = valid & is_home_team.fillna(False).astype(bool)
    away = valid & ~is_home_team.fillna(True).astype(bool)
    defending_goalie[home] = code[home].str[0]  # la locale tire -> gardien visiteur
    defending_goalie[away] = code[away].str[3]  # la visiteuse tire -> gardien local

    empty = pd.Series(pd.NA, index=situation.index, dtype="boolean")
    empty[valid] = defending_goalie[valid].eq("0")
    return empty


def build_season(season: int, game_type: str = REGULAR, data_dir=None) -> pd.DataFrame:
    """Construit le dataframe des tirs d'une saison (sans cache)."""
    files = _iter_game_files(season, game_type, data_dir)
    if not files:
        raise FileNotFoundError(
            f"Aucun match {game_type} téléchargé pour {season} dans {get_data_dir(data_dir)}. "
            f"Lancez : uv run download-nhl --first {season} --last {season}"
        )

    rows = []
    for f in files:
        rows.extend(_shot_rows(json.loads(f.read_text())))

    df = pd.DataFrame(rows)
    df["season"] = season
    df["is_goal"] = df["event_type"].eq("goal")
    df["is_home_team"] = df["is_home_team"].astype("boolean")

    # Direction d'attaque : même fonction que tidy_data.py (cohérent avec le milestone 1),
    # mais appelée une seule fois pour toute la saison au lieu d'une fois par match.
    df["attack_dir"] = infer_attack_direction(df)

    # Distance et angle par rapport au filet attaqué (même formule que compute_distance / compute_angle)
    dx = NET_X - df["x"] * df["attack_dir"]
    df["distance"] = np.sqrt(dx**2 + df["y"] ** 2)
    df["angle"] = np.rad2deg(np.arctan2(np.abs(df["y"]), dx))

    df["empty_net"] = _empty_net_from_situation(df["situation"], df["is_home_team"])

    cols = ["season", "game_id", "event_id", "period", "team_id", "is_home_team",
            "event_type", "x", "y", "distance", "angle", "is_goal", "empty_net"]
    return df[cols]


# --------------------------------------------------------------------------------------
# 3. Cache Parquet par saison
# --------------------------------------------------------------------------------------
def _cache_path(season: int, game_type: str, cache_dir) -> Path:
    cache_dir = Path(cache_dir or os.environ.get("FEAT_ENG_1_CACHE", DEFAULT_CACHE_DIR))
    return cache_dir / f"{season}_{game_type}_v{CACHE_VERSION}.parquet"


def load_season_shots(season: int, game_type: str = REGULAR, data_dir=None,
                      cache_dir=None, rebuild: bool = False) -> pd.DataFrame:
    """Une saison : lit le cache s'il existe, sinon construit puis sauvegarde."""
    path = _cache_path(season, game_type, cache_dir)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    df = build_season(season, game_type, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_feat_eng_1_df(start: int = 2016, end: int = 2022, game_type: str = REGULAR,
                       data_dir=None, cache_dir=None, rebuild: bool = False,
                       drop_missing_coords: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Dataframe de la partie 2 pour les saisons start..end (incluses).

    Par défaut 2016-2022 : la saison 2023-24 est le jeu de test, on ne la charge pas ici.
    drop_missing_coords : retire les tirs sans coordonnées (distance/angle impossibles).
    empty_net garde ses valeurs manquantes (demandé par l'énoncé) : elles sont comptées si verbose.
    """
    df = pd.concat(
        [load_season_shots(s, game_type, data_dir, cache_dir, rebuild) for s in range(start, end + 1)],
        ignore_index=True,
    )

    if drop_missing_coords:
        missing = df["distance"].isna() | df["angle"].isna()
        if verbose and missing.any():
            print(f"{missing.sum()} tirs sans coordonnées retirés (sur {len(df)}).")
        df = df.loc[~missing].reset_index(drop=True)

    if verbose:
        print(f"{len(df)} tirs, saisons {start}-{end} ({game_type}). "
              f"empty_net indéterminé : {df['empty_net'].isna().sum()} événements.")
    return df


if __name__ == "__main__":
    # Construire tous les caches d'un coup :  uv run python -m ift6758.features.feat_eng_1_fast
    import time

    t0 = time.perf_counter()
    load_feat_eng_1_df(2016, 2022, rebuild=True)
    print(f"Construit en {time.perf_counter() - t0:.0f} s")
