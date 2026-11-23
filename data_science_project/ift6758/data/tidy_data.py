from pathlib import Path
import os
import pandas as pd
import numpy as np

from ift6758.data import load_season

def game_to_pandas(game: dict,
                   export: bool | None = None,
                   verbose: bool = False):
    """Extraire la liste d'événements d'une seule partie en dataframe pandas."""

    # Le workflow est principalement adapté des TPs
    # Liste d'événements
    plays = game["plays"]

    # Conversion en dataframe
    events = pd.json_normalize(plays)

    # Garder seulement les evenements de type tirs et buts
    keep = events["typeDescKey"].isin(["shot-on-goal", "goal"])
    shots = events.loc[keep].copy()

    # Game ID
    shots["game_id"] = game["id"]

    # Game type
    shots["game_type"] = game["gameType"]

    # Nommer les colonnes du dataframe
    column_names = {
        "game_id": "game_id",
        "eventId": "event_id",
        "game_type": "game_type",
        "periodDescriptor.number": "period",
        "periodDescriptor.periodType": "period_type",
        "timeInPeriod": "time_in_period",
        "situationCode": "situation",
        "homeTeamDefendingSide": "home_team_defending_side",
        "typeDescKey": "event_type",
        "details.eventOwnerTeamId": "team_id",
        "details.xCoord": "x",
        "details.yCoord": "y",
        "details.shotType": "shot_type",
        "details.shootingPlayerId": "shooter_id",
        "details.scoringPlayerId": "scorer_id",
        "details.goalieInNetId": "goalie_id",
        "details.zoneCode": "zone",
    }

    # Reordonner et renommer les colonnes
    shots = shots.reindex(columns=list(column_names)).rename(columns=column_names)

    # Saison: ce sont les 4 premiers carateres de "game_id"
    shots["season"] = shots["game_id"].apply(lambda df: str(df)[:4])

    # Noms des équipes
    team_names = {
        game["awayTeam"]["id"]: game["awayTeam"]["abbrev"],
        game["homeTeam"]["id"]: game["homeTeam"]["abbrev"],
    }
    shots["team"] = shots["team_id"].map(team_names)

    # Vérifier la table avant de l’utiliser
    assert len(events) == len(plays), "Flattening changed the number of events."
    assert len(shots) == int(keep.sum()), "Cleaning changed the number of retained events."
    assert shots[["game_id", "event_id"]].notna().all().all(), "Missing event identifier."
    assert not shots.duplicated(["game_id", "event_id"]).any(), "Duplicate event identifiers."

    if verbose:
        print("Checks passed.")
    #display(shots.isna().sum().rename("missing_count").to_frame())

    # Indicateur de but ou non
    shots["is_goal"] = shots["event_type"].eq("goal")

    # Indicateur de filet désert ou non
    shots["empty_net"] = shots["goalie_id"].isna()

    # Pour déterminer qui est en avantage numérique ou non a partir du situationCode, il faut savoir qui est l'équipe a domicile
    shots["is_home_team"] = (shots["team_id"] == game["homeTeam"]["id"])

    # Convertir en dtypes supportant les valeurs manquantes
    shots = shots.convert_dtypes()

    # Fusion de 'shooter_id' et 'scorer_id' en un nouvelle colonne 'player_id' en remplissant les NA dans 'shooter_id' avec la valeur dans 'scorer_id'
    shots['player_id'] = shots['shooter_id'].fillna(shots['scorer_id'])


    # Préparation de la table des joueurs
    # On ne prend que les colonnes pertinentes
    players = pd.DataFrame(game["rosterSpots"], columns=['playerId', 'firstName', 'lastName'])

    # On réindexe avec des noms de colonnes canoniques
    column_names = {
      "playerId": "player_id",
      "firstName": "first_name",
      "lastName": "last_name",
    }
    players = players.reindex(columns=list(column_names)).rename(columns=column_names)
    players = players.convert_dtypes()

    # Fusion de first_name et last_name en une seule colonne contenant un dictionnaire indexé par le nom de la colonne
    players['player_name'] = players.apply(lambda df: {'first_name': df['first_name'], 'last_name': df['last_name']}, axis='columns')

    # On vérifie qu'il ne manque aucune données pour chaque colonne
    #assert not players.isna().sum().any()

    # Fusion des tables shots et players
    # Par la gauche sur la colonne 'player_id',
    # suivi de la meme chose pour 'goalie_id',
    # puis on drop les colonnes redondantes dues a la fusion,
    # et enfin on renomme 'player_name_goalie' en 'goalie_name'
    shots = (shots.drop(columns=['shooter_id', 'scorer_id'])
      .merge(players[['player_id', 'player_name']], how='left', on='player_id')
      .merge(players[['player_id', 'player_name']], how='left', left_on='goalie_id', right_on='player_id', suffixes=(None, '_goalie'))
      .drop(columns=['player_id_goalie'])
      .rename(columns={'player_name_goalie': 'goalie_name'}))

    # On peut vérifier qu'on a bien:
    # - retiré les colonnes 'shooter_id' et 'scorer_id';
    # - et rajouté deux colonnes 'player_name' et 'goalie_name'
    #assert shots.columns.isin(['shooter_id', 'scorer_id']).any()
    #assert 'player_name' in shots.columns
    #assert 'goalie_name' in shots.columns

    # Enfin, on vérifie qu'il ne manque aucune données pour chaque colonne
    #assert not shots.isna().sum().any()


    # Préparation de la table pour les données de force égales
    # On zip la liste des champs respectifs avec le situationCode (string) que l'on a transformé en une liste de int
    # pour en faire un dictionnaire que l'on normalise ensuite
    situations = (pd.json_normalize(
        shots['situation'].apply(
            lambda df: dict(zip(
                ['away_goalie_in_net', 'away_skaters', 'home_skaters', 'home_goalie_in_net'],
                map(int, list(df))))
            )))

    # Avantage numérique si #home_skaters - #away_skaters > 0 et événement de home_team
    situations['strength'] = (situations['home_skaters'] - situations['away_skaters']) * (shots["is_home_team"]*2 - 1)

    # Fusion des tables shots et situation_df
    shots = shots.merge(situations, how='left', left_index=True, right_index=True)


    # Convertir en valeurs manquantes
    shots = shots.convert_dtypes()

    # Remise en ordre
    column_order = ['season', 'game_id', 'event_id', 'game_type', 'period', 'period_type',
                    'time_in_period', 'team', 'team_id', 'is_home_team', 'home_team_defending_side',
                    'event_type', 'is_goal', 'x', 'y', 'shot_type', 'player_name', 'goalie_name',
                    'zone', 'empty_net', 'away_goalie_in_net', 'away_skaters', 'home_skaters',
                    'home_goalie_in_net', 'strength']
    shots = shots[column_order]

    if export:
        save_to_csv(shots, filename="shots")
    return shots

def season_to_pandas(season: int,
                     data_dir: str | os.PathLike | None = None,
                     export: bool | None = None,
                     verbose: bool = False):
    """Extraire le dataframe pandas de tous les matchs d'une saison"""

    # Si data_dir non defini, on prend $NHL_DATA_DIR par defaut
    if data_dir is None: data_dir = Path(__file__).resolve().parent / "raw"

    if verbose:
        print(f"Chargement des fichiers de donnees brutes de la saison {season}...")
    season_list = load_season(season, data_dir=data_dir) + load_season(season, "playoffs", data_dir=data_dir)
    if verbose:
        print("Chargement termine")
        print(f"Transformation des donnees brutes en tidy data...")

    # Liste de dataframes
    dataframes = []

    # Extraire chaque dataframe d'un match
    for game in season_list:
        # Claude.AI utilisé, cf annexe
        try:
            df = game_to_pandas(game)
        except Exception as e:
            print(f"Skipping game {game}: {e}")
            continue

        dataframes.append(df)

    # Soulever une erreur si aucun match n'est extrait
    if not dataframes:
        raise ValueError("Aucune partie trouvée.")


    if verbose:
        print(f"Transformation terminee")
        print(f"Concatenation des tidy data en un seul dataframe de la saison {season}...")
    season_df = pd.concat(dataframes, ignore_index=True)
    if verbose:
        print(f"Concatenation terminee")

    if export:
        save_to_csv(season_df, filename=f"{season}_shots")

    # Retourner le dataframe avec tous les matchs d'une saison
    return season_df

def save_to_csv(dataframe: pd.DataFrame,
                filename: str | os.PathLike,
                path: str | os.PathLike | None = None,
                overwrite: bool | None = None):
    """Sauvegarde un dataframe dans un CSV. Adapté du TP."""

    # Verifications des arguments
    if path is None:
        path = Path(__file__).resolve().parent / "processed"
    else:
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{filename}.csv"

    if csv_path.is_file() and not overwrite:
        raise FileExistsError("Utiliser overwrite=True pour réécrire le fichier.")

    dataframe.to_csv(csv_path, index=False)
    print("Saved processed table:", csv_path.resolve())

def infer_attack_direction(shots_df):
    """Se base sur une heuristique du nombre majoritaire de certaines combinaisons
    des valeurs de zone (O ou D), du signe de x (positif ou negatif), de la
    parite de periode de jeu (impaire ou paire) et si l'equipe qui effectue le
    est a domicile (pour donner une convention de direction).

    En donnant un score de 1 aux shots avec le label de zone O, -1 aux shots de
    la zone D, et 0 aux shots de la zone N, alors le signe de (x * zone_score)
    donne la direction de l'equipe a domicile en periode impaire.
    En effet, quand une equipe attaque vers la droite, la zone O est celle qui
    correspond a la droite de la patinoire (i.e. x > 0), et vice versa.

    Parametres:
        shots_df: DataFrame Pandas; necessite les colonnes 'game_id', 'team_id',
        'is_home_team', 'period', 'zone', et 'x'.

    Retourne:
        Series qui contient une colonne 'attack_dir' qui donne la direction
        de l'equipe qui attaque.
    """

    # Lookup table home team starting attack direction
    lut_htsad = (
        shots_df[['game_id', 'team_id', 'is_home_team', 'period', 'zone', 'x']]
        .assign(zone_score=lambda df: df.zone.map(zone_score_map),
                htsad_score=lambda df: np.sign(df.x*df.zone_score
                                                *((df.period%2 == 1)*2 - 1)
                                                *((df.is_home_team == True)*2 - 1))
                )
        [['game_id', 'htsad_score']]
        .groupby('game_id')
        .sum().transform(np.sign).astype(int)
    )['htsad_score'].to_dict()

    return (
        shots_df.game_id.map(lut_htsad)
        *((shots_df.is_home_team == True)*2 - 1)
        *((shots_df.period%2 == 1)*2 - 1)
        )
