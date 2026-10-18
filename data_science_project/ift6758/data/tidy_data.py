from pathlib import Path
import os
import pandas as pd

def game_to_pandas(game: dict,
                   export: bool | None = None):
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

    # Indicateur de but ou non
    shots["is_goal"] = shots["event_type"].eq("goal")

    # Indicateur de filet désert ou non
    shots["empty_net"] = shots["goalie_id"].isna()

    # Pour déterminer qui est en avantage numérique ou non a partir du situationCode, il faut savoir qui est l'équipe a domicile
    home_team = game["homeTeam"]["id"]

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

    # Avantage numérique si (#home_goalie_in_net + #home_skaters) - (#away_goalie_in_net + #away_skaters) > 0 et événement de home_team
    situations['strength'] = (
        ((situations['home_goalie_in_net']+situations['home_skaters']) - (situations['away_goalie_in_net']+situations['away_skaters']))
        *((shots["team_id"] == home_team).map({True: 1, False: -1})))

    # Fusion des tables shots et situation_df
    shots = shots.merge(situations['strength'], how='left', left_index=True, right_index=True)


    # Convertir en valeurs manquantes
    shots = shots.convert_dtypes()

    # Remise en ordre
    column_order = ['season', 'game_id', 'event_id', 'game_type', 'period', 'period_type',
                    'time_in_period', 'team', 'team_id', 'event_type', 'is_goal',
                    'x', 'y', 'player_name', 'goalie_name', 'shot_type', 'zone',
                    'empty_net', 'strength']
    shots = shots[column_order]

    if export:
        save_to_csv(shots, filename="shots")
    return shots

def season_to_pandas(season: int,
                     data_dir: str | os.PathLike | None = None,
                     export: bool | None = None):
    """Extraire le dataframe pandas de tous les matchs d'une saison"""

    # Si data_dir non defini, on prend $NHL_DATA_DIR par defaut
    if data_dir is None: data_dir = os.environ['NHL_DATA_DIR']

    season_list = load_season(season, data_dir=data_dir) + load_season(season, "playoffs", data_dir=data_dir)

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


    season_df = pd.concat(dataframes, ignore_index=True)
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
        path = Path("ift6758/data/processed")
    elif type(path) is str:
        path = Path(path)

    #if overwrite is None:

    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{filename}.csv"

    if csv_path.is_file() and not overwrite:
        raise FileExistsError("Utiliser overwrite=True pour réécrire le fichier.")

    dataframe.to_csv(csv_path, index=False)
    print("Saved processed table:", csv_path.resolve())
