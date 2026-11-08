from ift6758.data import download_season, load_season, game_to_pandas, save_to_csv, infer_attack_direction, season_to_pandas, get_game
from ift6758.visualizations.simple_viz import compute_distance
import pandas as pd
import os
import numpy as np

def compute_angle(df):
        
    try:
        # Calculer l'angle en radians en utilisant la fonction arctan2, y  en valeur absolue car l'angle est symétrique à droite et à gauche du but
        angle = np.rad2deg(np.arctan2(np.absolute(df["y"]), 89-df["x"]*df["attack_dir"]))
        
    # Si la donnée est manquante, remplir quand même et continuer
    except (TypeError, ValueError):
        angle = np.nan

    return angle

def tidy_game_for_engineering(game):
    # Adapté de game_to_pandas de tidy_data.py
    # Liste d'événements
    plays = game["plays"]

    # Conversion en dataframe
    events = pd.json_normalize(plays)

    # Garder seulement les evenements de type tirs et buts
    keep = events["typeDescKey"].isin(["shot-on-goal", "goal"])
    shots = events.loc[keep].copy()

    # Indicateur de but ou non
    shots["is_goal"] = shots["typeDescKey"].eq("goal")

    # Game ID
    shots["game_id"] = game["id"]

    # Indicateur de filet désert ou non
    shots["empty_net"] = shots["details.goalieInNetId"].isna()
    
    # Nommer les colonnes du dataframe
    column_names = {
        "game_id": "game_id",
        "periodDescriptor.number": "period",
        "details.eventOwnerTeamId": "team_id",
        "details.xCoord": "x",
        "details.yCoord": "y",
        "details.zoneCode": "zone",
        "is_goal": "is_goal",
        "empty_net": "empty_net"
    }

    # Reordonner et renommer les colonnes
    shots = shots.reindex(columns=list(column_names)).rename(columns=column_names)

    # Saison: ce sont les 4 premiers carateres de "game_id"
    shots["season"] = shots["game_id"].apply(lambda df: str(df)[:4])
    
    # Convertir en dtypes supportant les valeurs manquantes
    shots = shots.convert_dtypes()

    # Ajouter la colonne "distance"
    shots = compute_distance(shots)

    # Ajouter la colonne "angle"
    # TODO: Ajouter la colonne "angle" en utilisant les coordonnées x et y du tir    
    shots = shots.assign(angle=compute_angle(shots))
    # Enfin, on vérifie qu'il ne manque aucune données pour chaque colonne
    #assert not shots.isna().sum().any()/

    # Garder seulement les colonnes distance, angle, is_goal, empty_net
    #TODO ajouter colonne angle
    shots = shots.loc[:, ["distance", "angle", "is_goal", "empty_net"]]
    return shots

def tidy_season_for_engineering(season : int, export=False):

    # Adapté de season_to_pandas de tidy_data.py
    # Télécharger et charger le raw data pour les saisons 2016-2023
    path = f"../data/raw/{season}"
    if os.path.exists(path) == False:
        download_season(season)
    else:
        pass
    # Charger les données pour les saisons régulières
    games_list = load_season(season)
    # Lignes suivantes provenant de la fonction season_to_pandas de tidy_data.py
    # Liste de dataframes
    dataframes = []

    # Extraire chaque dataframe d'un match
    for game in games_list:
        try:
            df = tidy_game_for_engineering(game)
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
