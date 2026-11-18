from ift6758.data import download_season, load_season, save_to_csv
from ift6758.visualizations.simple_viz import compute_distance
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def tidy_game_for_engineering(game: dict):
    """
    Créer un dataframe nettoyé d'une partie pour le feature engineering
    Input: dictionnaire d'événements d'une partie
    Output: Dataframe avec les colonnes 'distance', 'angle', 'is_goal', 'empty_net' et les rangées de tous les événements de tirs d'une partie
    """
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
    shots = shots.assign(angle=compute_angle(shots))
    
    # Enfin, on vérifie qu'il ne manque aucune données pour chaque colonne
    #assert not shots.isna().sum().any()/

    # Garder seulement les colonnes distance, angle, is_goal, empty_net
    shots = shots.loc[:, ["distance", "angle", "is_goal", "empty_net"]]
    return shots

def tidy_season_for_engineering(season : int, export=False):
    """
    Données nettoyées des événements de tous les matchs d'une saison régulière pour le feature engineering.
    Utilise tidy_game_for_engineering.
    Input: Année de début de la saison (int)
    Output: Dataframe d'une saison régulière avec les colonnes 'distance', 'angle', 'is_goal', 'empty_net' et les rangées de tous les événements de tirs d'une saison
    """
    # Adapté de season_to_pandas de tidy_data.py
    # Télécharger et charger le raw data
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

def create_feat_engineering_1_df():
    df = pd.DataFrame()
    for i in range(2016, 2023):
        season_df = tidy_season_for_engineering(season=i, export=False)
        df = pd.concat([df, season_df], ignore_index=True)
    return df

def compute_angle(df):
    """
    Calculer l'angle par rapport au filet selon les coordonnées x et y ainsi que la direction de l'attaque.
    Input: Dataframe avec colonnes 'y', 'x' et 'attack_dir
    Output: 
    '"""
    try:
        # Calculer l'angle en radians en utilisant la fonction arctan2, y  en valeur absolue car l'angle est symétrique de chaque côté du but
        # 'attack_dir' renvoie +1 vers la droite, -1 vers la gauche, donc change le signe de x
        angle = np.rad2deg(np.arctan2(np.absolute(df["y"]), 89-df["x"]*df["attack_dir"]))
        
    # Si la donnée est manquante, remplir quand même et continuer
    except (TypeError, ValueError):
        angle = np.nan

    return angle

def to_list_data_by_column(df, critere: str, data: str = "is_goal"):
    """
    Retourne une liste d'événements selon un critère (nom de colonne)
    Input: pd.DataFrame, nom de colonne des données à représenter (str), nom de colonne du critère utilisé (str)
    Output: liste contenant toutes les occurences d'événements [1] et de non-événements [0] qui peut être utilisé dans un histogramme
    """
    # Séparer non événements et événements
    events_false = df[df[data] == False]
    events_true = df[df[data]]

    # Accéder aux tirs et non tirs selon le critère
    all_events = [events_false[critere], events_true[critere]]
    return all_events

#TODO: implémenter cette fonction pour illustrer les données sur un tableau
def to_array_goalrate(df, critere: str):
    """
    Retourne le taux de buts en fonction d'un critère (nom de colonne)
    Input:
    Output:
    """

#TODO: ajuster les labels une fois les graphiques fonctionnels
#TODO: implémenter la partie goalrate (bloc else)
def graph_per_distance(df, bins, rate: bool=False):
    # Accéder aux tirs et buts pour chaque distance
    if rate == False:
        x = to_list_data_by_column(df, critere="distance")
        plt.ylabel("Tirs et buts")
        plt.title("Nombre de tirs au but par distance dans la LNH (saisons 2016-2017 à 2022-2023)")
        label=["Tir sans but", "But"]
        plt.legend()
    """
    else:
        x = to_series_goalrate(df, critere="distance")
        plt.ylabel("Taux de buts (%)")
        plt.title("Taux de buts par distance dans la LNH (saisons 2016-2017 à 2022-2023)")
        label=None
    """

    plt.hist(x=x, bins=bins, stacked=True, label=label)        
    plt.xlabel("Distance (pi)")
    plt.show()

#TODO: implémenter la partie goalrate (bloc else)
def graph_per_angle(df, bins, rate: bool=False):
    bins = 20

    # Accéder aux tirs et buts pour chaque distance
    if rate == False:
        x = to_list_data_by_column(df, critere="angle")
        plt.ylabel("Tirs et buts")
        plt.title("Nombre de tirs au but par angle de tir dans la LNH (saisons 2016-2017 à 2022-2023)")
        label=["Tir sans but", "But"]
        plt.legend()
        plt.hist(x=x, bins=bins, stacked=True, label=label)
    """
    else:
        x = to_series_goalrate(df, critere="angle")
        plt.ylabel("Taux de buts (%)")
        plt.title("Taux de buts par angle de tir dans la LNH (saisons 2016-2017 à 2022-2023)")
        plt.bar(to_series_goalrate(df, "angle"))
    """
    plt.xlabel("Angle (deg)")
    plt.show()

def graph_emptynet_goals_per_distance(df, bins):
    x = to_list_data_by_column(df[df["is_goal"]], data="empty_net", critere="distance")

    # Étiquettes de distances
    kwargs = dict(alpha=0.5)
    plt.hist(histtype="stepfilled", x=x, bins=bins, label=["Gardien présent", "Filet désert"], log=True, alpha=0.75, **kwargs)
    plt.legend()
    plt.title("Logarithme du nombre de buts en filet désert par distance dans la LNH (saisons 2016-2017 à 2022-2023)")
    plt.xlabel("Distance (pi)")
    plt.ylabel("Nombre de buts (log)")
    plt.xlim(right=200)

    plt.show()

def graph_goals_per_distance_and_angle(df, bins):
    # Tiré de la documentation seaborn.jointplot et:
    # Source - https://stackoverflow.com/a/34140785
    # Posted by iayork, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-09, License - CC BY-SA 4.0
    joint_plot = sns.jointplot(data=df, x="distance", y="angle", kind="hist", cbar=True,
                xlim=(0, 200), ylim=(0,180),
                joint_kws=(dict(bins=bins, color ="b")),
                marginal_kws=dict(bins=bins, color ="b"),
                )
    joint_plot.figure.suptitle("Nombre de tirs au but dans la LNH agrégés selon la distance et l'angle", y=0.935)
    plt.xlabel("Distance (pi)")
    plt.ylabel("Angle (deg)")

    # Adjust colorbar to be out of the histogram
    # Source - https://stackoverflow.com/a/60849048
    # Posted by JohanC, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-15, License - CC BY-SA 4.0
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = joint_plot.ax_joint.get_position()
    pos_marg_x_ax = joint_plot.ax_marg_x.get_position()

    # reposition the joint ax so it has the same width as the marginal x ax
    joint_plot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])

    # reposition the colorbar using new x positions and y positions of the joint ax
    joint_plot.figure.axes[-1].set_position([0.86, pos_joint_ax.y0, 0.1, pos_joint_ax.height])
    #TODO: ajouter nom à la colorbar
    plt.show()

def graph_all_feat_eng_1(bins=20):
    df = create_feat_engineering_1_df()
    graph_per_distance(df, bins=bins, rate=False) # Goals per distance
    #TODO: implémenter les fonctions de goalrate pour l'angle et la distance
    #graph_per_distance(df, rate=True) # Goal rate per distance
    graph_per_angle(df, bins=bins, rate=False) # Goals per angle
    #TODO: implémenter les fonctions de goalrate pour l'angle et la distance
    #graph_per_angle(df, rate=True) # Goal rate per angle
    graph_emptynet_goals_per_distance(df, bins=bins) # Empty net goals per distance
    graph_goals_per_distance_and_angle(df, bins=bins) # Goals per distance and angle