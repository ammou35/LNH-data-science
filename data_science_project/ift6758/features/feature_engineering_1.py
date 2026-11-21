from ift6758.data import download_season, load_season, save_to_csv
from ift6758.visualizations.simple_viz import compute_distance

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def tidy_game_for_engineering(game: dict):
    """
    Créer le dataframe d'une partie nettoyée pour le feature engineering partie 1.
    Input: Dictionnaire d'événements d'une partie
    Output: Dataframe avec les colonnes 'distance', 'angle', 'is_goal', 'empty_net' pour chaque tir
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
    Données nettoyées des tirs de tous les matchs d'une saison régulière pour le feature engineering partie 1.
    Utilise tidy_game_for_engineering.
    Input: Année de début de la saison
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
        save_to_csv(season_df, filename=f"{season}_shots_feat_eng_1")

    # Retourner le dataframe avec tous les matchs d'une saison
    return season_df

def create_feat_engineering_1_df(start, end):
    """
    Crée le dataframe des données nettoyées d'entraînement pour le feature engineering 1.
    Input: Année de début de la première saison, année de début de la dernière saison
    Output: Dataframe nettoyé des saisons en question (colonnes 'distance', 'angle', 'empty_net', 'is_goal')
    """
    df = pd.DataFrame()
    for i in range(start, end+1):
        season_df = tidy_season_for_engineering(season=i, export=False)
        df = pd.concat([df, season_df], ignore_index=True)
    return df.dropna()

def compute_angle(df):
    """
    Calculer l'angle par rapport au filet selon les coordonnées x et y ainsi que la direction de l'attaque.
    Input: Dataframe avec colonnes 'y', 'x' et 'attack_dir'
    Output: Ajout de la colonne 'angle' avec les angles de tir
    """
    try:
        # Calculer l'angle en radians en utilisant la fonction arctan2, y  en valeur absolue car l'angle est symétrique de chaque côté du but
        # 'attack_dir' renvoie +1 vers la droite, -1 vers la gauche, donc change le signe de x
        angle = np.rad2deg(np.arctan2(np.absolute(df["y"]), 89-df["x"]*df["attack_dir"]))
        
    # Si la donnée est manquante, remplir quand même et continuer
    except (TypeError, ValueError):
        angle = np.nan

    return angle


def graph_goal_counts_by_column(df, column, start, end, yscale: str="log", bins=20, save_fig: bool=False):
    """
    Tracer l'histogramme des tirs sans buts et des buts (y) selon un critère (column)(x)
    Input: Dataframe avec la colonne, colonne, conversion en log ou non, bins
    Output: Histogramme
    """
    graph = sns.histplot(data=df, x=df[column], hue="is_goal", bins=bins)
    xlabel="Angle (deg)" if column == "angle" else "Distance (pi)"
    graph.set(yscale=yscale, xlabel=xlabel, ylabel=f"Nombre de tirs {'(échelle log)' if yscale=='log' else ''}")
    graph.set_title(f"Nombre de buts et non-buts selon le critère '{column}' (bins = {bins})")
    plt.suptitle(f"Source des données: API officielle de la LNH (saisons {start}-{start+1} à {end}-{end+1})", x=0.08, y=0, ha="left", fontsize=10, color="gray")
    if save_fig==True:
        plt.savefig(f"../../figures/{start}_{end}_counts_by_{column}.svg", format="svg", bbox_inches="tight")    
    plt.show()

def graph_goals_per_distance_and_angle(df,start, end, bins=20, save_fig: bool=False):
    """
    Tracer la heatmap du nombre de tirs (colorbar) selon la distance (x) et l'angle (y)
    Input: Dataframe avec colonnes 'distance' et 'angle', bins
    Output: Heatmap
    """
    # Tiré de la documentation seaborn.jointplot et:
    # Source - https://stackoverflow.com/a/34140785
    # Posted by iayork, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-09, License - CC BY-SA 4.0
    joint_plot = sns.jointplot(data=df, x="distance", y="angle", kind="hist", cbar=True, xlim=(0, 200), ylim=(0,180),
                joint_kws=(dict(bins=bins, color ="dodgerblue", thresh=1)),
                marginal_kws=dict(bins=bins, color ="dodgerblue"),
                cbar_kws=dict(label='Nombre de tirs'))
    joint_plot.figure.suptitle(f"Nombre de tirs au but selon la distance et l'angle (bins = {bins})", y=0.95)
    plt.xlabel("Distance (pi)")
    plt.ylabel("Angle (deg)")
    joint_plot.figure.text(0.05, -0.02, f"Source des données: API officielle de la LNH (saisons {start}-{start+1} à {end}-{end+1})", ha="left", fontsize=10, color="gray")
    # Syntaxe de la colorbar: Source - https://stackoverflow.com/a/60849048
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

    if save_fig==True:
        plt.savefig(f"../../figures/{start}_{end}_goals_dist_ang.svg", format="svg", bbox_inches="tight")
    plt.show()

def to_bins(df, column: str, bin_size=10):
    """
    Regrouper les valeurs d'une colonne en bins
    Input: Dataframe, colonne, bin_size
    Output: Series de bins
    """
    bins = np.arange(0, df[column].max()+bin_size, bin_size)   
    bin_centers = (bins[:-1] + bins[1:]) / 2 
    binned = pd.cut(df[column], bins=bins, labels=bin_centers, right=False)
    return binned

def graph_goalrate_by_column(df, start, end, column, bin_size, save_fig: bool=False):
    """
    Tracer la courbe du taux de buts selon un critère (colonne)
    Input: Dataframe, colonne, bin_size
    Output: Courbe du taux de buts selon le critère choisi
    """
    df_copy = df.copy()
    df_copy["bins"]=to_bins(df, column, bin_size=bin_size)
    graph = sns.lineplot(data=df_copy, x="bins", y="is_goal")
    xlabel="Angle (deg)" if column == "angle" else "Distance (pi)"
    graph.set(xlabel=xlabel, ylabel="Taux de buts")
    graph.set_title(f"Taux de buts selon le critère '{column}' (bin size = {bin_size})")
    plt.suptitle(f"Source des données: API officielle de la LNH (saisons {start}-{start+1} à {end}-{end+1})", x=0.05, y=-0.02, ha="left", fontsize=10, color="gray")

    if save_fig==True:
        plt.savefig(f"../../figures/{start}_{end}_goalrate_by_{column}.svg", format="svg", bbox_inches="tight")
    plt.show()

def graph_emptynet_goals_per_distance(df, start, end, yscale: str="log", bins=20, save_fig: bool=False):
    """
    Tracer l'histogramme des buts en filet désert vs non désert selon la distance
    Input: Dataframe, conversion en log ou non, bins
    Output: Histogramme de buts en filet désert ou non selon la distance
    """
    df = df.loc[df["is_goal"]]
    graph = sns.histplot(data=df, x=df["distance"], hue="empty_net", bins=bins)
    graph.set(yscale=yscale, xlabel="Distance (pi)", ylabel=f"Nombre de tirs {'(échelle log)' if yscale=='log' else ''}")
    graph.set_title(f"Nombre de buts en filet désert selon la distance (bins = {bins})")
    plt.suptitle(f"Source des données: API officielle de la LNH (saisons {start}-{start+1} à {end}-{end+1})", x=0.08, y=0, ha="left", fontsize=10, color="gray")

    if save_fig==True:
        plt.savefig(f"../../figures/{start}_{end}_emptynet_dist.svg", format="svg", bbox_inches="tight")
    plt.show()

def graph_all_feature_engineering_1(df, start, end, bins=20, bin_size=10, save_fig: bool=False):
    """
    Tracer tous les graphiques de la section 'Feature engineering 1'
    Input: Dataframe, bins pour 4 des 6 graphiques, année de début de la première saison (ENTRÉE MANUELLE selon df utilisé), année de début de la dernière saison (ENTRÉE MANUELLE selon df utilisé), bin_size pour les courbes de taux de buts
    Output: 6 graphiques
    """
    graph_goal_counts_by_column(df, column="distance", start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goal_counts_by_column(df, column="angle", start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goals_per_distance_and_angle(df, start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goalrate_by_column(df, column="distance", start=start, end=end, bin_size=bin_size, save_fig=save_fig)
    graph_goalrate_by_column(df, column="angle", start=start, end=end, bin_size=bin_size, save_fig=save_fig)
    graph_emptynet_goals_per_distance(df, start=start, end=end, bins=bins, save_fig=save_fig)