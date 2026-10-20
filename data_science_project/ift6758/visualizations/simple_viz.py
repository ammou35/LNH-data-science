"""

Code pour créer différentes figures pour la section 4: Visualisations simples

Endpoint: Figures (1 pour question 1, 3 pour question 2, 1 pour question 3)


"""

#--------------------------------------------------------------------------------- Setup
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns

from ift6758.data.tidy_data import (
    game_to_pandas,
    season_to_pandas,
    save_to_csv
)

DATA_DIR_ENV = "NHL_DATA_DIR"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw"


#---------------------------------------------------------------------------------
# Utilisé pour create_Q1_figure

def get_shots_and_goals(df, column: str, count: bool=False):
    """Séparer les tirs et les buts, avec l'option de retourner la colonne ou le décompte.
    Input: dataframe
    Output: liste de value_counts() ou liste de colonnes avec index 0 = shots, index 1 = buts"""

    # Garder les tirs et les buts
    keep_shots = df["is_goal"] == False
    keep_goals = df["is_goal"] == True

    # Donner le décompte ou seulement les séparer
    if count == True:
        goals = df.loc[keep_goals].copy()[column].value_counts()
        shots = df.loc[keep_shots].copy()[column].value_counts()
    else:
        goals = df.loc[keep_goals].copy()[column]
        shots = df.loc[keep_shots].copy()[column]
    
    return [shots, goals]

#------------------------------------------------------------------------------------------------------------------------ Figure 3
# Utilisé pour create_Q2_figure, create_Q3_figure

def compute_distance(df):
    """Redonner une copie du dataframe et ajouter la colonne distance
    Input: dataframe
    Output: nouveau dataframe avec df["distance"] = [les distances pour chaque événement]"""
    # Ajouter la valeur 0
    distances = [0]

    # Convertir les valeurs manquantes du dataframe en NaN
    dist_df = df.convert_dtypes()

    # Boucle pour 
    for i in range(1, len(dist_df)):
        # Ajouter la clé 0
            try:
                x = dist_df["x"].iloc[i]
                y = dist_df["y"].iloc[i]

                # Filets à 89 et -89 pieds, y = 0
                distance_from_net1 = np.sqrt((x+89)**2+y**2)
                distance_from_net2 = np.sqrt((x-89)**2+y**2)

                # En zone défensive, le filet est celui le plus loin.
                # En zone neutre, nous assumons que le filet est celui le plus près.
                # Distances arrondies pour agréger tous les tirs près les uns des autres
                if dist_df["zone"].iloc[i] == "D":
                    distance = max(distance_from_net1, distance_from_net2)
                else:
                    distance = min(distance_from_net1, distance_from_net2)
                distances.append(int(distance))

            # Si la donnée est manquante, remplir quand même et continuer
            except (TypeError, ValueError):
                distance = np.nan
                distances.append(distance)
                continue
        
    
    dist_df["distance"] = distances
    return dist_df

#------------------------------------------------------------------------------------------------------------------------ 
# Utilisé pour create_Q2_figure, create_Q3_figure

def compute_shots_by_distance(df, bin_size=5):
    """Calculer le nombre de tirs et de buts selon la distance du filet en fonction des coordonnées x et y
    et le type de zone (offensive, défensive, neutre)
    Input: dataframe sans colonne distance, bin size(opt)
    Output: liste de deux dictionnaires avec index 0 = tirs, index 1 = buts, chaque dict: Clé = distance, valeur = pourcentage"""
    
    df_distances = compute_distance(df)[["distance", "is_goal"]]
    df_distances_dropna = df_distances.dropna()

    # Utilisation de Claude AI pour diviser les données en bins, après plusieurs heures d'essais.
    # Voir annexe Jérémie 1 pour justification
    # Ajoute la colonne distance au dataframe et retire les NaN
    bins = (df_distances_dropna["distance"].astype(int) // bin_size) * bin_size



    # Dictionnaire de buts et de tirs pour chaque distance
    goals = {}
    shots = {}


    #Pour chaque valeur de distance, créer une clé 0
    for key in bins:
        goals[key]= 0
        shots[key]= 0

    # Dans le même prompt, utilisation de Claude AI pour diviser les données en bins, après plusieurs heures d'essais.
    # Voir annexe Jérémie 1 pour justification.
    # Pour chaque distance, ajouter le but ou le tir
    for keys, is_goal in zip(bins, df_distances_dropna["is_goal"]):
        if is_goal:
            goals[keys] += 1
        else:
            shots[keys] += 1
            
    # Retourne une liste comportant deux dictionnaires. Clés = distances, valeurs = nombre de tirs ou de buts
    return [shots, goals]

#------------------------------------------------------------------------------------------------------------------------
# Utilisé pour create_Q2_figure, create_Q3_figure

def compute_goal_percent(distance_dataframe, bin_size, df_columns):
    """Calculer le nombre de tirs [0] et de buts [1] par distance avec compute_shots_by_distance
    Input: Dataframe avec une colonne distance, bin_size, df_columns = nombre entier correspondant aux distances possibles.
    Output: liste de pourcentages de buts, en ordre de distance. Doit matcher le nombre de distances.
    Un minimum de 10 tirs par bins est nécessaire pour participer au décompte, afin de minimiser les données aberrantes et le hasard"""

    # Calculer les tirs et buts selon la distance
    shots_goals = compute_shots_by_distance(distance_dataframe, bin_size)
    # Liste des pourcentage de buts pour chaque distance
    goal_percent = []

    # Pour chaque colonne, calculer le pourcentage
    # Claude IA utilisé pour éviter les divisions par 0 et TypeError avec missing values
    # Voir annexe Jérémie 2 pour justification
    for j in df_columns:
        s = shots_goals[0].get(j)
        g = shots_goals[1].get(j, 0)
   

    # Nombre de tirs minimal pour être considéré
        min_shots = 10

        # Sauter les NaN, SURTOUT PAS = 0
        try:
            goal_percent.append(100*g/(s+g) if (s+g) >= min_shots else np.nan)

        except TypeError:
            goal_percent.append(np.nan)

    return goal_percent

#------------------------------------------------------------------------------------------------------------------------
# Utilisé pour create_Q3_figure

def create_dict_by_type_w_dist(df):
    """Créer un dictionnaire de dataframes par type de tir
    Input: dataframe avec ue colonne distances
    Output: dictionnaire {"shot_type" : dataframe}"""

    # Séparer par type de tir
    shot_types_df_dict = {}
    all_shot_types = df["shot_type"].dropna().unique()
    for i in all_shot_types:

        # Garder toutes les rangées du bon type
        shot_types_df_dict[f"{i}"]= df[df["shot_type"] == f"{i}"]

    return shot_types_df_dict

#------------------------------------------------------------------------------------------------------------------------ 
# Figure 1

def create_Q1_figure(season):
    """Créer la première figure de pourcentage et nombre de buts et de tirs sans but en fonction du type de tir
    Input: Année de début de saison, bin_size(opt)
    Output: Diagramme à bandes"""

    # Extraire les données de la saison dans un dataframe
    season_df = season_to_pandas(season)

    # Value_counts des tirs et des buts selon le critère "Type de tir"
    shots_and_goals = get_shots_and_goals(season_df, column="shot_type", count=True)
    shot_count = shots_and_goals[0]
    goal_count = shots_and_goals[1]

    # Claude IA utilisé pour indexer les tirs sur le bon label.
    # Voir annexe Jérémie 3 et 3.5 pour justification.
    # Aligner les buts sur le même index que les tirs (0 si aucun but)
    goal_count = goal_count.reindex(shot_count.index, fill_value=0)
    
    # Transformer en array
    goals = goal_count.to_numpy()
    shots = shot_count.to_numpy()

    # Pourcentage de buts
    goal_percent = []
    for i in range(0, len(shot_count.index)):
        if shots[i]+goals[i] == 0:
            goal_percent.append(np.nan)
        else:
            goal_percent.append(100*goals[i]/(shots[i]+goals[i]))

    # Nombre de colonnes déterminé par le nombre de labels pour chaque tir
    x = np.arange(len(shot_count))
    width = 0.35

    # Figure
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Premier axe y: les tirs et les buts
    ax1.bar(x, shots, width, color="#0e2d9c")
    ax1.bar(x, goals, width, bottom=shots, color="#008f30")

    # Étiquettes dérivées des colonnes de type de tir
    ax1.set_xticks(x+0.1, shot_count.index)
    ax1.tick_params(length=0)
    ax1.set_ylabel("# de tirs")
    fig.suptitle(f"Tirs sans but, buts et pourcentage de buts par type de tir \nSaison {season_df['season'].iloc[0]}-{int(season_df['season'].iloc[0])+1}", y = 0.94)
    fig.supxlabel("Type de tir", size = "medium", y=0.035)

    # Deuxième axe y: le pourcentage de buts
    ax2 = ax1.twinx()
    ax2.bar(x+0.3, goal_percent, width-0.2, color="#cc0202")
    ax2.set_ylabel("% de buts")
    ax2.set_ylim(0, 30)
    fig.legend(["Tirs sans but", "Buts", "% Buts"], loc='upper right', bbox_to_anchor=(0.88, 0.85))

    plt.show()

#------------------------------------------------------------------------------------------------------------------------ Figure 2
# Figure 2

def create_Q2_figure(season, bin_size=5):
    """Figure qui représente la relation entre la distance au filet et la proportion de tirs qui sont des buts
    Input: année de début de la saison, bin_size(opt)
    Output: plot"""

    # Extraire les données de la saison dans un dataframe
    season_df = season_to_pandas(season)
    
    # Distance maximale théorique d'un joueur par rapport au filet
    max_distance = np.sqrt(189**2 + 42.5**2)
    
    # Axe horizontal = toutes les clés, qui équivalent aux distances
    x = np.arange(0, int(max_distance)+bin_size, bin_size)
    
    goal_percent = compute_goal_percent(season_df, bin_size=bin_size, df_columns=x)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xticks(np.arange(0, int(max_distance)+1, step=25))
    ax.tick_params(length=0)
    ax.set_ylim(0, 50)
    ax.set_ylabel("% de buts")
    ax.set_xlabel("Distance du filet (pi)")
    fig.suptitle(f"Proportion de buts par rapport au nombre de tirs selon la distance du filet \nSaison {season_df['season'].iloc[0]}-{int(season_df['season'].iloc[0])+1} (bin size={bin_size})", y = 0.96)

    plt.plot(x, goal_percent)
    plt.show()

#------------------------------------------------------------------------------------------------------------------------ Figure 3
# Figure 3

def create_Q3_figure(season, bin_size=15):
    """Fonction qui crée une heatmap du pourcentage de buts selon la distance, classée par type de tir
    Input: dataframe d'une saison complète, bin_size(opt)
    Output: heatmap"""

    # Extraire les données de la saison dans un dataframe
    season_df = season_to_pandas(season)

    # Liste de pourcentages de buts comptés pour chaque type de tir, selon la distance
    goal_percent_by_dist_type = []

    # Créer l'index de la heatmap
    figure_index = []

    # Créer un dictionnaire pour chaque type de tir: {"shot-type": dataframe de tous les tirs de ce type}
    shot_types_dict = create_dict_by_type_w_dist(season_df)

    # Distance maximale théorique d'un joueur par rapport au filet
    max_distance = np.sqrt(189**2 + 42.5**2)
    
    # Calculer le nombre de colonnes de distances, une par distance jusqu'au maximum de la patinoire
    df_columns = [i for i in range(0, int(max_distance), bin_size)]

    # Pour chaque type de tir, ajouter la liste des pourcentages à une liste plus grande
    for i in shot_types_dict:

        # Ajouter le type de tir à l'index de la figure
        figure_index.append(f"{i}")

        # Calculer les pourcentages de buts et l'ajouter
        computed_percent = compute_goal_percent(distance_dataframe=shot_types_dict[f"{i}"], bin_size=bin_size, df_columns=df_columns)
        goal_percent_by_dist_type.append(computed_percent)

    # Étiquettes de distances
    fig_labels = [f"[{i}-{i+bin_size}[" for i in df_columns]

    # Créer le dataframe pour la heatmap
    figure_df = pd.DataFrame(goal_percent_by_dist_type, columns=fig_labels, index=figure_index)

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.set_xlabel("Distance du filet (pi)")
    fig.suptitle(f"Pourcentage de buts sur le nombre de tirs selon la distance et le type de lancer \nSaison {season_df['season'][0]}-{int(season_df['season'][0])+1} (bin size={bin_size})", y = 0.96)
    sns.heatmap(figure_df, ax=ax, cmap ='YlOrBr', linewidths = 0.30, annot = False, vmin=0, vmax=50)
    plt.show()