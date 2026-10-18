"""Code pour créer différentes figures pour la section 4: Visualisations simples

Endpoint: Figures (1 pour question 1, 3 pour question 2, 1 pour question 3)


Exemple:


"""


#--------------------------------------------------------------------------------- Setup

import json
import os
from ift6758.data import load_season
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if p.name == "data_science_project")
os.environ["NHL_DATA_DIR"] = str(ROOT / "ift6758" / "data" / "data" / "raw")


#--------------------------------------------------------------------------------- Fonction pour séparer les buts et les tirs

def get_shots_or_goals(df, critere: str, goal_or_not: bool, count: bool=False):
    """Séparer les tirs et les buts selon un critère, avec l'option de les compter ou non avec get_count"""
    keep = df["is_goal"] == goal_or_not
    if count == False:
        goals_or_shots = df.loc[keep].copy()
    else:
        goals_or_shots = df.loc[keep].copy()[critere].value_counts()
    
    return goals_or_shots

#---------------------------------------------------------------------------------  Figure de la question 1

def create_Q1_figure(df):
    """Créer la première figure"""
    # Value_counts des tirs et des buts selon le critère "Type de tir"
    goal_count = get_shots_or_goals(df, critere="shot_type", goal_or_not=True, count=True)
    shot_count =  get_shots_or_goals(df, critere="shot_type", goal_or_not=False, count=True)

    # Transformer en array
    goals = goal_count.to_numpy()
    shots = shot_count.to_numpy()

    # Pourcentage de buts
    goal_percent = [100*goals[i]/(shots[i]+goals[i]) for i in range(0, len(shot_count.index))]

    # Nombre de colonnes déterminé par le nombre de labels pour chaque tir
    x = np.arange(len(shot_count.dropna()))
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
    fig.suptitle(f"Tirs et buts par type de tir pendant la saison {df['season'][0]}", y = 0.93)
    fig.supxlabel("Type de tir", size = "medium", y=0.035)

    # Deuxième axe y: le pourcentage de buts
    ax2 = ax1.twinx()
    ax2.bar(x+0.3, goal_percent, width-0.2, color="#cc0202")
    ax2.set_ylabel("% buts")
    ax2.set_ylim(0, 30)
    fig.legend(["Tirs au but", "Buts", "% Buts"], loc='upper right', bbox_to_anchor=(0.88, 0.85))

    plt.show()

#--------------------------------------------------------------------------------- Calculer la distance du filet

def compute_distance(df):
    distances = []
    "Calculer la distance entre le tir et le filet et convertir en dataframe"
    
    for i in range(len(df)):
        try:
            x = df["x"][i]
            y = df["y"][i]

            # Filets à 89 et -89 pieds, y = 0
            distance_from_net1 = np.sqrt((x+89)**2+y**2)
            distance_from_net2 = np.sqrt((x-89)**2+y**2)


            # En zone défensive, le filet est celui le plus loin.
            # En zone neutre, nous assumons que le filet est celui le plus près.
            if df["zone"][i] == "D":
                distance = max(distance_from_net1, distance_from_net2)
            else:
                distance = min(distance_from_net1, distance_from_net2)

            # Distances arrondies pour agréger tous les tirs près les uns des autres
            distances.append(int(distance))

        # Si la donnée est manquante, remplir quand même et continuer la boucle
        except TypeError:
            distance = np.nan
            distances.append(distance)
            continue
        
    
    df["distance"] = distances

#--------------------------------------------------------------------------------- Créer une des trois figures pour la question 2

def create_Q2_figure(df):
    """Figure qui représente la relation entre la distance au filet et la proportion de tirs qui sont des buts"""
    compute_distance(df)
    df_distances = df[["distance", "is_goal"]].dropna()

    # Dictionnaire de buts et de tirs pour chaque distance
    goals_by_distance = {}
    shots_by_distance = {}

    #Pour chaque valeur de distance, créer une clé
    for i in df_distances["distance"]:
        goals_by_distance[int(i)]= 0
        shots_by_distance[int(i)]= 0

    # Pour chaque distance, ajouter le but ou le tir
    for distance, is_goal in zip(df_distances["distance"], df_distances["is_goal"]):
        key = int(distance)
        if is_goal:
            goals_by_distance[key] += 1
        else:
            shots_by_distance[key] += 1

    # Pourcentage de buts pour chaque distance
    goal_percent = []
    for i in sorted(goals_by_distance):
            goal_percent.append(100*goals_by_distance.get(i)/(shots_by_distance.get(i)+goals_by_distance.get(i)))
    
    # Axe horizontal = toutes les clés, qui équivalent aux distances
    x = sorted(goals_by_distance.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    max_possible_distance = np.sqrt(189**2 + 42.5**2)
    ax.set_xticks(np.arange(0, max_possible_distance, step=25))
    ax.tick_params(length=0)
    ax.set_ylabel("% de buts")
    ax.set_xlabel("Distance du filet (pi)")
    fig.suptitle(f"Proportion de buts par rapport au nombre de tirs selon la distance du filet pour la saison {df['season'][0]}", y = 0.93)

    plt.step(x, goal_percent, where='pre')
    plt.show()