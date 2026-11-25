"""Construction rapide (et mise en cache) du dataframe de la partie 2 (Feature Engineering I).

But du fichier
    Produire la table « une ligne = un tir » (distance, angle, but ou non, filet désert)
    pour les saisons 2016-17 à 2022-23. C'est la base des parties 2, 3 et 6, et celle
    que la partie 4 enrichit.

Pourquoi une version « rapide » ?
    L'ancienne version traitait chaque match séparément (json_normalize, convert_dtypes,
    infer_attack_direction... x ~9000 matchs) et recommençait tout à chaque exécution :
    4 à 8 minutes. Ici :
      1. on lit les matchs en Python pur et on met tous les tirs d'une saison dans une liste ;
      2. on transforme cette liste en UN dataframe et on fait chaque calcul UNE fois par saison ;
      3. on sauvegarde chaque saison en Parquet : les appels suivants prennent ~1 seconde.

Organisation (de bas en haut) :
    _shot_rows                  JSON d'un match   -> liste de tirs (dictionnaires)
    _iter_game_files            saison            -> liste des fichiers JSON à lire
    _empty_net_from_situation   situationCode     -> colonne empty_net
    build_season                saison            -> dataframe complet (sans cache)
    _cache_path                 saison            -> chemin du fichier Parquet
    load_season_shots           saison            -> dataframe (avec cache)
    load_feat_eng_1_df          plusieurs saisons -> dataframe final  <- LA fonction à utiliser

Utilisation :
    from ift6758.features.feat_eng_1_fast import load_feat_eng_1_df

    df = load_feat_eng_1_df(2016, 2022)                # 1re fois : construit + met en cache
    df = load_feat_eng_1_df(2016, 2022)                # ensuite : lit le cache (~1 s)
    df = load_feat_eng_1_df(2016, 2022, rebuild=True)  # forcer la reconstruction (code modifié)

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

# Ignorer les warnings de seaborn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns


# Types d'événements gardés : les tirs cadrés et les buts (comme demandé à la partie 2)
SHOT_TYPES = ("shot-on-goal", "goal")

# Les filets sont à x = ±89 pi ; on les approxime par un point (permis par l'énoncé)
NET_X = 89

# Dossier où les saisons déjà calculées sont sauvegardées
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "feat_eng_1"

# Incrémenter quand la logique change : les anciens caches (v1, v2...) sont alors ignorés
# automatiquement et tout le monde reconstruit avec le nouveau code.
CACHE_VERSION = 1


# --------------------------------------------------------------------------------------
# 1. Extraction : JSON brut -> lignes (dictionnaires), sans pandas
# --------------------------------------------------------------------------------------
def _shot_rows(game: dict) -> list[dict]:
    """Extrait les tirs et buts d'UN match, sous forme de liste de dictionnaires.

    Entrée : le JSON brut d'un match (tel que sauvegardé par download-nhl).
    Sortie : une liste avec un dictionnaire par tir, ex.
             [{"game_id": 2017020001, "event_id": 54, "x": 84, "y": -6, ...}, ...]

    On utilise une simple boucle Python (pas pandas) : pour ~300 événements dont on garde
    ~60, c'est beaucoup moins coûteux que de tout convertir en dataframe puis de filtrer.
    Les calculs (distance, angle...) ne sont PAS faits ici : on les fait plus tard,
    une seule fois pour toute la saison (voir build_season).
    """
    home_id = game["homeTeam"]["id"]  # pour savoir si le tireur joue à domicile
    rows = []
    for play in game.get("plays", []):
        # On saute tout ce qui n'est pas un tir cadré ou un but (mises au jeu, mises en échec...)
        if play.get("typeDescKey") not in SHOT_TYPES:
            continue

        # .get(...) or {} : certains événements n'ont pas de bloc "details" -> on évite une erreur
        details = play.get("details") or {}
        team_id = details.get("eventOwnerTeamId")  # équipe qui a tiré

        rows.append({
            "game_id": game["id"],
            "event_id": play.get("eventId"),  # identifiant unique de l'événement dans le match
            "period": (play.get("periodDescriptor") or {}).get("number"),
            "team_id": team_id,
            # None si l'équipe est inconnue (on ne devine pas)
            "is_home_team": None if team_id is None else team_id == home_id,
            # côté défendu par l'équipe locale ("left"/"right") ; absent avant 2019-20
            "home_team_defending_side": play.get("homeTeamDefendingSide"),
            "event_type": play["typeDescKey"],
            "x": details.get("xCoord"),  # coordonnées sur la patinoire (pi), None si absentes
            "y": details.get("yCoord"),
            "zone": details.get("zoneCode"),  # O / N / D : sert à deviner la direction avant 2019
            "situation": play.get("situationCode"),  # ex. "1551" : sert pour empty_net
        })
    return rows


def _iter_game_files(season: int, game_type: str, data_dir):
    """Liste (triée) des fichiers JSON déjà téléchargés pour une saison.

    Ex. pour 2017 en saison régulière : raw/2017/201702*.json
    (le code "02" = saison régulière, "03" = séries ; voir GAME_TYPE_CODE dans nhl.py).
    Ne télécharge rien : lit seulement ce qui est sur le disque.
    """
    folder = get_data_dir(data_dir) / str(season)
    return sorted(folder.glob(f"{season}{GAME_TYPE_CODE[game_type]}*.json"))


# --------------------------------------------------------------------------------------
# 2. Calculs vectorisés sur toute la saison
# --------------------------------------------------------------------------------------
def _empty_net_from_situation(situation: pd.Series, is_home_team: pd.Series) -> pd.Series:
    """Calcule empty_net (filet désert de l'équipe qui DÉFEND) à partir du situationCode.

    Le situationCode a 4 chiffres :
        [gardien visiteur][patineurs visiteurs][patineurs locaux][gardien local]
        ex. "1551" = 5 contre 5, deux gardiens ; "0651" = filet visiteur désert (6 patineurs)

    Le filet qui compte est celui de l'équipe qui subit le tir :
        - l'équipe locale tire    -> on regarde le gardien visiteur (1er chiffre)
        - l'équipe visiteuse tire -> on regarde le gardien local (4e chiffre)
    Chiffre "0" = pas de gardien = filet désert.

    Valeur manquante (pd.NA) si le code est absent/mal formé ou si on ne sait pas quelle
    équipe a tiré : l'énoncé demande de NE PAS remplacer ces cas par 0.

    Tout est vectorisé : une seule opération par colonne pour toute la saison.
    """
    code = situation.astype("string")

    # Lignes utilisables : code d'exactement 4 chiffres ET équipe du tireur connue
    valid = code.str.fullmatch(r"\d{4}").fillna(False) & is_home_team.notna()

    # Chiffre du gardien de l'équipe qui défend (vide au départ)
    defending_goalie = pd.Series(pd.NA, index=situation.index, dtype="string")
    home = valid & is_home_team.fillna(False).astype(bool)   # tirs de l'équipe locale
    away = valid & ~is_home_team.fillna(True).astype(bool)   # tirs de l'équipe visiteuse
    defending_goalie[home] = code[home].str[0]  # la locale tire -> gardien visiteur
    defending_goalie[away] = code[away].str[3]  # la visiteuse tire -> gardien local

    # Résultat : True / False pour les lignes valides, pd.NA pour les autres
    empty = pd.Series(pd.NA, index=situation.index, dtype="boolean")
    empty[valid] = defending_goalie[valid].eq("0")
    return empty


def build_season(season: int, game_type: str = REGULAR, data_dir=None) -> pd.DataFrame:
    """Construit le dataframe des tirs d'UNE saison, à partir des JSON (sans cache).

    Étapes :
      1. lire chaque match et accumuler ses tirs dans une liste Python (rapide) ;
      2. convertir la liste en UN dataframe (~80 000 lignes) ;
      3. faire chaque calcul une seule fois sur toute la saison :
         direction d'attaque -> distance -> angle -> empty_net.

    C'est l'étape 2-3 qui rend cette version ~9x plus rapide : une poignée d'appels pandas
    sur un gros tableau, au lieu de ~6 appels par match sur de petits tableaux.
    """
    files = _iter_game_files(season, game_type, data_dir)
    if not files:
        raise FileNotFoundError(
            f"Aucun match {game_type} téléchargé pour {season} dans {get_data_dir(data_dir)}. "
            f"Lancez : uv run download-nhl --first {season} --last {season}"
        )

    # 1. Lecture des matchs un par un ; tous les tirs vont dans la même liste
    rows = []
    for f in files:
        rows.extend(_shot_rows(json.loads(f.read_text())))

    # 2. Un seul dataframe pour toute la saison
    df = pd.DataFrame(rows)
    df["season"] = season
    df["is_goal"] = df["event_type"].eq("goal")              # True si but
    df["is_home_team"] = df["is_home_team"].astype("boolean")  # booléen qui accepte pd.NA

    # 3a. Direction d'attaque (+1 = l'équipe attaque vers la droite, -1 = vers la gauche).
    #     Même fonction que tidy_data.py (cohérent avec le milestone 1), mais appelée une
    #     seule fois pour toute la saison au lieu d'une fois par match.
    #     Nécessaire parce que les équipes changent de côté à chaque période.
    df["attack_dir"] = infer_attack_direction(df)

    # 3b. Distance et angle par rapport au filet attaqué (mêmes formules que
    #     compute_distance / compute_angle). On « ramène » x du côté du filet visé :
    #     dx = distance horizontale entre le tir et le filet.
    dx = NET_X - df["x"] * df["attack_dir"]
    df["distance"] = np.sqrt(dx**2 + df["y"] ** 2)                # Pythagore
    df["angle"] = np.rad2deg(np.arctan2(np.abs(df["y"]), dx))     # 0° = en face du filet ; |y| car symétrique

    # 3c. Filet désert de l'équipe qui défend (voir la fonction plus haut)
    df["empty_net"] = _empty_net_from_situation(df["situation"], df["is_home_team"])

    # On garde les identifiants requis (pour le découpage par match et pour joindre les
    # caractéristiques de la partie 4) + les caractéristiques de la partie 2
    # Version précédente: cols = ["season", "game_id", "event_id", "period", "team_id", "is_home_team", "event_type", "x", "y", "distance", "angle", "is_goal", "empty_net"]
    cols = ["distance", "angle", "is_goal", "empty_net"]
    return df[cols]


# --------------------------------------------------------------------------------------
# 3. Cache Parquet par saison
# --------------------------------------------------------------------------------------
def _cache_path(season: int, game_type: str, cache_dir) -> Path:
    """Chemin du fichier cache d'une saison, ex. processed/feat_eng_1/2017_regular_v1.parquet.

    Dossier choisi dans cet ordre : argument cache_dir, variable d'environnement
    FEAT_ENG_1_CACHE, puis le dossier par défaut. Le "_v1" vient de CACHE_VERSION.
    """
    cache_dir = Path(cache_dir or os.environ.get("FEAT_ENG_1_CACHE", DEFAULT_CACHE_DIR))
    return cache_dir / f"{season}_{game_type}_v{CACHE_VERSION}.parquet"


def load_season_shots(season: int, game_type: str = REGULAR, data_dir=None,
                      cache_dir=None, rebuild: bool = False) -> pd.DataFrame:
    """Une saison, avec cache : lit le fichier Parquet s'il existe, sinon construit puis sauvegarde.

    rebuild=True : ignore le cache et reconstruit (à faire si on modifie build_season
    sans changer CACHE_VERSION).
    Parquet garde les types (booléens nullables, entiers...) et se lit en ~0,1 s.
    """
    path = _cache_path(season, game_type, cache_dir)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)               # cas rapide : déjà calculé

    df = build_season(season, game_type, data_dir)  # cas lent (1re fois) : on calcule
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)               # ...et on sauvegarde pour la prochaine fois
    return df


def load_feat_eng_1_df(start: int = 2016, end: int = 2022, game_type: str = REGULAR,
                       data_dir=None, cache_dir=None, rebuild: bool = False,
                       drop_missing_coords: bool = True, verbose: bool = True) -> pd.DataFrame:
    """POINT D'ENTRÉE : dataframe de la partie 2 pour les saisons start..end (incluses).

    Par défaut 2016-2022 : la saison 2023-24 est le jeu de test, on ne la charge pas ici.
    drop_missing_coords : retire les tirs sans coordonnées (distance/angle impossibles).
    empty_net garde ses valeurs manquantes (demandé par l'énoncé) : elles sont comptées si verbose,
    pour pouvoir rapporter ce nombre dans le blogue.
    """
    # Charge chaque saison (cache ou construction) puis empile les saisons en un seul tableau
    df = pd.concat(
        [load_season_shots(s, game_type, data_dir, cache_dir, rebuild) for s in range(start, end + 1)],
        ignore_index=True,
    )

    # Tirs sans coordonnées : on ne peut pas calculer distance/angle -> on les retire (et on le dit)
    if drop_missing_coords:
        missing = df["distance"].isna() | df["angle"].isna()
        if verbose and missing.any():
            print(f"{missing.sum()} tirs sans coordonnées retirés (sur {len(df)}).")
        df = df.loc[~missing].reset_index(drop=True)

    # Tableau final
        print(f"{len(df)} tirs, saisons {start}-{end} ({game_type}). "
              f"empty_net indéterminé : {df['empty_net'].isna().sum()} événements.")
    return df


if __name__ == "__main__":
    # Construire tous les caches d'un coup :  uv run python -m ift6758.features.feat_eng_1_fast
    import time

    t0 = time.perf_counter()
    load_feat_eng_1_df(2016, 2022, rebuild=True)
    print(f"Construit en {time.perf_counter() - t0:.0f} s")

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
        plt.savefig(f"../../../data_science_blog/images/{start}_{end}_counts_by_{column}.svg", format="svg", bbox_inches="tight")    
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
        plt.savefig(f"../../../data_science_blog/images/{start}_{end}_goals_dist_ang.svg", format="svg", bbox_inches="tight")
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
        plt.savefig(f"../../../data_science_blog/images/{start}_{end}_goalrate_by_{column}.svg", format="svg", bbox_inches="tight")
    plt.show()

def graph_emptynet_goals_per_distance(df, start, end, yscale: str="log", bins=20, save_fig: bool=False, path: str=None):
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
        plt.savefig(f"../../../data_science_blog/images/{start}_{end}_emptynet_dist.svg", format="svg", bbox_inches="tight")
    plt.show()

def graph_all_feature_engineering_1(df=load_feat_eng_1_df(2016,2022), start=2016, end=2022, bins=20, bin_size=10, save_fig: bool=False):
    """
    Tracer tous les graphiques de la section 'Feature engineering 1'
    Input: Dataframe, bins pour 4 des 6 graphiques, année de début de la première saison (ENTRÉE MANUELLE selon df utilisé), année de début de la dernière saison (ENTRÉE MANUELLE selon df utilisé), bin_size pour les courbes de taux de buts
    Output: 6 graphiques
    Attention: les fonctions doivent provenir du dossier 'features' (chemin relatif) pour être sauvegardées au bon dossier.
    """
    graph_goal_counts_by_column(df, column="distance", start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goal_counts_by_column(df, column="angle", start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goals_per_distance_and_angle(df, start=start, end=end, bins=bins, save_fig=save_fig)
    graph_goalrate_by_column(df, column="distance", start=start, end=end, bin_size=bin_size, save_fig=save_fig)
    graph_goalrate_by_column(df, column="angle", start=start, end=end, bin_size=bin_size, save_fig=save_fig)
    graph_emptynet_goals_per_distance(df, start=start, end=end, bins=bins, save_fig=save_fig)