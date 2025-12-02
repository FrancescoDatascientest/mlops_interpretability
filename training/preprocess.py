import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_initial(df: pd.DataFrame):
    """Préprocessing commun pour toutes les données"""
    df['is_home_pitcher'] = df['inning_topbot'].apply(lambda x: 1 if x == 'Top' else 0)

    variables = ['description', 'player_name', 'is_home_pitcher', 
                 "release_speed", "release_pos_x", "release_pos_y", 
                 "release_pos_z", "release_extension", "release_spin_rate", 
                 "spin_axis", "p_throws", "pitch_name", "pitch_number", 
                 "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z", 
                 "effective_speed", "sz_top", "sz_bot", "arm_angle",
                 "game_type", "stand","age_bat", "age_bat_legacy", "n_priorpa_thisgame_player_at_bat",
                 "of_fielding_alignment", "if_fielding_alignment",   "balls", "strikes", "outs_when_up",
                 "inning", "inning_topbot", "home_score", "away_score", "at_bat_number"]

    df = df[variables].copy()
    df.drop(columns="description", inplace = True)
    df.dropna(inplace=True)

    # Conversion colonnes numériques
    cols_to_convert = [
        'pitch_number','age_bat','age_bat_legacy','n_priorpa_thisgame_player_at_bat',
        'balls','strikes','outs_when_up','inning','home_score','away_score','at_bat_number'
    ]
    for col in cols_to_convert:
        df[col] = df[col].astype(float)

    return df

def preprocess_for_training(df: pd.DataFrame):
    """Preprocessing complet pour le training (fit OHE)"""
    df = preprocess_initial(df)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    numeric_columns = [col for col in df.columns if col not in categorical_columns]

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_cat = pd.DataFrame(ohe.fit_transform(df[categorical_columns]),
                          columns=ohe.get_feature_names_out(categorical_columns),
                          index=df.index)
    df_final = pd.concat([df[numeric_columns], df_cat], axis=1)
    return df_final, ohe

def preprocess_for_inference(df: pd.DataFrame, ohe):
    """Preprocessing pour l'API (transformer avec OHE existant)"""
    df = preprocess_initial(df)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    numeric_columns = [col for col in df.columns if col not in categorical_columns]

    df_cat = pd.DataFrame(ohe.transform(df[categorical_columns]),
                          columns=ohe.get_feature_names_out(categorical_columns),
                          index=df.index)
    df_final = pd.concat([df[numeric_columns], df_cat], axis=1)
    return df_final