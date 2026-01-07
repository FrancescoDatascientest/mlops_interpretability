import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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

def preprocess_for_inference(df: pd.DataFrame, ohe: OneHotEncoder):
    df = preprocess_initial(df)

    categorical_columns = list(ohe.feature_names_in_)

    for col in categorical_columns:
        if col not in df.columns:
            df[col] = "unknown"

    numeric_columns = [col for col in df.columns if col not in categorical_columns]

    # OHE
    df_cat = pd.DataFrame(
        ohe.transform(df[categorical_columns]),
        columns=ohe.get_feature_names_out(categorical_columns),
        index=df.index
    )

    df_final = pd.concat([df[numeric_columns], df_cat], axis=1)
    return df_final


def preprocess_train(df: pd.DataFrame):
    """Préprocessing strict pour le training (supprime la cible)"""
    df['is_home_pitcher'] = df['inning_topbot'].apply(lambda x: 1 if x == 'Top' else 0)
    
    variables = ['player_name', 'is_home_pitcher', 
                 "release_speed", "release_pos_x", "release_pos_y", 
                 "release_pos_z", "release_extension", "release_spin_rate", 
                 "spin_axis", "p_throws", "pitch_number", 
                 "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z", 
                 "effective_speed", "sz_top", "sz_bot", "arm_angle",
                 "game_type", "stand","age_bat", "age_bat_legacy", "n_priorpa_thisgame_player_at_bat",
                 "of_fielding_alignment", "if_fielding_alignment", "balls","strikes","outs_when_up",
                 "inning", "inning_topbot", "home_score", "away_score","at_bat_number", "pitch_name","description"]
    
    df = df[variables].copy()
    df.dropna(inplace=True)
    
    # Conversion colonnes numériques
    cols_to_convert = [
        'pitch_number','age_bat','age_bat_legacy','n_priorpa_thisgame_player_at_bat',
        'balls','strikes','outs_when_up','inning','home_score','away_score','at_bat_number'
    ]
    for col in cols_to_convert:
        df[col] = df[col].astype(float)
    
    y = df['description']
    X = df.drop(columns = 'description')
    le = LabelEncoder()
    y = le.fit_transform(y)
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    numeric_columns = [col for col in X.columns if col not in categorical_columns]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_cat = pd.DataFrame(ohe.fit_transform(X[categorical_columns]),
                          columns=ohe.get_feature_names_out(categorical_columns),
                          index=df.index)
    df_final = pd.concat([X[numeric_columns], df_cat], axis=1)
    return df_final, ohe, le, y