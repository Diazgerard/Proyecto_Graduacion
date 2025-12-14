"""
TrainPremierLeague.py
=====================

Script de entrenamiento para modelos de predicción de la Premier League.
Entrena modelos XGBoost y Poisson, guarda archivos .pkl.

Uso: python TrainPremierLeague.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import PoissonRegressor
import warnings
warnings.filterwarnings('ignore')

# Importar generadores de features avanzadas
from elo_features import EloFeatureGenerator
from momentum_features import MomentumFeatureGenerator
from h2h_features import H2HFeatureGenerator
from features.league_position_features import LeaguePositionFeatures
from features.rest_days_features import RestDaysFeatures
from features.advanced_stats_features import AdvancedStatsFeatures

# Configuración
DATA_PATH = "../Data_Mining/eda_outputsMatchesPremierLeague/match_data_cleaned.csv"
MODELS_DIR = "models/premierleague/"
RANDOM_STATE = 42

# Diccionario de normalización de nombres de equipos
TEAM_NAME_MAPPING = {
    'Brighton & Hove Albion': 'Brighton',
    'Manchester Utd': 'Manchester United',
    'Newcastle Utd': 'Newcastle United',
    "Nott'ham Forest": 'Nottingham Forest',
    'Tottenham Hotspur': 'Tottenham',
    'Sheffield Utd': 'Sheffield United',
    'West Brom': 'West Bromwich Albion',
    'Wolves': 'Wolverhampton Wanderers'
}

def normalize_team_name(team_name):
    """Normalizar nombres de equipos para consistencia."""
    return TEAM_NAME_MAPPING.get(team_name, team_name)

def transform_to_match_format(df):
    """Transformar datos de formato largo a formato de partidos."""
    print("Transformando datos a formato de partidos...")
    
    # Convertir date_game a datetime si existe
    if 'date_game' in df.columns:
        df['date_game'] = pd.to_datetime(df['date_game'])
    
    # Separar equipos locales y visitantes
    home_df = df[df['home_away'] == 'home'].copy()
    away_df = df[df['home_away'] == 'away'].copy()
    
    # Agrupar por match_id para unir local y visitante
    matches = []
    for match_id in df['match_id'].unique():
        home_row = home_df[home_df['match_id'] == match_id]
        away_row = away_df[away_df['match_id'] == match_id]
        
        if len(home_row) > 0 and len(away_row) > 0:
            home_row = home_row.iloc[0]
            away_row = away_row.iloc[0]
            
            match = {
                'match_id': match_id,
                'date_game': home_row.get('date_game', pd.NaT),
                'home_team': normalize_team_name(home_row['team_name']),
                'away_team': normalize_team_name(away_row['team_name']),
                'home_goals': home_row['goals_for'],
                'away_goals': away_row['goals_for'],
                'result': home_row['match_result'],
                'home_xg': home_row['ttl_xg'],
                'away_xg': away_row['ttl_xg'],
                'home_possession': home_row['avg_poss'],
                'away_possession': away_row['avg_poss'],
                'home_shots': home_row['ttl_sh'],
                'away_shots': away_row['ttl_sh'],
                'home_shots_on_target': home_row['ttl_sot'],
                'away_shots_on_target': away_row['ttl_sot'],
            }
            matches.append(match)
    
    matches_df = pd.DataFrame(matches)
    
    # Si no hay date_game o están vacías, crear fechas secuenciales
    if 'date_game' not in matches_df.columns or matches_df['date_game'].isna().all():
        matches_df['date_game'] = pd.date_range(start='2017-08-01', periods=len(matches_df), freq='3D')
    
    print(f"   {len(matches_df)} partidos creados")
    return matches_df

def generate_advanced_features(df):
    """Generar TODAS las features avanzadas disponibles."""
    print("🔧 Generando TODAS las features avanzadas...")
    
    # 1. Features ELO Rating (CRÍTICO - Siempre usado)
    print("1. Features ELO Rating...")
    elo_gen = EloFeatureGenerator(k_factor=20, home_advantage=100, initial_rating=1500)
    df_elo = elo_gen.calculate_elo_history(df)
    
    # 2. Features de Momentum
    print("2. Features de Momentum...")
    momentum_gen = MomentumFeatureGenerator(windows=[3, 5, 10])
    df_momentum = momentum_gen.calculate_momentum_features(df)
    
    # 3. Features Head-to-Head
    print("3. Features Head-to-Head...")
    h2h_gen = H2HFeatureGenerator(n_h2h=5)
    df_h2h = h2h_gen.calculate_h2h_features(df)
    
    # 4. Features de Posición en Tabla (NUEVO)
    print("4. Features de Posición en Tabla...")
    position_gen = LeaguePositionFeatures()
    df_position = position_gen.generate_features(df)
    
    # 5. Features de Días de Descanso (NUEVO)
    print("5. Features de Días de Descanso...")
    rest_gen = RestDaysFeatures()
    df_rest = rest_gen.generate_features(df)
    
    # 6. Features Estadísticas Avanzadas (NUEVO)
    print("6. Features Estadísticas Avanzadas...")
    stats_gen = AdvancedStatsFeatures()
    df_stats = stats_gen.generate_features(df)
    
    # 7. Unir todas las features
    print("7. Integrando features...")
    df_merged = df.reset_index(drop=True).copy()
    
    # Preparar dataframes de features
    df_elo_clean = df_elo.drop(['date_game', 'home_team', 'away_team', 'match_id'], axis=1, errors='ignore').reset_index(drop=True)
    df_momentum_clean = df_momentum.drop(['date_game', 'home_team', 'away_team', 'match_id'], axis=1, errors='ignore').reset_index(drop=True)
    df_h2h_clean = df_h2h.drop(['date_game', 'home_team', 'away_team', 'match_id'], axis=1, errors='ignore').reset_index(drop=True)
    df_position_clean = df_position.drop(['match_id'], axis=1, errors='ignore').reset_index(drop=True)
    df_rest_clean = df_rest.drop(['match_id'], axis=1, errors='ignore').reset_index(drop=True)
    df_stats_clean = df_stats.drop(['match_id'], axis=1, errors='ignore').reset_index(drop=True)
    
    # Concatenar features horizontalmente
    df_merged = pd.concat([df_merged, df_elo_clean, df_momentum_clean, df_h2h_clean, 
                          df_position_clean, df_rest_clean, df_stats_clean], axis=1)
    
    total_new_features = (len(df_elo_clean.columns) + len(df_momentum_clean.columns) + 
                         len(df_h2h_clean.columns) + len(df_position_clean.columns) + 
                         len(df_rest_clean.columns) + len(df_stats_clean.columns))
    
    print(f"\n✅ Features integradas: {total_new_features} nuevas features")
    print(f"   - ELO: {len(df_elo_clean.columns)} features")
    print(f"   - Momentum: {len(df_momentum_clean.columns)} features")
    print(f"   - H2H: {len(df_h2h_clean.columns)} features")
    print(f"   - Posición Tabla: {len(df_position_clean.columns)} features")
    print(f"   - Días Descanso: {len(df_rest_clean.columns)} features")
    print(f"   - Stats Avanzadas: {len(df_stats_clean.columns)} features")
    
    return df_merged

def create_features(df):
    """Seleccionar features para ML."""
    print("\n5. Seleccionando features para ML...")
    
    # Seleccionar todas las columnas numéricas excepto identificadores y targets
    exclude_cols = ['match_id', 'date_game', 'home_team', 'away_team', 'result', 'home_goals', 'away_goals']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"   Features seleccionadas: {len(feature_cols)}")
    print(f"   Principales features:")
    for i, col in enumerate(feature_cols[:15]):
        print(f"     - {col}")
    if len(feature_cols) > 15:
        print(f"     ... y {len(feature_cols) - 15} más")
    
    return feature_cols

def train_models():
    """Entrenar todos los modelos."""
    
    print("="*60)
    print("ENTRENAMIENTO DE MODELOS - PREMIER LEAGUE")
    print("="*60)
    
    # 1. Cargar datos
    print(f"\n1. Cargando datos desde: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No se encuentra el archivo {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    # Normalizar nombres de equipos
    df['team_name'] = df['team_name'].apply(normalize_team_name)
    print(f"   Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    print(f"   Equipos únicos: {df['team_name'].nunique()}")
    
    # 2. Transformar a formato de partidos
    print("\n2. Transformando datos...")
    df_matches = transform_to_match_format(df)
    print(f"   Equipos únicos después de transformación:")
    print(f"   - Locales: {df_matches['home_team'].nunique()}")
    print(f"   - Visitantes: {df_matches['away_team'].nunique()}")
    all_teams = set(df_matches['home_team'].unique()) | set(df_matches['away_team'].unique())
    print(f"   - Total único: {len(all_teams)}")
    
    # 3. Limpiar datos
    print("\n3. Limpiando datos...")
    df_clean = df_matches.dropna(subset=['home_team', 'away_team', 'result', 'home_goals', 'away_goals'])
    df_clean['date_game'] = pd.to_datetime(df_clean['date_game'])
    print(f"   Registros después de limpieza: {len(df_clean)}")
    
    # 4. Generar features avanzadas
    df_with_features = generate_advanced_features(df_clean)
    
    # 5. Crear features
    feature_cols = create_features(df_with_features)
    
    # Rellenar NaN en features con 0
    for col in feature_cols:
        if col in df_with_features.columns:
            df_with_features[col] = df_with_features[col].fillna(0)
    
    # 6. Preparar datos para entrenamiento
    print("\n6. Preparando datasets...")
    
    X = df_with_features[feature_cols].values
    y_result = df_with_features['result'].values
    y_home_goals = df_with_features['home_goals'].values
    y_away_goals = df_with_features['away_goals'].values
    
    # Encoder para resultados
    le = LabelEncoder()
    y_result_encoded = le.fit_transform(y_result) # type: ignore
    
    # Normalizar features
    print("   Normalizando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split datos
    X_train, X_test, y_result_train, y_result_test = train_test_split(
        X_scaled, y_result_encoded, test_size=0.2, random_state=RANDOM_STATE
    )
    
    _, _, y_home_train, y_home_test = train_test_split(
        X_scaled, y_home_goals, test_size=0.2, random_state=RANDOM_STATE
    )
    
    _, _, y_away_train, y_away_test = train_test_split(
        X_scaled, y_away_goals, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 7. Entrenar modelo XGBoost mejorado (resultados)
    print("\n7. Entrenando modelo XGBoost mejorado para resultados...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        early_stopping_rounds=20
    )
    xgb_model.fit(
        X_train, y_result_train,
        eval_set=[(X_test, y_result_test)],
        verbose=False
    )
    
    train_acc = xgb_model.score(X_train, y_result_train)
    test_acc = xgb_model.score(X_test, y_result_test)
    print(f"   ✓ Accuracy Train: {train_acc:.3f} | Test: {test_acc:.3f}")
    
    # 8. Entrenar modelos Poisson (goles)
    print("\n8. Entrenando modelos Poisson para goles...")
    
    home_goals_model = PoissonRegressor(max_iter=300)
    home_goals_model.fit(X_train, y_home_train)
    print("   ✓ Modelo goles locales entrenado")
    
    away_goals_model = PoissonRegressor(max_iter=300)
    away_goals_model.fit(X_train, y_away_train)
    print("   ✓ Modelo goles visitantes entrenado")
    
    # 9. Guardar modelos
    print("\n9. Guardando modelos...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Guardar XGBoost
    with open(os.path.join(MODELS_DIR, 'xgb_production.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"   ✓ {MODELS_DIR}xgb_production.pkl")
    
    # Guardar modelos de goles
    with open(os.path.join(MODELS_DIR, 'goals_models.pkl'), 'wb') as f:
        pickle.dump({'home': home_goals_model, 'away': away_goals_model}, f)
    print(f"   ✓ {MODELS_DIR}goals_models.pkl")
    
    # Guardar pipeline (encoder y scaler)
    with open(os.path.join(MODELS_DIR, 'pipeline.pkl'), 'wb') as f:
        pickle.dump({'label_encoder': le, 'scaler': scaler, 'feature_cols': feature_cols}, f)
    print(f"   ✓ {MODELS_DIR}pipeline.pkl")
    
    # Guardar datos de referencia
    equipos_disponibles = sorted(list(set(df_with_features['home_team'].unique()) | set(df_with_features['away_team'].unique())))
    
    with open(os.path.join(MODELS_DIR, 'reference_data.pkl'), 'wb') as f:
        pickle.dump({
            'matches_final': df_with_features[['home_team', 'away_team', 'home_goals', 'away_goals', 'result']],
            'equipos_disponibles': equipos_disponibles,
            'X_sample': X_test[0:1],  # Una muestra para predicciones
            'feature_names': feature_cols
        }, f)
    print(f"   ✓ {MODELS_DIR}reference_data.pkl")
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nResumen:")
    print(f"  - Registros procesados: {len(df_with_features)}")
    print(f"  - Features utilizadas: {len(feature_cols)}")
    print(f"  - Equipos disponibles: {len(equipos_disponibles)}")
    print(f"  - Accuracy modelo resultado: {test_acc:.1%}")
    print(f"  - Modelos guardados en: {MODELS_DIR}")
    print(f"\nPara hacer predicciones, ejecuta: python PredictionPremierLeague.py")
    
    return True


if __name__ == "__main__":
    success = train_models()
    
    if not success:
        print("\n❌ Error en el entrenamiento")
        exit(1)
    
    print("\n✓ Todo listo para predicciones!")
