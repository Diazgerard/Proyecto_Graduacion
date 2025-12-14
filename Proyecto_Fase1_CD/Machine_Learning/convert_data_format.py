"""
Script para convertir datos a formato por partido
==================================================

Convierte el formato actual (una fila por equipo) a formato de partido
(una fila por partido con home_team, away_team, etc.)

Este script usa el CSV limpio del EDA que YA está actualizado desde PostgreSQL.
"""

import pandas as pd
import os

# Mapeo de nombres de equipos para normalización
TEAM_NAME_MAPPING = {
    "Nott'ham Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Nottham Forest": "Nottingham Forest",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "West Ham": "West Ham United",
    "West Ham United": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Leeds United": "Leeds United",
    "Leicester City": "Leicester City",
    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
}

def normalize_team_name(team_name):
    """Normalizar nombre de equipo usando el mapeo"""
    if pd.isna(team_name):
        return team_name
    return TEAM_NAME_MAPPING.get(team_name, team_name)

# Leer datos del EDA (YA está actualizado desde PostgreSQL)
data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Data_Mining",
    "eda_outputsMatchesPremierLeague",
    "match_data_cleaned.csv"
)

print("📂 Leyendo datos del EDA (actualizado desde PostgreSQL)...")
print(f"   Ruta: {data_path}")
df = pd.read_csv(data_path)

print(f"   Filas originales: {len(df):,}")

# Separar home y away
home_df = df[df['home_away'] == 'home'].copy()
away_df = df[df['home_away'] == 'away'].copy()

print(f"   Partidos home: {len(home_df):,}")
print(f"   Partidos away: {len(away_df):,}")

# Merge por match_id y date
matches = home_df.merge(
    away_df,
    on=['match_id', 'date_game', 'matchday', 'season_id'],
    suffixes=('_home', '_away')
)

print(f"   Partidos únicos: {len(matches):,}")

# NORMALIZAR NOMBRES DE EQUIPOS
print("\n🔧 Normalizando nombres de equipos...")
matches['team_name_home'] = matches['team_name_home'].apply(normalize_team_name)
matches['team_name_away'] = matches['team_name_away'].apply(normalize_team_name)

# Verificar equipos únicos después de normalización
home_teams = set(matches['team_name_home'].unique())
away_teams = set(matches['team_name_away'].unique())
all_teams = sorted(list(home_teams | away_teams))
print(f"   ✓ Equipos únicos después de normalización: {len(all_teams)}")
print(f"   Equipos: {', '.join(all_teams[:10])}...")

# Crear DataFrame simplificado para ML
match_df = pd.DataFrame({
    'match_id': matches['match_id'],
    'date_game': pd.to_datetime(matches['date_game']),
    'season': matches['season_id'],
    'matchday': matches['matchday'],
    
    # Equipos (ya normalizados)
    'home_team': matches['team_name_home'],
    'away_team': matches['team_name_away'],
    
    # Resultado
    'home_goals': matches['goals_for_home'],
    'away_goals': matches['goals_for_away'],
    'result': matches['match_result_home'].map({'Win': 'H', 'Loss': 'A', 'Draw': 'D'}),
    
    # Estadísticas home
    'home_xg': matches['ttl_xg_home'],
    'home_shots': matches['total_shots_home'],
    'home_shots_on_target': matches['ttl_sot_home'],
    'home_possession': matches['avg_poss_home'],
    'home_passes': matches['ttl_pass_att_home'],
    'home_pass_accuracy': matches['pct_pass_cmp_home'],
    'home_tackles': matches['ttl_tkl_home'],
    'home_fouls': matches['ttl_fls_for_home'],
    'home_corners': matches['ttl_ck_home'],
    'home_yellow_cards': matches['ttl_yellow_cards_home'],
    'home_red_cards': matches['ttl_red_cards_home'],
    
    # Estadísticas away
    'away_xg': matches['ttl_xg_away'],
    'away_shots': matches['total_shots_away'],
    'away_shots_on_target': matches['ttl_sot_away'],
    'away_possession': matches['avg_poss_away'],
    'away_passes': matches['ttl_pass_att_away'],
    'away_pass_accuracy': matches['pct_pass_cmp_away'],
    'away_tackles': matches['ttl_tkl_away'],
    'away_fouls': matches['ttl_fls_for_away'],
    'away_corners': matches['ttl_ck_away'],
    'away_yellow_cards': matches['ttl_yellow_cards_away'],
    'away_red_cards': matches['ttl_red_cards_away'],
})

# Ordenar por fecha
match_df = match_df.sort_values('date_game').reset_index(drop=True)

# Guardar
output_path = os.path.join(
    os.path.dirname(__file__),
    "match_data_for_ml.csv"
)

match_df.to_csv(output_path, index=False)

print(f"\n✅ Datos convertidos exitosamente")
print(f"   Archivo guardado: {output_path}")
print(f"   Shape: {match_df.shape}")
print(f"\n📊 Columnas:")
for col in match_df.columns:
    print(f"   • {col}")

print(f"\n🔍 Primeras filas:")
print(match_df.head())

print(f"\n📈 Distribución de resultados:")
print(match_df['result'].value_counts())
