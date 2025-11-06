#!/usr/bin/env python3
"""
Pipeline ETL para Football ML usando datos CSV existentes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class FootballETLPipelineCSV:
    """
    Pipeline ETL para Machine Learning de fútbol usando archivos CSV
    """
    
    def __init__(self, data_dir):
        """
        Inicializar pipeline con directorio de datos
        """
        self.data_dir = data_dir
        self.csv_file = os.path.join(data_dir, 'match_data_cleaned.csv')
        self.raw_data = None
        self.matches_df = None
        self.elo_ratings = {}
        
        print(f"Pipeline ETL inicializado con datos de: {self.csv_file}")
    
    def extract_raw_data(self):
        """Extraer datos desde archivo CSV"""
        try:
            print("Cargando datos desde CSV...")
            self.raw_data = pd.read_csv(self.csv_file)
            
            # Convertir fecha
            self.raw_data['date_game'] = pd.to_datetime(self.raw_data['date_game'])
            
            print(f"Datos cargados: {len(self.raw_data):,} registros")
            print(f"Rango de fechas: {self.raw_data['date_game'].min()} - {self.raw_data['date_game'].max()}")
            print(f"Partidos únicos: {self.raw_data['match_id'].nunique():,}")
            print(f"Equipos únicos: {self.raw_data['team_name'].nunique():,}")
            
            return self.raw_data
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            return None
    
    def create_match_pairs(self):
        """Crear pares de partidos (home vs away) para cada match_id"""
        if self.raw_data is None:
            print("No hay datos raw disponibles")
            return None
        
        matches = []
        matches_skipped = 0
        
        print(f"Procesando {self.raw_data['match_id'].nunique()} partidos únicos...")
        
        # Agrupar por match_id para crear pares
        for match_id, group in self.raw_data.groupby('match_id'):
            # Verificar que tenemos exactamente 2 registros (home + away)
            if len(group) != 2:
                matches_skipped += 1
                continue
            
            # Verificar que tenemos tanto Home como Away
            home_records = group[group['home_away'] == 'home']
            away_records = group[group['home_away'] == 'away']
            
            if len(home_records) != 1 or len(away_records) != 1:
                matches_skipped += 1
                continue
                
            home_team = home_records.iloc[0]
            away_team = away_records.iloc[0]
            
            # Crear diccionario de datos del partido
            match_record = {
                'match_id': match_id,
                'date_game': home_team['date_game'],
                'season_id': home_team['season_id'],
                'matchday': home_team['matchday'],
                
                # Teams
                'home_team_id': home_team['team_id'],
                'away_team_id': away_team['team_id'],
                'home_team': home_team['team_name'],
                'away_team': away_team['team_name'],
                
                # Results - usar goles reales del partido
                'home_goals': home_team.get('ttl_gls', 0),
                'away_goals': away_team.get('ttl_gls', 0),
            }
            
            # Calcular resultado
            home_goals = match_record['home_goals']
            away_goals = match_record['away_goals']
            match_record['result'] = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
            
            # Home team stats con valores por defecto
            home_stats = {
                'home_shots': home_team.get('ttl_sh', 0),
                'home_shots_on_target': home_team.get('ttl_sot', 0),
                'home_possession': home_team.get('avg_poss', 50),
                'home_passes': home_team.get('ttl_pass_cmp', 0),
                'home_pass_accuracy': home_team.get('pct_pass_cmp', 0),
                'home_tackles': home_team.get('ttl_tkl', 0),
                'home_cards': home_team.get('ttl_yellow_cards', 0) + home_team.get('ttl_red_cards', 0) * 2,
                'home_xg': home_team.get('ttl_xg', 0),
                'home_assists': home_team.get('ttl_ast', 0),
                'home_interceptions': home_team.get('ttl_int', 0),
                'home_blocks': home_team.get('ttl_blocks', 0),
                'home_crosses': home_team.get('ttl_crosses', 0),
                'home_corners': home_team.get('ttl_ck', 0),
                'home_fouls': home_team.get('ttl_fls_for', 0),
                'home_key_passes': home_team.get('ttl_key_passes', 0),
                'home_air_duels_won': home_team.get('ttl_air_dual_won', 0),
                'home_ball_recovery': home_team.get('ttl_ball_recov', 0),
                'home_progressive_passes': home_team.get('ttl_pass_prog', 0)
            }
            
            # Away team stats con valores por defecto
            away_stats = {
                'away_shots': away_team.get('ttl_sh', 0),
                'away_shots_on_target': away_team.get('ttl_sot', 0),
                'away_possession': away_team.get('avg_poss', 50),
                'away_passes': away_team.get('ttl_pass_cmp', 0),
                'away_pass_accuracy': away_team.get('pct_pass_cmp', 0),
                'away_tackles': away_team.get('ttl_tkl', 0),
                'away_cards': away_team.get('ttl_yellow_cards', 0) + away_team.get('ttl_red_cards', 0) * 2,
                'away_xg': away_team.get('ttl_xg', 0),
                'away_assists': away_team.get('ttl_ast', 0),
                'away_interceptions': away_team.get('ttl_int', 0),
                'away_blocks': away_team.get('ttl_blocks', 0),
                'away_crosses': away_team.get('ttl_crosses', 0),
                'away_corners': away_team.get('ttl_ck', 0),
                'away_fouls': away_team.get('ttl_fls_for', 0),
                'away_key_passes': away_team.get('ttl_key_passes', 0),
                'away_air_duels_won': away_team.get('ttl_air_dual_won', 0),
                'away_ball_recovery': away_team.get('ttl_ball_recov', 0),
                'away_progressive_passes': away_team.get('ttl_pass_prog', 0)
            }
            
            # Combinar todo
            match_record.update(home_stats)
            match_record.update(away_stats)
            matches.append(match_record)
        
        # Crear DataFrame final
        matches_df = pd.DataFrame(matches)
        matches_df['date_game'] = pd.to_datetime(matches_df['date_game'])
        matches_df = matches_df.sort_values('date_game').reset_index(drop=True)
        
        print(f"Pares de partidos creados: {len(matches_df):,} matches")
        if matches_skipped > 0:
            print(f"Partidos omitidos: {matches_skipped} (sin par completo Home/Away)")
        
        self.matches_df = matches_df
        return matches_df
    
    def initialize_elo_ratings(self, matches_df, initial_rating=1500.0):
        """Inicializar ratings Elo para todos los equipos"""
        teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
        self.elo_ratings = {team: float(initial_rating) for team in teams}
        print(f"Ratings Elo inicializados para {len(teams)} equipos")
    
    def calculate_elo_features(self, matches_df, k_factor=20):
        """Calcular features basados en sistema Elo"""
        matches_with_elo = matches_df.copy()
        
        # Columnas para Elo ratings
        matches_with_elo['home_elo_before'] = 0.0
        matches_with_elo['away_elo_before'] = 0.0
        matches_with_elo['elo_diff'] = 0.0
        matches_with_elo['elo_home_prob'] = 0.0
        
        for idx, match in matches_with_elo.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Elo antes del partido
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            matches_with_elo.at[idx, 'home_elo_before'] = home_elo
            matches_with_elo.at[idx, 'away_elo_before'] = away_elo
            matches_with_elo.at[idx, 'elo_diff'] = home_elo - away_elo
            
            # Probabilidad basada en Elo (con ventaja de local)
            home_advantage = 100  # Ventaja de local en puntos Elo
            expected_home = 1 / (1 + 10**((away_elo - home_elo - home_advantage) / 400))
            matches_with_elo.at[idx, 'elo_home_prob'] = expected_home
            
            # Actualizar Elo basado en resultado
            if match['result'] == 'H':
                actual_home = 1.0
            elif match['result'] == 'A':
                actual_home = 0.0
            else:  # Draw
                actual_home = 0.5
            
            # Nuevos ratings
            home_new = home_elo + k_factor * (actual_home - expected_home)
            away_new = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))
            
            self.elo_ratings[home_team] = home_new
            self.elo_ratings[away_team] = away_new
        
        print("Features Elo calculados")
        return matches_with_elo
    
    def calculate_form_features(self, matches_df, n_games=5):
        """Calcular features de forma reciente de equipos"""
        matches_with_form = matches_df.copy()
        
        # Inicializar columnas de forma
        form_cols = ['home_form_points', 'away_form_points', 
                    'home_form_goals_for', 'home_form_goals_against',
                    'away_form_goals_for', 'away_form_goals_against']
        
        for col in form_cols:
            matches_with_form[col] = 0.0
        
        for idx, match in matches_with_form.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date_game']
            
            # Obtener partidos anteriores para cada equipo
            for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
                previous_matches = matches_with_form[
                    (matches_with_form['date_game'] < match_date) &
                    ((matches_with_form['home_team'] == team) |
                     (matches_with_form['away_team'] == team))
                ].tail(n_games)
                
                if len(previous_matches) > 0:
                    # Calcular puntos de forma
                    points = 0
                    goals_for = 0
                    goals_against = 0
                    
                    for _, prev_match in previous_matches.iterrows():
                        if prev_match['home_team'] == team:
                            # Equipo jugó en casa
                            goals_for += prev_match['home_goals']
                            goals_against += prev_match['away_goals']
                            if prev_match['result'] == 'H':
                                points += 3
                            elif prev_match['result'] == 'D':
                                points += 1
                        else:
                            # Equipo jugó de visitante
                            goals_for += prev_match['away_goals']
                            goals_against += prev_match['home_goals']
                            if prev_match['result'] == 'A':
                                points += 3
                            elif prev_match['result'] == 'D':
                                points += 1
                    
                    matches_with_form.at[idx, f'{prefix}_form_points'] = points
                    matches_with_form.at[idx, f'{prefix}_form_goals_for'] = goals_for
                    matches_with_form.at[idx, f'{prefix}_form_goals_against'] = goals_against
        
        print("Features de forma calculados")
        return matches_with_form
    
    def calculate_advanced_features(self, matches_df):
        """Calcular features avanzados y ratios"""
        matches_advanced = matches_df.copy()
        
        # Ratios de eficiencia
        matches_advanced['home_shot_accuracy'] = matches_advanced['home_shots_on_target'] / matches_advanced['home_shots'].replace(0, 1)
        matches_advanced['away_shot_accuracy'] = matches_advanced['away_shots_on_target'] / matches_advanced['away_shots'].replace(0, 1)
        
        matches_advanced['home_goal_conversion'] = matches_advanced['home_goals'] / matches_advanced['home_shots_on_target'].replace(0, 1)
        matches_advanced['away_goal_conversion'] = matches_advanced['away_goals'] / matches_advanced['away_shots_on_target'].replace(0, 1)
        
        # Diferencias entre equipos
        matches_advanced['shot_diff'] = matches_advanced['home_shots'] - matches_advanced['away_shots']
        matches_advanced['sot_diff'] = matches_advanced['home_shots_on_target'] - matches_advanced['away_shots_on_target']
        matches_advanced['xg_diff'] = matches_advanced['home_xg'] - matches_advanced['away_xg']
        matches_advanced['possession_diff'] = matches_advanced['home_possession'] - matches_advanced['away_possession']
        matches_advanced['pass_acc_diff'] = matches_advanced['home_pass_accuracy'] - matches_advanced['away_pass_accuracy']
        
        # Features combinados
        matches_advanced['total_goals'] = matches_advanced['home_goals'] + matches_advanced['away_goals']
        matches_advanced['total_shots'] = matches_advanced['home_shots'] + matches_advanced['away_shots']
        matches_advanced['total_cards'] = matches_advanced['home_cards'] + matches_advanced['away_cards']
        
        # Clasificación de intensidad
        matches_advanced['high_scoring'] = (matches_advanced['total_goals'] >= 3).astype(int)
        matches_advanced['low_scoring'] = (matches_advanced['total_goals'] <= 1).astype(int)
        
        print("Features avanzados calculados")
        return matches_advanced
    
    def run_full_pipeline(self):
        """Ejecutar pipeline completo de ETL y feature engineering"""
        print("Iniciando pipeline completo de ETL...")
        
        # 1. Extraer datos
        raw_data = self.extract_raw_data()
        if raw_data is None:
            return None
        
        # 2. Crear pares de partidos
        matches_df = self.create_match_pairs()
        if matches_df is None:
            return None
        
        # 3. Inicializar Elo ratings
        self.initialize_elo_ratings(matches_df)
        
        # 4. Calcular features Elo
        matches_df = self.calculate_elo_features(matches_df)
        
        # 5. Calcular features de forma
        matches_df = self.calculate_form_features(matches_df)
        
        # 6. Calcuar features avanzados
        matches_df = self.calculate_advanced_features(matches_df)
        
        print(f"Pipeline completado! Dataset final: {len(matches_df):,} partidos con {len(matches_df.columns)} features")
        
        return matches_df

if __name__ == "__main__":
    # Directorio de datos
    data_dir = r"c:\Users\gerar\OneDrive\Desktop\Proyecto_Graduacion\Proyecto_Fase1_CD\Data_Mining\eda_outputsMatchesPremierLeague"
    
    # Crear y ejecutar pipeline
    pipeline = FootballETLPipelineCSV(data_dir)
    final_dataset = pipeline.run_full_pipeline()
    
    if final_dataset is not None:
        print("\nResumen del dataset final:")
        print(f"  • Partidos: {len(final_dataset):,}")
        print(f"  • Features: {len(final_dataset.columns)}")
        print(f"  • Rango fechas: {final_dataset['date_game'].min()} - {final_dataset['date_game'].max()}")
        print(f"  • Temporadas: {final_dataset['season_id'].nunique()}")
        print(f"  • Equipos: {final_dataset['home_team'].nunique()}")
        
        # Mostrar muestra
        print(f"\nMuestra de datos:")
        sample_cols = ['date_game', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'elo_diff', 'home_form_points']
        print(final_dataset[sample_cols].head(10).to_string(index=False))