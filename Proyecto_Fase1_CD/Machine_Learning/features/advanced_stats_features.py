"""
Advanced Stats Features
=======================
Genera features estadísticas avanzadas como varianza de rendimiento,
tasas de clean sheets, y promedios móviles de xG.
"""

import pandas as pd
import numpy as np


class AdvancedStatsFeatures:
    """Generador de features estadísticas avanzadas."""
    
    def calculate_rolling_variance(self, df, team, n_games=5):
        """
        Calcula varianza en el rendimiento reciente (goles anotados).
        Una varianza alta indica inconsistencia.
        
        Args:
            df: DataFrame con partidos
            team: Nombre del equipo
            n_games: Ventana de partidos a considerar
        
        Returns:
            Varianza de goles en últimos n_games
        """
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(n_games)
        
        goals_for = []
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                goals_for.append(match['home_goals'])
            else:
                goals_for.append(match['away_goals'])
        
        return np.var(goals_for) if len(goals_for) > 0 else 0
    
    def generate_features(self, df):
        """
        Genera features estadísticas avanzadas.
        
        Args:
            df: DataFrame con columnas home_team, away_team, home_goals, away_goals, 
                home_xg, away_xg, match_id
        
        Returns:
            DataFrame con features avanzadas
        """
        print("   📈 Generando features estadísticas avanzadas...")
        
        df = df.sort_values('date_game').reset_index(drop=True)
        
        features_list = []
        
        for idx, row in df.iterrows():
            # Obtener partidos previos
            prev_matches = df.iloc[:idx]
            
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Partidos previos de cada equipo
            home_prev = prev_matches[(prev_matches['home_team'] == home_team) | (prev_matches['away_team'] == home_team)]
            away_prev = prev_matches[(prev_matches['home_team'] == away_team) | (prev_matches['away_team'] == away_team)]
            
            # Features de varianza/consistencia (últimos 5 partidos)
            home_goal_variance = self.calculate_rolling_variance(prev_matches, home_team, 5)
            away_goal_variance = self.calculate_rolling_variance(prev_matches, away_team, 5)
            
            # Average xG (Expected Goals)
            home_xg_list = []
            for _, match in home_prev.iterrows():
                if match['home_team'] == home_team:
                    home_xg_list.append(match['home_xg'])
                else:
                    home_xg_list.append(match['away_xg'])
            
            away_xg_list = []
            for _, match in away_prev.iterrows():
                if match['home_team'] == away_team:
                    away_xg_list.append(match['home_xg'])
                else:
                    away_xg_list.append(match['away_xg'])
            
            home_xg_avg = np.mean(home_xg_list) if len(home_xg_list) > 0 else 1.0
            away_xg_avg = np.mean(away_xg_list) if len(away_xg_list) > 0 else 1.0
            
            # Clean sheets ratio (últimos 10 partidos)
            home_recent = home_prev.tail(10)
            away_recent = away_prev.tail(10)
            
            home_clean_sheets = 0
            for _, match in home_recent.iterrows():
                if match['home_team'] == home_team:
                    if match['away_goals'] == 0:
                        home_clean_sheets += 1
                else:
                    if match['home_goals'] == 0:
                        home_clean_sheets += 1
            
            away_clean_sheets = 0
            for _, match in away_recent.iterrows():
                if match['home_team'] == away_team:
                    if match['away_goals'] == 0:
                        away_clean_sheets += 1
                else:
                    if match['home_goals'] == 0:
                        away_clean_sheets += 1
            
            home_clean_sheet_rate = home_clean_sheets / max(len(home_recent), 1)
            away_clean_sheet_rate = away_clean_sheets / max(len(away_recent), 1)
            
            # Over 2.5 goals ratio (últimos 10 partidos)
            home_over25 = 0
            for _, match in home_recent.iterrows():
                total_goals = match['home_goals'] + match['away_goals']
                if total_goals > 2.5:
                    home_over25 += 1
            
            away_over25 = 0
            for _, match in away_recent.iterrows():
                total_goals = match['home_goals'] + match['away_goals']
                if total_goals > 2.5:
                    away_over25 += 1
            
            home_over25_rate = home_over25 / max(len(home_recent), 1)
            away_over25_rate = away_over25 / max(len(away_recent), 1)
            
            features = {
                'match_id': row['match_id'],
                'home_goal_variance': home_goal_variance,
                'away_goal_variance': away_goal_variance,
                'goal_variance_diff': home_goal_variance - away_goal_variance,
                'home_xg_avg': home_xg_avg,
                'away_xg_avg': away_xg_avg,
                'xg_avg_diff': home_xg_avg - away_xg_avg,
                'home_clean_sheet_rate': home_clean_sheet_rate,
                'away_clean_sheet_rate': away_clean_sheet_rate,
                'clean_sheet_diff': home_clean_sheet_rate - away_clean_sheet_rate,
                'home_over25_rate': home_over25_rate,
                'away_over25_rate': away_over25_rate,
                'over25_rate_avg': (home_over25_rate + away_over25_rate) / 2
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"   ✓ {len(features_df.columns)-1} features estadísticas avanzadas generadas")
        
        return features_df
