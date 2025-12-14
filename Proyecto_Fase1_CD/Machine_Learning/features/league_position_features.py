"""
League Position Features
========================
Genera features basadas en la posición en la tabla de la liga.
"""

import pandas as pd
import numpy as np


class LeaguePositionFeatures:
    """Generador de features de posición en la tabla."""
    
    def __init__(self):
        self.standings_cache = {}
    
    def calculate_standings(self, df, up_to_date=None):
        """
        Calcula la tabla de posiciones hasta una fecha específica.
        
        Args:
            df: DataFrame con partidos
            up_to_date: Fecha límite (None = todas las fechas)
        
        Returns:
            DataFrame con la tabla de posiciones
        """
        if up_to_date:
            matches = df[df['date_game'] < up_to_date].copy()
        else:
            matches = df.copy()
        
        if len(matches) == 0:
            return pd.DataFrame()
        
        # Crear tabla vacía
        teams = pd.concat([matches['home_team'], matches['away_team']]).unique()
        standings = pd.DataFrame({
            'team': teams,
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_diff': 0
        })
        
        # Calcular estadísticas por equipo
        for team in teams:
            home_matches = matches[matches['home_team'] == team]
            away_matches = matches[matches['away_team'] == team]
            
            # Partidos de local
            home_wins = (home_matches['home_goals'] > home_matches['away_goals']).sum()
            home_draws = (home_matches['home_goals'] == home_matches['away_goals']).sum()
            home_losses = (home_matches['home_goals'] < home_matches['away_goals']).sum()
            home_gf = home_matches['home_goals'].sum()
            home_ga = home_matches['away_goals'].sum()
            
            # Partidos de visitante
            away_wins = (away_matches['away_goals'] > away_matches['home_goals']).sum()
            away_draws = (away_matches['away_goals'] == away_matches['home_goals']).sum()
            away_losses = (away_matches['away_goals'] < away_matches['home_goals']).sum()
            away_gf = away_matches['away_goals'].sum()
            away_ga = away_matches['home_goals'].sum()
            
            # Totales
            idx = standings[standings['team'] == team].index[0]
            standings.at[idx, 'played'] = len(home_matches) + len(away_matches)
            standings.at[idx, 'wins'] = home_wins + away_wins
            standings.at[idx, 'draws'] = home_draws + away_draws
            standings.at[idx, 'losses'] = home_losses + away_losses
            standings.at[idx, 'goals_for'] = home_gf + away_gf
            standings.at[idx, 'goals_against'] = home_ga + away_ga
            standings.at[idx, 'points'] = (home_wins + away_wins) * 3 + (home_draws + away_draws)
            standings.at[idx, 'goal_diff'] = standings.at[idx, 'goals_for'] - standings.at[idx, 'goals_against']
        
        # Ordenar por puntos, diferencia de goles, goles a favor
        standings = standings.sort_values(['points', 'goal_diff', 'goals_for'], ascending=[False, False, False])
        standings['position'] = range(1, len(standings) + 1)
        
        return standings
    
    def generate_features(self, df):
        """
        Genera features de posición en tabla para cada partido.
        
        Args:
            df: DataFrame con columnas date_game, home_team, away_team, home_goals, away_goals, match_id
        
        Returns:
            DataFrame con features de posición
        """
        print("   📊 Generando features de posición en tabla...")
        
        df = df.sort_values('date_game').reset_index(drop=True)
        
        features_list = []
        
        for idx, row in df.iterrows():
            # Calcular tabla hasta el partido anterior
            standings = self.calculate_standings(df, up_to_date=row['date_game'])
            
            home_team = row['home_team']
            away_team = row['away_team']
            
            if len(standings) > 0:
                home_stats = standings[standings['team'] == home_team].iloc[0] if len(standings[standings['team'] == home_team]) > 0 else None
                away_stats = standings[standings['team'] == away_team].iloc[0] if len(standings[standings['team'] == away_team]) > 0 else None
            else:
                home_stats = None
                away_stats = None
            
            if home_stats is not None and away_stats is not None:
                features = {
                    'match_id': row['match_id'],
                    'home_position': home_stats['position'],
                    'away_position': away_stats['position'],
                    'position_diff': home_stats['position'] - away_stats['position'],
                    'home_points': home_stats['points'],
                    'away_points': away_stats['points'],
                    'points_diff': home_stats['points'] - away_stats['points'],
                    'home_goal_diff': home_stats['goal_diff'],
                    'away_goal_diff': away_stats['goal_diff'],
                    'home_ppg': home_stats['points'] / max(home_stats['played'], 1),
                    'away_ppg': away_stats['points'] / max(away_stats['played'], 1),
                    'home_win_rate': home_stats['wins'] / max(home_stats['played'], 1),
                    'away_win_rate': away_stats['wins'] / max(away_stats['played'], 1)
                }
            else:
                # Valores por defecto si no hay datos previos
                features = {
                    'match_id': row['match_id'],
                    'home_position': 10,
                    'away_position': 10,
                    'position_diff': 0,
                    'home_points': 0,
                    'away_points': 0,
                    'points_diff': 0,
                    'home_goal_diff': 0,
                    'away_goal_diff': 0,
                    'home_ppg': 0,
                    'away_ppg': 0,
                    'home_win_rate': 0,
                    'away_win_rate': 0
                }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"   ✓ {len(features_df.columns)-1} features de posición generadas")
        
        return features_df
