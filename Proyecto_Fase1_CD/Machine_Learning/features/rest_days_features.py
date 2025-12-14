"""
Rest Days Features
==================
Genera features basadas en días de descanso entre partidos.
El descanso afecta el rendimiento físico y táctico de los equipos.
"""

import pandas as pd
import numpy as np


class RestDaysFeatures:
    """Generador de features de días de descanso."""
    
    def generate_features(self, df):
        """
        Calcula días de descanso para cada equipo.
        
        Args:
            df: DataFrame con columnas date_game, home_team, away_team, match_id
        
        Returns:
            DataFrame con features de descanso
        """
        print("   😴 Generando features de días de descanso...")
        
        df = df.sort_values('date_game').reset_index(drop=True)
        
        # Convertir a datetime si no lo es
        df['date_game'] = pd.to_datetime(df['date_game'])
        
        features_list = []
        
        # Diccionario para trackear último partido de cada equipo
        last_match_date = {}
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date_game']
            
            # Calcular días de descanso para equipo local
            if home_team in last_match_date:
                home_rest_days = (match_date - last_match_date[home_team]).days
            else:
                home_rest_days = 7  # Valor por defecto (1 semana)
            
            # Calcular días de descanso para equipo visitante
            if away_team in last_match_date:
                away_rest_days = (match_date - last_match_date[away_team]).days
            else:
                away_rest_days = 7  # Valor por defecto
            
            features = {
                'match_id': row['match_id'],
                'home_rest_days': home_rest_days,
                'away_rest_days': away_rest_days,
                'rest_days_diff': home_rest_days - away_rest_days,
                'home_is_rested': 1 if home_rest_days >= 5 else 0,
                'away_is_rested': 1 if away_rest_days >= 5 else 0,
                'home_is_tired': 1 if home_rest_days <= 3 else 0,
                'away_is_tired': 1 if away_rest_days <= 3 else 0
            }
            
            features_list.append(features)
            
            # Actualizar últimos partidos
            last_match_date[home_team] = match_date
            last_match_date[away_team] = match_date
        
        features_df = pd.DataFrame(features_list)
        print(f"   ✓ {len(features_df.columns)-1} features de descanso generadas")
        
        return features_df
