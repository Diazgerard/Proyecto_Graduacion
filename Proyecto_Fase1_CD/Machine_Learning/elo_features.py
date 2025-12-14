"""
Generador de Features basadas en ELO Rating
============================================

Este módulo genera features avanzadas basadas en el sistema ELO Rating
para mejorar la predicción de Machine Learning.

Features generadas:
- home_elo: Rating ELO del equipo local
- away_elo: Rating ELO del equipo visitante
- elo_diff: Diferencia de ELO (home - away)
- elo_ratio: Ratio de ELO (home / away)
- elo_sum: Suma de ambos ELOs
- elo_avg: Promedio de ambos ELOs
- elo_expected_home: Probabilidad esperada de victoria local según ELO
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime


class EloFeatureGenerator:
    """Generador de features basadas en ELO Rating"""
    
    def __init__(self, k_factor=20, home_advantage=100, initial_rating=1500):
        """
        Inicializar generador de features ELO
        
        Args:
            k_factor (int): Factor K para actualizaciones Elo (default: 20)
            home_advantage (int): Ventaja de local en puntos Elo (default: 100)
            initial_rating (int): Rating inicial para equipos nuevos (default: 1500)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.elo_ratings = {}
        self.elo_history = []  # Historial de todos los ratings
        
    def calculate_elo_history(self, match_data):
        """
        Calcula el historial de ELO para todos los partidos
        
        Args:
            match_data (pd.DataFrame): DataFrame con columnas:
                - date_game: Fecha del partido
                - home_team: Equipo local
                - away_team: Equipo visitante
                - result: Resultado ('H', 'D', 'A')
                - home_goals: Goles del local (opcional)
                - away_goals: Goles del visitante (opcional)
        
        Returns:
            pd.DataFrame: DataFrame con features de ELO añadidas
        """
        print("🔄 Generando features ELO...")
        
        # Inicializar equipos
        teams = set(match_data['home_team'].unique()) | set(match_data['away_team'].unique())
        self.elo_ratings = {team: float(self.initial_rating) for team in teams}
        
        # Lista para almacenar features
        elo_features = []
        
        # Ordenar por fecha
        match_data_sorted = match_data.sort_values('date_game').reset_index(drop=True)
        
        print(f"   Procesando {len(match_data_sorted):,} partidos...")
        print(f"   Equipos únicos: {len(teams)}")
        
        for idx, match in match_data_sorted.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Obtener ELO actual ANTES del partido
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            # Calcular features derivadas
            elo_diff = home_elo - away_elo
            elo_ratio = home_elo / away_elo if away_elo > 0 else 1.0
            elo_sum = home_elo + away_elo
            elo_avg = elo_sum / 2
            
            # Probabilidad esperada según ELO (con ventaja de local)
            expected_home = 1 / (1 + 10**((away_elo - home_elo - self.home_advantage) / 400))
            expected_away = 1 - expected_home
            
            # Guardar features
            elo_features.append({
                'match_id': idx,
                'date_game': match['date_game'],
                'home_team': home_team,
                'away_team': away_team,
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'elo_ratio': elo_ratio,
                'elo_sum': elo_sum,
                'elo_avg': elo_avg,
                'elo_expected_home': expected_home,
                'elo_expected_away': expected_away
            })
            
            # Actualizar ELO después del partido
            if 'result' in match:
                result = match['result']
                
                if result == 'H':
                    actual_home, actual_away = 1.0, 0.0
                elif result == 'A':
                    actual_home, actual_away = 0.0, 1.0
                else:  # Draw
                    actual_home, actual_away = 0.5, 0.5
                
                # Actualizar ratings
                self.elo_ratings[home_team] += self.k_factor * (actual_home - expected_home)
                self.elo_ratings[away_team] += self.k_factor * (actual_away - expected_away)
                
                # Guardar en historial
                self.elo_history.append({
                    'date': match['date_game'],
                    'team': home_team,
                    'elo': self.elo_ratings[home_team],
                    'match_id': idx
                })
                self.elo_history.append({
                    'date': match['date_game'],
                    'team': away_team,
                    'elo': self.elo_ratings[away_team],
                    'match_id': idx
                })
        
        elo_df = pd.DataFrame(elo_features)
        
        # Estadísticas finales
        final_elos = list(self.elo_ratings.values())
        print(f"   ✓ Features ELO generadas")
        print(f"   ELO más alto: {max(final_elos):.0f}")
        print(f"   ELO más bajo: {min(final_elos):.0f}")
        print(f"   ELO promedio: {np.mean(final_elos):.0f}")
        
        return elo_df
    
    def get_current_ratings(self):
        """
        Obtener ratings actuales de todos los equipos
        
        Returns:
            dict: Diccionario {equipo: rating}
        """
        return self.elo_ratings.copy()
    
    def get_team_elo(self, team_name):
        """
        Obtener ELO de un equipo específico
        
        Args:
            team_name (str): Nombre del equipo
        
        Returns:
            float: Rating ELO del equipo
        """
        return self.elo_ratings.get(team_name, self.initial_rating)
    
    def predict_match_probability(self, home_team, away_team):
        """
        Predecir probabilidades de un partido usando ELO
        
        Args:
            home_team (str): Equipo local
            away_team (str): Equipo visitante
        
        Returns:
            dict: Probabilidades {home_win, draw, away_win}
        """
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        # Probabilidad básica de victoria local
        expected_home = 1 / (1 + 10**((away_elo - home_elo - self.home_advantage) / 400))
        expected_away = 1 / (1 + 10**((home_elo - away_elo + self.home_advantage) / 400))
        
        # Empate como residuo (simplificado)
        prob_draw = max(0.1, 1 - expected_home - expected_away)
        
        # Normalizar
        total = expected_home + prob_draw + expected_away
        
        return {
            'home_win': expected_home / total,
            'draw': prob_draw / total,
            'away_win': expected_away / total
        }
    
    def save_ratings(self, filepath):
        """
        Guardar ratings actuales a archivo
        
        Args:
            filepath (str): Ruta del archivo
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'elo_ratings': self.elo_ratings,
                'k_factor': self.k_factor,
                'home_advantage': self.home_advantage,
                'initial_rating': self.initial_rating,
                'elo_history': self.elo_history
            }, f)
        print(f"   ✓ Ratings ELO guardados en: {filepath}")
    
    def load_ratings(self, filepath):
        """
        Cargar ratings desde archivo
        
        Args:
            filepath (str): Ruta del archivo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.elo_ratings = data['elo_ratings']
            self.k_factor = data['k_factor']
            self.home_advantage = data['home_advantage']
            self.initial_rating = data['initial_rating']
            self.elo_history = data.get('elo_history', [])
        print(f"   ✓ Ratings ELO cargados desde: {filepath}")
    
    def get_elo_history_df(self):
        """
        Obtener historial de ELO como DataFrame
        
        Returns:
            pd.DataFrame: Historial de ratings por equipo
        """
        return pd.DataFrame(self.elo_history)


def generate_elo_features_from_csv(input_csv, output_csv=None, save_ratings=True):
    """
    Función helper para generar features ELO desde un CSV
    
    Args:
        input_csv (str): Ruta del CSV de entrada con datos de partidos
        output_csv (str, optional): Ruta del CSV de salida. Si None, se genera automáticamente
        save_ratings (bool): Si guardar los ratings en pickle
    
    Returns:
        pd.DataFrame: DataFrame con features ELO añadidas
    """
    print(f"\n{'='*60}")
    print("GENERADOR DE FEATURES ELO")
    print(f"{'='*60}")
    
    # Leer datos
    print(f"\n📂 Leyendo datos desde: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Verificar columnas requeridas
    required_cols = ['date_game', 'home_team', 'away_team', 'result']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Columnas faltantes en CSV: {missing_cols}")
    
    # Convertir fecha a datetime
    if df['date_game'].dtype == 'object':
        df['date_game'] = pd.to_datetime(df['date_game'])
    
    # Generar features
    elo_gen = EloFeatureGenerator(k_factor=20, home_advantage=100)
    elo_features_df = elo_gen.calculate_elo_history(df)
    
    # Merge con datos originales
    result_df = df.merge(
        elo_features_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
        left_index=True,
        right_on='match_id',
        how='left'
    )
    result_df.drop('match_id', axis=1, errors='ignore', inplace=True)
    
    # Guardar ratings
    if save_ratings:
        ratings_dir = os.path.join(os.path.dirname(input_csv), 'elo_ratings')
        os.makedirs(ratings_dir, exist_ok=True)
        ratings_file = os.path.join(ratings_dir, 'elo_ratings.pkl')
        elo_gen.save_ratings(ratings_file)
    
    # Guardar CSV
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_with_elo.csv"
    
    result_df.to_csv(output_csv, index=False)
    print(f"\n💾 Datos con features ELO guardados en: {output_csv}")
    print(f"\n{'='*60}")
    print("✅ FEATURES ELO GENERADAS EXITOSAMENTE")
    print(f"{'='*60}\n")
    
    return result_df


if __name__ == "__main__":
    """
    Ejemplo de uso:
    
    python elo_features.py
    """
    
    # Buscar datos de entrenamiento
    import sys
    
    # Ruta por defecto
    default_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "Data_Mining",
        "eda_outputsMatchesPremierLeague",
        "match_data_cleaned.csv"
    )
    
    if os.path.exists(default_data_path):
        print(f"\n🎯 Usando datos: {default_data_path}")
        df_with_elo = generate_elo_features_from_csv(default_data_path)
        print(f"\n📊 Shape del DataFrame resultante: {df_with_elo.shape}")
        print(f"\n🔍 Columnas ELO añadidas:")
        elo_cols = [col for col in df_with_elo.columns if 'elo' in col.lower()]
        for col in elo_cols:
            print(f"   • {col}")
    else:
        print(f"\n⚠️  No se encontró el archivo de datos en: {default_data_path}")
        print(f"\n💡 Uso desde código Python:")
        print(f"    from elo_features import generate_elo_features_from_csv")
        print(f"    df = generate_elo_features_from_csv('ruta/a/tus/datos.csv')")
