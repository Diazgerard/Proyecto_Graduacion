"""
Generador de Features de Momentum y Forma
==========================================

Este módulo genera features basadas en la forma reciente y momentum de los equipos
para mejorar la predicción de Machine Learning.

Features generadas por equipo (home/away):
- points_last_N: Puntos obtenidos en últimos N partidos
- goals_for_last_N: Goles a favor en últimos N partidos
- goals_against_last_N: Goles en contra en últimos N partidos
- goal_diff_last_N: Diferencia de goles en últimos N partidos
- ppg_last_N: Puntos por partido en últimos N partidos
- gpg_last_N: Goles por partido en últimos N partidos
- current_streak: Racha actual (positiva=victorias, negativa=derrotas)
- wins_last_N: Victorias en últimos N partidos
- draws_last_N: Empates en últimos N partidos
- losses_last_N: Derrotas en últimos N partidos
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict


class MomentumFeatureGenerator:
    """Generador de features de momentum y forma reciente"""
    
    def __init__(self, windows=[3, 5, 10]):
        """
        Inicializar generador de features de momentum
        
        Args:
            windows (list): Ventanas temporales para calcular estadísticas (default: [3, 5, 10])
        """
        self.windows = windows
        self.team_history = defaultdict(lambda: {
            'results': [],
            'goals_for': [],
            'goals_against': [],
            'dates': []
        })
        
    def calculate_momentum_features(self, match_data):
        """
        Calcula features de momentum para todos los partidos
        
        Args:
            match_data (pd.DataFrame): DataFrame con columnas:
                - date_game: Fecha del partido
                - home_team: Equipo local
                - away_team: Equipo visitante
                - home_goals: Goles del local
                - away_goals: Goles del visitante
                - result: Resultado ('H', 'D', 'A')
        
        Returns:
            pd.DataFrame: DataFrame con features de momentum añadidas
        """
        print("🔄 Generando features de Momentum...")
        
        # Ordenar por fecha
        match_data_sorted = match_data.sort_values('date_game').reset_index(drop=True)
        
        print(f"   Procesando {len(match_data_sorted):,} partidos...")
        print(f"   Ventanas: {self.windows}")
        
        momentum_features = []
        
        for idx, match in match_data_sorted.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Calcular features ANTES del partido actual
            home_features = self._calculate_team_momentum(home_team, venue='home')
            away_features = self._calculate_team_momentum(away_team, venue='away')
            
            # Guardar con prefijos
            features_dict = {
                'match_id': idx,
                'date_game': match['date_game'],
                'home_team': home_team,
                'away_team': away_team
            }
            
            # Añadir features de local
            for key, value in home_features.items():
                features_dict[f'home_{key}'] = value
            
            # Añadir features de visitante
            for key, value in away_features.items():
                features_dict[f'away_{key}'] = value
            
            # Features comparativas
            for window in self.windows:
                features_dict[f'points_diff_last_{window}'] = (
                    home_features.get(f'points_last_{window}', 0) - 
                    away_features.get(f'points_last_{window}', 0)
                )
                features_dict[f'goal_diff_advantage_last_{window}'] = (
                    home_features.get(f'goal_diff_last_{window}', 0) - 
                    away_features.get(f'goal_diff_last_{window}', 0)
                )
            
            momentum_features.append(features_dict)
            
            # Actualizar historial DESPUÉS del partido
            self._update_team_history(match)
        
        momentum_df = pd.DataFrame(momentum_features)
        
        print(f"   ✓ Features de Momentum generadas")
        print(f"   Total de features: {len([col for col in momentum_df.columns if col not in ['match_id', 'date_game', 'home_team', 'away_team']])}")
        
        return momentum_df
    
    def _calculate_team_momentum(self, team, venue='all'):
        """
        Calcula métricas de momentum para un equipo
        
        Args:
            team (str): Nombre del equipo
            venue (str): Venue filter ('home', 'away', 'all')
        
        Returns:
            dict: Diccionario con features de momentum
        """
        features = {}
        history = self.team_history[team]
        
        # Si no hay historial, retornar valores por defecto
        if not history['results']:
            for window in self.windows:
                features[f'points_last_{window}'] = 0
                features[f'goals_for_last_{window}'] = 0
                features[f'goals_against_last_{window}'] = 0
                features[f'goal_diff_last_{window}'] = 0
                features[f'ppg_last_{window}'] = 0
                features[f'gpg_last_{window}'] = 0
                features[f'gapg_last_{window}'] = 0
                features[f'wins_last_{window}'] = 0
                features[f'draws_last_{window}'] = 0
                features[f'losses_last_{window}'] = 0
            features['current_streak'] = 0
            features['total_matches'] = 0
            return features
        
        # Calcular para cada ventana
        for window in self.windows:
            recent_results = history['results'][-window:] if len(history['results']) >= window else history['results']
            recent_gf = history['goals_for'][-window:] if len(history['goals_for']) >= window else history['goals_for']
            recent_ga = history['goals_against'][-window:] if len(history['goals_against']) >= window else history['goals_against']
            
            n_matches = len(recent_results)
            
            if n_matches > 0:
                # Puntos y goles
                features[f'points_last_{window}'] = sum(recent_results)
                features[f'goals_for_last_{window}'] = sum(recent_gf)
                features[f'goals_against_last_{window}'] = sum(recent_ga)
                features[f'goal_diff_last_{window}'] = sum(recent_gf) - sum(recent_ga)
                
                # Promedios
                features[f'ppg_last_{window}'] = sum(recent_results) / n_matches
                features[f'gpg_last_{window}'] = sum(recent_gf) / n_matches
                features[f'gapg_last_{window}'] = sum(recent_ga) / n_matches
                
                # Resultados
                features[f'wins_last_{window}'] = sum(1 for p in recent_results if p == 3)
                features[f'draws_last_{window}'] = sum(1 for p in recent_results if p == 1)
                features[f'losses_last_{window}'] = sum(1 for p in recent_results if p == 0)
            else:
                # Sin datos
                for stat in ['points', 'goals_for', 'goals_against', 'goal_diff', 'ppg', 'gpg', 'gapg', 'wins', 'draws', 'losses']:
                    features[f'{stat}_last_{window}'] = 0
        
        # Racha actual (victorias/derrotas consecutivas)
        if history['results']:
            current_streak = 0
            for points in reversed(history['results']):
                if points == 3:  # Victoria
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        break
                elif points == 0:  # Derrota
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        break
                else:  # Empate
                    break
            features['current_streak'] = current_streak
        else:
            features['current_streak'] = 0
        
        # Total de partidos jugados
        features['total_matches'] = len(history['results'])
        
        return features
    
    def _update_team_history(self, match):
        """
        Actualiza el historial de un equipo después de un partido
        
        Args:
            match (pd.Series): Serie con datos del partido
        """
        home_team = match['home_team']
        away_team = match['away_team']
        result = match['result']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        date = match['date_game']
        
        # Puntos: 3 victoria, 1 empate, 0 derrota
        home_points = 3 if result == 'H' else (1 if result == 'D' else 0)
        away_points = 3 if result == 'A' else (1 if result == 'D' else 0)
        
        # Actualizar local
        self.team_history[home_team]['results'].append(home_points)
        self.team_history[home_team]['goals_for'].append(home_goals)
        self.team_history[home_team]['goals_against'].append(away_goals)
        self.team_history[home_team]['dates'].append(date)
        
        # Actualizar visitante
        self.team_history[away_team]['results'].append(away_points)
        self.team_history[away_team]['goals_for'].append(away_goals)
        self.team_history[away_team]['goals_against'].append(home_goals)
        self.team_history[away_team]['dates'].append(date)
    
    def get_team_form(self, team, last_n=5):
        """
        Obtener forma reciente de un equipo en formato legible
        
        Args:
            team (str): Nombre del equipo
            last_n (int): Últimos N partidos
        
        Returns:
            str: String con forma (ej: "WWDLW")
        """
        history = self.team_history[team]
        if not history['results']:
            return "N/A"
        
        recent = history['results'][-last_n:]
        form_map = {3: 'W', 1: 'D', 0: 'L'}
        return ''.join([form_map[p] for p in recent])


def generate_momentum_features_from_csv(input_csv, output_csv=None, windows=[3, 5, 10]):
    """
    Función helper para generar features de momentum desde un CSV
    
    Args:
        input_csv (str): Ruta del CSV de entrada con datos de partidos
        output_csv (str, optional): Ruta del CSV de salida
        windows (list): Ventanas temporales para calcular estadísticas
    
    Returns:
        pd.DataFrame: DataFrame con features de momentum añadidas
    """
    print(f"\n{'='*60}")
    print("GENERADOR DE FEATURES DE MOMENTUM")
    print(f"{'='*60}")
    
    # Leer datos
    print(f"\n📂 Leyendo datos desde: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Verificar columnas requeridas
    required_cols = ['date_game', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Columnas faltantes en CSV: {missing_cols}")
    
    # Convertir fecha a datetime
    if df['date_game'].dtype == 'object':
        df['date_game'] = pd.to_datetime(df['date_game'])
    
    # Generar features
    momentum_gen = MomentumFeatureGenerator(windows=windows)
    momentum_features_df = momentum_gen.calculate_momentum_features(df)
    
    # Merge con datos originales
    result_df = df.merge(
        momentum_features_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
        left_index=True,
        right_on='match_id',
        how='left'
    )
    result_df.drop('match_id', axis=1, errors='ignore', inplace=True)
    
    # Guardar CSV
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_with_momentum.csv"
    
    result_df.to_csv(output_csv, index=False)
    print(f"\n💾 Datos con features de Momentum guardados en: {output_csv}")
    print(f"\n{'='*60}")
    print("✅ FEATURES DE MOMENTUM GENERADAS EXITOSAMENTE")
    print(f"{'='*60}\n")
    
    return result_df


if __name__ == "__main__":
    """
    Ejemplo de uso:
    
    python momentum_features.py
    """
    
    # Buscar datos de entrenamiento
    default_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "Data_Mining",
        "eda_outputsMatchesPremierLeague",
        "match_data_cleaned.csv"
    )
    
    if os.path.exists(default_data_path):
        print(f"\n🎯 Usando datos: {default_data_path}")
        df_with_momentum = generate_momentum_features_from_csv(default_data_path)
        print(f"\n📊 Shape del DataFrame resultante: {df_with_momentum.shape}")
        print(f"\n🔍 Columnas Momentum añadidas (primeras 10):")
        momentum_cols = [col for col in df_with_momentum.columns if any(x in col.lower() for x in ['points', 'streak', 'wins', 'ppg', 'gpg'])]
        for col in momentum_cols[:10]:
            print(f"   • {col}")
        if len(momentum_cols) > 10:
            print(f"   ... y {len(momentum_cols) - 10} más")
    else:
        print(f"\n⚠️  No se encontró el archivo de datos en: {default_data_path}")
        print(f"\n💡 Uso desde código Python:")
        print(f"    from momentum_features import generate_momentum_features_from_csv")
        print(f"    df = generate_momentum_features_from_csv('ruta/a/tus/datos.csv')")
