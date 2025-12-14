"""
Generador de Features Head-to-Head (H2H)
========================================

Este módulo genera features basadas en enfrentamientos directos entre equipos
para mejorar la predicción de Machine Learning.

Features generadas:
- h2h_home_wins: Victorias del local en últimos N enfrentamientos
- h2h_away_wins: Victorias del visitante en últimos N enfrentamientos
- h2h_draws: Empates en últimos N enfrentamientos
- h2h_avg_goals: Promedio de goles totales en H2H
- h2h_home_avg_goals: Promedio de goles del local en H2H
- h2h_away_avg_goals: Promedio de goles del visitante en H2H
- h2h_matches: Número de enfrentamientos previos considerados
- h2h_home_dominance: Ratio de dominio del local (wins / total)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict


class H2HFeatureGenerator:
    """Generador de features Head-to-Head"""
    
    def __init__(self, n_h2h=5):
        """
        Inicializar generador de features H2H
        
        Args:
            n_h2h (int): Número de enfrentamientos previos a considerar (default: 5)
        """
        self.n_h2h = n_h2h
        self.h2h_history = defaultdict(list)
        
    def calculate_h2h_features(self, match_data):
        """
        Calcula features de enfrentamientos directos para todos los partidos
        
        Args:
            match_data (pd.DataFrame): DataFrame con columnas:
                - date_game: Fecha del partido
                - home_team: Equipo local
                - away_team: Equipo visitante
                - home_goals: Goles del local
                - away_goals: Goles del visitante
                - result: Resultado ('H', 'D', 'A')
        
        Returns:
            pd.DataFrame: DataFrame con features H2H añadidas
        """
        print("🔄 Generando features Head-to-Head (H2H)...")
        
        # Ordenar por fecha
        match_data_sorted = match_data.sort_values('date_game').reset_index(drop=True)
        
        print(f"   Procesando {len(match_data_sorted):,} partidos...")
        print(f"   Considerando últimos {self.n_h2h} enfrentamientos H2H")
        
        h2h_features = []
        
        for idx, match in match_data_sorted.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Calcular features H2H ANTES del partido
            features = self._calculate_h2h_stats(home_team, away_team)
            
            features_dict = {
                'match_id': idx,
                'date_game': match['date_game'],
                'home_team': home_team,
                'away_team': away_team,
                **features
            }
            
            h2h_features.append(features_dict)
            
            # Actualizar historial H2H DESPUÉS del partido
            self._update_h2h_history(match)
        
        h2h_df = pd.DataFrame(h2h_features)
        
        # Estadísticas
        matches_with_h2h = h2h_df[h2h_df['h2h_matches'] > 0].shape[0]
        pct_with_h2h = (matches_with_h2h / len(h2h_df)) * 100 if len(h2h_df) > 0 else 0
        
        print(f"   ✓ Features H2H generadas")
        print(f"   Partidos con historial H2H: {matches_with_h2h:,} ({pct_with_h2h:.1f}%)")
        
        return h2h_df
    
    def _calculate_h2h_stats(self, home_team, away_team):
        """
        Calcula estadísticas H2H entre dos equipos
        
        Args:
            home_team (str): Equipo local
            away_team (str): Equipo visitante
        
        Returns:
            dict: Diccionario con estadísticas H2H
        """
        # Clave ordenada para buscar historial (independiente del orden)
        teams_key = tuple(sorted([home_team, away_team]))
        
        # Obtener historial reciente
        history = self.h2h_history[teams_key]
        recent_h2h = history[-self.n_h2h:] if len(history) > self.n_h2h else history
        
        # Si no hay historial
        if not recent_h2h:
            return {
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_draws': 0,
                'h2h_avg_goals': 0.0,
                'h2h_home_avg_goals': 0.0,
                'h2h_away_avg_goals': 0.0,
                'h2h_matches': 0,
                'h2h_home_dominance': 0.5,
                'h2h_goals_variance': 0.0
            }
        
        # Contar resultados considerando quién es local en el partido actual
        home_wins = 0
        away_wins = 0
        draws = 0
        total_goals = []
        home_goals_list = []
        away_goals_list = []
        
        for h2h_match in recent_h2h:
            total_goals.append(h2h_match['total_goals'])
            
            # Identificar quién ganó desde la perspectiva del partido actual
            if h2h_match['home_team_in_h2h'] == home_team:
                # El equipo local actual era local en el H2H
                home_goals_list.append(h2h_match['home_goals'])
                away_goals_list.append(h2h_match['away_goals'])
                
                if h2h_match['winner'] == home_team:
                    home_wins += 1
                elif h2h_match['winner'] == away_team:
                    away_wins += 1
                else:
                    draws += 1
            else:
                # El equipo local actual era visitante en el H2H
                home_goals_list.append(h2h_match['away_goals'])
                away_goals_list.append(h2h_match['home_goals'])
                
                if h2h_match['winner'] == home_team:
                    home_wins += 1
                elif h2h_match['winner'] == away_team:
                    away_wins += 1
                else:
                    draws += 1
        
        # Calcular métricas
        n_matches = len(recent_h2h)
        avg_goals = np.mean(total_goals)
        avg_home_goals = np.mean(home_goals_list)
        avg_away_goals = np.mean(away_goals_list)
        
        # Dominancia del local (0 a 1, donde 1 = siempre gana el local)
        home_dominance = (home_wins + 0.5 * draws) / n_matches if n_matches > 0 else 0.5
        
        # Varianza de goles (indica lo predecibles que son los partidos)
        goals_variance = np.var(total_goals) if len(total_goals) > 1 else 0.0
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_avg_goals': float(avg_goals),
            'h2h_home_avg_goals': float(avg_home_goals),
            'h2h_away_avg_goals': float(avg_away_goals),
            'h2h_matches': n_matches,
            'h2h_home_dominance': float(home_dominance),
            'h2h_goals_variance': float(goals_variance)
        }
    
    def _update_h2h_history(self, match):
        """
        Actualiza el historial H2H después de un partido
        
        Args:
            match (pd.Series): Serie con datos del partido
        """
        home_team = match['home_team']
        away_team = match['away_team']
        result = match['result']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        
        # Clave ordenada
        teams_key = tuple(sorted([home_team, away_team]))
        
        # Determinar ganador
        if result == 'H':
            winner = home_team
        elif result == 'A':
            winner = away_team
        else:
            winner = None  # Empate
        
        # Añadir al historial
        self.h2h_history[teams_key].append({
            'home_team_in_h2h': home_team,
            'away_team_in_h2h': away_team,
            'winner': winner,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'total_goals': home_goals + away_goals,
            'date': match['date_game']
        })
    
    def get_h2h_record(self, team1, team2, last_n=None):
        """
        Obtener récord H2H entre dos equipos
        
        Args:
            team1 (str): Primer equipo
            team2 (str): Segundo equipo
            last_n (int, optional): Últimos N enfrentamientos
        
        Returns:
            dict: Récord H2H
        """
        teams_key = tuple(sorted([team1, team2]))
        history = self.h2h_history[teams_key]
        
        if last_n:
            history = history[-last_n:]
        
        team1_wins = sum(1 for m in history if m['winner'] == team1)
        team2_wins = sum(1 for m in history if m['winner'] == team2)
        draws = len(history) - team1_wins - team2_wins
        
        return {
            'team1': team1,
            'team2': team2,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'draws': draws,
            'total_matches': len(history)
        }


def generate_h2h_features_from_csv(input_csv, output_csv=None, n_h2h=5):
    """
    Función helper para generar features H2H desde un CSV
    
    Args:
        input_csv (str): Ruta del CSV de entrada con datos de partidos
        output_csv (str, optional): Ruta del CSV de salida
        n_h2h (int): Número de enfrentamientos previos a considerar
    
    Returns:
        pd.DataFrame: DataFrame con features H2H añadidas
    """
    print(f"\n{'='*60}")
    print("GENERADOR DE FEATURES HEAD-TO-HEAD (H2H)")
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
    h2h_gen = H2HFeatureGenerator(n_h2h=n_h2h)
    h2h_features_df = h2h_gen.calculate_h2h_features(df)
    
    # Merge con datos originales
    result_df = df.merge(
        h2h_features_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
        left_index=True,
        right_on='match_id',
        how='left'
    )
    result_df.drop('match_id', axis=1, errors='ignore', inplace=True)
    
    # Guardar CSV
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_with_h2h.csv"
    
    result_df.to_csv(output_csv, index=False)
    print(f"\n💾 Datos con features H2H guardados en: {output_csv}")
    print(f"\n{'='*60}")
    print("✅ FEATURES H2H GENERADAS EXITOSAMENTE")
    print(f"{'='*60}\n")
    
    return result_df


if __name__ == "__main__":
    """
    Ejemplo de uso:
    
    python h2h_features.py
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
        df_with_h2h = generate_h2h_features_from_csv(default_data_path)
        print(f"\n📊 Shape del DataFrame resultante: {df_with_h2h.shape}")
        print(f"\n🔍 Columnas H2H añadidas:")
        h2h_cols = [col for col in df_with_h2h.columns if 'h2h' in col.lower()]
        for col in h2h_cols:
            print(f"   • {col}")
    else:
        print(f"\n⚠️  No se encontró el archivo de datos en: {default_data_path}")
        print(f"\n💡 Uso desde código Python:")
        print(f"    from h2h_features import generate_h2h_features_from_csv")
        print(f"    df = generate_h2h_features_from_csv('ruta/a/tus/datos.csv')")
