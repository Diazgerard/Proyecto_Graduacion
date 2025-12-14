"""
PredictionPremierLeague.py
==========================

Script independiente para predicción de partidos de la Premier League.
Genera features dinámicas actualizadas para cada predicción.

Uso: python PredictionPremierLeague.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar generadores de features
sys.path.append(str(Path(__file__).parent))
from elo_features import EloFeatureGenerator
from momentum_features import MomentumFeatureGenerator
from h2h_features import H2HFeatureGenerator
from features.league_position_features import LeaguePositionFeatures
from features.rest_days_features import RestDaysFeatures
from features.advanced_stats_features import AdvancedStatsFeatures

class PremierLeaguePredictor:
    """
    Predictor de partidos de la Premier League usando modelos entrenados.
    Genera features dinámicas en tiempo real para predicciones precisas.
    """
    
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
    
    def __init__(self, models_path="models/premierleague/", data_path="../Data_Mining/eda_outputsMatchesPremierLeague/match_data_cleaned.csv"):
        self.models_path = models_path
        self.data_path = data_path
        self.models_loaded = False
        
        # Modelos y datos
        self.xgb_model = None
        self.home_goals_model = None
        self.away_goals_model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        self.teams_list = None
        self.matches_data = None
        
        # Generadores de features
        self.elo_system = None
        self.momentum_gen = None
        self.h2h_gen = None
        self.position_gen = None
        self.rest_gen = None
        self.stats_gen = None
    
    @staticmethod
    def normalize_team_name(team_name):
        """Normalizar nombres de equipos para consistencia."""
        return PremierLeaguePredictor.TEAM_NAME_MAPPING.get(team_name, team_name)
    
    def load_models(self):
        """Cargar todos los modelos y datos guardados."""
        try:
            print("Cargando modelos de la Premier League...")
            
            # Cargar modelo XGBoost
            with open(os.path.join(self.models_path, 'xgb_production.pkl'), 'rb') as f:
                self.xgb_model = pickle.load(f)
            print("✓ Modelo XGBoost cargado")
            
            # Cargar modelos de goles
            with open(os.path.join(self.models_path, 'goals_models.pkl'), 'rb') as f:
                goals_data = pickle.load(f)
                self.home_goals_model = goals_data['home']
                self.away_goals_model = goals_data['away']
            print("✓ Modelos de goles cargados")
            
            # Cargar pipeline (encoder + scaler + feature_cols)
            with open(os.path.join(self.models_path, 'pipeline.pkl'), 'rb') as f:
                pipeline_data = pickle.load(f)
                self.label_encoder = pipeline_data['label_encoder']
                self.scaler = pipeline_data['scaler']
                self.feature_columns = pipeline_data['feature_cols']
            print(f"✓ Pipeline cargado ({len(self.feature_columns)} features)")
            
            # Cargar datos históricos completos desde CSV
            print("Cargando datos históricos...")
            df_raw = pd.read_csv(self.data_path)
            # Normalizar nombres de equipos en los datos raw
            df_raw['team_name'] = df_raw['team_name'].apply(self.normalize_team_name)
            self.matches_data = self._transform_to_match_format(df_raw)
            # Normalizar nombres en matches_data también
            self.matches_data['home_team'] = self.matches_data['home_team'].apply(self.normalize_team_name)
            self.matches_data['away_team'] = self.matches_data['away_team'].apply(self.normalize_team_name)
            self.teams_list = sorted(self.matches_data['home_team'].unique())
            print(f"✓ Datos históricos cargados: {len(self.matches_data)} partidos, {len(self.teams_list)} equipos")
            
            # Inicializar generadores de features
            self._initialize_feature_generators()
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _transform_to_match_format(self, df):
        """Transformar datos de formato largo a formato de partidos."""
        # Convertir fecha si existe
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
                    'home_team': home_row['team_name'],
                    'away_team': away_row['team_name'],
                    'home_goals': home_row['goals_for'],
                    'away_goals': away_row['goals_for'],
                    'result': home_row['match_result'],
                    'home_xg': home_row.get('ttl_xg', 1.5),
                    'away_xg': away_row.get('ttl_xg', 1.5),
                    'home_possession': home_row.get('avg_poss', 50),
                    'away_possession': away_row.get('avg_poss', 50),
                    'home_shots': home_row.get('ttl_sh', 12),
                    'away_shots': away_row.get('ttl_sh', 12),
                    'home_shots_on_target': home_row.get('ttl_sot', 5),
                    'away_shots_on_target': away_row.get('ttl_sot', 5),
                }
                matches.append(match)
        
        matches_df = pd.DataFrame(matches)
        
        # Si no hay date_game, crear secuencial
        if 'date_game' not in matches_df.columns or matches_df['date_game'].isna().all():
            matches_df['date_game'] = pd.date_range(start='2017-08-01', periods=len(matches_df), freq='3D')
        
        matches_df = matches_df.sort_values('date_game').reset_index(drop=True)
        
        return matches_df
    
    def _initialize_feature_generators(self):
        """Inicializar todos los generadores de features."""
        self.elo_system = EloFeatureGenerator(k_factor=20, home_advantage=100)
        self.momentum_gen = MomentumFeatureGenerator(windows=[3, 5, 10])
        self.h2h_gen = H2HFeatureGenerator(n_h2h=5)
        self.league_pos = LeaguePositionFeatures()
        self.rest_days = RestDaysFeatures()
        self.advanced_stats = AdvancedStatsFeatures()
        
        print("✓ Generadores de features inicializados")
    
    def _generate_features_for_match(self, home_team, away_team):
        """
        Generar TODAS las features (126) para un partido usando datos históricos actualizados.
        NO necesita match_id, solo los nombres de los equipos.
        """
        print(f"   Generando features para {home_team} vs {away_team}...")
        
        # Crear un registro temporal con los datos básicos del partido
        # Usamos un match_id temporal solo para el procesamiento interno
        temp_match_id = 'PRED_NEW'
        
        # Crear fila base con información del partido (formato match después de transformación)
        base_row = pd.DataFrame([{
            'match_id': temp_match_id,
            'date_game': pd.Timestamp.now(),
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': 0,
            'away_goals': 0,
            'result': 'D',  # Placeholder
            'home_xg': 1.5,
            'away_xg': 1.5,
            'home_possession': 50,
            'away_possession': 50,
            'home_shots': 12,
            'away_shots': 12,
            'home_shots_on_target': 5,
            'away_shots_on_target': 5
        }])
        
        # SOLUCIÓN: Usar índice numérico en lugar de match_id para merge
        print("   → Calculando features sobre datos históricos...")
        
        # Calcular todas las features
        elo_features = self.elo_system.calculate_elo_history(self.matches_data)
        momentum_features = self.momentum_gen.calculate_momentum_features(self.matches_data)
        h2h_features = self.h2h_gen.calculate_h2h_features(self.matches_data)
        league_features = self.league_pos.generate_features(self.matches_data)
        rest_features = self.rest_days.generate_features(self.matches_data)
        advanced_features = self.advanced_stats.generate_features(self.matches_data)
        
        # Combinar usando ÍNDICE en lugar de match_id
        print("   → Combinando features por índice...")
        all_features = self.matches_data.copy().reset_index(drop=True)
        
        for features_df, name in [
            (elo_features, 'ELO'),
            (momentum_features, 'Momentum'),
            (h2h_features, 'H2H'),
            (league_features, 'League'),
            (rest_features, 'Rest'),
            (advanced_features, 'Advanced')
        ]:
            if features_df is not None and len(features_df) > 0:
                features_df = features_df.copy().reset_index(drop=True)
                # Eliminar match_id y otras columnas que ya existen
                cols_to_drop = [col for col in features_df.columns if col in all_features.columns]
                features_df = features_df.drop(columns=cols_to_drop, errors='ignore')
                # Combinar por índice (concat horizontal)
                all_features = pd.concat([all_features, features_df], axis=1)
        
        # Eliminar columnas duplicadas que puedan quedar
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Extraer el ÚLTIMO partido de cada equipo como base para features
        print(f"   → Extrayendo features de últimos partidos...")
        
        # Buscar último partido del equipo local (como local o visitante)
        home_matches = all_features[
            (all_features['home_team'] == home_team) | 
            (all_features['away_team'] == home_team)
        ]
        if len(home_matches) > 0:
            home_last = home_matches.iloc[-1].copy()
        else:
            home_last = None
        
        # Buscar último partido del equipo visitante
        away_matches = all_features[
            (all_features['home_team'] == away_team) | 
            (all_features['away_team'] == away_team)
        ]
        if len(away_matches) > 0:
            away_last = away_matches.iloc[-1].copy()
        else:
            away_last = None
        
        # Crear fila de predicción combinando las features de ambos equipos
        if home_last is not None:
            pred_row = pd.DataFrame([home_last])
        elif away_last is not None:
            pred_row = pd.DataFrame([away_last])
        else:
            # Fallback: usar última fila disponible
            pred_row = all_features.tail(1).copy()
        
        # Asegurar que tiene todas las columnas necesarias
        missing_cols = []
        for col in self.feature_columns:
            if col not in pred_row.columns:
                pred_row[col] = 0
                missing_cols.append(col)
        
        if len(missing_cols) > 0:
            print(f"   ⚠️  {len(missing_cols)} columnas faltantes rellenadas con 0")
            print(f"   Ejemplos: {missing_cols[:5]}")
        
        # Seleccionar solo las columnas de features que usa el modelo
        X_pred = pred_row[self.feature_columns]
        
        # Rellenar NaN con 0 (común para H2H cuando no hay historial, etc.)
        nan_count = X_pred.isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠️  {nan_count} valores NaN rellenados con 0")
        X_pred = X_pred.fillna(0)
        
        print(f"   ✓ {len(X_pred.columns)} features calculadas")
        
        return X_pred
    
    def get_available_teams(self):
        """Obtener lista de equipos disponibles."""
        return self.teams_list or []
    
    def predict_match(self, home_team, away_team):
        """
        Predecir resultado y marcador de un partido.
        
        Args:
            home_team (str): Equipo local
            away_team (str): Equipo visitante
            
        Returns:
            dict: Predicción completa
        """
        if not self.models_loaded:
            return {'error': 'Modelos no cargados. Ejecuta load_models() primero.'}
        
        # Normalizar nombres de equipos ingresados
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)
        
        # Validar equipos
        if home_team not in self.teams_list or away_team not in self.teams_list:
            return {'error': f'Equipos no válidos. Usa get_available_teams()'}
        
        try:
            print(f"\n🔄 Generando predicción...")
            
            # Generar features dinámicamente para este partido específico
            X_pred = self._generate_features_for_match(home_team, away_team)
            
            # Normalizar con el scaler entrenado
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predicción de resultado
            result_proba = self.xgb_model.predict_proba(X_pred_scaled)[0]
            result_pred_encoded = self.xgb_model.predict(X_pred_scaled)[0]
            result_pred = self.label_encoder.inverse_transform([result_pred_encoded])[0]
            
            # Predicción de goles
            home_goals_pred = max(0, round(self.home_goals_model.predict(X_pred_scaled)[0]))
            away_goals_pred = max(0, round(self.away_goals_model.predict(X_pred_scaled)[0]))
            
            # Ajustar marcador para que coincida con resultado predicho
            # Soportar ambos formatos: H/D/A y Win/Draw/Loss
            if result_pred in ['H', 'Win'] and home_goals_pred <= away_goals_pred:
                home_goals_pred = away_goals_pred + 1
            elif result_pred in ['A', 'Loss'] and away_goals_pred <= home_goals_pred:
                away_goals_pred = home_goals_pred + 1
            elif result_pred in ['D', 'Draw']:
                avg_goals = (home_goals_pred + away_goals_pred) // 2
                home_goals_pred = away_goals_pred = max(1, avg_goals)
            
            # Confianza basada en probabilidad máxima
            max_proba = max(result_proba)
            if max_proba >= 0.65:
                confidence = "Alta ⭐⭐⭐⭐"
            elif max_proba >= 0.45:
                confidence = "Media ⭐⭐⭐"
            else:
                confidence = "Baja ⭐⭐"
            
            # Extraer features clave para mostrar al usuario
            features_info = {
                'home_elo': round(X_pred['home_elo'].values[0], 1) if 'home_elo' in X_pred.columns else 'N/A',
                'away_elo': round(X_pred['away_elo'].values[0], 1) if 'away_elo' in X_pred.columns else 'N/A',
                'elo_diff': round(X_pred['elo_diff'].values[0], 1) if 'elo_diff' in X_pred.columns else 'N/A',
                'home_form': round(X_pred['home_ppg_last_5'].values[0], 2) if 'home_ppg_last_5' in X_pred.columns else 'N/A',
                'away_form': round(X_pred['away_ppg_last_5'].values[0], 2) if 'away_ppg_last_5' in X_pred.columns else 'N/A',
            }
            
            # Mapear probabilidades según el orden real de las clases
            # label_encoder.classes_ = ['Draw' 'Loss' 'Win']
            # Necesitamos reordenar a [Home/Win, Draw, Away/Loss]
            class_to_index = {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)}
            
            proba_home = result_proba[class_to_index.get('Win', class_to_index.get('H', 0))]
            proba_draw = result_proba[class_to_index.get('Draw', class_to_index.get('D', 1))]
            proba_away = result_proba[class_to_index.get('Loss', class_to_index.get('A', 2))]
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_result': result_pred,
                'predicted_score': f"{home_goals_pred}-{away_goals_pred}",
                'probabilities': {
                    'Home': round(proba_home, 3),
                    'Draw': round(proba_draw, 3),
                    'Away': round(proba_away, 3)
                },
                'confidence': confidence,
                'features_info': features_info
            }
            
        except Exception as e:
            import traceback
            return {'error': f"Error en predicción: {str(e)}\n{traceback.format_exc()}"}


def get_result_text(result_code):
    """Convertir código de resultado a texto."""
    mapping = {
        # Formato antiguo (H/D/A)
        'H': 'Victoria Local',
        'D': 'Empate',
        'A': 'Victoria Visitante',
        # Formato nuevo (Win/Draw/Loss)
        'Win': 'Victoria Local',
        'Draw': 'Empate',
        'Loss': 'Victoria Visitante'
    }
    return mapping.get(result_code, 'Desconocido')


if __name__ == "__main__":
    print("=== PREDICTOR DE PARTIDOS PREMIER LEAGUE ===")
    print("(Usando modelos entrenados con features dinámicas)")
    
    predictor = PremierLeaguePredictor()
    
    if not predictor.load_models():
        print("Error: No se pudieron cargar los modelos.")
        print("Ejecuta el script de entrenamiento primero: python TrainPremierLeague.py")
        exit()
    
    teams = predictor.get_available_teams()
    print(f"\nEquipos disponibles ({len(teams)}):")
    for i, team in enumerate(teams):
        print(f"  {i+1:2d}. {team}")
    
    print("\n" + "="*50)
    
    while True:
        try:
            print("\nIngresa 'salir' para terminar")
            home_input = input("Equipo LOCAL: ").strip()
            
            if home_input.lower() == 'salir':
                break
            
            away_input = input("Equipo VISITANTE: ").strip()
            
            if away_input.lower() == 'salir':
                break
            
            # Hacer predicción
            prediction = predictor.predict_match(home_input, away_input)
            
            # Mostrar resultado
            if 'error' in prediction:
                print(f"\n❌ {prediction['error']}")
            else:
                print("\n" + "="*50)
                print(f"📊 PREDICCIÓN: {prediction['home_team']} vs {prediction['away_team']}")
                print("="*50)
                print(f"\n🏆 Resultado: {get_result_text(prediction['predicted_result'])}")
                print(f"⚽ Marcador: {prediction['predicted_score']}")
                print(f"\nProbabilidades:")
                print(f"  Local:     {prediction['probabilities']['Home']:.1%}")
                print(f"  Empate:    {prediction['probabilities']['Draw']:.1%}")
                print(f"  Visitante: {prediction['probabilities']['Away']:.1%}")
                print(f"\nConfianza: {prediction['confidence']}")
                
                # Mostrar features clave
                if 'features_info' in prediction:
                    info = prediction['features_info']
                    print(f"\n📈 Factores clave:")
                    print(f"  ELO {prediction['home_team']}: {info['home_elo']}")
                    print(f"  ELO {prediction['away_team']}: {info['away_elo']}")
                    print(f"  Diferencia ELO: {info['elo_diff']}")
                    print(f"  Forma {prediction['home_team']} (PPG últimos 5): {info['home_form']}")
                    print(f"  Forma {prediction['away_team']} (PPG últimos 5): {info['away_form']}")
                
                print("="*50)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\n¡Gracias por usar el predictor!")
