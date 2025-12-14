"""
base_predictor.py
=================

Clase base para predicción de partidos de fútbol.
Proporciona funcionalidad común para todas las ligas.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BaseFootballPredictor:
    """
    Clase base para predicción de partidos de fútbol.
    Proporciona funcionalidad común reutilizable para todas las ligas.
    """
    
    def __init__(self, league_name, data_path, models_path="models/improved_models/"):
        """
        Inicializar predictor.
        
        Args:
            league_name (str): Nombre de la liga (e.g., 'LaLiga', 'Bundesliga')
            data_path (str): Ruta al archivo CSV con datos limpios
            models_path (str): Ruta a los modelos entrenados
        """
        self.league_name = league_name
        self.data_path = data_path
        self.models_path = models_path
        self.models_loaded = False
        
        # Modelos y datos
        self.xgb_model = None
        self.home_goals_model = None
        self.away_goals_model = None
        self.label_encoder = None
        self.teams_list = None
        self.matches_data = None
        self.X_sample = None
    
    def load_models(self):
        """Cargar todos los modelos y datos guardados."""
        try:
            print(f"Cargando modelos para {self.league_name}...")
            
            # Cargar modelo XGBoost
            xgb_path = os.path.join(self.models_path, f'xgb_{self.league_name.lower()}.pkl')
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print(f"✓ Modelo XGBoost cargado desde {xgb_path}")
            else:
                print(f"⚠ Modelo XGBoost no encontrado: {xgb_path}")
                print("  Usando modelo genérico...")
                xgb_generic = os.path.join(self.models_path, 'xgb_production.pkl')
                if os.path.exists(xgb_generic):
                    with open(xgb_generic, 'rb') as f:
                        self.xgb_model = pickle.load(f)
            
            # Cargar modelos de goles
            goals_path = os.path.join(self.models_path, f'goals_models_{self.league_name.lower()}.pkl')
            if os.path.exists(goals_path):
                with open(goals_path, 'rb') as f:
                    goals_data = pickle.load(f)
                    self.home_goals_model = goals_data['home']
                    self.away_goals_model = goals_data['away']
                print(f"✓ Modelos de goles cargados desde {goals_path}")
            else:
                print(f"⚠ Modelos de goles no encontrados: {goals_path}")
                print("  Usando modelos genéricos...")
                goals_generic = os.path.join(self.models_path, 'goals_models.pkl')
                if os.path.exists(goals_generic):
                    with open(goals_generic, 'rb') as f:
                        goals_data = pickle.load(f)
                        self.home_goals_model = goals_data['home']
                        self.away_goals_model = goals_data['away']
            
            # Cargar encoder y datos
            encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("✓ Label encoder cargado")
            
            self.models_loaded = True
            print(f"\n✅ Modelos de {self.league_name} listos para predicción\n")
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            self.models_loaded = False
    
    def load_data(self):
        """Cargar datos de partidos."""
        try:
            print(f"Cargando datos de {self.league_name}...")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")
            
            self.matches_data = pd.read_csv(self.data_path)
            
            # Obtener lista de equipos únicos
            if 'team_name' in self.matches_data.columns:
                self.teams_list = sorted(self.matches_data['team_name'].unique())
            
            print(f"✓ Datos cargados: {len(self.matches_data)} registros")
            print(f"✓ Equipos únicos: {len(self.teams_list) if self.teams_list else 0}")
            print(f"✓ Columnas disponibles: {len(self.matches_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def prepare_features(self, home_team, away_team):
        """
        Preparar características para predicción.
        
        Args:
            home_team (str): Nombre del equipo local
            away_team (str): Nombre del equipo visitante
            
        Returns:
            pd.DataFrame: Features preparados
        """
        # Esta función debe ser implementada por las clases hijas
        # o puede usar características genéricas
        
        features = {
            'home_team': home_team,
            'away_team': away_team,
            # Agregar más características según disponibilidad
        }
        
        return pd.DataFrame([features])
    
    def predict_match(self, home_team, away_team):
        """
        Predecir resultado de un partido.
        
        Args:
            home_team (str): Equipo local
            away_team (str): Equipo visitante
            
        Returns:
            dict: Predicciones y probabilidades
        """
        if not self.models_loaded:
            print("⚠ Modelos no cargados. Ejecute load_models() primero.")
            return None
        
        try:
            # Preparar características
            X = self.prepare_features(home_team, away_team)
            
            # Predicción de resultado
            if self.xgb_model:
                result_proba = self.xgb_model.predict_proba(X)[0]
                result_pred = self.xgb_model.predict(X)[0]
            else:
                result_proba = [0.33, 0.33, 0.34]  # Default uniforme
                result_pred = 1  # Empate por defecto
            
            # Predicción de goles
            if self.home_goals_model and self.away_goals_model:
                home_goals = int(round(self.home_goals_model.predict(X)[0]))
                away_goals = int(round(self.away_goals_model.predict(X)[0]))
            else:
                home_goals = 1
                away_goals = 1
            
            # Decodificar resultado
            if self.label_encoder:
                result_label = self.label_encoder.inverse_transform([result_pred])[0]
            else:
                result_map = {0: 'Away', 1: 'Draw', 2: 'Home'}
                result_label = result_map.get(result_pred, 'Draw')
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_result': result_label,
                'home_win_prob': result_proba[2] if len(result_proba) > 2 else 0.33,
                'draw_prob': result_proba[1] if len(result_proba) > 1 else 0.33,
                'away_win_prob': result_proba[0],
                'predicted_home_goals': home_goals,
                'predicted_away_goals': away_goals,
                'confidence': max(result_proba)
            }
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    def predict_multiple_matches(self, matches_list):
        """
        Predecir múltiples partidos.
        
        Args:
            matches_list (list): Lista de tuplas (home_team, away_team)
            
        Returns:
            pd.DataFrame: Predicciones para todos los partidos
        """
        predictions = []
        
        print(f"\n{'='*60}")
        print(f"PREDICCIONES - {self.league_name}")
        print(f"{'='*60}\n")
        
        for i, (home, away) in enumerate(matches_list, 1):
            pred = self.predict_match(home, away)
            if pred:
                predictions.append(pred)
                
                # Mostrar predicción
                print(f"{i}. {home} vs {away}")
                print(f"   Resultado: {pred['predicted_result']}")
                print(f"   Marcador: {pred['predicted_home_goals']}-{pred['predicted_away_goals']}")
                print(f"   Probabilidades: H:{pred['home_win_prob']:.2%} | "
                      f"E:{pred['draw_prob']:.2%} | V:{pred['away_win_prob']:.2%}")
                print(f"   Confianza: {pred['confidence']:.2%}")
                print()
        
        return pd.DataFrame(predictions)
    
    def save_predictions(self, predictions_df, output_path=None):
        """Guardar predicciones a CSV."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"predictions/predictions_{self.league_name.lower()}_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        print(f"\n💾 Predicciones guardadas en: {output_path}")
        
        return output_path
    
    def get_recent_form(self, team_name, last_n=5):
        """Obtener forma reciente de un equipo."""
        if self.matches_data is None:
            print("⚠ Datos no cargados")
            return None
        
        team_matches = self.matches_data[
            self.matches_data['team_name'] == team_name
        ].sort_values('date_game' if 'date_game' in self.matches_data.columns else 'matchday', 
                      ascending=False).head(last_n)
        
        return team_matches
    
    def show_teams_list(self):
        """Mostrar lista de equipos disponibles."""
        if not self.teams_list:
            print("⚠ Lista de equipos no disponible")
            return
        
        print(f"\n📋 EQUIPOS EN {self.league_name}")
        print(f"{'='*50}")
        for i, team in enumerate(self.teams_list, 1):
            print(f"{i:2}. {team}")
        print()
