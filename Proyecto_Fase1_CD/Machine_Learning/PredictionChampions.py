"""
PredictionChampions.py
======================

Script para predicción de partidos de UEFA Champions League (inter-liga).
Combina datos históricos de las 5 grandes ligas europeas para predecir
enfrentamientos entre equipos de diferentes países.

Ligas incluidas:
- Bundesliga (Alemania)
- La Liga (España)
- Ligue 1 (Francia)
- Premier League (Inglaterra)
- Serie A (Italia)

Uso: python PredictionChampions.py
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

class ChampionsPredictor:
    """
    Predictor de partidos de Champions League usando datos de todas las ligas.
    Genera features dinámicas en tiempo real para predicciones inter-liga.
    """
    
    def __init__(self, 
                 models_base_path="models/",
                 data_base_path="../Data_Mining/"):
        """
        Inicializar predictor de Champions League.
        
        Args:
            models_base_path: Ruta base a carpeta de modelos
            data_base_path: Ruta base a carpeta de datos
        """
        self.models_base_path = models_base_path
        self.data_base_path = data_base_path
        self.models_loaded = False
        
        # Configuración de ligas
        self.leagues_config = {
            'Bundesliga': {
                'model_path': 'bundesliga/',
                'data_file': 'eda_outputsMatchesBundesliga/match_data_cleaned.csv',
                'country': 'Germany'
            },
            'La Liga': {
                'model_path': 'laliga/',
                'data_file': 'eda_outputsMatchesLaLiga/match_data_cleaned.csv',
                'country': 'Spain'
            },
            'Ligue 1': {
                'model_path': 'ligue1/',
                'data_file': 'eda_outputsMatchesLigue1/match_data_cleaned.csv',
                'country': 'France'
            },
            'Premier League': {
                'model_path': 'premierleague/',
                'data_file': 'eda_outputsMatchesPremierLeague/match_data_cleaned.csv',
                'country': 'England'
            },
            'Serie A': {
                'model_path': 'seriea/',
                'data_file': 'eda_outputsMatchesSerieA/match_data_cleaned.csv',
                'country': 'Italy'
            }
        }
        
        # Modelos (usaremos ensemble de todas las ligas)
        self.league_models = {}
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        
        # Datos combinados
        self.all_matches_data = None
        self.teams_by_league = {}
        self.all_teams = []
        
        # Generadores de features (a nivel europeo)
        self.elo_system = None
        self.momentum_gen = None
        self.h2h_gen = None
        self.position_gen = None
        self.rest_gen = None
        self.stats_gen = None
        
        # Cache de features pre-calculadas POR LIGA (OPTIMIZACIÓN)
        self.features_by_league = {}
    
    def load_models(self):
        """Cargar todos los modelos y datos de todas las ligas."""
        try:
            print("=" * 60)
            print("🏆 CARGANDO PREDICTOR UEFA CHAMPIONS LEAGUE 🏆")
            print("=" * 60)
            
            # 1. Cargar datos históricos de todas las ligas
            print("\n📊 Cargando datos históricos de las 5 grandes ligas...")
            all_league_data = []
            
            for league_name, config in self.leagues_config.items():
                try:
                    data_path = os.path.join(self.data_base_path, config['data_file'])
                    df_raw = pd.read_csv(data_path)
                    df_matches = self._transform_to_match_format(df_raw)
                    
                    # Agregar metadatos de liga
                    df_matches['league'] = league_name
                    df_matches['country'] = config['country']
                    
                    # Guardar equipos por liga
                    teams = sorted(df_matches['home_team'].unique())
                    self.teams_by_league[league_name] = teams
                    
                    all_league_data.append(df_matches)
                    print(f"  ✓ {league_name:20s}: {len(df_matches):4d} partidos, {len(teams):2d} equipos")
                    
                except Exception as e:
                    print(f"  ⚠️  {league_name}: Error cargando datos - {e}")
            
            # Combinar todos los datos
            self.all_matches_data = pd.concat(all_league_data, ignore_index=True)
            self.all_matches_data = self.all_matches_data.sort_values('date_game').reset_index(drop=True)
            self.all_teams = sorted(self.all_matches_data['home_team'].unique())
            
            print(f"\n  📦 Total combinado: {len(self.all_matches_data)} partidos, {len(self.all_teams)} equipos")
            
            # 2. Cargar modelos de cada liga
            print(f"\n🤖 Cargando modelos de predicción...")
            models_loaded = 0
            
            for league_name, config in self.leagues_config.items():
                try:
                    model_path = os.path.join(self.models_base_path, config['model_path'])
                    
                    # Cargar XGBoost
                    with open(os.path.join(model_path, 'xgb_production.pkl'), 'rb') as f:
                        xgb_model = pickle.load(f)
                    
                    # Cargar modelos de goles
                    with open(os.path.join(model_path, 'goals_models.pkl'), 'rb') as f:
                        goals_data = pickle.load(f)
                    
                    # Cargar pipeline
                    with open(os.path.join(model_path, 'pipeline.pkl'), 'rb') as f:
                        pipeline_data = pickle.load(f)
                    
                    self.league_models[league_name] = {
                        'xgb': xgb_model,
                        'home_goals': goals_data['home'],
                        'away_goals': goals_data['away'],
                        'encoder': pipeline_data['label_encoder'],
                        'scaler': pipeline_data['scaler'],
                        'features': pipeline_data['feature_cols']
                    }
                    
                    models_loaded += 1
                    print(f"  ✓ {league_name:20s}: Modelo cargado ({len(pipeline_data['feature_cols'])} features)")
                    
                except Exception as e:
                    print(f"  ⚠️  {league_name}: Error cargando modelo - {e}")
            
            if models_loaded == 0:
                print("\n❌ No se pudo cargar ningún modelo")
                return False
            
            # Usar el pipeline de la primera liga cargada como referencia
            first_league = list(self.league_models.keys())[0]
            self.label_encoder = self.league_models[first_league]['encoder']
            self.scaler = self.league_models[first_league]['scaler']
            self.feature_columns = self.league_models[first_league]['features']
            
            print(f"\n  📊 Pipeline de referencia: {first_league} ({len(self.feature_columns)} features)")
            
            # 3. Inicializar generadores de features con datos europeos
            print("\n⚙️  Inicializando generadores de features (nivel europeo)...")
            self._initialize_feature_generators()
            
            self.models_loaded = True
            print("\n" + "=" * 60)
            print("✅ Sistema Champions League listo para predicciones")
            print("   ⚡ Primera predicción por liga: ~30 seg (ELO + features)")
            print("   ⚡ Predicciones posteriores: instantáneas (usa cache)")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ Error crítico cargando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _transform_to_match_format(self, df):
        """Transformar datos de formato largo a formato de partidos."""
        if 'date_game' in df.columns:
            df['date_game'] = pd.to_datetime(df['date_game'])
        
        home_df = df[df['home_away'] == 'home'].copy()
        away_df = df[df['home_away'] == 'away'].copy()
        
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
        
        if 'date_game' not in matches_df.columns or matches_df['date_game'].isna().all():
            matches_df['date_game'] = pd.date_range(start='2017-08-01', periods=len(matches_df), freq='3D')
        
        matches_df = matches_df.sort_values('date_game').reset_index(drop=True)
        
        return matches_df
    
    def _initialize_feature_generators(self):
        """Inicializar todos los generadores de features."""
        self.elo_system = EloFeatureGenerator(k_factor=20, home_advantage=100)
        self.momentum_gen = MomentumFeatureGenerator(windows=[3, 5, 10])
        self.h2h_gen = H2HFeatureGenerator(n_h2h=5)
        self.position_gen = LeaguePositionFeatures()
        self.rest_gen = RestDaysFeatures()
        self.stats_gen = AdvancedStatsFeatures()
        
        print("  ✓ Generadores de features inicializados")
    
    def _calculate_features_for_league(self, league_name):
        """Calcular features para UNA liga específica (bajo demanda)."""
        if league_name in self.features_by_league:
            return  # Ya calculadas
        
        print(f"\n   📊 Calculando features para {league_name}...")
        
        league_data = self.all_matches_data[self.all_matches_data['league'] == league_name].copy()
        league_data = league_data.reset_index(drop=True)
        
        # Calcular ELO europeo si no existe
        if not hasattr(self, 'elo_features_global'):
            print(f"      → Calculando ELO europeo (base común)...")
            self.elo_features_global = self.elo_system.calculate_elo_history(self.all_matches_data)
        
        # Calcular features específicas de la liga
        momentum_features = self.momentum_gen.calculate_momentum_features(league_data)
        h2h_features = self.h2h_gen.calculate_h2h_features(league_data)
        league_features = self.position_gen.generate_features(league_data)
        rest_features = self.rest_gen.generate_features(league_data)
        advanced_features = self.stats_gen.generate_features(league_data)
        
        # Combinar features
        all_features = league_data.copy()
        
        # Agregar ELO (basado en índice original)
        league_elo = self.elo_features_global.loc[self.all_matches_data[self.all_matches_data['league'] == league_name].index]
        league_elo = league_elo.reset_index(drop=True)
        cols_to_drop = [col for col in league_elo.columns if col in all_features.columns]
        league_elo = league_elo.drop(columns=cols_to_drop, errors='ignore')
        all_features = pd.concat([all_features, league_elo], axis=1)
        
        for features_df in [momentum_features, h2h_features, league_features, rest_features, advanced_features]:
            if features_df is not None and len(features_df) > 0:
                features_df = features_df.copy().reset_index(drop=True)
                cols_to_drop = [col for col in features_df.columns if col in all_features.columns]
                features_df = features_df.drop(columns=cols_to_drop, errors='ignore')
                all_features = pd.concat([all_features, features_df], axis=1)
        
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        self.features_by_league[league_name] = all_features
        
        print(f"      ✅ Features de {league_name} listas")
    
    def _generate_features_for_match(self, home_team, away_team):
        """
        Generar TODAS las features (126) para un partido Champions League.
        OPTIMIZADO: Calcula features bajo demanda solo para ligas necesarias.
        """
        print(f"\n   🔍 Generando features para: {home_team} vs {away_team}")
        
        # Obtener ligas de los equipos
        home_league = self.get_team_league(home_team)
        away_league = self.get_team_league(away_team)
        
        # Calcular features de las ligas necesarias (bajo demanda)
        self._calculate_features_for_league(home_league)
        if away_league != home_league:
            self._calculate_features_for_league(away_league)
        
        # Usar features de la liga del equipo local
        all_features = self.features_by_league[home_league]
        
        # Extraer features del último partido de cada equipo
        print(f"   → Extrayendo últimos partidos...")
        
        home_matches = all_features[
            (all_features['home_team'] == home_team) | 
            (all_features['away_team'] == home_team)
        ]
        home_last = home_matches.iloc[-1].copy() if len(home_matches) > 0 else None
        
        away_matches = all_features[
            (all_features['home_team'] == away_team) | 
            (all_features['away_team'] == away_team)
        ]
        away_last = away_matches.iloc[-1].copy() if len(away_matches) > 0 else None
        
        # Crear fila de predicción
        if home_last is not None:
            pred_row = pd.DataFrame([home_last])
        elif away_last is not None:
            pred_row = pd.DataFrame([away_last])
        else:
            pred_row = all_features.tail(1).copy()
        
        # Asegurar todas las columnas
        missing_cols = []
        for col in self.feature_columns:
            if col not in pred_row.columns:
                pred_row[col] = 0
                missing_cols.append(col)
        
        if len(missing_cols) > 0:
            print(f"   ⚠️  {len(missing_cols)} columnas faltantes rellenadas con 0")
        
        X_pred = pred_row[self.feature_columns]
        
        nan_count = X_pred.isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠️  {nan_count} valores NaN rellenados con 0")
        X_pred = X_pred.fillna(0)
        
        print(f"   ✓ {len(X_pred.columns)} features calculadas")
        
        return X_pred
    
    def get_team_league(self, team_name):
        """Determinar la liga de un equipo."""
        for league, teams in self.teams_by_league.items():
            if team_name in teams:
                return league
        return None
    
    def get_available_teams(self, league=None):
        """
        Obtener lista de equipos disponibles.
        
        Args:
            league: Nombre de liga específica, o None para todos
        """
        if league and league in self.teams_by_league:
            return self.teams_by_league[league]
        return self.all_teams
    
    def predict_match(self, home_team, away_team, use_ensemble=True):
        """
        Predecir resultado de un partido Champions League.
        
        Args:
            home_team (str): Equipo local
            away_team (str): Equipo visitante
            use_ensemble (bool): Si True, usa promedio de modelos; si False, usa modelo de liga local
            
        Returns:
            dict: Predicción completa con información de ligas
        """
        if not self.models_loaded:
            return {'error': 'Modelos no cargados. Ejecuta load_models() primero.'}
        
        # Validar equipos
        if home_team not in self.all_teams:
            return {'error': f'Equipo local no encontrado: {home_team}'}
        if away_team not in self.all_teams:
            return {'error': f'Equipo visitante no encontrado: {away_team}'}
        
        try:
            # Identificar ligas
            home_league = self.get_team_league(home_team)
            away_league = self.get_team_league(away_team)
            
            print(f"\n{'='*60}")
            print(f"🏆 PREDICCIÓN CHAMPIONS LEAGUE")
            print(f"{'='*60}")
            print(f"🏠 {home_team:30s} ({home_league})")
            print(f"✈️  {away_team:30s} ({away_league})")
            print(f"{'='*60}")
            
            # Generar features
            X_pred = self._generate_features_for_match(home_team, away_team)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            if use_ensemble:
                # ENSEMBLE: Promediar solo las 2 ligas involucradas
                leagues_to_use = list(set([home_league, away_league]))
                print(f"\n   🤖 Usando predicción ENSEMBLE ({len(leagues_to_use)} liga(s): {', '.join(leagues_to_use)})...")
                
                all_probas = []
                all_home_goals = []
                all_away_goals = []
                
                for league_name in leagues_to_use:
                    models = self.league_models[league_name]
                    proba = models['xgb'].predict_proba(X_pred_scaled)[0]
                    home_g = models['home_goals'].predict(X_pred_scaled)[0]
                    away_g = models['away_goals'].predict(X_pred_scaled)[0]
                    
                    all_probas.append(proba)
                    all_home_goals.append(home_g)
                    all_away_goals.append(away_g)
                
                # Promediar
                result_proba = np.mean(all_probas, axis=0)
                home_goals_pred = max(0, round(np.mean(all_home_goals)))
                away_goals_pred = max(0, round(np.mean(all_away_goals)))
                
            else:
                # Usar modelo de la liga del equipo local
                print(f"\n   🤖 Usando modelo de: {home_league}...")
                models = self.league_models[home_league]
                
                result_proba = models['xgb'].predict_proba(X_pred_scaled)[0]
                home_goals_pred = max(0, round(models['home_goals'].predict(X_pred_scaled)[0]))
                away_goals_pred = max(0, round(models['away_goals'].predict(X_pred_scaled)[0]))
            
            # Determinar resultado predicho
            result_pred_idx = np.argmax(result_proba)
            result_pred = self.label_encoder.inverse_transform([result_pred_idx])[0]
            
            # Ajustar marcador
            if result_pred in ['H', 'Win'] and home_goals_pred <= away_goals_pred:
                home_goals_pred = away_goals_pred + 1
            elif result_pred in ['A', 'Loss'] and away_goals_pred <= home_goals_pred:
                away_goals_pred = home_goals_pred + 1
            elif result_pred in ['D', 'Draw']:
                avg_goals = (home_goals_pred + away_goals_pred) // 2
                home_goals_pred = away_goals_pred = max(1, avg_goals)
            
            # Confianza
            max_proba = max(result_proba)
            if max_proba >= 0.65:
                confidence = "Alta ⭐⭐⭐⭐"
            elif max_proba >= 0.45:
                confidence = "Media ⭐⭐⭐"
            else:
                confidence = "Baja ⭐⭐"
            
            # Features clave
            features_info = {
                'home_elo': round(X_pred['home_elo'].values[0], 1) if 'home_elo' in X_pred.columns else 'N/A',
                'away_elo': round(X_pred['away_elo'].values[0], 1) if 'away_elo' in X_pred.columns else 'N/A',
                'elo_diff': round(X_pred['elo_diff'].values[0], 1) if 'elo_diff' in X_pred.columns else 'N/A',
                'home_form': round(X_pred['home_ppg_last_5'].values[0], 2) if 'home_ppg_last_5' in X_pred.columns else 'N/A',
                'away_form': round(X_pred['away_ppg_last_5'].values[0], 2) if 'away_ppg_last_5' in X_pred.columns else 'N/A',
            }
            
            # Mapear probabilidades
            class_to_index = {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)}
            proba_home = result_proba[class_to_index.get('Win', class_to_index.get('H', 0))]
            proba_draw = result_proba[class_to_index.get('Draw', class_to_index.get('D', 1))]
            proba_away = result_proba[class_to_index.get('Loss', class_to_index.get('A', 2))]
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_league': home_league,
                'away_league': away_league,
                'is_inter_league': home_league != away_league,
                'predicted_result': result_pred,
                'predicted_score': f"{home_goals_pred}-{away_goals_pred}",
                'probabilities': {
                    'Home': round(proba_home, 3),
                    'Draw': round(proba_draw, 3),
                    'Away': round(proba_away, 3)
                },
                'confidence': confidence,
                'features_info': features_info,
                'method': 'Ensemble' if use_ensemble else home_league
            }
            
        except Exception as e:
            import traceback
            return {'error': f"Error en predicción: {str(e)}\n{traceback.format_exc()}"}


def get_result_text(result_code):
    """Convertir código de resultado a texto."""
    mapping = {
        'H': 'Victoria Local',
        'D': 'Empate',
        'A': 'Victoria Visitante',
        'Win': 'Victoria Local',
        'Draw': 'Empate',
        'Loss': 'Victoria Visitante'
    }
    return mapping.get(result_code, 'Desconocido')


if __name__ == "__main__":
    print("\n" + "🏆" * 30)
    print("   PREDICTOR UEFA CHAMPIONS LEAGUE")
    print("   Predicciones inter-liga entre las 5 grandes de Europa")
    print("🏆" * 30 + "\n")
    
    predictor = ChampionsPredictor()
    
    if not predictor.load_models():
        print("\n❌ Error: No se pudieron cargar los modelos.")
        print("Asegúrate de haber entrenado los modelos de cada liga primero.")
        exit()
    
    # Mostrar TODOS los equipos por liga
    print("\n📋 EQUIPOS DISPONIBLES POR LIGA:")
    print("=" * 60)
    for league, teams in predictor.teams_by_league.items():
        print(f"\n{league} ({len(teams)} equipos):")
        for i, team in enumerate(teams):
            print(f"  {i+1:2d}. {team}")
    
    print("\n" + "=" * 60)
    print("💡 TIP: Puedes predecir partidos entre equipos de CUALQUIER liga")
    print("=" * 60)
    
    while True:
        try:
            print("\n" + "-" * 60)
            print("Ingresa 'salir' para terminar")
            home_input = input("🏠 Equipo LOCAL: ").strip()
            
            if home_input.lower() == 'salir':
                break
            
            away_input = input("✈️  Equipo VISITANTE: ").strip()
            
            if away_input.lower() == 'salir':
                break
            
            # Opción de método
            method_input = input("Método (ensemble/local) [ensemble]: ").strip().lower()
            use_ensemble = method_input != 'local'
            
            # Hacer predicción
            prediction = predictor.predict_match(home_input, away_input, use_ensemble=use_ensemble)
            
            # Mostrar resultado
            if 'error' in prediction:
                print(f"\n❌ {prediction['error']}")
            else:
                print("\n" + "=" * 60)
                print(f"📊 PREDICCIÓN CHAMPIONS LEAGUE")
                print("=" * 60)
                print(f"🏠 {prediction['home_team']} ({prediction['home_league']})")
                print(f"✈️  {prediction['away_team']} ({prediction['away_league']})")
                
                if prediction['is_inter_league']:
                    print(f"\n🌍 Enfrentamiento INTER-LIGA")
                else:
                    print(f"\n🏴 Enfrentamiento dentro de {prediction['home_league']}")
                
                print(f"\n🏆 Resultado: {get_result_text(prediction['predicted_result'])}")
                print(f"⚽ Marcador: {prediction['predicted_score']}")
                print(f"\nProbabilidades:")
                print(f"  Local:     {prediction['probabilities']['Home']:.1%}")
                print(f"  Empate:    {prediction['probabilities']['Draw']:.1%}")
                print(f"  Visitante: {prediction['probabilities']['Away']:.1%}")
                print(f"\nConfianza: {prediction['confidence']}")
                print(f"Método: {prediction['method']}")
                
                if 'features_info' in prediction:
                    info = prediction['features_info']
                    print(f"\n📈 Factores clave:")
                    print(f"  ELO {prediction['home_team']}: {info['home_elo']}")
                    print(f"  ELO {prediction['away_team']}: {info['away_elo']}")
                    print(f"  Diferencia ELO: {info['elo_diff']}")
                    print(f"  Forma {prediction['home_team']} (PPG últimos 5): {info['home_form']}")
                    print(f"  Forma {prediction['away_team']} (PPG últimos 5): {info['away_form']}")
                
                print("=" * 60)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🏆 ¡Gracias por usar el predictor Champions League! 🏆\n")
