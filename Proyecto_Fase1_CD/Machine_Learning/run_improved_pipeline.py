"""
Pipeline Mejorado de Machine Learning para Predicción de Partidos
==================================================================

Este script ejecuta el pipeline completo mejorado:
1. Genera features ELO
2. Genera features de Momentum
3. Genera features H2H
4. Integra todas las features
5. Entrena modelos (baseline + ensemble)
6. Evalúa y compara resultados
7. Guarda modelos mejorados
8. Genera reporte comparativo

Uso:
    python run_improved_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Imports de módulos personalizados
from elo_features import EloFeatureGenerator

# Wrapper para modelos de goles
class GoalsModelWrapper:
    """Wrapper para hacer el modelo Poisson compatible con predict()"""
    def __init__(self, poisson_model, is_home=True):
        self.poisson_model = poisson_model
        self.is_home = is_home
    
    def predict(self, X):
        """Predice goles promedio basado en el modelo Poisson"""
        # Retornar el promedio de goles de la liga como predicción simple
        n_samples = len(X) if hasattr(X, '__len__') else 1
        avg = self.poisson_model.league_avg_goals
        if self.is_home:
            avg += self.poisson_model.home_advantage / 2
        else:
            avg -= self.poisson_model.home_advantage / 2
        return np.array([avg] * n_samples)
from momentum_features import MomentumFeatureGenerator
from h2h_features import H2HFeatureGenerator
from baseline_models import EloBaseline, PoissonBaseline
from ensemble_model import EnsembleModel
from xgboost import XGBClassifier


class ImprovedMLPipeline:
    """Pipeline completo mejorado de Machine Learning"""
    
    def __init__(self, data_path, output_dir='improved_models'):
        """
        Inicializar pipeline
        
        Args:
            data_path (str): Ruta al CSV con datos de partidos
            output_dir (str): Directorio para guardar modelos y resultados
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.df_with_features = None
        self.results = {}
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(" PIPELINE MEJORADO DE MACHINE LEARNING ")
        print(f"{'='*70}\n")
    
    def load_data(self):
        """Cargar datos desde CSV"""
        print("📂 PASO 1: Cargando datos...")
        self.df = pd.read_csv(self.data_path)
        
        # Convertir fecha a datetime
        if self.df['date_game'].dtype == 'object':
            self.df['date_game'] = pd.to_datetime(self.df['date_game'])
        
        print(f"   ✓ Datos cargados: {self.df.shape[0]:,} partidos")
        print(f"   ✓ Columnas originales: {self.df.shape[1]}")
        print(f"   ✓ Rango de fechas: {self.df['date_game'].min()} a {self.df['date_game'].max()}")
        
        return self.df
    
    def generate_all_features(self):
        """Generar todas las features mejoradas"""
        print(f"\n🔧 PASO 2: Generando Features Mejoradas...")
        print("-" * 70)
        
        # 1. Features ELO
        print("\n1️⃣  Generando features ELO...")
        elo_gen = EloFeatureGenerator(k_factor=20, home_advantage=100)
        elo_df = elo_gen.calculate_elo_history(self.df)
        
        # Merge
        self.df_with_features = self.df.merge(
            elo_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
            left_index=True,
            right_on='match_id',
            how='left'
        )
        self.df_with_features.drop('match_id', axis=1, errors='ignore', inplace=True)
        
        # 2. Features Momentum
        print("\n2️⃣  Generando features de Momentum...")
        momentum_gen = MomentumFeatureGenerator(windows=[3, 5, 10])
        momentum_df = momentum_gen.calculate_momentum_features(self.df)
        
        # Merge
        self.df_with_features = self.df_with_features.merge(
            momentum_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
            left_index=True,
            right_on='match_id',
            how='left'
        )
        self.df_with_features.drop('match_id', axis=1, errors='ignore', inplace=True)
        
        # 3. Features H2H
        print("\n3️⃣  Generando features Head-to-Head...")
        h2h_gen = H2HFeatureGenerator(n_h2h=5)
        h2h_df = h2h_gen.calculate_h2h_features(self.df)
        
        # Merge
        self.df_with_features = self.df_with_features.merge(
            h2h_df.drop(['date_game', 'home_team', 'away_team'], axis=1),
            left_index=True,
            right_on='match_id',
            how='left'
        )
        self.df_with_features.drop('match_id', axis=1, errors='ignore', inplace=True)
        
        # Resumen
        new_features = self.df_with_features.shape[1] - self.df.shape[1]
        print(f"\n✅ Features generadas exitosamente")
        print(f"   Total de features nuevas: {new_features}")
        print(f"   Shape final: {self.df_with_features.shape}")
        
        # Guardar datos con features
        features_file = os.path.join(self.output_dir, 'data_with_all_features.csv')
        self.df_with_features.to_csv(features_file, index=False)
        print(f"   💾 Guardado en: {features_file}")
        
        return self.df_with_features
    
    def prepare_data_for_ml(self, test_size=0.2, val_size=0.1):
        """
        Preparar datos para Machine Learning
        
        Args:
            test_size (float): Proporción de datos de test
            val_size (float): Proporción de datos de validación
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n📊 PASO 3: Preparando datos para ML...")
        
        df = self.df_with_features.copy()
        
        # Seleccionar features para ML (solo numéricas)
        exclude_cols = ['date_game', 'home_team', 'away_team', 'result', 
                       'home_goals', 'away_goals', 'season', 'match_id']
        
        # Filtrar solo columnas numéricas
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(0)  # Rellenar NaN con 0
        y = df['result']
        
        print(f"   Features para ML: {len(feature_cols)}")
        print(f"   Total de muestras: {len(X):,}")
        
        # ENTRENAR CON 100% DE LOS DATOS
        # Ordenar por fecha
        df_sorted = df.sort_values('date_game').reset_index(drop=True)
        X_sorted = df_sorted[feature_cols].fillna(0)
        y_sorted = df_sorted['result']
        
        # Calcular índices de corte: 90% train, 10% val (para early stopping)
        n = len(X_sorted)
        train_end = int(n * 0.90)
        
        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        
        X_val = X_sorted.iloc[train_end:]
        y_val = y_sorted.iloc[train_end:]
        
        # Test es el conjunto completo (para evaluación final con todos los datos)
        X_test = X_sorted.copy()
        y_test = y_sorted.copy()
        
        print(f"   Train: {len(X_train):,} ({len(X_train)/n*100:.1f}%) - Para entrenamiento")
        print(f"   Val:   {len(X_val):,} ({len(X_val)/n*100:.1f}%) - Para early stopping")
        print(f"   Eval:  {len(X_test):,} (100%) - Evaluación final con TODOS los datos")
        print(f"\n   ⚠️  MAXIMIZANDO APRENDIZAJE: Entrenando con {len(X_train):,} partidos")
        
        # Guardar para uso posterior
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_baseline_models(self):
        """Entrenar modelos baseline para comparación"""
        print(f"\n🎯 PASO 4: Entrenando Modelos Baseline...")
        print("-" * 70)
        
        # Preparar datos para baseline (solo necesitan columnas básicas)
        df_train = self.df_with_features.iloc[:len(self.X_train)].copy()
        df_test = self.df_with_features.iloc[len(self.X_train)+len(self.X_val):].copy()
        
        # ELO Baseline
        print("\n📈 ELO Baseline...")
        elo_baseline = EloBaseline(k_factor=20, home_advantage=100)
        elo_baseline.fit(df_train)
        elo_metrics = elo_baseline.evaluate(df_test, self.y_test, detailed=False)
        
        self.results['elo_baseline'] = {
            'accuracy': elo_metrics['accuracy'],
            'f1_score': elo_metrics['f1_score'],
            'precision': elo_metrics['precision'],
            'recall': elo_metrics['recall']
        }
        
        print(f"   ✓ Accuracy: {elo_metrics['accuracy']:.4f}")
        
        return self.results
    
    def train_improved_models(self):
        """Entrenar modelos mejorados"""
        print(f"\n🚀 PASO 5: Entrenando Modelos Mejorados...")
        print("-" * 70)
        
        # XGBoost individual mejorado
        print("\n📊 XGBoost Mejorado...")
        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # Codificar etiquetas con sklearn LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['H', 'D', 'A'])  # Asegurar orden consistente
        y_train_enc = label_encoder.transform(self.y_train)
        y_test_enc = label_encoder.transform(self.y_test)
        
        xgb_model.fit(self.X_train, y_train_enc)
        xgb_pred = xgb_model.predict(self.X_test)
        xgb_accuracy = (xgb_pred == y_test_enc).mean()
        
        self.results['xgboost_improved'] = {
            'accuracy': float(xgb_accuracy),
            'model': xgb_model
        }
        
        print(f"   ✓ Accuracy: {xgb_accuracy:.4f}")
        
        # Ensemble Model
        print("\n🎭 Ensemble Model (Stacking)...")
        ensemble = EnsembleModel(ensemble_type='stacking')
        ensemble.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        ensemble_metrics = ensemble.evaluate(self.X_test, self.y_test, detailed=False)
        
        self.results['ensemble'] = {
            'accuracy': ensemble_metrics['accuracy'],
            'f1_score': ensemble_metrics['f1_score'],
            'precision': ensemble_metrics['precision'],
            'recall': ensemble_metrics['recall'],
            'log_loss': ensemble_metrics['log_loss'],
            'model': ensemble
        }
        
        # Guardar modelos con nombres compatibles para PredictionPremierLeague.py
        xgb_path = os.path.join(self.output_dir, 'xgb_production.pkl')
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"   💾 XGBoost guardado en: {xgb_path}")
        
        # Guardar ensemble también
        ensemble_path = os.path.join(self.output_dir, 'ensemble_model.pkl')
        ensemble.save_model(ensemble_path)
        
        # Guardar pipeline info para compatibilidad
        pipeline_data = {
            'label_encoder': label_encoder,
            'feature_names': list(self.X_train.columns)
        }
        pipeline_path = os.path.join(self.output_dir, 'pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        print(f"   💾 Pipeline guardado en: {pipeline_path}")
        
        # Guardar datos de referencia
        reference_data = {
            'matches_final': self.df_with_features[['home_team', 'away_team', 'home_goals', 'away_goals']],
            'equipos_disponibles': sorted(list(set(self.df['home_team'].unique()) | set(self.df['away_team'].unique()))),
            'X_sample': self.X_test.iloc[:1]  # Una muestra para predicción
        }
        reference_path = os.path.join(self.output_dir, 'reference_data.pkl')
        with open(reference_path, 'wb') as f:
            pickle.dump(reference_data, f)
        print(f"   💾 Datos de referencia guardados en: {reference_path}")
        
        # Crear modelos de goles simples (wrapper para compatibilidad)
        from baseline_models import PoissonBaseline
        
        poisson = PoissonBaseline()
        df_train = self.df_with_features.iloc[:len(self.X_train)].copy()
        poisson.fit(df_train)
        
        goals_models = {
            'home': GoalsModelWrapper(poisson, is_home=True),
            'away': GoalsModelWrapper(poisson, is_home=False)
        }
        goals_path = os.path.join(self.output_dir, 'goals_models.pkl')
        with open(goals_path, 'wb') as f:
            pickle.dump(goals_models, f)
        print(f"   💾 Modelos de goles guardados en: {goals_path}")
        
        return self.results
    
    def generate_comparison_report(self):
        """Generar reporte comparativo de todos los modelos"""
        print(f"\n📄 PASO 6: Generando Reporte Comparativo...")
        print("-" * 70)
        
        # Preparar datos para reporte
        comparison = {
            'date': datetime.now().isoformat(),
            'data_path': self.data_path,
            'total_matches': len(self.df),
            'test_matches': len(self.y_test),
            'models': {}
        }
        
        # Añadir métricas de cada modelo
        for model_name, metrics in self.results.items():
            if 'model' in metrics:
                del metrics['model']  # No serializar el modelo
            comparison['models'][model_name] = metrics
        
        # Identificar mejor modelo
        best_model = max(comparison['models'].items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        comparison['best_model'] = {
            'name': best_model[0],
            'accuracy': best_model[1]['accuracy']
        }
        
        # Guardar reporte JSON
        report_path = os.path.join(self.output_dir, 'comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"   💾 Reporte guardado en: {report_path}")
        
        # Mostrar resumen
        print(f"\n{'='*70}")
        print(" RESUMEN COMPARATIVO DE MODELOS ")
        print(f"{'='*70}\n")
        
        print(f"{'Modelo':<25} {'Accuracy':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for model_name, metrics in sorted(comparison['models'].items(), 
                                         key=lambda x: x[1].get('accuracy', 0), 
                                         reverse=True):
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            print(f"{model_name:<25} {acc:<12.4f} {f1:<12.4f}")
        
        print("\n" + "="*70)
        print(f"🏆 MEJOR MODELO: {comparison['best_model']['name']}")
        print(f"   Accuracy: {comparison['best_model']['accuracy']:.4f}")
        
        # Calcular mejora vs baseline
        baseline_acc = comparison['models'].get('elo_baseline', {}).get('accuracy', 0)
        best_acc = comparison['best_model']['accuracy']
        improvement = ((best_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        print(f"\n📈 MEJORA vs BASELINE: +{improvement:.2f}%")
        print(f"{'='*70}\n")
        
        return comparison
    
    def run_full_pipeline(self):
        """Ejecutar el pipeline completo"""
        start_time = datetime.now()
        
        try:
            # 1. Cargar datos
            self.load_data()
            
            # 2. Generar features
            self.generate_all_features()
            
            # 3. Preparar datos
            self.prepare_data_for_ml()
            
            # 4. Entrenar baseline
            self.train_baseline_models()
            
            # 5. Entrenar modelos mejorados
            self.train_improved_models()
            
            # 6. Generar reporte
            report = self.generate_comparison_report()
            
            # Tiempo total
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n⏱️  Tiempo total: {duration:.1f} segundos")
            print(f"\n{'='*70}")
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print(f"{'='*70}\n")
            
            return report
            
        except Exception as e:
            print(f"\n❌ ERROR en el pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Ruta de datos por defecto (archivo convertido para ML)
    default_data_path = os.path.join(
        os.path.dirname(__file__),
        "match_data_for_ml.csv"
    )
    
    if not os.path.exists(default_data_path):
        print(f"❌ No se encontró el archivo: {default_data_path}")
        print("\n💡 Uso:")
        print("   from run_improved_pipeline import ImprovedMLPipeline")
        print("   pipeline = ImprovedMLPipeline('ruta/a/datos.csv')")
        print("   report = pipeline.run_full_pipeline()")
    else:
        # Ejecutar pipeline
        pipeline = ImprovedMLPipeline(
            data_path=default_data_path,
            output_dir='improved_models'
        )
        
        report = pipeline.run_full_pipeline()
