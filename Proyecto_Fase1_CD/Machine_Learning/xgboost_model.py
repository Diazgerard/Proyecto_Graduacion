"""
Modelo XGBoost Avanzado para Predicción de Partidos de Fútbol
=============================================================

Este módulo implementa:
1. Modelo XGBoost para predicción de resultados
2. Modelo XGBoost para predicción de goles
3. Optimización de hiperparámetros
4. Feature importance y análisis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class XGBoostFootballModel:
    """Modelo XGBoost avanzado para predicción de partidos de fútbol"""
    
    def __init__(self, task='classification', random_state=42):
        """
        Inicializar modelo XGBoost

        """
        self.task = task
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_importance = None
        self.is_fitted = False
        self.best_params = None
        
        # Configuraciones por defecto
        if task == 'classification':
            self.default_params = {
                'objective': 'multi:softprob',
                'num_class': 3,  # H, D, A
                'eval_metric': 'mlogloss',
                'random_state': random_state,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:  # regression
            self.default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'random_state': random_state,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
    
    def prepare_features(self, X, fit_scaler=False):
        """
        Preparar features para entrenamiento/predicción

        """
        # Copiar y limpiar datos
        X_clean = X.copy()
        
        # Manejar valores infinitos y NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Escalar features numéricas
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_clean)
            self.feature_names = list(X_clean.columns)
        else:
            X_scaled = self.scaler.transform(X_clean)
        
        return X_scaled
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, optimize_params=True, cv_folds=5, search_method='randomized'):

        print(f" Entrenando modelo XGBoost ({self.task})...")
        
        # Preparar features
        X_train_scaled = self.prepare_features(X_train, fit_scaler=True)
        
        # Preparar target
        if self.task == 'classification':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Optimizar hiperparámetros si se solicita
        if optimize_params:
            print(f" Optimizando hiperparámetros usando {search_method}...")
            self.best_params = self._optimize_hyperparameters_advanced(X_train_scaled, y_train_encoded, cv_folds, search_method)
        else:
            self.best_params = self.default_params.copy()
        
        # Entrenar modelo final
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**self.best_params)
        else:
            if len(y_train.shape) > 1 or (hasattr(y_train, 'ndim') and y_train.ndim > 1):
                # Multi-output regression (para predecir goles home y away)
                self.model = MultiOutputRegressor(xgb.XGBRegressor(**self.best_params))
            else:
                self.model = xgb.XGBRegressor(**self.best_params)
        
        # Datos de validación para early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.prepare_features(X_val, fit_scaler=False)
            if self.task == 'classification':
                y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val.values if hasattr(y_val, 'values') else y_val
            eval_set = [(X_val_scaled, y_val_encoded)]
        
        # Entrenar
        if eval_set and not isinstance(self.model, MultiOutputRegressor):
            self.model.fit(
                X_train_scaled, y_train_encoded,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, np.asarray(y_train_encoded))
        
        # Feature importance
        self._calculate_feature_importance()
        
        self.is_fitted = True
        print(f" Modelo XGBoost entrenado")
        print(f"   Features: {len(self.feature_names) if self.feature_names is not None else 0}")
        if not isinstance(self.model, MultiOutputRegressor) and hasattr(self.model, 'best_iteration'):
            print(f"   Best iteration: {self.model.best_iteration}")
        
    def _optimize_hyperparameters(self, X, y, cv_folds):
        """Optimizar hiperparámetros usando RandomizedSearchCV"""
        
        if self.task == 'classification':
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=self.random_state
            )
            scoring = 'neg_log_loss'
        else:
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state
            )
            scoring = 'neg_mean_absolute_error'
        
        # Espacio de búsqueda
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5],
            'reg_lambda': [0, 0.01, 0.05, 0.1, 0.5]
        }
        
        # Búsqueda randomizada
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=50,  # Número de combinaciones a probar
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        
        print(f"    Mejor score CV: {random_search.best_score_:.4f}")
        print(f"    Mejores parámetros encontrados:")
        for param, value in random_search.best_params_.items():
            print(f"      {param}: {value}")
        
        return random_search.best_params_
    
    def _optimize_hyperparameters_advanced(self, X, y, cv_folds, search_method='randomized'):
        """
        Optimización avanzada de hiperparámetros con múltiples métodos
        """
        from sklearn.model_selection import TimeSeriesSplit
        from scipy.stats import randint, uniform
        
        if self.task == 'classification':
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'neg_log_loss'
        else:
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'neg_mean_absolute_error'
        
        # Espacio de búsqueda expandido
        if search_method == 'randomized':
            param_distributions = {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 12),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1),
                'gamma': uniform(0, 0.5),
                'min_child_weight': randint(1, 10)
            }
            
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=100,  # Más iteraciones
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
        elif search_method == 'grid':
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [0, 0.01, 0.1]
            }
            
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
        else:  # 'bayesian' o método por defecto
            try:
                from skopt import BayesSearchCV
                from skopt.space import Real, Integer
                
                param_space = {
                    'n_estimators': Integer(100, 1000),
                    'max_depth': Integer(3, 12),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0, 1, prior='log-uniform'),
                    'reg_lambda': Real(0, 1, prior='log-uniform'),
                    'gamma': Real(0, 0.5),
                    'min_child_weight': Integer(1, 10)
                }
                
                search = BayesSearchCV(
                    base_model,
                    param_space,
                    n_iter=50,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
                
            except ImportError:
                print("    scikit-optimize no disponible, usando RandomizedSearchCV...")
                return self._optimize_hyperparameters(X, y, cv_folds)
        
        # Ejecutar búsqueda
        print(f"    Ejecutando {search_method} search con {cv_folds}-fold CV...")
        search.fit(X, y)
        
        # Resultados
        print(f"    Mejor score CV: {search.best_score_:.4f}")
        print(f"    Mejores parámetros encontrados:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
            
        # Guardar información de la búsqueda
        self.search_results_ = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_ if hasattr(search, 'cv_results_') else None
        }
        
        return search.best_params_
    
    def predict(self, X_test):

        if not self.is_fitted or self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        X_test_scaled = self.prepare_features(X_test, fit_scaler=False)
        predictions = self.model.predict(X_test_scaled)
        
        if self.task == 'classification':
            # Decodificar predicciones
            predictions = self.label_encoder.inverse_transform(np.asarray(predictions))
        
        return predictions
    
    def predict_proba(self, X_test):

        if self.task != 'classification':
            raise ValueError("predict_proba solo disponible para clasificación")
        
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
            
        if not isinstance(self.model, xgb.XGBClassifier):
            raise ValueError("predict_proba requiere un modelo de clasificación XGBoost")
        
        X_test_scaled = self.prepare_features(X_test, fit_scaler=False)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        return probabilities
    
    def _calculate_feature_importance(self):
        """Calcular importancia de features"""
        if self.model is None:
            return
            
        try:
            # Simplificar: solo usar feature_importances_ directo del modelo
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_ # type: ignore
            else:
                # Para MultiOutputRegressor, no calcular importancias complejas
                print("No se pueden calcular importancias para este tipo de modelo")
                return
        except Exception as e:
            print(f"Error calculando importancias: {e}")
            return
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8), save_path=None):
        if self.feature_importance is None:
            print("Feature importance no calculada")
            return
        
        plt.figure(figsize=figsize)
        
        top_features = self.feature_importance.head(top_n)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'].tolist())
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Features - XGBoost ({self.task.title()})', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Agregar valores
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(top_features['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance guardado en: {save_path}")
        
        plt.show()
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        Evaluación completa del modelo con métricas avanzadas
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        
        predictions = self.predict(X_test)
        
        if self.task == 'classification':
            probabilities = self.predict_proba(X_test)
            
            # Métricas básicas
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            # Métricas por clase
            precision_per_class = precision_score(y_test, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
            recall_per_class = recall_score(y_test, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
            f1_per_class = f1_score(y_test, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, predictions, labels=['H', 'D', 'A']) # type: ignore
            
            # Preparar para log_loss y brier score
            y_test_encoded = []
            for result in y_test:
                if result == 'H':
                    y_test_encoded.append([1, 0, 0])
                elif result == 'D':
                    y_test_encoded.append([0, 1, 0])
                else:
                    y_test_encoded.append([0, 0, 1])
            
            y_test_encoded = np.array(y_test_encoded)
            
            try:
                logloss = log_loss(y_test_encoded, probabilities)
                # Brier Score (para calibración)
                brier_scores = []
                for i, clase in enumerate(['H', 'D', 'A']):
                    y_binary = (y_test == clase).astype(int)
                    brier_scores.append(brier_score_loss(y_binary, probabilities[:, i]))
                brier_avg = np.mean(brier_scores)
            except:
                logloss = 999.0
                brier_avg = 999.0
            
            # Feature importance si está disponible
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = list(zip(self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))], # type: ignore
                                            self.model.feature_importances_))# type: ignore
                feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
                self.feature_importance_ = feature_importance
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'precision_per_class': dict(zip(['H', 'D', 'A'], precision_per_class)), # type: ignore
                'recall_per_class': dict(zip(['H', 'D', 'A'], recall_per_class)),# type: ignore
                'f1_per_class': dict(zip(['H', 'D', 'A'], f1_per_class)),# type: ignore
                'confusion_matrix': cm,
                'log_loss': logloss,
                'brier_score': brier_avg,
                'brier_scores_per_class': dict(zip(['H', 'D', 'A'], brier_scores)) if 'brier_scores' in locals() else {},
                'predictions': predictions,
                'probabilities': probabilities,
                'feature_importance': feature_importance
            }
            
            if detailed:
                print(f" EVALUACIÓN COMPLETA XGBOOST ({self.task.upper()}):")
                print(f"    Accuracy:  {accuracy:.3f}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall:    {recall:.3f}")
                print(f"    F1-Score:  {f1:.3f}")
                print(f"    Log Loss:  {logloss:.3f}")
                print(f"    Brier Score: {brier_avg:.3f}")
                
                print(f"\n   Métricas por clase:")
                for clase in ['H', 'D', 'A']:
                    desc = {'H': 'Local', 'D': 'Empate', 'A': 'Visitante'}[clase]
                    print(f"     {clase} ({desc:<9}): P={metrics['precision_per_class'][clase]:.3f}, R={metrics['recall_per_class'][clase]:.3f}, F1={metrics['f1_per_class'][clase]:.3f}")
                
                print(f"\n   Matriz de Confusión:")
                print(f"                Predicho")
                print(f"              H    D    A")
                print(f"    Real  H  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
                print(f"          D  {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
                print(f"          A  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
                
                print("\n Reporte de Clasificación:")
                print(classification_report(y_test, predictions, target_names=['Away', 'Draw', 'Home']))
        
        else:  # regression
            if len(y_test.shape) > 1:  # Multi-output
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
            else:
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions
            }
            
            if detailed:
                print(f" EVALUACIÓN XGBOOST ({self.task.upper()}):")
                print(f"    MAE: {mae:.3f}")
                print(f"    RMSE: {rmse:.3f}")
                print(f"    R²: {r2:.3f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Guardar modelo entrenado"""
        if not self.is_fitted:
            print(" Modelo no entrenado")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if self.task == 'classification' else None,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'task': self.task
        }
        
        joblib.dump(model_data, filepath)
        print(f" Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Cargar modelo guardado"""
        model_data = joblib.load(filepath)
        
        # Crear instancia
        instance = cls(task=model_data['task'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']
        instance.best_params = model_data['best_params']
        instance.is_fitted = True
        
        print(f" Modelo cargado desde: {filepath}")
        return instance


class XGBoostEnsemble:
    """Ensemble de modelos XGBoost para predicción completa"""
    
    def __init__(self, random_state=42):
        """Inicializar ensemble"""
        self.result_model = XGBoostFootballModel('classification', random_state)
        self.goals_model = XGBoostFootballModel('regression', random_state)
        self.is_fitted = False
    
    def fit(self, X_train, y_results_train, y_goals_train, X_val=None, y_results_val=None, y_goals_val=None):

        print(" ENTRENANDO ENSEMBLE XGBOOST")
        print("="*35)
        
        # Entrenar modelo de resultados
        self.result_model.fit(X_train, y_results_train, X_val, y_results_val)
        
        print()
        
        # Entrenar modelo de goles
        self.goals_model.fit(X_train, y_goals_train, X_val, y_goals_val)
        
        self.is_fitted = True
        print("\nEnsemble XGBoost entrenado completamente")
    
    def predict_complete(self, X_test):

        if not self.is_fitted:
            raise ValueError("Ensemble no entrenado")
        
        result_predictions = self.result_model.predict(X_test)
        result_probabilities = self.result_model.predict_proba(X_test)
        goals_predictions = self.goals_model.predict(X_test)
        
        # Simplificar manejo de predicciones
        try:
            # Convertir a numpy array si es necesario
            if hasattr(goals_predictions, 'toarray'):
                goals_predictions = goals_predictions.toarray() # type: ignore
            
            # Extraer goles home y away si es posible
            home_goals = None
            away_goals = None
            if hasattr(goals_predictions, 'shape') and len(goals_predictions.shape) > 1 and goals_predictions.shape[1] >= 2:
                home_goals = goals_predictions[:, 0] # type: ignore
                away_goals = goals_predictions[:, 1] # type: ignore
            elif hasattr(goals_predictions, '__len__'):
                home_goals = goals_predictions
                
        except Exception:
            home_goals = goals_predictions
            away_goals = None
        
        return {
            'results': result_predictions,
            'result_probabilities': result_probabilities,
            'goals': goals_predictions,
            'home_goals': home_goals,
            'away_goals': away_goals
        }
    
    def evaluate_complete(self, X_test, y_results_test, y_goals_test):
        """Evaluación completa del ensemble"""
        print(" EVALUACIÓN COMPLETA ENSEMBLE XGBOOST")
        print("="*45)
        
        # Evaluar resultados
        result_metrics = self.result_model.evaluate(X_test, y_results_test)
        
        print()
        
        # Evaluar goles
        goals_metrics = self.goals_model.evaluate(X_test, y_goals_test)
        
        return {
            'results': result_metrics,
            'goals': goals_metrics
        }


if __name__ == "__main__":
    print("Modelos XGBoost implementados:")
    print("   • XGBoostFootballModel - Modelo individual")
    print("   • XGBoostEnsemble - Ensemble completo")
    print("   • Optimización automática de hiperparámetros")
    print("   • Feature importance y evaluación detallada")