"""
Modelo Ensemble Avanzado para Predicción de Partidos
=====================================================

Este módulo implementa un ensemble de múltiples algoritmos de ML:
- XGBoost
- LightGBM
- CatBoost

Usando Stacking con Logistic Regression como meta-learner.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, log_loss, roc_auc_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("WARNING: LightGBM no disponible. Instalar con: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: CatBoost no disponible. Instalar con: pip install catboost")
    CATBOOST_AVAILABLE = False


class EnsembleModel:
    """Modelo Ensemble con múltiples algoritmos"""
    
    def __init__(self, ensemble_type='stacking', voting='soft'):
        """
        Inicializar modelo ensemble
        
        Args:
            ensemble_type (str): Tipo de ensemble ('stacking' o 'voting')
            voting (str): Tipo de voting ('soft' o 'hard') - solo para VotingClassifier
        """
        self.ensemble_type = ensemble_type
        self.voting = voting
        self.model = None
        self.label_encoder = {'H': 0, 'D': 1, 'A': 2}
        self.label_decoder = {0: 'H', 1: 'D', 2: 'A'}
        self.feature_names = None
        self.is_fitted = False
        
    def create_base_models(self):
        """
        Crear modelos base para el ensemble
        
        Returns:
            list: Lista de tuplas (nombre, modelo)
        """
        base_models = []
        
        # XGBoost (siempre disponible)
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        base_models.append(('xgb', xgb))
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            base_models.append(('lgbm', lgbm))
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            catboost = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=False,
                loss_function='MultiClass'
            )
            base_models.append(('catboost', catboost))
        
        return base_models
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrenar el modelo ensemble
        
        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Etiquetas de entrenamiento
            X_val (pd.DataFrame, optional): Features de validación
            y_val (pd.Series, optional): Etiquetas de validación
        """
        print(f"\n{'='*60}")
        print(f"ENTRENANDO MODELO ENSEMBLE ({self.ensemble_type.upper()})")
        print(f"{'='*60}\n")
        
        # Guardar nombres de features
        self.feature_names = list(X_train.columns)
        
        # Codificar etiquetas
        y_train_encoded = y_train.map(self.label_encoder)
        
        # Crear modelos base
        base_models = self.create_base_models()
        print(f"📊 Modelos base:")
        for name, _ in base_models:
            print(f"   • {name}")
        
        print(f"\n🔧 Configuración:")
        print(f"   Tipo: {self.ensemble_type}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Crear ensemble según tipo
        if self.ensemble_type == 'stacking':
            # Meta-learner
            meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
            
            self.model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1,
                verbose=0
            )
            print(f"   Meta-learner: Logistic Regression")
            
        elif self.ensemble_type == 'voting':
            self.model = VotingClassifier(
                estimators=base_models,
                voting=self.voting,
                n_jobs=-1
            )
            print(f"   Voting: {self.voting}")
        
        # Entrenar
        print(f"\n⏳ Entrenando ensemble...")
        self.model.fit(X_train, y_train_encoded)
        
        self.is_fitted = True
        
        # Evaluar en validación si está disponible
        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.map(self.label_encoder)
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val_encoded, val_pred)
            print(f"\n✅ Modelo entrenado exitosamente")
            print(f"   Accuracy en validación: {val_accuracy:.4f}")
        else:
            print(f"\n✅ Modelo entrenado exitosamente")
    
    def predict(self, X):
        """
        Predecir clases
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            np.array: Predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        predictions_encoded = self.model.predict(X)
        predictions = [self.label_decoder[pred] for pred in predictions_encoded]
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predecir probabilidades
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            np.array: Probabilidades [H, D, A]
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        Evaluar el modelo con métricas completas
        
        Args:
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Etiquetas de test
            detailed (bool): Mostrar reporte detallado
        
        Returns:
            dict: Diccionario con métricas
        """
        # Codificar etiquetas
        y_test_encoded = y_test.map(self.label_encoder)
        
        # Predicciones
        predictions_encoded = self.model.predict(X_test)
        predictions = [self.label_decoder[pred] for pred in predictions_encoded]
        probabilities = self.model.predict_proba(X_test)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        precision = precision_score(y_test_encoded, predictions_encoded, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, predictions_encoded, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, predictions_encoded, average='weighted', zero_division=0)
        
        # Log loss
        logloss = log_loss(y_test_encoded, probabilities)
        
        # Métricas por clase
        precision_per_class = precision_score(y_test_encoded, predictions_encoded, average=None, zero_division=0)
        recall_per_class = recall_score(y_test_encoded, predictions_encoded, average=None, zero_division=0)
        f1_per_class = f1_score(y_test_encoded, predictions_encoded, average=None, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_encoded, predictions_encoded)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'precision_per_class': dict(zip(['H', 'D', 'A'], precision_per_class)),
            'recall_per_class': dict(zip(['H', 'D', 'A'], recall_per_class)),
            'f1_per_class': dict(zip(['H', 'D', 'A'], f1_per_class)),
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        if detailed:
            print(f"\n{'='*60}")
            print(f"EVALUACIÓN MODELO ENSEMBLE ({self.ensemble_type.upper()})")
            print(f"{'='*60}")
            print(f"\n📊 Métricas Generales:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            print(f"   Log Loss:  {logloss:.4f}")
            
            print(f"\n📈 Métricas por Clase:")
            for clase in ['H', 'D', 'A']:
                desc = {'H': 'Local', 'D': 'Empate', 'A': 'Visitante'}[clase]
                print(f"   {clase} ({desc:<9}): P={metrics['precision_per_class'][clase]:.3f}, "
                      f"R={metrics['recall_per_class'][clase]:.3f}, "
                      f"F1={metrics['f1_per_class'][clase]:.3f}")
            
            print(f"\n🎯 Matriz de Confusión:")
            print(f"              Predicho")
            print(f"            H    D    A")
            print(f"  Real  H  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
            print(f"        D  {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
            print(f"        A  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
            print(f"{'='*60}\n")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Guardar modelo a archivo
        
        Args:
            filepath (str): Ruta del archivo
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. No hay nada que guardar.")
        
        model_data = {
            'model': self.model,
            'ensemble_type': self.ensemble_type,
            'voting': self.voting,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """
        Cargar modelo desde archivo
        
        Args:
            filepath (str): Ruta del archivo
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.ensemble_type = model_data['ensemble_type']
        self.voting = model_data.get('voting', 'soft')
        self.feature_names = model_data['feature_names']
        self.label_encoder = model_data['label_encoder']
        self.label_decoder = model_data['label_decoder']
        self.is_fitted = True
        
        print(f"✅ Modelo cargado desde: {filepath}")


if __name__ == "__main__":
    print("Modelo Ensemble para Prediccion de Partidos")
    print("\nModelos disponibles:")
    print("   - XGBoost: OK")
    print(f"   - LightGBM: {'OK' if LIGHTGBM_AVAILABLE else 'NO (pip install lightgbm)'}")
    print(f"   - CatBoost: {'OK' if CATBOOST_AVAILABLE else 'NO (pip install catboost)'}")
    print("\nUso:")
    print("   from ensemble_model import EnsembleModel")
    print("   model = EnsembleModel(ensemble_type='stacking')")
    print("   model.fit(X_train, y_train)")
    print("   predictions = model.predict(X_test)")
