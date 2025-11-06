"""
Calibración de Modelos para Predicción de Fútbol
===============================================

Este módulo implementa técnicas de calibración para modelos de ML:
1. CalibratedClassifierCV con Platt Scaling e Isotonic Regression
2. Reliability Diagrams para visualizar calibración
3. Brier Score para medir calibración
4. Funciones de utilidad para análisis de calibración
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')


class ModelCalibrator:
    """Clase para calibrar modelos de clasificación"""
    
    def __init__(self, base_model, method='sigmoid', cv=3):
        """
        Inicializar calibrador
        
        Args:
            base_model: Modelo base para calibrar
            method: 'sigmoid' (Platt) o 'isotonic' 
            cv: Número de folds para cross-validation
        """
        self.base_model = base_model
        self.method = method if method in ['sigmoid', 'isotonic'] else 'sigmoid'
        self.cv = cv
        self.calibrated_model = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Entrenar modelo calibrado
        """
        print(f"Entrenando modelo calibrado ({self.method})...")
        
        # Crear modelo calibrado
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, 
            method=self.method, # type: ignore
            cv=self.cv
        )
        
        # Entrenar
        self.calibrated_model.fit(X, y)
        self.is_fitted = True
        
        print(f"Modelo calibrado entrenado con {self.cv}-fold CV")
        
    def predict(self, X):
        """Predicciones del modelo calibrado"""
        if not self.is_fitted or self.calibrated_model is None:
            raise ValueError("Modelo no entrenado")
        return self.calibrated_model.predict(X)
    
    def predict_proba(self, X):
        """Probabilidades calibradas"""
        if not self.is_fitted or self.calibrated_model is None:
            raise ValueError("Modelo no entrenado")
        return self.calibrated_model.predict_proba(X)
    
    def evaluate_calibration(self, X_test, y_test, n_bins=10, plot=True):
        """
        Evaluar calidad de calibración
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
            
        # Probabilidades calibradas y sin calibrar
        probs_calibrated = self.predict_proba(X_test)
        probs_uncalibrated = self.base_model.predict_proba(X_test)
        
        # Métricas de calibración por clase
        calibration_metrics = {}
        
        classes = ['H', 'D', 'A']
        class_names = ['Local', 'Empate', 'Visitante']
        
        for i, (class_label, class_name) in enumerate(zip(classes, class_names)):
            # Convertir a binario para esta clase
            y_binary = (y_test == class_label).astype(int)
            
            # Brier Score
            brier_uncalibrated = brier_score_loss(y_binary, probs_uncalibrated[:, i])
            brier_calibrated = brier_score_loss(y_binary, probs_calibrated[:, i])
            
            # Curva de calibración
            fraction_pos_uncal, mean_pred_uncal = calibration_curve(
                y_binary, probs_uncalibrated[:, i], n_bins=n_bins
            )
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_binary, probs_calibrated[:, i], n_bins=n_bins
            )
            
            calibration_metrics[class_label] = {
                'brier_uncalibrated': brier_uncalibrated,
                'brier_calibrated': brier_calibrated,
                'brier_improvement': brier_uncalibrated - brier_calibrated,
                'calibration_curve_uncal': (fraction_pos_uncal, mean_pred_uncal),
                'calibration_curve_cal': (fraction_pos_cal, mean_pred_cal),
                'class_name': class_name
            }
        
        # Log Loss global
        logloss_uncalibrated = log_loss(y_test, probs_uncalibrated)
        logloss_calibrated = log_loss(y_test, probs_calibrated)
        
        metrics = {
            'logloss_uncalibrated': logloss_uncalibrated,
            'logloss_calibrated': logloss_calibrated,
            'logloss_improvement': logloss_uncalibrated - logloss_calibrated,
            'calibration_per_class': calibration_metrics
        }
        
        # Mostrar resultados
        print("\nEVALUACIÓN DE CALIBRACIÓN:")
        print(f"   Log Loss sin calibrar: {logloss_uncalibrated:.4f}")
        print(f"   Log Loss calibrado:    {logloss_calibrated:.4f}")
        print(f"   Mejora Log Loss:       {metrics['logloss_improvement']:.4f}")
        
        print("\n   Brier Score por clase:")
        for class_label in classes:
            metrics_class = calibration_metrics[class_label]
            print(f"     {class_label} ({metrics_class['class_name']:<9}): {metrics_class['brier_uncalibrated']:.4f} → {metrics_class['brier_calibrated']:.4f} (Δ: {metrics_class['brier_improvement']:.4f})")
        
        # Gráfico de calibración
        if plot:
            self._plot_calibration_curves(calibration_metrics, n_bins)
            
        return metrics
    
    def _plot_calibration_curves(self, calibration_metrics, n_bins):
        """
        Crear reliability diagrams (curvas de calibración)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Reliability Diagrams - Calibración {self.method.title()}', fontsize=14)
        
        classes = ['H', 'D', 'A']
        colors = ['blue', 'green', 'red']
        
        for i, (class_label, color) in enumerate(zip(classes, colors)):
            ax = axes[i]
            metrics = calibration_metrics[class_label]
            
            # Datos de calibración
            fraction_pos_uncal, mean_pred_uncal = metrics['calibration_curve_uncal']
            fraction_pos_cal, mean_pred_cal = metrics['calibration_curve_cal']
            
            # Línea perfecta
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Calibración perfecta')
            
            # Curvas de calibración
            ax.plot(mean_pred_uncal, fraction_pos_uncal, 'o-', color=color, alpha=0.7, 
                   label=f'Sin calibrar (Brier: {metrics["brier_uncalibrated"]:.3f})')
            ax.plot(mean_pred_cal, fraction_pos_cal, 's-', color=color, 
                   label=f'Calibrado (Brier: {metrics["brier_calibrated"]:.3f})')
            
            ax.set_xlabel('Probabilidad Predicha')
            ax.set_ylabel('Fracción Positiva')
            ax.set_title(f'Clase {class_label} - {metrics["class_name"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()


class FootballModelCalibrator:
    """Calibrador específico para modelos de fútbol"""
    
    def __init__(self):
        """Inicializar calibrador de fútbol"""
        self.calibrators = {}
        self.models = {}
        
    def add_model(self, name, model, X_train, y_train):
        """
        Agregar modelo para calibrar
        """
        print(f"\nAgregando modelo '{name}' para calibración...")
        
        # Guardar modelo original
        self.models[name] = {
            'original': model,
            'X_train': X_train,
            'y_train': y_train
        }
        
        # Crear calibradores con diferentes métodos
        self.calibrators[name] = {
            'sigmoid': ModelCalibrator(model, method='sigmoid', cv=3),
            'isotonic': ModelCalibrator(model, method='isotonic', cv=3)
        }
        
        # Entrenar calibradores
        for method, calibrator in self.calibrators[name].items():
            try:
                calibrator.fit(X_train, y_train)
                print(f"  ✅ Calibrador {method} entrenado")
            except (ValueError, RuntimeError) as e:
                print(f"  ❌ Error en calibrador {method}: {e}")
    
    def compare_calibration(self, model_name, X_test, y_test, plot=True):
        """
        Comparar calibración entre métodos
        """
        if model_name not in self.calibrators:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        print(f"\nCOMPARACIÓN DE CALIBRACIÓN - {model_name.upper()}")
        print("=" * 60)
        
        results = {}
        
        # Evaluar modelo original
        original = self.models[model_name]['original']
        probs_original = original.predict_proba(X_test)
        logloss_original = log_loss(y_test, probs_original)
        
        print("Modelo Original:")
        print(f"   Log Loss: {logloss_original:.4f}")
        
        results['original'] = {
            'logloss': logloss_original,
            'probabilities': probs_original
        }
        
        # Evaluar calibradores
        for method, calibrator in self.calibrators[model_name].items():
            if calibrator.is_fitted:
                print(f"\nCalibrador {method.title()}:")
                metrics = calibrator.evaluate_calibration(X_test, y_test, plot=False)
                results[method] = metrics
        
        # Determinar mejor método
        best_method = min(results.keys(), 
                         key=lambda x: results[x]['logloss_calibrated'] if 'logloss_calibrated' in results[x] else results[x]['logloss'])
        
        print(f"\n🏆 MEJOR MÉTODO: {best_method.upper()}")
        
        # Gráfico comparativo si se solicita
        if plot:
            self._plot_comparison(results, model_name)
        
        return results
    
    def _plot_comparison(self, results, model_name):
        """
        Gráfico comparativo de métodos de calibración
        """
        methods = list(results.keys())
        logloss_values = []
        
        for method in methods:
            if 'logloss_calibrated' in results[method]:
                logloss_values.append(results[method]['logloss_calibrated'])
            else:
                logloss_values.append(results[method]['logloss'])
        
        # Gráfico de barras
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, logloss_values, 
                      color=['gray', 'blue', 'green'], alpha=0.7)
        
        plt.title(f'Comparación Log Loss - {model_name}')
        plt.ylabel('Log Loss')
        plt.xlabel('Método de Calibración')
        
        # Agregar valores en las barras
        for bar, value in zip(bars, logloss_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_best_calibrated_model(self, model_name, X_test, y_test):
        """
        Obtener el mejor modelo calibrado
        """
        results = self.compare_calibration(model_name, X_test, y_test, plot=False)
        
        # Encontrar mejor método
        best_method = min(results.keys(), 
                         key=lambda x: results[x]['logloss_calibrated'] if 'logloss_calibrated' in results[x] else results[x]['logloss'])
        
        if best_method == 'original':
            return self.models[model_name]['original']
        else:
            return self.calibrators[model_name][best_method]


def calibrate_football_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Función utilitaria para calibrar múltiples modelos de fútbol
    
    Args:
        models_dict: Diccionario {'nombre': modelo}
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        Diccionario con modelos calibrados
    """
    calibrator = FootballModelCalibrator()
    calibrated_models = {}
    
    # Agregar y calibrar cada modelo
    for name, model in models_dict.items():
        calibrator.add_model(name, model, X_train, y_train)
        
        # Obtener mejor modelo calibrado
        best_model = calibrator.get_best_calibrated_model(name, X_test, y_test)
        calibrated_models[name] = best_model
        
        print(f"\n{'='*50}")
    
    return calibrated_models


# Ejemplo de uso
if __name__ == "__main__":
    print("Módulo de Calibración de Modelos de Fútbol")
    print("==========================================")
    print("Funcionalidades disponibles:")
    print("• ModelCalibrator: Calibración individual")
    print("• FootballModelCalibrator: Calibración múltiple")  
    print("• Reliability Diagrams")
    print("• Métricas de calibración (Brier Score, Log Loss)")
    print("• Comparación de métodos (Sigmoid vs Isotonic)")