"""
Modelos Baseline para Predicción de Partidos de Fútbol
======================================================

Este módulo implementa modelos baseline:
1. Modelo Elo Rating - Para predicción de resultados
2. Modelo Poisson - Para predicción de goles
3. Evaluación y métricas de rendimiento
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EloBaseline:
    """Modelo baseline usando sistema Elo Rating"""
    
    def __init__(self, k_factor=20, home_advantage=100):
        """
        Inicializar modelo Elo
        
        Args:
            k_factor (int): Factor K para actualizaciones Elo
            home_advantage (int): Ventaja de local en puntos Elo
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.elo_ratings = {}
        self.predictions = []
        self.actuals = []
        self.is_fitted = False
        
    def fit(self, train_data):
        """
        Entrenar modelo Elo con datos históricos
        
        Args:
            train_data (pd.DataFrame): Datos de entrenamiento con resultados
        """
        print("Entrenando modelo Elo Baseline...")
        
        # Inicializar ratings para todos los equipos
        teams = set(train_data['home_team'].unique()) | set(train_data['away_team'].unique())
        self.elo_ratings = {team: 1500.0 for team in teams}  # Rating inicial estándar como float
        
        # Procesar partidos cronológicamente para actualizar Elo
        train_sorted = train_data.sort_values('date_game').reset_index(drop=True)
        
        for _, match in train_sorted.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            result = match['result']  # 'H', 'A', 'D'
            
            # Calcular probabilidades antes de actualizar
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            # Probabilidad de victoria local (con ventaja de casa)
            prob_home = 1 / (1 + 10**((away_elo - home_elo - self.home_advantage) / 400))
            prob_away = 1 / (1 + 10**((home_elo - away_elo + self.home_advantage) / 400))
            prob_draw = 1 - prob_home - prob_away
            
            # Ajustar probabilidades para que sumen 1
            total_prob = prob_home + prob_away + prob_draw
            prob_home /= total_prob
            prob_away /= total_prob  
            prob_draw /= total_prob
            
            # Actualizar Elo basado en resultado real
            if result == 'H':
                actual_home = 1.0
                actual_away = 0.0
            elif result == 'A':
                actual_home = 0.0
                actual_away = 1.0
            else:  # Draw
                actual_home = 0.5
                actual_away = 0.5
            
            # Nuevos ratings Elo
            expected_home = prob_home + 0.5 * prob_draw
            expected_away = prob_away + 0.5 * prob_draw
            
            self.elo_ratings[home_team] += self.k_factor * (actual_home - expected_home)
            self.elo_ratings[away_team] += self.k_factor * (actual_away - expected_away)
        
        self.is_fitted = True
        print(f"Modelo Elo entrenado con {len(train_sorted):,} partidos")
        print(f"   Equipos: {len(teams)}")
        print(f"   Rating más alto: {max(self.elo_ratings.values()):.0f}")
        print(f"   Rating más bajo: {min(self.elo_ratings.values()):.0f}")
        
    def predict_proba(self, test_data):
        """
        Predecir probabilidades de resultados
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        probabilities = []
        
        for _, match in test_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Obtener ratings actuales
            home_elo = self.elo_ratings.get(home_team, 1500)
            away_elo = self.elo_ratings.get(away_team, 1500)
            
            # Calcular probabilidades usando método más estable
            elo_diff = home_elo - away_elo + self.home_advantage
            
            # Probabilidad básica de victoria local
            prob_home = 1 / (1 + 10**(-elo_diff / 400))
            prob_away = 1 / (1 + 10**(elo_diff / 400))
            
            # Calcular empate como factor de las probabilidades extremas
            prob_draw = max(0.1, 1 - prob_home - prob_away)
            
            # Asegurar valores positivos
            prob_home = max(0.01, prob_home)
            prob_away = max(0.01, prob_away) 
            prob_draw = max(0.01, prob_draw)
            
            # Normalizar para asegurar que sumen 1
            total = prob_home + prob_draw + prob_away
            prob_home /= total
            prob_draw /= total
            prob_away /= total
            
            # Clip final para evitar errores numéricos
            probs = np.array([prob_home, prob_draw, prob_away])
            probs = np.clip(probs, 1e-15, 1-1e-15)
            probs = probs / probs.sum()  # Renormalizar
            
            probabilities.append(probs.tolist())
        
        return np.array(probabilities)
    
    def predict(self, test_data):
        """
        Predecir resultados más probables
        """
        probas = self.predict_proba(test_data)
        predictions = []
        
        for proba in probas:
            if proba[0] > proba[1] and proba[0] > proba[2]:
                predictions.append('H')
            elif proba[2] > proba[1]:
                predictions.append('A') 
            else:
                predictions.append('D')
                
        return predictions
    
    def evaluate(self, test_data, y_true, detailed=True):
        """
        Evaluación completa del modelo con métricas avanzadas
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        predictions = self.predict(test_data)
        probabilities = self.predict_proba(test_data)
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        
        # Métricas por clase
        precision_per_class = precision_score(y_true, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        recall_per_class = recall_score(y_true, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        f1_per_class = f1_score(y_true, predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, predictions, labels=['H', 'D', 'A'])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': dict(zip(['H', 'D', 'A'], precision_per_class)), # type: ignore
            'recall_per_class': dict(zip(['H', 'D', 'A'], recall_per_class)), # type: ignore
            'f1_per_class': dict(zip(['H', 'D', 'A'], f1_per_class)), # type: ignore
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        if detailed:
            print("EVALUACIÓN COMPLETA ELO BASELINE:")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
            
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
        
        return metrics


class PoissonBaseline:
    """Modelo baseline usando distribución Poisson para goles"""
    
    def __init__(self):
        """Inicializar modelo Poisson"""
        self.team_attack_strength = {}
        self.team_defense_strength = {}
        self.home_advantage = 0
        self.league_avg_goals = 0
        self.is_fitted = False
        
    def fit(self, train_data):
        """
        Entrenar modelo Poisson

        """
        print(" Entrenando modelo Poisson Baseline...")
        
        # Calcular promedios de liga
        total_goals = train_data['home_goals'].sum() + train_data['away_goals'].sum()
        total_games = len(train_data) * 2  # Cada partido cuenta como 2 "actuaciones"
        self.league_avg_goals = total_goals / total_games
        
        # Calcular ventaja de local
        home_goals_avg = train_data['home_goals'].mean()
        away_goals_avg = train_data['away_goals'].mean()
        self.home_advantage = home_goals_avg - away_goals_avg
        
        # Calcular strength de ataque y defensa para cada equipo
        teams = set(train_data['home_team'].unique()) | set(train_data['away_team'].unique())
        
        # Inicializar
        team_goals_for = {team: [] for team in teams}
        team_goals_against = {team: [] for team in teams}
        
        # Recopilar goles a favor y en contra
        for _, match in train_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            # Goals para equipo local
            team_goals_for[home_team].append(home_goals)
            team_goals_against[home_team].append(away_goals)
            
            # Goals para equipo visitante
            team_goals_for[away_team].append(away_goals)
            team_goals_against[away_team].append(home_goals)
        
        # Calcular strengths
        for team in teams:
            goals_for_avg = np.mean(team_goals_for[team]) if team_goals_for[team] else self.league_avg_goals
            goals_against_avg = np.mean(team_goals_against[team]) if team_goals_against[team] else self.league_avg_goals
            
            self.team_attack_strength[team] = goals_for_avg / self.league_avg_goals
            self.team_defense_strength[team] = goals_against_avg / self.league_avg_goals
        
        self.is_fitted = True
        print(f" Modelo Poisson entrenado con {len(train_data):,} partidos")
        print(f"    Promedio goles liga: {self.league_avg_goals:.2f}")
        print(f"    Ventaja local: {self.home_advantage:.2f} goles")
        print(f"    Mejor ataque: {max(self.team_attack_strength.values()):.2f}")
        print(f"    Mejor defensa: {min(self.team_defense_strength.values()):.2f}")
        
    def predict_goals(self, test_data):
        """
        Predecir goles esperados usando Poisson
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecutar fit() primero.")
        
        home_goals_expected = []
        away_goals_expected = []
        
        for _, match in test_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Obtener strengths
            home_attack = self.team_attack_strength.get(home_team, 1.0)
            home_defense = self.team_defense_strength.get(home_team, 1.0)
            away_attack = self.team_attack_strength.get(away_team, 1.0)
            away_defense = self.team_defense_strength.get(away_team, 1.0)
            
            # Calcular goles esperados
            expected_home = self.league_avg_goals * home_attack * away_defense * (1 + self.home_advantage/self.league_avg_goals)
            expected_away = self.league_avg_goals * away_attack * home_defense
            
            home_goals_expected.append(expected_home)
            away_goals_expected.append(expected_away)
        
        return np.array(home_goals_expected), np.array(away_goals_expected)
    
    def predict_result_probabilities(self, test_data):
        """
        Predecir probabilidades de resultado usando distribución Poisson
        """
        home_expected, away_expected = self.predict_goals(test_data)
        probabilities = []
        
        for home_exp, away_exp in zip(home_expected, away_expected):
            # Calcular probabilidades para diferentes scorelines
            prob_home = 0
            prob_draw = 0
            prob_away = 0
            
            # Considerar hasta 10 goles por equipo (suficiente para la mayoría de casos)
            for home_goals in range(11):
                for away_goals in range(11):
                    prob_score = poisson.pmf(home_goals, home_exp) * poisson.pmf(away_goals, away_exp)
                    
                    if home_goals > away_goals:
                        prob_home += prob_score
                    elif home_goals == away_goals:
                        prob_draw += prob_score
                    else:
                        prob_away += prob_score
            
            probabilities.append([prob_home, prob_draw, prob_away])
        
        return np.array(probabilities)
    
    def evaluate(self, test_data, y_home_goals, y_away_goals, y_results, detailed=True):
        """
        Evaluación completa del modelo Poisson con métricas avanzadas
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import classification_report, confusion_matrix
        
        home_pred, away_pred = self.predict_goals(test_data)
        result_probs = self.predict_result_probabilities(test_data)
        
        # Métricas para goles
        mae_home = mean_absolute_error(y_home_goals, home_pred)
        mae_away = mean_absolute_error(y_away_goals, away_pred)
        rmse_home = np.sqrt(mean_squared_error(y_home_goals, home_pred))
        rmse_away = np.sqrt(mean_squared_error(y_away_goals, away_pred))
        
        # Predicciones de resultado
        result_predictions = []
        for proba in result_probs:
            if proba[0] > proba[1] and proba[0] > proba[2]:
                result_predictions.append('H')
            elif proba[2] > proba[1]:
                result_predictions.append('A')
            else:
                result_predictions.append('D')
        
        # Métricas básicas para resultados
        accuracy = accuracy_score(y_results, result_predictions)
        precision = precision_score(y_results, result_predictions, average='weighted', zero_division=0)
        recall = recall_score(y_results, result_predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_results, result_predictions, average='weighted', zero_division=0)
        
        # Métricas por clase
        precision_per_class = precision_score(y_results, result_predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        recall_per_class = recall_score(y_results, result_predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        f1_per_class = f1_score(y_results, result_predictions, average=None, zero_division=0, labels=['H', 'D', 'A'])
        
        # Matriz de confusión
        cm = confusion_matrix(y_results, result_predictions, labels=['H', 'D', 'A'])
        
        # Log loss para resultados
        y_results_encoded = []
        for result in y_results:
            if result == 'H':
                y_results_encoded.append([1, 0, 0])
            elif result == 'D':
                y_results_encoded.append([0, 1, 0])
            else:
                y_results_encoded.append([0, 0, 1])
        
        # Corregir probabilidades para evitar errores de precisión numérica
        result_probs_clipped = np.clip(result_probs, 1e-15, 1-1e-15)
        try:
            logloss = log_loss(np.array(y_results_encoded), result_probs_clipped)
        except:
            logloss = 999.0  # Valor alto si no se puede calcular
        
        metrics = {
            'mae_home_goals': mae_home,
            'mae_away_goals': mae_away,
            'rmse_home_goals': rmse_home,
            'rmse_away_goals': rmse_away,
            'accuracy': accuracy,  # Para compatibilidad
            'result_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': dict(zip(['H', 'D', 'A'], precision_per_class)), # type: ignore
            'recall_per_class': dict(zip(['H', 'D', 'A'], recall_per_class)), # type: ignore
            'f1_per_class': dict(zip(['H', 'D', 'A'], f1_per_class)), # type: ignore
            'confusion_matrix': cm,
            'result_log_loss': logloss,
            'home_goals_pred': home_pred,
            'away_goals_pred': away_pred,
            'result_probabilities': result_probs,
            'result_predictions': result_predictions
        }
        
        if detailed:
            print(f" EVALUACIÓN COMPLETA POISSON BASELINE:")
            print(f"   Goles - MAE Local: {mae_home:.3f}, MAE Visitante: {mae_away:.3f}")
            print(f"   Goles - RMSE Local: {rmse_home:.3f}, RMSE Visitante: {rmse_away:.3f}")
            print(f"   Resultados - Accuracy: {accuracy:.3f}")
            print(f"   Resultados - Precision: {precision:.3f}")
            print(f"   Resultados - Recall: {recall:.3f}")
            print(f"   Resultados - F1-Score: {f1:.3f}")
            
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
        
        return metrics


class BaselineEvaluator:
    """Clase para evaluar y comparar modelos baseline"""
    
    def __init__(self):
        """Inicializar evaluador"""
        self.elo_model = None
        self.poisson_model = None
        self.results = {}
        
    def train_models(self, train_data):
        """
        Entrenar ambos modelos baseline

        """
        print(" ENTRENANDO MODELOS BASELINE")
        print("="*40)
        
        # Entrenar Elo
        self.elo_model = EloBaseline()
        self.elo_model.fit(train_data)
        
        print()
        
        # Entrenar Poisson
        self.poisson_model = PoissonBaseline()
        self.poisson_model.fit(train_data)
        
    def evaluate_models(self, test_data, y_results, y_home_goals, y_away_goals):
        """
        Evaluar ambos modelos en datos de test

        """
        print("\n EVALUANDO MODELOS BASELINE")
        print("="*35)
        
        if self.elo_model is None or self.poisson_model is None:
            raise ValueError("Los modelos no están inicializados. Ejecute train_models primero.")
            
        # Evaluar Elo
        elo_metrics = self.elo_model.evaluate(test_data, y_results)
        
        print()
        
        # Evaluar Poisson
        poisson_metrics = self.poisson_model.evaluate(test_data, y_home_goals, y_away_goals, y_results)
        
        self.results = {
            'elo': elo_metrics,
            'poisson': poisson_metrics
        }
        
        return self.results
        
    def create_comparison_plots(self, test_data, y_results, save_dir='baseline_plots'):
        """
        Crear visualizaciones comparativas
        

        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print(" No hay resultados para visualizar")
            return
        
        # 1. Comparación de accuracy
        plt.figure(figsize=(10, 6))
        
        models = ['Elo', 'Poisson']
        accuracies = [self.results['elo']['accuracy'], self.results['poisson']['result_accuracy']]
        
        bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        plt.ylabel('Accuracy')
        plt.title('Comparación de Accuracy - Modelos Baseline', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Agregar valores en barras
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'baseline_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Matriz de confusión para Elo
        from sklearn.metrics import confusion_matrix
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elo confusion matrix
        cm_elo = confusion_matrix(y_results, self.results['elo']['predictions'], labels=['H', 'D', 'A'])
        sns.heatmap(cm_elo, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Home', 'Draw', 'Away'], yticklabels=['Home', 'Draw', 'Away'],
                   ax=axes[0])
        axes[0].set_title('Matriz de Confusión - ELO', fontweight='bold')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Actual')
        
        # Poisson confusion matrix
        cm_poisson = confusion_matrix(y_results, self.results['poisson']['result_predictions'], labels=['H', 'D', 'A'])
        sns.heatmap(cm_poisson, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=['Home', 'Draw', 'Away'], yticklabels=['Home', 'Draw', 'Away'],
                   ax=axes[1])
        axes[1].set_title('Matriz de Confusión - Poisson', fontweight='bold')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Visualizaciones guardadas en: {save_dir}")
        
    def generate_summary_report(self):
        """Generar reporte resumen de modelos baseline"""
        if not self.results:
            print(" No hay resultados para reportar")
            return
        
        print("\n" + "="*50)
        print(" REPORTE RESUMEN - MODELOS BASELINE")
        print("="*50)
        
        print("\n ELO RATING MODEL:")
        print(f"   • Accuracy: {self.results['elo']['accuracy']:.3f}")
        print(f"   • Log Loss: {self.results['elo']['log_loss']:.3f}")
        
        print("\n POISSON MODEL:")
        print(f"   • Accuracy Resultados: {self.results['poisson']['result_accuracy']:.3f}")
        print(f"   • Log Loss Resultados: {self.results['poisson']['result_log_loss']:.3f}")
        print(f"   • MAE Goles Local: {self.results['poisson']['mae_home_goals']:.3f}")
        print(f"   • MAE Goles Visitante: {self.results['poisson']['mae_away_goals']:.3f}")
        
        print("\n MEJOR MODELO:")
        if self.results['elo']['accuracy'] > self.results['poisson']['result_accuracy']:
            print("   ELO Rating (mayor accuracy)")
        else:
            print("   Poisson (mayor accuracy)")
        
        print("="*50)


if __name__ == "__main__":
    # Ejemplo de uso - requiere datos del pipeline ETL
    print(" Modelos Baseline implementados:")
    print("   • EloBaseline - Sistema de rating Elo")
    print("   • PoissonBaseline - Distribución Poisson para goles") 
    print("   • BaselineEvaluator - Comparación y evaluación")