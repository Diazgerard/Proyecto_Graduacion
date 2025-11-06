"""
Evaluación Avanzada para Modelos de Predicción de Fútbol
=======================================================

Este módulo implementa técnicas avanzadas de evaluación:
1. Cross-validation temporal para datos de series de tiempo
2. Curvas ROC y métricas AUC para clasificación multiclase
3. Métricas específicas de apuestas (Kelly Criterion, ROI, Profit)
4. Análisis de confianza en predicciones
5. Comparación estadística entre modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, log_loss
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TemporalCrossValidator:
    """Cross-validation temporal para datos de series de tiempo"""
    
    def __init__(self, n_splits=5, test_size_months=2):
        """
        Inicializar validador temporal
        
        Args:
            n_splits: Número de divisiones temporales
            test_size_months: Meses para conjunto de prueba
        """
        self.n_splits = n_splits
        self.test_size_months = test_size_months
        
    def split_by_date(self, X, y, date_column):
        """
        Dividir datos por fechas (ideal para fútbol)
        """
        # Convertir fechas
        dates = pd.to_datetime(X[date_column])
        
        # Ordenar por fecha
        sort_idx = dates.argsort()
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y.iloc[sort_idx] if hasattr(y, 'iloc') else y[sort_idx]
        dates_sorted = dates.iloc[sort_idx]
        
        # Crear divisiones
        splits = []
        total_months = (dates_sorted.max() - dates_sorted.min()).days / 30.44
        months_per_split = total_months / self.n_splits
        
        for i in range(self.n_splits):
            # Punto de corte para entrenamiento
            train_end = dates_sorted.min() + pd.DateOffset(months=int((i+1) * months_per_split - self.test_size_months))
            test_start = train_end
            test_end = dates_sorted.min() + pd.DateOffset(months=int((i+1) * months_per_split))
            
            # Índices de entrenamiento y prueba
            train_idx = dates_sorted <= train_end
            test_idx = (dates_sorted > test_start) & (dates_sorted <= test_end)
            
            if train_idx.sum() > 0 and test_idx.sum() > 0:
                splits.append((
                    X_sorted.index[train_idx].tolist(),
                    X_sorted.index[test_idx].tolist()
                ))
        
        return splits
    
    def split_sequential(self, X, y=None):
        """
        División secuencial simple (sin fechas)
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return list(tscv.split(X))
    
    def evaluate_model(self, model, X, y, date_column=None, metrics=None):
        """
        Evaluar modelo con cross-validation temporal
        """
        if metrics is None:
            metrics = ['accuracy', 'logloss']
            
        print(f"Evaluación temporal con {self.n_splits} divisiones...")
        
        # Elegir método de división
        if date_column and date_column in X.columns:
            splits = self.split_by_date(X, y, date_column)
            print(f"Usando división por fechas (columna: {date_column})")
        else:
            splits = self.split_sequential(X, y)
            print("Usando división secuencial")
        
        results = {metric: [] for metric in metrics}
        results['split_info'] = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"  Split {i+1}/{len(splits)}: Train={len(train_idx)}, Test={len(test_idx)}")
            
            # Dividir datos
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calcular métricas
            if 'accuracy' in metrics:
                results['accuracy'].append(accuracy_score(y_test, y_pred))
            
            if 'logloss' in metrics and y_proba is not None:
                results['logloss'].append(log_loss(y_test, y_proba))
            
            # Información del split
            results['split_info'].append({
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start': train_idx[0] if train_idx else None,
                'test_end': test_idx[-1] if test_idx else None
            })
        
        # Calcular estadísticas
        for metric in metrics:
            if results[metric]:
                mean_val = np.mean(results[metric])
                std_val = np.std(results[metric])
                print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return results


class ROCAnalyzer:
    """Análisis de curvas ROC para clasificación multiclase"""
    
    def __init__(self, classes=None, class_names=None):
        """
        Inicializar analizador ROC
        """
        if classes is None:
            classes = ['H', 'D', 'A']
        if class_names is None:
            class_names = ['Local', 'Empate', 'Visitante']
            
        self.classes = classes
        self.class_names = class_names
        self.n_classes = len(classes)
        
    def compute_roc_multiclass(self, y_true, y_proba):
        """
        Calcular ROC para clasificación multiclase
        """
        # Calcular ROC para cada clase usando one-vs-rest
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i, class_label in enumerate(self.classes):
            # Crear etiquetas binarias para esta clase
            y_binary = (y_true == class_label).astype(int)
            
            # Calcular ROC para esta clase
            fpr[i], tpr[i], _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # ROC micro-average: convertir a problema binario global
        y_true_all = []
        y_proba_all = []
        
        for i, class_label in enumerate(self.classes):
            y_binary = (y_true == class_label).astype(int)
            y_true_all.extend(y_binary)
            y_proba_all.extend(y_proba[:, i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_all, y_proba_all)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # ROC macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        return fpr, tpr, roc_auc
    
    def plot_roc_curves(self, y_true, y_proba, title="Curvas ROC"):
        """
        Graficar curvas ROC
        """
        fpr, tpr, roc_auc = self.compute_roc_multiclass(y_true, y_proba)
        
        plt.figure(figsize=(12, 8))
        
        # Colores para cada clase
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # ROC por clase
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
        
        # ROC promedio
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
                label=f'Micro-promedio (AUC = {roc_auc["micro"]:.3f})')
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
                label=f'Macro-promedio (AUC = {roc_auc["macro"]:.3f})')
        
        # Línea diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Clasificador aleatorio')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return roc_auc


class BettingAnalyzer:
    """Análisis de métricas de apuestas"""
    
    def __init__(self, initial_bankroll=1000):
        """
        Inicializar analizador de apuestas
        """
        self.initial_bankroll = initial_bankroll
        
    def kelly_criterion(self, probabilities, odds, min_edge=0.01):
        """
        Calcular apuesta óptima según Kelly Criterion
        
        Args:
            probabilities: Probabilidades del modelo [P(H), P(D), P(A)]
            odds: Cuotas de la casa [odds_H, odds_D, odds_A]
            min_edge: Ventaja mínima para apostar
            
        Returns:
            Fracciones de bankroll a apostar en cada resultado
        """
        probabilities = np.array(probabilities)
        odds = np.array(odds)
        
        # Calcular ventaja (edge)
        implied_probs = 1 / odds
        edges = probabilities - implied_probs
        
        # Kelly fractions
        kelly_fractions = np.zeros_like(edges)
        
        for i, (edge, odd) in enumerate(zip(edges, odds)):
            if edge > min_edge:  # Solo apostar si hay ventaja
                kelly_fractions[i] = edge / (odd - 1)
            
        return kelly_fractions, edges
    
    def simulate_betting(self, predictions_proba, actual_results, odds_data, strategy='kelly'):
        """
        Simular apuestas con diferentes estrategias
        """
        bankroll_history = [self.initial_bankroll]
        bet_history = []
        
        current_bankroll = self.initial_bankroll
        
        for i, (probs, actual, odds) in enumerate(zip(predictions_proba, actual_results, odds_data)):
            
            if strategy == 'kelly':
                bet_fractions, _ = self.kelly_criterion(probs, odds)
                bet_amounts = bet_fractions * current_bankroll
            elif strategy == 'fixed':
                # Apostar cantidad fija al resultado más probable
                best_prob_idx = np.argmax(probs)
                bet_amounts = np.zeros(3)
                if probs[best_prob_idx] > 0.4:  # Solo si confianza > 40%
                    bet_amounts[best_prob_idx] = 50  # $50 fijos
            else:
                bet_amounts = np.zeros(3)
            
            # Determinar resultado ganador
            result_map = {'H': 0, 'D': 1, 'A': 2}
            winning_idx = result_map[actual]
            
            # Calcular ganancias/pérdidas
            total_bet = np.sum(bet_amounts)
            if total_bet > 0:
                payout = bet_amounts[winning_idx] * odds[winning_idx]
                profit = payout - total_bet
                current_bankroll += profit
            else:
                profit = 0
            
            # Guardar historial
            bankroll_history.append(current_bankroll)
            bet_history.append({
                'match': i,
                'bet_amounts': bet_amounts,
                'profit': profit,
                'bankroll': current_bankroll,
                'actual_result': actual,
                'predicted_probs': probs
            })
        
        return bankroll_history, bet_history
    
    def analyze_betting_performance(self, bankroll_history, bet_history):
        """
        Analizar rendimiento de apuestas
        """
        initial = bankroll_history[0]
        final = bankroll_history[-1]
        
        # Métricas básicas
        total_roi = (final - initial) / initial * 100
        total_profit = final - initial
        
        # Número de apuestas
        actual_bets = [bet for bet in bet_history if np.sum(bet['bet_amounts']) > 0]
        n_bets = len(actual_bets)
        
        if n_bets > 0:
            # Apuestas ganadoras
            winning_bets = [bet for bet in actual_bets if bet['profit'] > 0]
            win_rate = len(winning_bets) / n_bets * 100
            
            # Profit promedio por apuesta
            avg_profit_per_bet = np.mean([bet['profit'] for bet in actual_bets])
            
            # Máximo drawdown
            running_max = np.maximum.accumulate(bankroll_history)
            drawdowns = (running_max - bankroll_history) / running_max * 100
            max_drawdown = np.max(drawdowns)
            
        else:
            win_rate = 0
            avg_profit_per_bet = 0
            max_drawdown = 0
        
        metrics = {
            'total_roi': total_roi,
            'total_profit': total_profit,
            'n_bets': n_bets,
            'win_rate': win_rate,
            'avg_profit_per_bet': avg_profit_per_bet,
            'max_drawdown': max_drawdown,
            'final_bankroll': final
        }
        
        return metrics
    
    def plot_betting_performance(self, bankroll_history, metrics, title="Rendimiento de Apuestas"):
        """
        Graficar rendimiento de apuestas
        """
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Evolución del bankroll
        ax1.plot(bankroll_history, linewidth=2, color='blue')
        ax1.axhline(y=self.initial_bankroll, color='red', linestyle='--', alpha=0.7, label='Bankroll inicial')
        ax1.set_title(f'{title}\nROI Total: {metrics["total_roi"]:.2f}%')
        ax1.set_ylabel('Bankroll ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Métricas resumen
        metrics_text = [
            f'Profit Total: ${metrics["total_profit"]:.2f}',
            f'Número de Apuestas: {metrics["n_bets"]}',
            f'Tasa de Acierto: {metrics["win_rate"]:.1f}%',
            f'Profit/Apuesta: ${metrics["avg_profit_per_bet"]:.2f}',
            f'Max Drawdown: {metrics["max_drawdown"]:.1f}%'
        ]
        
        ax2.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax2.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


class ModelComparator:
    """Comparación estadística entre modelos"""
    
    def __init__(self):
        """Inicializar comparador"""
        self.results = {}
        
    def add_model_results(self, name, y_true, y_pred, y_proba=None):
        """
        Agregar resultados de un modelo
        """
        self.results[name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy_score(y_true, y_pred),
            'logloss': log_loss(y_true, y_proba) if y_proba is not None else None
        }
    
    def mcnemar_test(self, model1_name, model2_name):
        """
        Test de McNemar para comparar dos modelos
        """
        if model1_name not in self.results or model2_name not in self.results:
            raise ValueError("Modelos no encontrados")
        
        y_true = self.results[model1_name]['y_true']
        pred1 = self.results[model1_name]['y_pred']
        pred2 = self.results[model2_name]['y_pred']
        
        # Tabla de contingencia
        correct1 = (pred1 == y_true)
        correct2 = (pred2 == y_true)
        
        # Casos donde los modelos difieren
        model1_correct_model2_wrong = np.sum(correct1 & ~correct2)
        model1_wrong_model2_correct = np.sum(~correct1 & correct2)
        
        # Test de McNemar
        if model1_correct_model2_wrong + model1_wrong_model2_correct > 0:
            chi2 = (abs(model1_correct_model2_wrong - model1_wrong_model2_correct) - 1)**2 / (model1_correct_model2_wrong + model1_wrong_model2_correct)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        else:
            chi2 = 0
            p_value = 1.0
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'model1_better': model1_correct_model2_wrong,
            'model2_better': model1_wrong_model2_correct,
            'significant': p_value < 0.05
        }
    
    def compare_all_models(self):
        """
        Comparar todos los modelos entre sí
        """
        model_names = list(self.results.keys())
        n_models = len(model_names)
        
        print("COMPARACIÓN ENTRE MODELOS")
        print("=" * 50)
        
        # Tabla de accuracy
        print("\nAccuracy por modelo:")
        for name in model_names:
            acc = self.results[name]['accuracy']
            logloss = self.results[name]['logloss']
            print(f"  {name:<20}: {acc:.4f} (LogLoss: {logloss:.4f if logloss else 'N/A'})")
        
        # Tests pareados
        print("\nTests de McNemar (significancia < 0.05):")
        for i in range(n_models):
            for j in range(i+1, n_models):
                name1, name2 = model_names[i], model_names[j]
                test_result = self.mcnemar_test(name1, name2)
                
                if test_result['significant']:
                    better_model = name1 if test_result['model1_better'] > test_result['model2_better'] else name2
                    print(f"  {name1} vs {name2}: p={test_result['p_value']:.4f} *** ({better_model} es mejor)")
                else:
                    print(f"  {name1} vs {name2}: p={test_result['p_value']:.4f} (no significativo)")
        
        return self.results


# Función utilitaria principal
def advanced_evaluation_pipeline(models_dict, X, y, date_column=None, odds_data=None):
    """
    Pipeline completo de evaluación avanzada
    
    Args:
        models_dict: Diccionario {'nombre': modelo}
        X, y: Datos de features y target
        date_column: Columna de fecha para CV temporal
        odds_data: Datos de cuotas para análisis de apuestas
        
    Returns:
        Diccionario con todos los resultados
    """
    print("PIPELINE DE EVALUACIÓN AVANZADA")
    print("=" * 50)
    
    results = {}
    
    # 1. Cross-validation temporal
    cv_temporal = TemporalCrossValidator(n_splits=5)
    
    for name, model in models_dict.items():
        print(f"\n🔄 Evaluando {name}...")
        cv_results = cv_temporal.evaluate_model(model, X, y, date_column)
        results[name] = {'cv_temporal': cv_results}
    
    # 2. Análisis ROC (usando último split como ejemplo)
    print("\n📊 Análisis ROC...")
    roc_analyzer = ROCAnalyzer()
    
    # Dividir datos para ROC
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    comparator = ModelComparator()
    
    for name, model in models_dict.items():
        # Entrenar y predecir
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # ROC Analysis
        if y_proba is not None:
            roc_auc = roc_analyzer.plot_roc_curves(y_test, y_proba, f"ROC - {name}")
            results[name]['roc_auc'] = roc_auc
        
        # Agregar a comparador
        comparator.add_model_results(name, y_test, y_pred, y_proba)
    
    # 3. Comparación estadística
    print("\n📈 Comparación entre modelos...")
    comparison_results = comparator.compare_all_models()
    results['model_comparison'] = comparison_results
    
    # 4. Análisis de apuestas (si hay datos de cuotas)
    if odds_data is not None:
        print("\n💰 Análisis de apuestas...")
        betting_analyzer = BettingAnalyzer()
        
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                # Simular apuestas
                bankroll_hist, bet_hist = betting_analyzer.simulate_betting(
                    model.predict_proba(X_test), 
                    y_test, 
                    odds_data[-len(y_test):]  # Últimas cuotas
                )
                
                # Analizar rendimiento
                betting_metrics = betting_analyzer.analyze_betting_performance(bankroll_hist, bet_hist)
                betting_analyzer.plot_betting_performance(bankroll_hist, betting_metrics, f"Apuestas - {name}")
                
                results[name]['betting'] = {
                    'metrics': betting_metrics,
                    'bankroll_history': bankroll_hist
                }
    
    print("\n✅ Evaluación avanzada completada!")
    return results


# Ejemplo de uso
if __name__ == "__main__":
    print("Módulo de Evaluación Avanzada para Fútbol")
    print("=========================================")
    print("Funcionalidades disponibles:")
    print("• TemporalCrossValidator: CV temporal")
    print("• ROCAnalyzer: Curvas ROC multiclase")
    print("• BettingAnalyzer: Métricas de apuestas")
    print("• ModelComparator: Comparación estadística")
    print("• Pipeline completo de evaluación")