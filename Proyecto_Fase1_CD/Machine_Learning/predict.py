"""
Script de Predicción Rápida
============================

Permite hacer predicciones rápidas sin menú interactivo.

Uso:
    python predict.py Arsenal Chelsea
    python predict.py "Manchester City" Liverpool
"""

import sys
import os

# Asegurarse de que se puede importar GoalsModelWrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PredictionPremierLeague import PremierLeaguePredictor, GoalsModelWrapper

def print_prediction(result):
    """Imprimir predicción formateada"""
    if 'error' in result:
        print(f"\n❌ {result['error']}\n")
        return
    
    print("\n" + "="*60)
    print("⚽ PREDICCIÓN DE PARTIDO")
    print("="*60)
    print(f"\n🏟️  {result['home_team']} vs {result['away_team']}")
    print(f"\n🎯 Resultado: {result['predicted_result']}")
    
    result_names = {'H': 'Victoria Local', 'D': 'Empate', 'A': 'Victoria Visitante'}
    print(f"   ({result_names[result['predicted_result']]})")
    
    print(f"\n⚽ Marcador Predicho: {result['predicted_score']}")
    print(f"\n📊 Confianza: {result['confidence']}")
    
    print(f"\n📈 Probabilidades:")
    probs = result['probabilities']
    print(f"   🏠 Local (H):     {probs['Home']:.1%}")
    print(f"   🤝 Empate (D):    {probs['Draw']:.1%}")
    print(f"   ✈️  Visitante (A): {probs['Away']:.1%}")
    print("\n" + "="*60 + "\n")

def main():
    """Función principal"""
    if len(sys.argv) < 3:
        print("\n❌ Error: Debes proporcionar 2 equipos")
        print("\nUso:")
        print("  python predict.py Arsenal Chelsea")
        print("  python predict.py \"Manchester City\" Liverpool\n")
        sys.exit(1)
    
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    print("\n🔄 Cargando modelos...")
    try:
        predictor = PremierLeaguePredictor()
        print("✅ Modelos cargados correctamente")
        
        print(f"\n🔮 Prediciendo: {home_team} vs {away_team}...")
        result = predictor.predict_match(home_team, away_team)
        
        print_prediction(result)
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
