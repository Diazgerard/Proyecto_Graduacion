"""
Script de Automatización - Pipeline Completo ML
================================================

Este script automatiza todo el flujo:
1. Convierte datos (si es necesario)
2. Entrena modelos con todos los datos
3. Muestra resumen de resultados

Uso:
    python train.py
"""

import os
import sys
from datetime import datetime

def check_data_file():
    """Verificar y actualizar datos desde el EDA"""
    eda_csv = os.path.join('..', 'Data_Mining', 'eda_outputsMatchesPremierLeague', 'match_data_cleaned.csv')
    
    if not os.path.exists('match_data_for_ml.csv') or not os.path.exists(eda_csv):
        print("❌ ERROR: No se encuentra match_data_for_ml.csv o el CSV del EDA")
        print("\n¿Deseas convertir los datos desde match_data_cleaned.csv del EDA? (s/n): ", end="")
        response = input().strip().lower()
        
        if response == 's':
            print("\n🔄 Convirtiendo datos desde EDA...")
            exit_code = os.system('python convert_data_format.py')
            if exit_code != 0 or not os.path.exists('match_data_for_ml.csv'):
                print("❌ Error en la conversión. Abortando.")
                sys.exit(1)
        else:
            print("❌ No se puede continuar sin datos. Abortando.")
            sys.exit(1)
    
    # Verificar si el EDA tiene datos más recientes
    if os.path.exists(eda_csv):
        import pandas as pd
        df_ml = pd.read_csv('match_data_for_ml.csv')
        df_eda = pd.read_csv(eda_csv)
        
        print(f"✅ Datos ML: {len(df_ml):,} partidos (fecha: {df_ml['date_game'].max()})")
        print(f"📊 Datos EDA: {len(df_eda):,} registros")
        
        # Si el EDA tiene más datos, preguntar si desea actualizar
        if len(df_eda) > len(df_ml) * 2:  # EDA tiene 2x porque son 2 filas por partido
            print("\n⚠️  El EDA tiene datos más recientes!")
            print("   ¿Deseas actualizar match_data_for_ml.csv? (s/n): ", end="")
            response = input().strip().lower()
            if response == 's':
                print("\n🔄 Actualizando desde EDA...")
                os.system('python convert_data_format.py')
    else:
        import pandas as pd
        df = pd.read_csv('match_data_for_ml.csv')
        print(f"✅ Datos cargados: {len(df):,} partidos")
        print(f"   Fecha más reciente: {df['date_game'].max()}")

def train_models():
    """Entrenar modelos con el pipeline mejorado"""
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
    print("="*70)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Ejecutar pipeline
    exit_code = os.system('python run_improved_pipeline.py')
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("❌ ERROR EN EL ENTRENAMIENTO")
        print("="*70)
        return False

def show_next_steps():
    """Mostrar siguientes pasos"""
    print("\n📋 SIGUIENTES PASOS:")
    print("="*70)
    print("\n1️⃣  Para hacer predicciones:")
    print("   python PredictionPremierLeague.py")
    print("\n2️⃣  Los modelos están en:")
    print("   improved_models/")
    print("\n3️⃣  Para re-entrenar con datos actualizados:")
    print("   python train.py")
    print("\n" + "="*70)

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🤖 PIPELINE AUTOMÁTICO DE MACHINE LEARNING")
    print("   Premier League Match Prediction")
    print("="*70 + "\n")
    
    # Paso 1: Verificar datos
    print("📊 PASO 1: Verificando datos...")
    check_data_file()
    
    # Paso 2: Entrenar modelos
    print("\n🧠 PASO 2: Entrenando modelos...")
    success = train_models()
    
    # Paso 3: Mostrar siguientes pasos
    if success:
        show_next_steps()
    else:
        print("\n⚠️  Revisa los errores arriba y vuelve a intentar.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        sys.exit(1)
