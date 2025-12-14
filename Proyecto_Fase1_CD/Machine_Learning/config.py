"""
config.py
=========

Configuración central para el sistema de Machine Learning.
"""

import os
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).resolve().parent
DATA_MINING_DIR = BASE_DIR.parent / 'Data_Mining'

# Rutas a datos de cada liga
LEAGUE_DATA_PATHS = {
    'laliga': DATA_MINING_DIR / 'eda_outputsMatchesLaLiga' / 'match_data_cleaned.csv',
    'bundesliga': DATA_MINING_DIR / 'eda_outputsMatchesBundesliga' / 'match_data_cleaned.csv',
    'ligue1': DATA_MINING_DIR / 'eda_outputsMatchesLigue1' / 'match_data_cleaned.csv',
    'premierleague': DATA_MINING_DIR / 'eda_outputsMatchesPremierLeague' / 'match_data_cleaned.csv',
    'seriea': DATA_MINING_DIR / 'eda_outputsMatchesSeriaA' / 'match_data_cleaned.csv'
}

# Directorios de modelos
MODELS_DIR = BASE_DIR / 'models'
IMPROVED_MODELS_DIR = MODELS_DIR / 'improved_models'
BASELINE_MODELS_DIR = MODELS_DIR / 'models_v1_baseline'

# Directorios de salida
PREDICTIONS_DIR = BASE_DIR / 'predictions'
DATA_DIR = BASE_DIR / 'data'

# Configuración de modelos
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1
}

# Características a usar
FEATURE_COLUMNS = [
    'ttl_gls', 'ttl_ast', 'ttl_xg', 'ttl_xag',
    'ttl_sh', 'ttl_sot', 'avg_poss',
    'ttl_pass_cmp', 'pct_pass_cmp',
    'ttl_tkl', 'ttl_tkl_won',
    'ttl_yellow_cards', 'ttl_red_cards'
]

# Nombres de ligas (para display)
LEAGUE_DISPLAY_NAMES = {
    'laliga': 'La Liga (España)',
    'bundesliga': 'Bundesliga (Alemania)',
    'ligue1': 'Ligue 1 (Francia)',
    'premierleague': 'Premier League (Inglaterra)',
    'seriea': 'Serie A (Italia)'
}


def ensure_directories():
    """Crear directorios necesarios si no existen."""
    directories = [
        MODELS_DIR,
        IMPROVED_MODELS_DIR,
        BASELINE_MODELS_DIR,
        PREDICTIONS_DIR,
        DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_league_data_path(league_name):
    """Obtener ruta de datos para una liga específica."""
    league_key = league_name.lower().replace(' ', '').replace('_', '')
    return LEAGUE_DATA_PATHS.get(league_key)


def validate_data_files():
    """Validar que existan los archivos de datos necesarios."""
    missing_files = []
    
    for league, path in LEAGUE_DATA_PATHS.items():
        if not path.exists():
            missing_files.append((league, str(path)))
    
    return missing_files


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONFIGURACIÓN DEL SISTEMA DE MACHINE LEARNING")
    print("="*70 + "\n")
    
    print("📁 DIRECTORIOS:")
    print(f"  Base:              {BASE_DIR}")
    print(f"  Data Mining:       {DATA_MINING_DIR}")
    print(f"  Modelos:           {MODELS_DIR}")
    print(f"  Predicciones:      {PREDICTIONS_DIR}")
    print(f"  Datos:             {DATA_DIR}")
    
    print("\n📊 LIGAS CONFIGURADAS:")
    for league, path in LEAGUE_DATA_PATHS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {LEAGUE_DISPLAY_NAMES[league]}")
        print(f"    {path}")
    
    print("\n🔧 VALIDACIÓN:")
    missing = validate_data_files()
    if missing:
        print("  ⚠️  Archivos faltantes:")
        for league, path in missing:
            print(f"    - {league}: {path}")
    else:
        print("  ✅ Todos los archivos de datos encontrados")
    
    print("\n📦 CREANDO DIRECTORIOS...")
    ensure_directories()
    print("  ✅ Directorios creados/verificados")
    
    print("\n" + "="*70 + "\n")
