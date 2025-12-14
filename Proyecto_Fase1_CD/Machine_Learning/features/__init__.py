"""
Features module for Machine Learning.
Contains scripts for feature engineering.

Módulos disponibles:
- elo_features: Sistema de rating ELO para equipos
- h2h_features: Historial de enfrentamientos directos
- momentum_features: Racha y momentum reciente de equipos
- league_position_features: Posición en tabla y estadísticas de liga
- rest_days_features: Días de descanso entre partidos
- advanced_stats_features: Estadísticas avanzadas (varianza, clean sheets, xG, etc.)
"""

__all__ = [
    'elo_features',
    'h2h_features',
    'momentum_features',
    'league_position_features',
    'rest_days_features',
    'advanced_stats_features'
]
