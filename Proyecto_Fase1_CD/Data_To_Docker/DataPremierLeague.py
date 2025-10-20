import requests
import psycopg2
import json
import os
import sys
import subprocess
import time
from typing import Optional, Dict, List, Any 
from datetime import datetime, timedelta

###################################################################################

class DatabaseConfig:
    HOST = "localhost"
    PORT = 5432
    DATABASE = "2. PremierLeague"
    USER = "admin"
    PASSWORD = "GadumUNITEC123"

###################################################################################

class FBRAPIClient:

    def __init__(self):
        self.base_url = "https://fbrapi.com"
        self.api_key = None
        self.session = requests.Session()
        self.load_api_key()
        self.league_id = 9  # League ID

    def load_api_key(self) -> bool:

        try:
            config_path = os.path.join(os.path.dirname(__file__), "api_config.txt")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    content = f.read().strip()
                    if content.startswith("FBR_API_KEY="):
                        self.api_key = content.split("=", 1)[1]
                        print(f"API Key cargada desde: {config_path}")
                        return True
            print("No se encontró archivo de configuración de API key.")
            return False
        except Exception as e:
            print(f"Error cargando API key: {e}")
            return False


    def get_general_data_league(self) -> Optional[List[Dict]]:

        try:
            url = f"{self.base_url}/league-seasons"
            print(f"Haciendo request a: {url}")

            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }

            payload = {'league_id': self.league_id}

            response = self.session.get(url, headers = headers, data = payload, timeout = 30)

            print(f"Response Status Code: {response.status_code}")

            if response.status_code in [200, 201]:
                # Intentar parsear como JSON
                try:
                    data = response.json()
                    
                    #Los datos vienen en el campo 'data'
                    if isinstance(data, dict) and 'data' in data:
                        leagues = data['data']
                    
                    elif isinstance(data, list):
                        leagues = data
                    
                    else:
                        print("Formato de datos inesperado en la respuesta JSON")
                        return None
                    
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en la API:  {response.status_code}")
                print(f"Contenido del error: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None  

        # Return leagues if successfully fetched
        return leagues if 'leagues' in locals() else None
    

    # API para obtener los id de los todos los equipos durante 2017-2024
    def get_team_ids(self, season_id: str) -> Optional[List[int]]:

        try:
            url = f"{self.base_url}/team-season-stats?league_id={self.league_id}&season_id={season_id}"
            print(f"Haciendo request a: {url}")

            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }

            payload = {
            }

            response = self.session.get(url, headers = headers, data = payload, timeout = 30)

            print(f"Response Status Code: {response.status_code}")

            if response.status_code in [200, 201]:
                # Intentar parsear como JSON
                try:
                    data = response.json()
                    
                    #Los datos vienen en el campo 'data'
                    if isinstance(data, dict) and 'data' in data:
                        teams_data = data['data']
                    
                    elif isinstance(data, list):
                        teams_data = data
                    
                    else:
                        print("Formato de datos inesperado en la respuesta JSON")
                        return None
                    

                    #Extraer Meta_Data
                    meta_data_list = []
                    for team in teams_data:
                        if 'meta_data' in team:
                            meta_data = team['meta_data']
                            # Extraer solo tema_id y team_name
                            team_info = {
                                'team_id': meta_data.get('team_id'),
                                'team_name': meta_data.get('team_name')
                            }
                            meta_data_list.append(team_info)

                    return meta_data_list
                
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en la API:  {response.status_code}")
                print(f"Contenido del error: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None 

    def multiple_seasons(self) -> List[Dict]:
        target_seasons = [
            '2017-2018', '2018-2019', '2019-2020', 
            '2020-2021', '2021-2022', '2022-2023', 
            '2023-2024', '2024-2025', '2025-2026'
        ]

        # Usar un diccionario para evitar duplicados por team_id
        unique_teams = {}

        for season in target_seasons:
            print(f"Obteniendo datos para la temporada: {season}")
            teams_data = self.get_team_ids(season)

            if teams_data:
                # Agregar season_id a cada equipo
                for team in teams_data:
                    team_id = team.get('team_id') # pyright: ignore[reportAttributeAccessIssue]
                    team_name = team.get('team_name') # pyright: ignore[reportAttributeAccessIssue]

                    if team_id not in unique_teams:
                        unique_teams[team_id] = {
                            'team_id': team_id,
                            'team_name': team_name,
                        }
                        print(f"    + Nuevo equipo: {team_name} (ID: {team_id})")
                    elif team_id in unique_teams:
                        # Si el equipo ya existe, solo actualizar team_name si es None
                        if not unique_teams[team_id]['team_name'] and team_name:
                            unique_teams[team_id]['team_name'] = team_name
                        print(f"    = Equipo existente: {team_name}")
            else:
                print(f"No se pudieron obtener datos para la temporada {season}")
        
        # Convertir el diccionario de vuelta a lista
        all_teams_data = list(unique_teams.values())

        return all_teams_data
    

    def get_match_ids(self, season_id: str, team_id: str) -> Optional[List[int]]:
        try:
            # Agregar delay para evitar rate limiting
            time.sleep(1)  # Esperar 1 segundo entre peticiones
            
            url = f"{self.base_url}/teams?team_id={team_id}&season_id={season_id}"
            print(f"Haciendo request a: {url}")

            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }

            payload = {
            }

            response = self.session.get(url, headers = headers, data = payload, timeout = 30)

            print(f"Response Status Code: {response.status_code}")

            if response.status_code in [200, 201]:
                # Intentar parsear como JSON
                try:
                    data = response.json()
                    
                    #Los datos vienen en el campo 'team_schedule'
                    if isinstance(data, dict) and 'team_schedule' in data:
                        team_schedule_data = data['team_schedule']
                        
                        # Los partidos están en team_schedule['data']
                        if 'data' in team_schedule_data:
                            matches_data = team_schedule_data['data']
                        else:
                            print("No se encontró 'data' en team_schedule")
                            return None
                    else:
                        print("Formato de datos inesperado - no se encontró 'team_schedule'")
                        print(f"Claves disponibles: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                        return None
                    

                    ##Extraer datos de los partidos
                    match_data_list = []
                    for match in matches_data:
                        #Filtro por para que sean del League_ID

                        if match.get('league_id') == self.league_id:
                            match_info = {
                                'season_id': season_id,
                                'date': match.get('date'),
                                'match_id': match.get('match_id'),
                                'team_id': team_id,  # Agregar el team_id que consultamos
                                'opponent': match.get('opponent'),
                                'opponent_id': match.get('opponent_id'),
                                'home_away': match.get('home_away'),
                                'gf': match.get('gf'),  #Goles a favor
                                'ga': match.get('ga'),  #Goles en contra
                            }
                            match_data_list.append(match_info)

                    return match_data_list
                
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en la API:  {response.status_code}")
                print(f"Contenido del error: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None 
    
    def get_teams_from_database(self, db_manager) -> List[Dict]:
        """Obtiene todos los equipos desde la tabla team_meta de la base de datos"""
        try:
            # Verificar que tenemos conexión a la base de datos
            if not db_manager.connection or not db_manager.cursor:
                print("No hay conexión activa a la base de datos")
                return []
            
            query = "SELECT team_id, team_name FROM team_meta ORDER BY team_name"
            db_manager.cursor.execute(query)
            
            teams_data = []
            for row in db_manager.cursor.fetchall():
                team_info = {
                    'team_id': row[0],
                    'team_name': row[1]
                }
                teams_data.append(team_info)
            
            print(f"Obtenidos {len(teams_data)} equipos de la base de datos")
            return teams_data
            
        except psycopg2.Error as e:
            print(f"Error obteniendo equipos de la base de datos: {e}")
            return []
        except Exception as e:
            print(f"Error inesperado obteniendo equipos: {e}")
            return []

    def multiple_seasons_matches(self, db_manager=None) -> List[Dict]:
        target_seasons = [
            '2017-2018', '2018-2019', '2019-2020', 
            '2020-2021', '2021-2022', '2022-2023', 
            '2023-2024', '2024-2025'
        ]
        
        all_matches_data = []
        
        # Obtener equipos desde la base de datos
        print("Obteniendo equipos desde la base de datos...")
        teams_from_db = self.get_teams_from_database(db_manager)
        
        if not teams_from_db:
            print("No se pudieron obtener equipos de la base de datos")
            return []
        
        print(f"Total de equipos en la base de datos: {len(teams_from_db)}")
        
        # Procesar por temporada con breaks entre temporadas
        for season_idx, season in enumerate(target_seasons):
            print(f"\n=== Obteniendo partidos para la temporada: {season} ===")
            season_matches = 0
            teams_processed = 0
            teams_no_data = 0
            
            # Break más largo entre temporadas
            if season_idx > 0:
                print("Pausa entre temporadas...")
                time.sleep(5)
            
            # Procesar equipos en lotes de 5
            batch_size = 5
            for i in range(0, len(teams_from_db), batch_size):
                batch = teams_from_db[i:i+batch_size]
                
                print(f"  Procesando lote {i//batch_size + 1} ({len(batch)} equipos)")
                
                for team in batch:
                    team_id = team['team_id']
                    team_name = team['team_name']
                    
                    print(f"    Procesando {team_name} ({team_id}) - {season}")
                    teams_processed += 1
                    
                    # Obtener partidos con retry logic
                    matches_data = self.get_match_ids(season, team_id)
                    
                    if matches_data and len(matches_data) > 0:
                        for match in matches_data:
                            match_id = match.get('match_id') # pyright: ignore[reportAttributeAccessIssue]
                            
                            match_exists = any(
                                existing_match.get('match_id') == match_id 
                                for existing_match in all_matches_data
                            )
                            
                            if not match_exists:
                                all_matches_data.append(match)
                                season_matches += 1
                        
                        print(f"      -> {len(matches_data)} partidos encontrados")
                    else:
                        teams_no_data += 1
                        print(f"      -> Sin datos")
                
                # Pausa entre lotes
                if i + batch_size < len(teams_from_db):
                    print(f"    Pausa entre lotes... (procesados {i + batch_size}/{len(teams_from_db)})")
                    time.sleep(3)
            
            print(f"\nResumen {season}:")
            print(f"  - Equipos procesados: {teams_processed}")
            print(f"  - Equipos sin datos: {teams_no_data}")
            print(f"  - Partidos únicos agregados: {season_matches}")
        
        return all_matches_data
    


    def team_stats_per_season(self, season_id: str) -> Optional[List[Dict]]:
        try:
            url = f"{self.base_url}/team-season-stats?league_id={self.league_id}&season_id={season_id}"
            print(f"Haciendo request a: {url}")

            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }

            payload = {}

            response = self.session.get(url, headers=headers, data=payload, timeout=30)

            print(f"Response Status Code: {response.status_code}")

            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    
                    # Los datos vienen en el campo 'data'
                    if isinstance(data, dict) and 'data' in data:
                        teams_data = data['data']
                    elif isinstance(data, list):
                        teams_data = data
                    else:
                        print("Formato de datos inesperado en la respuesta JSON")
                        return None

                    # Lista para almacenar todos los datos combinados
                    complete_teams_data = []

                    for team in teams_data:
                        # Inicializar el diccionario del equipo
                        team_complete_data = {}
                        
                        # Extraer meta_data
                        if 'meta_data' in team:
                            meta_data = team['meta_data']
                            team_complete_data.update({
                                'season_id': season_id,
                                'team_id': meta_data.get('team_id'),
                                'team_name': meta_data.get('team_name')
                            })
                        
                        # Extraer stats generales (navegando a través de stats.stats)
                        if 'stats' in team and 'stats' in team['stats']:
                            stats = team['stats']['stats']
                            
                            # Agregar estadísticas generales
                            team_complete_data.update({
                                'matches_played': stats.get('matches_played'),
                                'ttl_gls': stats.get('ttl_gls'),
                                'ttl_ast': stats.get('ttl_ast'),
                                'ttl_non_pen_gls': stats.get('ttl_non_pen_gls'),
                                'ttl_xg': stats.get('ttl_xg'),
                                'ttl_xag': stats.get('ttl_xag'),
                                'ttl_pk_made': stats.get('ttl_pk_made'),
                                'ttl_pk_att': stats.get('ttl_pk_att'),
                                'ttl_yellow_cards': stats.get('ttl_yellow_cards'),
                                'ttl_red_cards': stats.get('ttl_red_cards'),
                                'avg_gls': stats.get('avg_gls'),
                                'avg_ast': stats.get('avg_ast'),
                                'avg_non_pen_gls': stats.get('avg_non_pen_gls'),
                                'avg_xg': stats.get('avg_xg'),
                                'avg_xag': stats.get('avg_xag'),
                            })
                        
                        # Extraer estadísticas de porteros (keepers)
                        if 'stats' in team and 'keepers' in team['stats']:
                            keepers = team['stats']['keepers']
                            
                            # Agregar estadísticas de porteros con prefijo 'keeper_'
                            team_complete_data.update({
                                'ttl_gls_ag': keepers.get('ttl_gls_ag'),
                                'avg_gls_ag': keepers.get('avg_gls_ag'),
                                'sot_ag': keepers.get('sot_ag'),
                                'ttl_saves': keepers.get('ttl_saves'),
                                'clean_sheets': keepers.get('clean_sheets'),
                                'pk_att_ag': keepers.get('pk_att_ag'),
                                'ttl_pk_made_ag': keepers.get('pk_made_ag'),
                                'pk_saved': keepers.get('pk_saved'),
                            })
                        
                        # Extraser estadisticas de tiros
                        if 'stats' in team and 'shooting' in team['stats']:
                            shoting = team['stats']['shooting']
                            team_complete_data.update({
                                'ttl_sho': shoting.get('ttl_shot'),
                                'ttl_sot': shoting.get('ttl_sot'),
                                'pct_sot': shoting.get('pct_sot'),
                                'avg_sho': shoting.get('avg_shot'),
                                'gls_per_sot': shoting.get('gls_per_sot'),
                                'ttl_gls_xg_diff': shoting.get('ttl_gls_xg_diff'),
                            })

                        # Extraer estadisticas de pases
                        if 'stats' in team and 'passing' in team['stats']:
                            passing = team['stats']['passing']
                            team_complete_data.update({
                                'ttl_pass_cmp': passing.get('ttl_pass_cmp'),
                                'pct_pass_cmp': passing.get('pct_pass_cmp'),
                                'ttl_pass_prog': passing.get('ttl_pass_prog'),
                                'ttl_key_passes': passing.get('ttl_key_passes'),
                                'ttl_pass_opp_box': passing.get('ttl_pass_opp_box'),
                                'ttl_cross_opp_box': passing.get('ttl_cross_opp_box'),
                            })
                        
                        # Extraer estaditicas de tipo de pase
                        if 'stats' in team and 'passing_types' in team['stats']:
                            passing_types = team['stats']['passing_types']
                            team_complete_data.update({
                                'ttl_pass_live': passing_types.get('ttl_pass_live'),
                                'ttl_pass_dead': passing_types.get('ttl_pass_dead'),
                                'ttl_pass_fk': passing_types.get('ttl_pass_fk'),
                                'ttl_through_balls': passing_types.get('ttl_through_balls'),
                                'ttl_switches': passing_types.get('ttl_switches'),
                                'ttl_crosses': passing_types.get('ttl_crosses'),
                                'ttl_pass_offside': passing_types.get('ttl_pass_offside'),
                                'ttl_pass_blocked': passing_types.get('ttl_pass_blocked'),
                                'ttl_throw_ins': passing_types.get('ttl_throw_ins'),
                                'ttl_ck': passing_types.get('ttl_ck'),
                            })

                        #Extraer estadisticas defensivas
                        if 'stats' in team and 'defense' in team['stats']:
                            defense = team['stats']['defense']
                            team_complete_data.update({
                                'ttl_tkl': defense.get('ttl_tkl'),
                                'ttl_tkl_won': defense.get('ttl_tkl_won'),
                                'ttl_tkl_drb': defense.get('ttl_tkl_drb'),
                                'ttl_tkl_drb_att': defense.get('ttl_tkl_drb_att'),
                                'pct_tkl_drb_suc': defense.get('pct_tkl_drb_suc'),
                                'ttl_blocks': defense.get('ttl_blocks'),
                                'ttl_sh_blocked': defense.get('ttl_sh_blocked'),
                                'ttl_int': defense.get('ttl_int'),
                                'ttl_clearances': defense.get('ttl_clearances'),
                                'ttl_def_error': defense.get('ttl_def_error'),
                            })
                        
                        #Extraer estadisticas de posesion
                        if 'stats' in team and 'possession' in team['stats']:
                            possession = team['stats']['possession']
                            team_complete_data.update({
                                'avg_poss': possession.get('avg_poss'),
                                'ttl_touches': possession.get('ttl_touches'),
                                'ttl_take_on_att': possession.get('ttl_take_on_att'),
                                'ttl_take_on_suc': possession.get('ttl_take_on_suc'),
                                'ttl_carries': possession.get('ttl_carries'),
                                'ttl_carries_miscontrolled': possession.get('ttl_carries_miscontrolled'),
                                'ttl_carries_dispossessed': possession.get('ttl_carries_dispossessed'),
                                'ttl_pass_recvd': possession.get('ttl_pass_recvd'),
                                'ttl_pass_prog_rcvd': possession.get('ttl_pass_prog_rcvd'),
                            })

                        # Extraer estidisticas de playtime
                        if 'stats' in team and 'playtime' in team['stats']:
                            playtime = team['stats']['playingtime']
                            team_complete_data.update({
                                'avg_age': playtime.get('avg_age'),
                                'avg_min_starter': playtime.get('avg_min_starter'),
                            })
                        
                        # Extraer estidisticas de misc
                        if 'stats' in team and 'misc' in team['stats']:
                            misc = team['stats']['misc']
                            team_complete_data.update({
                                'ttl_fls_ag': misc.get('ttl_fls_ag'),
                                'ttl_fls_for': misc.get('ttl_fls_for'),
                                'ttl_offside': misc.get('ttl_offside'),
                                'ttl_og': misc.get('ttl_og'),
                                'ttl_ball_recov': misc.get('ttl_ball_recov'),
                                'ttl_air_dual_won': misc.get('ttl_air_dual_won'),
                                'ttl_air_dual_lost': misc.get('ttl_air_dual_lost'),
                            })
                        
                        # Solo agregar si tenemos al menos meta_data
                        if 'team_id' in team_complete_data:
                            complete_teams_data.append(team_complete_data)

                    return complete_teams_data
                
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en la API: {response.status_code}")
                print(f"Contenido del error: {response.text}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None
    

    def multiple_team_season_stats(self) -> List[Dict]:
        target_seasons = [
            '2017-2018', '2018-2019', '2019-2020', 
            '2020-2021', '2021-2022', '2022-2023', 
            '2023-2024', '2024-2025'
        ]
        
        all_team_stats_data = []
        
        print("Obteniendo estadísticas de equipos por temporada...")
        
        # Procesar por temporada con breaks entre temporadas
        for season_idx, season in enumerate(target_seasons):
            print(f"\n=== Obteniendo estadísticas para la temporada: {season} ===")
            season_stats = 0
            
            # Break más largo entre temporadas
            if season_idx > 0:
                print("Pausa entre temporadas...")
                time.sleep(5)
            
            # Obtener estadísticas de la temporada
            team_stats_data = self.team_stats_per_season(season)
            
            if team_stats_data and len(team_stats_data) > 0:
                # Agregar todas las estadísticas de la temporada
                for team_stat in team_stats_data:
                    all_team_stats_data.append(team_stat)
                    season_stats += 1
                
                print(f"  -> {len(team_stats_data)} equipos con estadísticas encontrados")
            else:
                print(f"  -> Sin datos de estadísticas para {season}")
            
            print(f"Resumen {season}:")
            print(f"  - Estadísticas de equipos agregadas: {season_stats}")
            
            # Pausa entre temporadas para evitar rate limiting
            if season_idx < len(target_seasons) - 1:  # No hacer pausa después de la última temporada
                print(f"Pausando antes de la siguiente temporada...")
                time.sleep(3)
        
        print(f"\n=== RESUMEN TOTAL ===")
        print(f"Total de registros de estadísticas obtenidos: {len(all_team_stats_data)}")
        
        return all_team_stats_data
        


###################################################################################

class DatabaseManager:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:

        try:
            self.connection = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                database=DatabaseConfig.DATABASE,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD
            )
            self.cursor = self.connection.cursor()
            print("Conexión a la base de datos exitosa")
            return True
        except psycopg2.Error as e:
            print(f"Error conectando a la base de datos: {e}")
            return False
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False

    def disconnect(self):
        """Desconectar de la base de datos"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("Desconectado de la base de datos")
        except Exception as e:
            print(f"Error al desconectar: {e}")

    #AQUI VAN LOS INSERTS 

    def insert_general_data(self, league__general_data: List[Dict]) -> int:

        try:
            inserted_count = 0
            
            for league in league__general_data:
                # Extraer campos según el formato de la API
                season_id = league.get('season_id')
                competition_name = league.get('competition_name')
                num_squads = league.get('#_squads')
                champion = league.get('champion')

                # Manejar top_scorer que puede ser un objeto complejo
                top_scorer_info = league.get('top_scorer', {})
                if top_scorer_info:
                    # Extraer el jugador (puede ser string o lista)
                    player = top_scorer_info.get('player')
                    goals_scored = top_scorer_info.get('goals_scored')
                    
                    # Si player es una lista, convertirla a string separado por comas
                    if isinstance(player, list):
                        top_scorer_player = ','.join(player)
                    else:
                        top_scorer_player = player
                else:
                    top_scorer_player = None
                    goals_scored = None
                
                insert_query = """
                INSERT INTO data_league
                (season_id, competition_name, squads_count, champion, top_scorer_player, top_scorer_goals, ingestion_time)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) 
                ON CONFLICT (season_id) DO UPDATE SET
                    competition_name = EXCLUDED.competition_name,
                    squads_count = EXCLUDED.squads_count,
                    champion = EXCLUDED.champion,
                    top_scorer_player = EXCLUDED.top_scorer_player,
                    top_scorer_goals = EXCLUDED.top_scorer_goals,
                    ingestion_time = CURRENT_TIMESTAMP
                """
                
                values = (season_id, competition_name, num_squads, champion, top_scorer_player, goals_scored)
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    inserted_count += 1
            
            if self.connection:
                self.connection.commit()
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando países: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0
        
    
    def insert_teams_data(self, teams_data: List[Dict]) -> int:
        try:
            inserted_count = 0
            
            for team in teams_data:
                team_id = team.get('team_id')
                team_name = team.get('team_name')
                
                insert_query = """
                INSERT INTO team_meta
                (team_id, team_name)
                VALUES (%s, %s) 
                ON CONFLICT (team_id) DO UPDATE SET
                    team_name = EXCLUDED.team_name;
                """
                
                values = (team_id, team_name)
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    inserted_count += 1
            
            if self.connection:
                self.connection.commit()
            print(f"Se insertaron {inserted_count} equipos exitosamente")
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando equipos: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0

  
    def insert_matches_data(self, matches_data: List[Dict]) -> int:
        try:
            inserted_count = 0
            updated_count = 0
            
            for match in matches_data:
                season_id = match.get('season_id')
                match_id = match.get('match_id')
                matchday = match.get('matchday', 1)  # Por defecto 1 si no está disponible
                date_game = match.get('date')
                team_id = match.get('team_id')
                # Necesitamos obtener team_name desde el match o desde la BD
                team_name = self.get_team_name_by_id(team_id)  # pyright: ignore[reportArgumentType] # Implementar esta función
                opponent = match.get('opponent')
                opponent_id = match.get('opponent_id')
                home_away = match.get('home_away')
                gf = match.get('gf')  # Goals for
                ga = match.get('ga')  # Goals against
                
                # Determinar si el equipo actual es local (Home) o visitante (Away)
                if home_away == 'Home':
                    # El equipo actual es local
                    home_team = team_name
                    home_team_id = team_id
                    home_team_score = gf
                    away_team = opponent
                    away_team_id = opponent_id
                    away_team_score = ga
                else:  # Away
                    # El equipo actual es visitante
                    home_team = opponent
                    home_team_id = opponent_id
                    home_team_score = ga
                    away_team = team_name
                    away_team_id = team_id
                    away_team_score = gf
                
                # Usar ON CONFLICT ahora que tienes UNIQUE(match_id)
                insert_query = """
                INSERT INTO matches_registered
                (season_id, match_id, matchday, home_team, home_team_id, home_team_score, away_team, away_team_id, away_team_score, date_game)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                ON CONFLICT (match_id) DO UPDATE SET
                    season_id = EXCLUDED.season_id,
                    matchday = EXCLUDED.matchday,
                    home_team = EXCLUDED.home_team,
                    home_team_id = EXCLUDED.home_team_id,
                    home_team_score = EXCLUDED.home_team_score,
                    away_team = EXCLUDED.away_team,
                    away_team_id = EXCLUDED.away_team_id,
                    away_team_score = EXCLUDED.away_team_score,
                    date_game = EXCLUDED.date_game
                """
                
                values = (season_id, match_id, matchday, home_team, home_team_id, home_team_score, 
                        away_team, away_team_id, away_team_score, date_game)
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    # Para distinguir entre insert y update, necesitamos verificar si existía
                    check_query = "SELECT COUNT(*) FROM matches_registered WHERE match_id = %s"
                    self.cursor.execute(check_query, (match_id,))
                    # Como usamos ON CONFLICT, siempre existirá después del INSERT
                    inserted_count += 1
        
            if self.connection:
                self.connection.commit()
            
            print(f"Se procesaron {inserted_count} partidos (insertados/actualizados)")
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando partidos: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0

    def get_team_name_by_id(self, team_id: str) -> str:
        """Obtiene el nombre del equipo desde la base de datos usando team_id"""
        try:
            if self.cursor:
                query = "SELECT team_name FROM team_meta WHERE team_id = %s"
                self.cursor.execute(query, (team_id,))
                result = self.cursor.fetchone()
                if result:
                    return result[0]
            return f"Team_{team_id}"  # Fallback si no se encuentra
        except Exception as e:
            print(f"Error obteniendo nombre del equipo {team_id}: {e}")
            return f"Team_{team_id}"
    

    def insert_team_stats_data(self, team_stats_data: List[Dict]) -> int:
        try:
            inserted_count = 0
            
            for team_stat in team_stats_data:
                # Extraer todos los campos del diccionario
                season_id = team_stat.get('season_id')
                team_id = team_stat.get('team_id')
                team_name = team_stat.get('team_name')
                
                # Estadísticas generales
                matches_played = team_stat.get('matches_played')
                ttl_gls = team_stat.get('ttl_gls')
                ttl_ast = team_stat.get('ttl_ast')
                ttl_non_pen_gls = team_stat.get('ttl_non_pen_gls')
                ttl_xg = team_stat.get('ttl_xg')
                ttl_xag = team_stat.get('ttl_xag')
                ttl_pk_made = team_stat.get('ttl_pk_made')
                ttl_pk_att = team_stat.get('ttl_pk_att')
                ttl_yellow_cards = team_stat.get('ttl_yellow_cards')
                ttl_red_cards = team_stat.get('ttl_red_cards')
                avg_gls = team_stat.get('avg_gls')
                avg_ast = team_stat.get('avg_ast')
                avg_non_pen_gls = team_stat.get('avg_non_pen_gls')
                avg_xg = team_stat.get('avg_xg')
                avg_xag = team_stat.get('avg_xag')
                
                # Estadísticas de porteros
                ttl_gls_ag = team_stat.get('ttl_gls_ag')
                avg_gls_ag = team_stat.get('avg_gls_ag')
                sot_ag = team_stat.get('sot_ag')
                ttl_saves = team_stat.get('ttl_saves')
                clean_sheets = team_stat.get('clean_sheets')
                pk_att_ag = team_stat.get('pk_att_ag')
                pk_made_ag = team_stat.get('ttl_pk_made_ag')
                pk_saved = team_stat.get('pk_saved')
                
                # Estadísticas de tiros
                ttl_sho = team_stat.get('ttl_sho')
                ttl_sot = team_stat.get('ttl_sot')
                pct_sot = team_stat.get('pct_sot')
                avg_sho = team_stat.get('avg_sho')
                gls_per_sot = team_stat.get('gls_per_sot')
                ttl_gls_xg_diff = team_stat.get('ttl_gls_xg_diff')
                
                # Estadísticas de pases
                ttl_pass_cmp = team_stat.get('ttl_pass_cmp')
                pct_pass_cmp = team_stat.get('pct_pass_cmp')
                ttl_pass_prog = team_stat.get('ttl_pass_prog')
                ttl_key_passes = team_stat.get('ttl_key_passes')
                ttl_pass_opp_box = team_stat.get('ttl_pass_opp_box')
                ttl_cross_opp_box = team_stat.get('ttl_cross_opp_box')
                
                # Estadísticas de tipos de pase
                ttl_pass_live = team_stat.get('ttl_pass_live')
                ttl_pass_dead = team_stat.get('ttl_pass_dead')
                ttl_pass_fk = team_stat.get('ttl_pass_fk')
                ttl_through_balls = team_stat.get('ttl_through_balls')
                ttl_switches = team_stat.get('ttl_switches')
                ttl_crosses = team_stat.get('ttl_crosses')
                ttl_pass_offsides = team_stat.get('ttl_pass_offside')
                ttl_pass_blocked = team_stat.get('ttl_pass_blocked')
                ttl_throw_ins = team_stat.get('ttl_throw_ins')
                ttl_cks = team_stat.get('ttl_cks')
                
                # Estadísticas defensivas
                ttl_tkl = team_stat.get('ttl_tkl')
                ttl_tkl_won = team_stat.get('ttl_tkl_won')
                ttl_tkl_drb = team_stat.get('ttl_tkl_drb')
                ttl_tkl_drb_att = team_stat.get('ttl_tkl_drb_att')
                pct_tkl_drb_suc = team_stat.get('pct_tkl_drb_suc')
                ttl_blocks = team_stat.get('ttl_blocks')
                ttl_sh_blocked = team_stat.get('ttl_sh_blocked')
                ttl_int = team_stat.get('ttl_int')
                ttl_clearances = team_stat.get('ttl_clearances')
                ttl_def_error = team_stat.get('ttl_def_error')
                
                # Estadísticas de posesión
                avg_poss = team_stat.get('avg_poss')
                ttl_touches = team_stat.get('ttl_touches')
                ttl_take_on_att = team_stat.get('ttl_take_on_att')
                ttl_take_on_suc = team_stat.get('ttl_take_on_suc')
                ttl_carries = team_stat.get('ttl_carries')
                ttl_carries_miscontrolled = team_stat.get('ttl_carries_miscontrolled')
                ttl_carries_dispossessed = team_stat.get('ttl_carries_dispossessed')
                ttl_pass_recvd = team_stat.get('ttl_pass_recvd')
                ttl_pass_prog_rcvd = team_stat.get('ttl_pass_prog_rcvd')
                
                # Estadísticas de tiempo de juego
                avg_age = team_stat.get('avg_age')
                avg_min_starter = team_stat.get('avg_min_starter')
                
                # Estadísticas misceláneas
                ttl_fls_ag = team_stat.get('ttl_fls_ag')
                ttl_fls_for = team_stat.get('ttl_fls_for')
                ttl_offside = team_stat.get('ttl_offside')
                ttl_og = team_stat.get('ttl_og')
                ttl_ball_recov = team_stat.get('ttl_ball_recov')
                ttl_air_dual_won = team_stat.get('ttl_air_dual_won')
                ttl_air_dual_lost = team_stat.get('ttl_air_dual_lost')
                
                # DEFINIR EL QUERY PRIMERO
                insert_query = """
                INSERT INTO team_season_stats
                (season_id, team_id, team_name, matches_played, ttl_gls, ttl_ast, ttl_non_pen_gls, 
                ttl_xg, ttl_xag, ttl_pk_made, ttl_pk_att, ttl_yellow_cards, ttl_red_cards,
                avg_gls, avg_ast, avg_non_pen_gls, avg_xg, avg_xag,
                ttl_gls_ag, avg_gls_ag, sot_ag, ttl_saves, clean_sheets, pk_att_ag, pk_made_ag, pk_saved,
                ttl_sho, ttl_sot, pct_sot, avg_sho, gls_per_sot, ttl_gls_xg_diff,
                ttl_pass_cmp, pct_pass_cmp, ttl_pass_prog, ttl_key_passes, ttl_pass_opp_box, ttl_cross_opp_box,
                ttl_pass_live, ttl_pass_dead, ttl_pass_fk, ttl_through_balls, ttl_switches, ttl_crosses,
                ttl_pass_offsides, ttl_pass_blocked, ttl_throw_ins, ttl_cks,
                ttl_tkl, ttl_tkl_won, ttl_tkl_drb, ttl_tkl_drb_att, pct_tkl_drb_suc, ttl_blocks,
                ttl_sh_blocked, ttl_int, ttl_clearances, ttl_def_error,
                avg_poss, ttl_touches, ttl_take_on_att, ttl_take_on_suc, ttl_carries,
                ttl_carries_miscontrolled, ttl_carries_dispossessed, ttl_pass_recvd, ttl_pass_prog_rcvd,
                avg_age, avg_min_starter,
                ttl_fls_ag, ttl_fls_for, ttl_offside, ttl_og, ttl_ball_recov, ttl_air_dual_won, ttl_air_dual_lost)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (season_id, team_id) DO UPDATE SET
                    team_name = EXCLUDED.team_name,
                    matches_played = EXCLUDED.matches_played,
                    ttl_gls = EXCLUDED.ttl_gls,
                    ttl_ast = EXCLUDED.ttl_ast,
                    ttl_non_pen_gls = EXCLUDED.ttl_non_pen_gls,
                    ttl_xg = EXCLUDED.ttl_xg,
                    ttl_xag = EXCLUDED.ttl_xag,
                    ttl_pk_made = EXCLUDED.ttl_pk_made,
                    ttl_pk_att = EXCLUDED.ttl_pk_att,
                    ttl_yellow_cards = EXCLUDED.ttl_yellow_cards,
                    ttl_red_cards = EXCLUDED.ttl_red_cards,
                    avg_gls = EXCLUDED.avg_gls,
                    avg_ast = EXCLUDED.avg_ast,
                    avg_non_pen_gls = EXCLUDED.avg_non_pen_gls,
                    avg_xg = EXCLUDED.avg_xg,
                    avg_xag = EXCLUDED.avg_xag,
                    ttl_gls_ag = EXCLUDED.ttl_gls_ag,
                    avg_gls_ag = EXCLUDED.avg_gls_ag,
                    sot_ag = EXCLUDED.sot_ag,
                    ttl_saves = EXCLUDED.ttl_saves,
                    clean_sheets = EXCLUDED.clean_sheets,
                    pk_att_ag = EXCLUDED.pk_att_ag,
                    pk_made_ag = EXCLUDED.pk_made_ag,
                    pk_saved = EXCLUDED.pk_saved,
                    ttl_sho = EXCLUDED.ttl_sho,
                    ttl_sot = EXCLUDED.ttl_sot,
                    pct_sot = EXCLUDED.pct_sot,
                    avg_sho = EXCLUDED.avg_sho,
                    gls_per_sot = EXCLUDED.gls_per_sot,
                    ttl_gls_xg_diff = EXCLUDED.ttl_gls_xg_diff,
                    ttl_pass_cmp = EXCLUDED.ttl_pass_cmp,
                    pct_pass_cmp = EXCLUDED.pct_pass_cmp,
                    ttl_pass_prog = EXCLUDED.ttl_pass_prog,
                    ttl_key_passes = EXCLUDED.ttl_key_passes,
                    ttl_pass_opp_box = EXCLUDED.ttl_pass_opp_box,
                    ttl_cross_opp_box = EXCLUDED.ttl_cross_opp_box,
                    ttl_pass_live = EXCLUDED.ttl_pass_live,
                    ttl_pass_dead = EXCLUDED.ttl_pass_dead,
                    ttl_pass_fk = EXCLUDED.ttl_pass_fk,
                    ttl_through_balls = EXCLUDED.ttl_through_balls,
                    ttl_switches = EXCLUDED.ttl_switches,
                    ttl_crosses = EXCLUDED.ttl_crosses,
                    ttl_pass_offsides = EXCLUDED.ttl_pass_offsides,
                    ttl_pass_blocked = EXCLUDED.ttl_pass_blocked,
                    ttl_throw_ins = EXCLUDED.ttl_throw_ins,
                    ttl_cks = EXCLUDED.ttl_cks,
                    ttl_tkl = EXCLUDED.ttl_tkl,
                    ttl_tkl_won = EXCLUDED.ttl_tkl_won,
                    ttl_tkl_drb = EXCLUDED.ttl_tkl_drb,
                    ttl_tkl_drb_att = EXCLUDED.ttl_tkl_drb_att,
                    pct_tkl_drb_suc = EXCLUDED.pct_tkl_drb_suc,
                    ttl_blocks = EXCLUDED.ttl_blocks,
                    ttl_sh_blocked = EXCLUDED.ttl_sh_blocked,
                    ttl_int = EXCLUDED.ttl_int,
                    ttl_clearances = EXCLUDED.ttl_clearances,
                    ttl_def_error = EXCLUDED.ttl_def_error,
                    avg_poss = EXCLUDED.avg_poss,
                    ttl_touches = EXCLUDED.ttl_touches,
                    ttl_take_on_att = EXCLUDED.ttl_take_on_att,
                    ttl_take_on_suc = EXCLUDED.ttl_take_on_suc,
                    ttl_carries = EXCLUDED.ttl_carries,
                    ttl_carries_miscontrolled = EXCLUDED.ttl_carries_miscontrolled,
                    ttl_carries_dispossessed = EXCLUDED.ttl_carries_dispossessed,
                    ttl_pass_recvd = EXCLUDED.ttl_pass_recvd,
                    ttl_pass_prog_rcvd = EXCLUDED.ttl_pass_prog_rcvd,
                    avg_age = EXCLUDED.avg_age,
                    avg_min_starter = EXCLUDED.avg_min_starter,
                    ttl_fls_ag = EXCLUDED.ttl_fls_ag,
                    ttl_fls_for = EXCLUDED.ttl_fls_for,
                    ttl_offside = EXCLUDED.ttl_offside,
                    ttl_og = EXCLUDED.ttl_og,
                    ttl_ball_recov = EXCLUDED.ttl_ball_recov,
                    ttl_air_dual_won = EXCLUDED.ttl_air_dual_won,
                    ttl_air_dual_lost = EXCLUDED.ttl_air_dual_lost;
                """
                
                # Crear tupla con exactamente 72 valores (sin duplicar values)
                values = (
                    season_id, team_id, team_name, matches_played, ttl_gls, ttl_ast, ttl_non_pen_gls,
                    ttl_xg, ttl_xag, ttl_pk_made, ttl_pk_att, ttl_yellow_cards, ttl_red_cards,
                    avg_gls, avg_ast, avg_non_pen_gls, avg_xg, avg_xag,
                    ttl_gls_ag, avg_gls_ag, sot_ag, ttl_saves, clean_sheets, pk_att_ag, pk_made_ag, pk_saved,
                    ttl_sho, ttl_sot, pct_sot, avg_sho, gls_per_sot, ttl_gls_xg_diff,
                    ttl_pass_cmp, pct_pass_cmp, ttl_pass_prog, ttl_key_passes, ttl_pass_opp_box, ttl_cross_opp_box,
                    ttl_pass_live, ttl_pass_dead, ttl_pass_fk, ttl_through_balls, ttl_switches, ttl_crosses,
                    ttl_pass_offsides, ttl_pass_blocked, ttl_throw_ins, ttl_cks,
                    ttl_tkl, ttl_tkl_won, ttl_tkl_drb, ttl_tkl_drb_att, pct_tkl_drb_suc, ttl_blocks,
                    ttl_sh_blocked, ttl_int, ttl_clearances, ttl_def_error,
                    avg_poss, ttl_touches, ttl_take_on_att, ttl_take_on_suc, ttl_carries,
                    ttl_carries_miscontrolled, ttl_carries_dispossessed, ttl_pass_recvd, ttl_pass_prog_rcvd,
                    avg_age, avg_min_starter,
                    ttl_fls_ag, ttl_fls_for, ttl_offside, ttl_og, ttl_ball_recov, ttl_air_dual_won, ttl_air_dual_lost
                )
                
                # Debug
                print(f"DEBUG: Número exacto de valores: {len(values)}")
                print(f"DEBUG: Equipo: {team_name}")
                placeholders_count = insert_query.count('%s')
                print(f"DEBUG: Número de placeholders %s: {placeholders_count}")
                
                if len(values) != placeholders_count:
                    print(f"ERROR: Desbalance - {len(values)} valores vs {placeholders_count} placeholders")
                    continue
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    inserted_count += 1
            
            if self.connection:
                self.connection.commit()
            
            print(f"Se procesaron {inserted_count} registros de estadísticas de equipos")
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando estadísticas de equipos: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0
        
###################################################################################

class MenuController:

    def __init__(self):
        self.api_client = FBRAPIClient()
        self.db_manager = DatabaseManager()
    
    def show_menu(self):
        print("\n" + "=" * 60)
        print("Mandar los datos a la base de datos sobre la Premier League - API FBR")
        print("=" * 60)
        print("1. Probar conexión a base de datos")
        print("2. Obtener y guardar datos de los datos generales de la Premier League")
        print("3. Obtener y guardar datos de los equipos de la Premier League (2017-2024)")
        print("4. Obtener y guardar partidos de todas las temporadas (2017-2024)")
        print("5. Obtener y guardar estadísticas de equipos por temporada (2017-2024)")
        print("0. Salir")
        print("=" * 60)

    def handle_test_db_connection(self):
        if self.db_manager.connect():
            print("Conexión exitosa a la base de datos")
            self.db_manager.disconnect()
        else:
            print("Error conectando a la base de datos")

    def handle_datos_generales(self):

        #Verificar conexión a BD
        if not self.db_manager.connect():
            print("No se puede continuar sin conexión a la base de datos")
            return
        
        # Obtener datos de la API
        league_general_data = self.api_client.get_general_data_league()

        if not league_general_data:
            print("No se obtuvieron datos de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(league_general_data, list):
            print("Formato de datos inesperado, se esperaba una lista")
            self.db_manager.disconnect()
            return

        # Insertar datos en la base de datos
        self.db_manager.insert_general_data(league_general_data)


        self.db_manager.disconnect()
    
    def handle_data_teams_meta(self):

        #Verificar conexion a BD
        if not self.db_manager.connect():
            print("No se puede continuar sin conexión a la base de datos")
            return
        
        # Obtener datos de la API
        teams_data = self.api_client.multiple_seasons()

        if not teams_data:
            print("No se obtuvieron datos de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(teams_data, list):
            print("Formato de datos inesperado, se esperaba una lista")
            self.db_manager.disconnect()
            return
        
        # Insertar datos en la base de datos
        self.db_manager.insert_teams_data(teams_data)

        self.db_manager.disconnect()
    


    def handle_data_matches(self):
        """Maneja la obtención y guardado de datos de partidos"""
        
        # Verificar conexión a BD
        if not self.db_manager.connect():
            print("No se puede continuar sin conexión a la base de datos")
            return
        
        # Obtener datos de partidos de la API pasando el db_manager
        matches_data = self.api_client.multiple_seasons_matches(self.db_manager)
        
        if not matches_data:
            print("No se obtuvieron datos de partidos de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(matches_data, list):
            print("Formato de datos inesperado, se esperaba una lista")
            self.db_manager.disconnect()
            return
        
        # Insertar datos en la base de datos
        self.db_manager.insert_matches_data(matches_data)
        
        self.db_manager.disconnect()

    def handle_team_season_stats(self):
        """Maneja la obtención y guardado de estadísticas de equipos por temporada"""
        
        # Verificar conexión a BD
        if not self.db_manager.connect():
            print("No se puede continuar sin conexión a la base de datos")
            return
        
        # Obtener datos de estadísticas de equipos de la API
        team_stats_data = self.api_client.multiple_team_season_stats()
        
        if not team_stats_data:
            print("No se obtuvieron datos de estadísticas de equipos de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(team_stats_data, list):
            print("Formato de datos inesperado, se esperaba una lista")
            self.db_manager.disconnect()
            return
        
        # Insertar datos en la base de datos
        self.db_manager.insert_team_stats_data(team_stats_data)
        
        self.db_manager.disconnect()

    def run(self):
        # Es el menú principal
        if not self.api_client.api_key:
            print("No se encontró API key. Por favor, genere o cargue una API key primero.")
            return
        
        while True:
            self.show_menu()
            choice = input("Seleccione una opción: ").strip()
            
            if choice == "1":
                self.handle_test_db_connection()
            elif choice == "2":
                self.handle_datos_generales()
            elif choice == "3":
                self.handle_data_teams_meta()
            elif choice == "4":
                self.handle_data_matches()
            elif choice == "5":
                self.handle_team_season_stats()
            elif choice == "0":
                print("Saliendo...")
                break
            else:
                print("Opción inválida, intente de nuevo.")


###################################################################################

def main():

    controller = MenuController()
    controller.run()

if __name__ == "__main__":
    main()
