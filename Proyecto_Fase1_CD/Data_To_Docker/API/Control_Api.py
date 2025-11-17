import requests
import psycopg2
import json
import os
import sys
import subprocess
from typing import Optional, List, Dict, Any
from datetime import datetime

###################################################################################

class DatabaseConfig:
    HOST = "localhost"
    PORT = 5432
    DATABASE = "ProyectoUNITEC"
    USER = "admin"
    PASSWORD = "GadumUNITEC123"

###################################################################################

class FBRAPIClient:
    def __init__(self):
        self.base_url = "https://fbrapi.com"
        self.api_key = None
        self.session = requests.Session()
        self.load_api_key()
    
    #Load API key from file
    def load_api_key(self) -> bool:
        """Carga la API key desde el archivo de configuración"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "api_config.txt")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read().strip()
                    if content.startswith("FBR_API_KEY="):
                        self.api_key = content.split("=", 1)[1]
                        print(f"API Key cargada correctamente")
                        return True
            print("No se encontró archivo de configuración de API")
            return False
        except Exception as e:
            print(f"Error cargando API key: {e}")
            return False
    
    # Get countries from API
    def get_countries(self) -> Optional[List[Dict]]:
        """Obtiene la lista de países desde la API"""
        
        try:
            url = f"{self.base_url}/countries"            
            print(f"Request a: {url}")
            
            # Usar x-api-key en headers
            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }
            
            # Sin parámetros adicionales
            payload = {}
            
            response = self.session.get(url, headers=headers, data=payload, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    
                    # Los datos vienen en el campo "data" según el formato proporcionado
                    if isinstance(data, dict) and 'data' in data:
                        countries = data['data']
                    elif isinstance(data, list):
                        countries = data
                    else:
                        print("Formato de respuesta inesperado")
                        return None
                    
                    print(f"{len(countries)} países obtenidos")
                    return countries
                    
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en API: {response.status_code}")
                print(f"Respuesta: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None
        

    # Siguientes métodos API
    
    def get_leagues_by_country(self, country_code: str) -> Optional[List[Dict]]:
        """Obtiene las ligas de un país específico desde la API"""
        
        try:
            url = f"{self.base_url}/leagues"            
            print(f"Request a: {url} con country_code: {country_code}")
            
            # Usar x-api-key en headers
            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json'
            }
            
            # Parámetros de query con el código de país
            params = {'country_code': country_code}
            
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    
                    # Los datos vienen en el campo "data" según el formato proporcionado
                    if isinstance(data, dict) and 'data' in data:
                        leagues_data = data['data']
                        
                        # Extraer todas las ligas de todos los tipos
                        all_leagues = []
                        for category in leagues_data:
                            if 'leagues' in category and isinstance(category['leagues'], list):
                                all_leagues.extend(category['leagues'])
                        
                        print(f"{len(all_leagues)} ligas obtenidas para {country_code}")
                        return all_leagues
                    else:
                        print("Formato de respuesta inesperado")
                        return None
                    
                except json.JSONDecodeError as e:
                    print(f"Error parseando JSON: {e}")
                    print(f"Respuesta: {response.text}")
                    return None
            else:
                print(f"Error en API: {response.status_code}")
                print(f"Respuesta: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None
    
    def get_leagues_for_multiple_countries(self) -> List[Dict]:
        """Obtiene ligas para los países especificados: ENG, GER, FRA, ITA, ESP, HON"""
        
        target_countries = ['ENG', 'GER', 'FRA', 'ITA', 'ESP', 'HON']
        all_leagues = []
        
        for country_code in target_countries:
            print(f"\nObteniendo ligas para {country_code}...")
            leagues = self.get_leagues_by_country(country_code)
            
            if leagues:
                # Agregar el country_code a cada liga para referencia
                for league in leagues:
                    league['country_code'] = country_code
                all_leagues.extend(leagues)
            else:
                print(f"No se obtuvieron ligas para {country_code}")
        
        return all_leagues 


class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establece conexión con la base de datos PostgreSQL"""
        
        try:
            self.connection = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                database=DatabaseConfig.DATABASE,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD
            )
            
            self.cursor = self.connection.cursor()
            print("Conexión a PostgreSQL establecida")
            return True
            
        except psycopg2.Error as e:
            print(f"Error conectando a PostgreSQL: {e}")
            return False
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False
    
    # Create Tables bools
    def create_countries_table(self) -> bool:
        """Crea la tabla para almacenar países si no existe"""
        
        try:
            # Borrar la tabla existente si tiene problemas de tipo y recrearla
            drop_table_query = "DROP TABLE IF EXISTS countries;"
            
            create_table_query = """
            CREATE TABLE countries (
                id SERIAL PRIMARY KEY,
                country_code VARCHAR(10) NOT NULL UNIQUE,
                country VARCHAR(255) NOT NULL,
                governing_body VARCHAR(50),
                num_clubs INTEGER,
                num_players INTEGER,
                national_teams TEXT,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Ejecutar ambas consultas
            if self.cursor and self.connection:
                self.cursor.execute(drop_table_query)
                self.cursor.execute(create_table_query)
                self.connection.commit()
                print("Tabla 'countries' creada/verificada")
                return True
            else:
                print("Error: No hay conexión activa")
                return False
            
        except psycopg2.Error as e:
            print(f"Error creando tabla: {e}")
            return False
    
    #Siguientes metodos de Table
    
    def create_leagues_table(self) -> bool:
        """Crea la tabla para almacenar ligas si no existe"""
        
        try:
            # Borrar la tabla existente si tiene problemas de tipo y recrearla
            drop_table_query = "DROP TABLE IF EXISTS leagues;"
            
            create_table_query = """
            CREATE TABLE leagues (
                id SERIAL PRIMARY KEY,
                league_id INTEGER NOT NULL UNIQUE,
                competition_name VARCHAR(255) NOT NULL,
                country_code VARCHAR(10),
                gender VARCHAR(1),
                first_season VARCHAR(20),
                last_season VARCHAR(20),
                tier VARCHAR(10),
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Ejecutar ambas consultas
            if self.cursor and self.connection:
                self.cursor.execute(drop_table_query)
                self.cursor.execute(create_table_query)
                self.connection.commit()
                print("Tabla 'leagues' creada/verificada")
                return True
            else:
                print("Error: No hay conexión activa")
                return False
            
        except psycopg2.Error as e:
            print(f"Error creando tabla leagues: {e}")
            return False

    # Los inserts 
    def insert_countries(self, countries_data: List[Dict]) -> int:
        """Inserta datos de países en la base de datos"""

        try:
            inserted_count = 0
            
            for country in countries_data:
                # Extraer campos según el nuevo formato de la API
                country_name = country.get('country')
                country_code = country.get('country_code')
                governing_body = country.get('governing_body')
                num_clubs = country.get('#_clubs')
                num_players = country.get('#_players')
                
                # Convertir la lista de national_teams a string separado por comas
                national_teams_list = country.get('national_teams', [])
                national_teams = ','.join(national_teams_list) if national_teams_list else None
                
                insert_query = """
                INSERT INTO countries 
                (country_code, country, governing_body, num_clubs, num_players, national_teams)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_code) DO UPDATE SET
                    country = EXCLUDED.country,
                    governing_body = EXCLUDED.governing_body,
                    num_clubs = EXCLUDED.num_clubs,
                    num_players = EXCLUDED.num_players,
                    national_teams = EXCLUDED.national_teams,
                    ingestion_time = CURRENT_TIMESTAMP
                """
                
                values = (
                    country_code, country_name, governing_body, 
                    num_clubs, num_players, national_teams
                )
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    inserted_count += 1
            
            if self.connection:
                self.connection.commit()
            print(f"{inserted_count} países insertados en la base de datos")
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando países: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0
        
    # Siguientes Inserts
    
    def insert_leagues(self, leagues_data: List[Dict]) -> int:
        """Inserta datos de ligas en la base de datos"""

        try:
            inserted_count = 0
            
            for league in leagues_data:
                # Extraer campos según el formato de la API
                league_id = league.get('league_id')
                competition_name = league.get('competition_name')
                country_code = league.get('country_code')  # Lo agregamos nosotros
                gender = league.get('gender')
                first_season = league.get('first_season')
                last_season = league.get('last_season')
                tier = league.get('tier')
                
                insert_query = """
                INSERT INTO leagues 
                (league_id, competition_name, country_code, gender, first_season, last_season, tier)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (league_id) DO UPDATE SET
                    competition_name = EXCLUDED.competition_name,
                    country_code = EXCLUDED.country_code,
                    gender = EXCLUDED.gender,
                    first_season = EXCLUDED.first_season,
                    last_season = EXCLUDED.last_season,
                    tier = EXCLUDED.tier,
                    ingestion_time = CURRENT_TIMESTAMP
                """
                
                values = (
                    league_id, competition_name, country_code, 
                    gender, first_season, last_season, tier
                )
                
                if self.cursor:
                    self.cursor.execute(insert_query, values)
                    inserted_count += 1
            
            if self.connection:
                self.connection.commit()
            print(f"{inserted_count} ligas insertadas en la base de datos")
            return inserted_count
            
        except psycopg2.Error as e:
            print(f"Error insertando ligas: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
        except Exception as e:
            print(f"Error inesperado: {e}")
            return 0
    
    def get_leagues_count(self) -> int:
        """Obtiene el número total de ligas en la base de datos"""
        
        try:
            if self.cursor:
                self.cursor.execute("SELECT COUNT(*) FROM leagues")
                result = self.cursor.fetchone()
                if result:
                    return result[0]
            return 0
        except:
            return 0
    
    def get_countries_count(self) -> int:
        """Obtiene el número total de países en la base de datos"""
        
        try:
            if self.cursor:
                self.cursor.execute("SELECT COUNT(*) FROM countries")
                result = self.cursor.fetchone()
                if result:
                    return result[0]
            return 0
        except:
            return 0
    

    
    def disconnect(self):
        """Cierra la conexión con la base de datos"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Desconectado de PostgreSQL")

class MenuController:
    """Controlador del menú principal"""
    
    def __init__(self):
        self.api_client = FBRAPIClient()
        self.db_manager = DatabaseManager()
    
    def show_menu(self):
        """Muestra el menú principal"""
        print("\n" + "=" * 60)
        print("SISTEMA DE GESTIÓN DE PAÍSES Y LIGAS - API FBR")
        print("=" * 60)
        print("1. Probar conexión a base de datos")
        print("2. Generar nueva API Key")
        print("3. Obtener y guardar datos de países")
        print("4. Obtener y guardar datos de ligas")
        print("0. Salir")
        print("=" * 60)
    
    def handle_countries_option(self):
        """Maneja la opción de obtener datos de países"""
        print("\nIniciando proceso de obtención de países...")
        
        # Verificar conexión a BD
        if not self.db_manager.connect():
            print("No se pudo conectar a la base de datos")
            return
        
        # Crear tabla si no existe
        if not self.db_manager.create_countries_table():
            print("No se pudo crear/verificar la tabla")
            self.db_manager.disconnect()
            return
        
        # Obtener datos de la API
        countries = self.api_client.get_countries()
        
        if not countries:
            print("No se pudieron obtener datos de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(countries, list):
            print("Error: Los datos obtenidos no tienen el formato esperado")
            self.db_manager.disconnect()
            return
        
        
        if len(countries) > 3:
            print(f"   ... y {len(countries) - 3} países más")
        
        # Guardar en base de datos
        inserted = self.db_manager.insert_countries(countries)
        
        
        self.db_manager.disconnect()
    
    def handle_leagues_option(self):
        """Maneja la opción de obtener datos de ligas"""
        print("\nIniciando proceso de obtención de ligas...")
        
        # Verificar conexión a BD
        if not self.db_manager.connect():
            print("No se pudo conectar a la base de datos")
            return
        
        # Crear tabla si no existe
        if not self.db_manager.create_leagues_table():
            print("No se pudo crear/verificar la tabla de ligas")
            self.db_manager.disconnect()
            return
        
        # Obtener datos de la API para múltiples países
        leagues = self.api_client.get_leagues_for_multiple_countries()
        
        if not leagues:
            print("No se pudieron obtener datos de ligas de la API")
            self.db_manager.disconnect()
            return
        
        if not isinstance(leagues, list):
            print("Error: Los datos de ligas obtenidos no tienen el formato esperado")
            self.db_manager.disconnect()
            return
        
        # Mostrar muestra de datos
        print(f"\nMuestra de ligas obtenidas:")
        for i, league in enumerate(leagues[:5]):
            print(f"   {i+1}. {league.get('competition_name')} ({league.get('country_code')}) - ID: {league.get('league_id')}")
        
        if len(leagues) > 5:
            print(f"   ... y {len(leagues) - 5} ligas más")
        
        # Guardar en base de datos
        inserted = self.db_manager.insert_leagues(leagues)
        
        # Mostrar estadísticas
        total_leagues = self.db_manager.get_leagues_count()
        print(f"\nEstadísticas finales:")
        print(f"   - Ligas obtenidas de API: {len(leagues)}")
        print(f"   - Ligas insertadas: {inserted}")
        print(f"   - Total en base de datos: {total_leagues}")
        
        self.db_manager.disconnect()
    
    def handle_test_connection(self):
        # Prueba la conexión a la base de datos
        
        if self.db_manager.connect():
            print("Conexión exitosa!")
            self.db_manager.disconnect()
        else:
            print("Fallo en la conexión")
    
    def handle_generate_api_key(self):
        # Ejecuta el script GenerarApiKey.py
        
        try:
            # Buscar el archivo GenerarApiKey.py
            current_dir = os.path.dirname(__file__)
            
            # Verificar posibles nombres del archivo
            possible_files = [
                os.path.join(current_dir, "GenerarApiKey.py"),
            ]
            
            script_path = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    script_path = file_path
                    break
            
            if not script_path:
                print("No se encontró el archivo GenerarApiKey.py")
                return
            
            print(f"Ejecutando: {os.path.basename(script_path)}")
            
            # Ejecutar el script usando el mismo intérprete de Python
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, script_path],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Mostrar la salida
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print(result.stderr)
            
            if result.returncode == 0:
                print("Script ejecutado exitosamente!")
                # Recargar la API key después de generar una nueva
                self.api_client.load_api_key()
            else:
                print(f"El script terminó con código de error: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("Timeout: El script tardó demasiado en ejecutarse")
        except Exception as e:
            print(f"Error ejecutando el script: {e}")
    
    def run(self):
        # Ejecuta el menú principal
        
        if not self.api_client.api_key:
            print("No se encontró API key. Usa la opción 2 para generar una nueva.")
        
        while True:
            try:
                self.show_menu()
                
                choice = input("\nSelecciona una opción: ").strip()
                
                if choice == "1":
                    self.handle_test_connection()
                elif choice == "2":
                    self.handle_generate_api_key()
                elif choice == "3":
                    self.handle_countries_option()
                elif choice == "4":
                    self.handle_leagues_option()
                elif choice == "0":
                    break
                else:
                    print("Opción no válida. Intenta de nuevo.")
                
                input("\nPresiona Enter para continuar...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error inesperado: {e}")
                input("\nPresiona Enter para continuar...")

def main():
    """Función principal"""
    controller = MenuController()
    controller.run()

if __name__ == "__main__":
    main()
