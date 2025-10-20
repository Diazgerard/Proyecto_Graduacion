import requests
import json
import os
from typing import Optional, Dict, Any

class FBRAPIHandler:
    def __init__(self):
        self.base_url = "https://fbrapi.com"
        self.api_key = None
        self.session = requests.Session()
        
    def generate_api_key(self) -> Optional[str]:
        
        try:
            url = f"{self.base_url}/generate_api_key"
            
            # Headers básicos para el request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Intentar primero con POST (según el error 405)
            response = self.session.post(url, headers=headers, timeout=30)
            
            
            if response.status_code in [200, 201]:
                # Intentar parsear como JSON
                try:
                    data = response.json()
                    print(f"Respuesta JSON: {data}")
                    
                    # Buscar la API key en diferentes campos posibles
                    api_key = None
                    possible_keys = ['api_key', 'apiKey', 'key', 'token', 'access_token']
                    
                    for key in possible_keys:
                        if key in data:
                            api_key = data[key]
                            break
                    
                    if api_key:
                        self.api_key = api_key
                        return api_key
                    else:
                        print("No se encontró API key en la respuesta JSON")
                        return None
                        
                except json.JSONDecodeError:
                    # Si no es JSON, mostrar el contenido como texto
                    text_content = response.text
                    
                    
                    return text_content
            else:
                print(f"Error en el request: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None
    
    def save_api_key(self, api_key: str, filename: str = "api_config.txt") -> bool:

        try:
            config_path = os.path.join(os.path.dirname(__file__), filename)
            with open(config_path, 'w') as f:
                f.write(f"FBR_API_KEY={api_key}\n")
            return True
        except Exception as e:
            print(f"Error guardando API key: {e}")
            return False
    
    def load_api_key(self, filename: str = "api_config.txt") -> Optional[str]:
        try:
            config_path = os.path.join(os.path.dirname(__file__), filename)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read().strip()
                    if content.startswith("FBR_API_KEY="):
                        api_key = content.split("=", 1)[1]
                        self.api_key = api_key
                        return api_key
            return None
        except Exception as e:
            print(f"Error cargando API key: {e}")
            return None

def main():
    print("*** Generador de API Key para FBR ***")
    print("=" * 40)
    
    # Crear instancia del manejador
    api_handler = FBRAPIHandler()
    
    # Intentar cargar API key existente
    existing_key = api_handler.load_api_key()
    
    if existing_key:
        print(f"API Key existente encontrada: {existing_key}")
        respuesta = input("¿Quieres generar una nueva API key? (s/n): ").lower().strip()
        if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
            print("Usando API key existente")
            return
    
    # Generar nueva API key
    print("\nGenerando nueva API key...")
    api_key = api_handler.generate_api_key()
    
    if api_key:
        print(f"\nExito! API Key obtenida: {api_key}")
        
        # Guardar la API key
        if api_handler.save_api_key(api_key):
            print("API Key guardada exitosamente en api_config.txt")
        else:
            print("No se pudo guardar la API key")
    else:
        print("No se pudo obtener la API key")
    
if __name__ == "__main__":
    main()
