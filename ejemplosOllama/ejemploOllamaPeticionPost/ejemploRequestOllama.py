import requests
import json

# Configurar servidor local que usa ollama por defecto
url = "http://localhost:11434/api/chat"

# Define payload 
payload = {
    "model": "llama3.2",  
    "messages": [
        {
            "role": "user",
            "content": "Generate code for a database used in a login system. Include table creation (e.g., for users with username and password) and example SQL queries for inserting and retrieving user data."
        }
    ]
}

# enviar peticion post con streaming habilitado
try:
    response = requests.post(url, json=payload, stream=True)
    
    # Check response status
    if response.status_code == 200:
        print("Streaming response from Ollama:")
        for line in response.iter_lines(decode_unicode=True):
            if line:  # Ignore empty lines
                try:
                    # Parse each line into JSON
                    json_data = json.loads(line)
                    # Extrer e imprimir la respuesta
                    if "message" in json_data and "content" in json_data["message"]:
                        print(json_data["message"]["content"], end="", flush=True)  
                except json.JSONDecodeError as e:
                    print(f"\nFailed to parse line: {line} - Error: {e}")
        print()  
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")