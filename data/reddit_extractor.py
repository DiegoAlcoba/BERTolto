import praw
import os
import csv
import time
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración ---
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

SUBREDDITS_SELECCIONADOS = ['learnpython', 'datascience', 'programming']  # Puedes cambiar o añadir más
LIMITE_PUBLICACIONES = 50  # Puedes aumentar este límite si lo deseas
ARCHIVO_SALIDA = 'comentarios_reddit.csv'


# --- Script Principal ---

def inicializar_reddit():
    """Inicializa y devuelve una instancia autenticada de PRAW."""
    if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
        print("Error: Asegúrate de que las variables de entorno de Reddit están en tu archivo .env")
        return None
    return praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)


def extraer_comentarios_a_csv(reddit, subreddits):
    """Extrae comentarios y los guarda progresivamente en un archivo CSV."""
    print(f"Iniciando la extracción. Los datos se guardarán en '{ARCHIVO_SALIDA}'")

    try:
        with open(ARCHIVO_SALIDA, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            # Escribir la cabecera del CSV
            csv_writer.writerow(
                ['Subreddit', 'ID_Publicacion', 'Titulo_Publicacion', 'Autor_Comentario', 'Cuerpo_Comentario'])

            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    print(f"\n--- Procesando Subreddit: r/{subreddit.display_name} ---")

                    for submission in subreddit.hot(limit=LIMITE_PUBLICACIONES):
                        print(f"  -> Publicación: {submission.title[:60]}...")
                        submission.comments.replace_more(limit=0)

                        for comment in submission.comments.list():
                            if comment.author:  # Solo procesar comentarios con autor
                                csv_writer.writerow([
                                    subreddit.display_name,
                                    submission.id,
                                    submission.title,
                                    comment.author.name,
                                    comment.body.replace('\n', ' ')  # Limpiar saltos de línea
                                ])

                        # Pausa cortés para no saturar la API innecesariamente
                        time.sleep(1)

                except Exception as e:
                    print(f"!! Error al procesar r/{subreddit_name}: {e}. Continuando con el siguiente...")

    except IOError as e:
        print(f"Error de E/S: No se pudo escribir en el archivo {ARCHIVO_SALIDA}. {e}")


if __name__ == "__main__":
    reddit_api = inicializar_reddit()
    if reddit_api:
        extraer_comentarios_a_csv(reddit_api, SUBREDDITS_SELECCIONADOS)
        print("\n¡Proceso de extracción completado!")
