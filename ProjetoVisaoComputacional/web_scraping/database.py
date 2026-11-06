import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "datasets.db")

def conectar_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Removido conn.row_factory = sqlite3 (isso costuma causar erro se não for sqlite3.Row)
    conn.row_factory = sqlite3.Row 
    return conn

def init_database():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Base_dados(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            letra TEXT,
            assunto TEXT,
            palavra TEXT,
            caminho_foto_mao TEXT,
            url_video TEXT)
    """)
    conn.commit()
    conn.close()
    print("Banco de dados inicializado! ✅")

def salvar_dados(letra, assunto, palavra, caminho_foto_mao, url_video):
    conn = conectar_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO Base_dados (letra, assunto, palavra, caminho_foto_mao, url_video) VALUES (?,?,?,?,?)",
            (letra, assunto, palavra, caminho_foto_mao, url_video)   
        )
        conn.commit()
    except Exception as e:
        print(f"❌ Erro ao inserir dados: {e}")
    finally:
        conn.close()

def verificar_existencia_video(url_video):
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Base_dados WHERE url_video = ?", (url_video,)) 
    resultado = cursor.fetchone()
    conn.close()
    return resultado is not None #Verificador para o caso do resultado ser nulo
