import os
import re
import sqlite3
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "libras_videos.db")
FRAMES_DIR = os.path.join(BASE_DIR, "dataset_teachable_machine")

def setup_frames_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            palavra_id INTEGER,
            palavra TEXT NOT NULL,
            caminho_frame TEXT NOT NULL,
            FOREIGN KEY (palavra_id) REFERENCES palavras_videos(id)
        )
    """)

def normalizar_nome_classe(palavra):
    nome = palavra.strip().lower()
    nome = re.sub(r"[^\w.-]+", "_", nome, flags=re.UNICODE)
    return nome.strip("._") or "sem_nome"

def resolver_caminho_video(caminho_video):
    if os.path.isabs(caminho_video):
        return caminho_video
    return os.path.join(BASE_DIR, caminho_video)

def limpar_dataset_anterior(cursor):
    cursor.execute("DELETE FROM frames")
    if not os.path.isdir(FRAMES_DIR):
        return

    for raiz, _, arquivos in os.walk(FRAMES_DIR):
        for arquivo in arquivos:
            os.remove(os.path.join(raiz, arquivo))

def extrair_frames_videos():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    setup_frames_table(cursor)
    limpar_dataset_anterior(cursor)
    conn.commit()
    
    # Busca os vídeos salvos com sucesso no banco
    cursor.execute("SELECT id, palavra, arquivo_local FROM palavras_videos WHERE arquivo_local IS NOT NULL AND arquivo_local != 'DOWNLOAD_FAIL'")
    registros = cursor.fetchall()
    
    print(f"Encontrados {len(registros)} vídeos para processar.")
    
    for registro_id, palavra, caminho_video in registros:
        caminho_video = resolver_caminho_video(caminho_video)
        if not os.path.isfile(caminho_video):
            print(f"Arquivo de vídeo não encontrado no disco: {caminho_video}")
            continue
            
        print(f"Extraindo frames para a palavra: {palavra}")
        
        # Cria uma pasta específica para a palavra (formato ideal para exportar/organizar)
        pasta_palavra = os.path.join(FRAMES_DIR, normalizar_nome_classe(palavra))
        os.makedirs(pasta_palavra, exist_ok=True)
        
        cap = cv2.VideoCapture(caminho_video)
        sucesso, frame = cap.read()
        count = 0
        saved_count = 0
        
        # Taxa de amostragem: Pula alguns frames para não gerar imagens demais idênticas
        # (ex: salva 1 a cada 3 frames, ajuste conforme necessário)
        frame_skip = 3 
        
        while sucesso:
            if count % frame_skip == 0:
                nome_frame = (
                    f"{normalizar_nome_classe(palavra)}_"
                    f"{registro_id}_{saved_count:04d}.jpg"
                )
                caminho_frame = os.path.join(pasta_palavra, nome_frame)
                
                # Salva a imagem do frame
                if not cv2.imwrite(caminho_frame, frame):
                    print(f"Falha ao salvar o frame: {caminho_frame}")
                    sucesso, frame = cap.read()
                    count += 1
                    continue
                
                # Salva o registro no banco de dados
                cursor.execute("""
                    INSERT INTO frames (palavra_id, palavra, caminho_frame)
                    VALUES (?, ?, ?)
                """, (registro_id, palavra, caminho_frame))
                
                saved_count += 1
                
            sucesso, frame = cap.read()
            count += 1
            
        cap.release()
        conn.commit()
        print(f"-> {saved_count} frames salvos para '{palavra}'.")
        
    conn.close()
    print("\nProcesso de extração de frames finalizado!")

if __name__ == "__main__":
    extrair_frames_videos()