import os
import sqlite3
import time
from urllib.parse import urljoin

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "libras_videos.db")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "videos_libras")

# 1. Configuração do Banco de Dados SQLite
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS palavras_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            palavra TEXT NOT NULL,
            video_url TEXT,
            arquivo_local TEXT
        )
    """)
    conn.commit()
    return conn, cursor

# 2. Configuração do Diretório de Download
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def main():
    conn, cursor = setup_database()
    
    # Configurar o Selenium 
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") #Se quiser rodar sem abrir a janela do navegador
    driver = webdriver.Chrome(options=options)
    base_url = "http://www.acessibilidadebrasil.org.br/libras_3/"
    
    try:
        driver.get(base_url)
        wait = WebDriverWait(driver, 10)
        
        # Lista de letras do alfabeto para percorrer
        letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        for letra in letras:
            print(f"\n--- Processando letra: {letra} ---")
            try:
                link_letra = wait.until((EC.presence_of_element_located(
                    (By.ID, f"letter-{letra}")
                )))
                driver.execute_script("arguments[0].click();", link_letra)
                wait.until(lambda browser: len(
                    browser.find_elements(By.CSS_SELECTOR, "#input-palavras option")
                ) > 1)
            except Exception as e:
                print(f"Letra {letra} não encontrada ou erro ao clicar: {e}")
                continue

            palavras = [
                (option.get_attribute("value"), option.text.strip())
                for option in driver.find_elements(By.CSS_SELECTOR, "#input-palavras option")
                if option.get_attribute("id") != "0"
            ]
            total_palavras = len(palavras)
            print(f"Encontradas {total_palavras} palavras para a letra {letra}.")

            for palavra_id, palavra_texto in palavras:
                try:
                    if not palavra_texto:
                        continue

                    print(f"-> Selecionando a palavra: {palavra_texto}")
                    select = Select(wait.until(
                        EC.presence_of_element_located((By.ID, "input-palavras"))
                    ))
                    driver.execute_script(
                        "document.getElementById('input-video').innerHTML = '';"
                    )
                    select.select_by_value(palavra_id)

                    video_elem = wait.until(
                        lambda browser: browser.find_element(
                            By.CSS_SELECTOR, "#input-video video source[src]"
                        )
                    )
                    video_url = video_elem.get_attribute("src")

                    caminho_arquivo = None
                    if video_url:
                        video_url = urljoin(base_url, video_url)
                        print(f"   URL do vídeo obtida: {video_url}")

                        # Faz o download do vídeo via requests
                        video_resposta = requests.get(video_url, stream=True, timeout=60)
                        nome_arquivo = f"{letra}_{palavra_texto}.mp4".replace("/", "_").replace(" ", "_").lower()
                        caminho_arquivo = os.path.join(DOWNLOAD_DIR, nome_arquivo)

                        if video_resposta.status_code == 200:
                            with open(caminho_arquivo, "wb") as f:
                                for chunk in video_resposta.iter_content(chunk_size=1024):
                                    f.write(chunk)
                            print(f"   Download salvo em: {caminho_arquivo}")
                        else:
                            print(f"   Falha ao baixar o vídeo. Status code: {video_resposta.status_code}")
                            caminho_arquivo = "DOWNLOAD_FAIL"
                        
                        # Salva os dados no Banco de Dados SQLite
                        cursor.execute("""
                            INSERT INTO palavras_videos (palavra, video_url, arquivo_local)
                            VALUES (?, ?, ?)
                        """, (palavra_texto, video_url, caminho_arquivo))
                        conn.commit()
                    else:
                        print(f"   Vídeo não encontrado para a palavra '{palavra_texto}'.")

                except Exception as e:
                    print(f"   Erro ao processar a palavra '{palavra_texto}': {e}")
                    continue
                    
    finally:
        driver.quit()
        conn.close()
        print("\nProcesso finalizado e conexões encerradas.")

if __name__ == "__main__":
    main()