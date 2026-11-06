from bs4 import BeautifulSoup
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from ProjetoVisaoComputacional.web_scraping.database import conectar_db, salvar_dados, verificar_existencia_video

#Função de WEb Scraping
#Array - carrega todas as letras, peccore ele. Em cada um pega o link correspondente a letra
#Transformar o rosto da pessoa em outro.

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #garantindo a pasta do DB
VIDEOS_DIR = os.path.join(BASE_DIR, "videos_libras")

if not os.path.exists(VIDEOS_DIR):
    os.makedirs(VIDEOS_DIR)

def configurar_navegador():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def capturar_dicionario_ines():
    driver = configurar_navegador()
    wait = WebDriverWait(driver, 10)
    
    url_base = "https://dicionario.ines.gov.br/"
    driver.get(url_base)

    # 1. Mapear as letras do alfabeto (os índices/botões)
    # No site do INES, as letras costumam estar em uma lista ou grid
    alfabeto_xpath = "//div[@id='filter-letter']//ul[contains(@class, 'list-inline')]//li//a"
    letras = wait.until(EC.presence_of_all_elements_located((By.XPATH, alfabeto_xpath)))
    
    qtd_letras = len(letras)

    for i in range(qtd_letras):
        # Re-localiza as letras para evitar StaleElementReferenceException
        letras = driver.find_elements(By.XPATH, alfabeto_xpath)
        letra_btn = letras[i]
        letra_texto = letra_btn.text.strip()
        
        print(f"\n--- 🔠 Processando Letra: {letra_texto} ---")
        letra_btn.click()
        time.sleep(2) # Espera carregar a lista de palavras

        # 2. Mapear todas as palavras daquela letra
        palavras_xpath = "//div[contains(@class, 'lista-palavras')]//a"
        palavras = driver.find_elements(By.XPATH, palavras_xpath)
        
        links_palavras = [p.get_attribute('href') for p in palavras]
        
        for link in links_palavras:
            try:
                driver.get(link)
                
                # 3. Extrair dados da palavra
                nome_palavra = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2"))).text
                
                # Exemplo: Capturar URL do vídeo de libras
                video_tag = driver.find_element(By.TAG_NAME, "video")
                video_url = video_tag.find_element(By.TAG_NAME, "source").get_attribute("src")
                
                print(f"📖 Palavra: {nome_palavra} | Vídeo: {video_url}")

                # Aqui você chamaria sua função de banco de dados:
                # salvar_no_banco(nome_palavra, letra_texto, video_url)

                driver.back() # Volta para a lista de palavras
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Erro ao processar palavra no link {link}: {e}")
                continue
        
    driver.quit()
    print("Processamento finalizado!")

if __name__ == "__main__":
    capturar_dicionario_ines()
    #chamar a função principal
