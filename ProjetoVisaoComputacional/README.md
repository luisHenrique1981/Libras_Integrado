# Projeto Visão Computacional - Tradutor de Libras em Tempo Real

![Status do Projeto](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Um aplicativo móvel multiplataforma (Flutter) que utiliza um backend Python com Visão Computacional e IA para fornecer tradução simultânea e em tempo real da Língua Brasileira de Sinais (Libras) para o português e vice-versa.

## 🎯 Sobre o Projeto

O objetivo deste projeto é criar uma ponte de comunicação acessível e instantânea entre a comunidade surda e os ouvintes. Utilizando a câmera do dispositivo móvel, o aplicativo captura os sinais de Libras, para fornecer uma tradução precisa em tempo real.

O projeto também visa permitir o caminho inverso: o usuário pode inserir texto ou falar em português, e o aplicativo exibirá o sinal correspondente em Libras (através de um avatar).

## ✨ Funcionalidades Principais

* **Tradução Libras -> Português:** Aponta a câmera e veja a tradução dos sinais em tempo real.
* **Tradução Português -> Libras:** Digite ou fale uma frase em português para ver a tradução em Libras.
* **Reconhecimento por Visão Computacional:** Identificação em tempo real de gestos, expressões faciais e movimentos corporais.
* **Processamento de IA:** Uso de modelos de inteligência artificial para interpretar o contexto dos sinais e fornecer uma tradução simultânea precisa.
* **Interface Acessível:** Interface de usuário limpa, rápida e intuitiva, construída com Flutter.
.

### 1. Frontend (Aplicativo Móvel)

* **Framework:** [Flutter](https://flutter.dev/) (Linguagem: Dart)
* **Função:** Responsável por toda a interface do usuário (UI), captura do stream de vídeo da câmera e comunicação via API REST (ou WebSockets) com o backend.

### 2. Backend (Servidor de IA)

* **Linguagem:** [Python](https://www.python.org/)
* **Framework** [Flask](https://flask.palletsprojects.com/) (para criar os *endpoints* que o Flutter irá consumir).
* **Visão Computacional (CV):** [OpenCV](https://opencv.org/)
    * Usado para o pré-processamento das imagens, detecção de mãos (hands tracking), pontos de referência corporais (pose estimation) e extração de características (features) do stream de vídeo.


## 🚀 Como Começar (Em breve)

*(Esta seção será atualizada com as instruções de instalação e execução do projeto.)*

### Pré-requisitos

* Flutter SDK
* Python 3.10+


