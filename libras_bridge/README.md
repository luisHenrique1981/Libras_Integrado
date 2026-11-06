# Libras Bridge

Módulo Android do projeto de visão computacional para comunicação em Libras.

Converte fala em sinais de Libras via avatar digital, e sinais capturados pela câmera em texto/fala.

---

## Estado atual

- Pipeline de resolução de frases para playlist de vídeos: funcional
- Integração ExoPlayer + Room: funcional
- Indicadores de 2026-05-01: `FRASE_OK=3`, `VIDEO_MISSING=0`, `PLAYER_ERROR=0`
- Lacuna de vocabulário em termos compostos (ex.: `bom dia`, `boa noite`) — próximo passo

---

## Requisitos

| Ferramenta | Versão |
|---|---|
| Android Studio | Hedgehog 2023.1.1 ou superior |
| JDK | 17 |
| Android SDK compileSdk | 35 |
| SDK Build-Tools | 35.0.0 |
| Emulador ou dispositivo | Android API 26+ |

### Instalar o JDK 17

**Debian/Ubuntu**
```bash
sudo apt update && sudo apt install -y openjdk-17-jdk
```

**Fedora**
```bash
sudo dnf install -y java-17-openjdk-devel
```

**Windows**
Baixe e instale o Temurin 17 em:
https://adoptium.net

---

## Como rodar

### 1. Clonar o repositório

```bash
git clone git@github.com:ProfRenatoOliveira/ProjetoVisaoComputacional.git
cd ProjetoVisaoComputacional/libras_bridge
```

### 2. Baixar o dataset de vídeos

Os vídeos (~225 MB, 5641 arquivos) não estão no repositório por limitação de tamanho do GitHub.

Link do Google Drive:
https://drive.google.com/drive/folders/1NpLRFw5nusoV0r4aWIgLgbBdE8K-vL0h?usp=sharing

Extraia de forma que a estrutura fique assim:

```text
~/Projetos/libras_ines_completo/libras_ines/libras_data/videos/
├── A/
│   └── ABACAXI/
│       └── abacaxiSm_Prog001.mp4
├── B/
├── C/
└── ...
```

- Linux/macOS: extraia em `~/Projetos/libras_ines_completo/`
- Windows: extraia em `C:\Users\SEU_USUARIO\Projetos\libras_ines_completo\`

### 3. Abrir no Android Studio

1. `File -> Open`
2. Selecione a pasta `libras_bridge`
3. Aguarde o Gradle Sync terminar
4. Se houver aviso de SDK, ajuste em `File -> Project Structure -> SDK Location`

### 4. Instalar os vídeos no emulador/device

Inicie o emulador no Android Studio. Depois rode:

**Linux / Debian / Fedora**
```bash
bash tools/setup_videos.sh
```

**Windows**
```bat
tools\setup_videos.bat
```

Se `adb` não estiver no PATH, adicione o diretório `platform-tools` do Android SDK:

- Linux: `~/Android/Sdk/platform-tools`
- Windows: `%LOCALAPPDATA%\Android\Sdk\platform-tools`

### 5. Compilar e executar

No Android Studio: botão **Run** (`Shift+F10`).

---

## Estrutura de branches

| Branch | Responsável | Finalidade |
|---|---|---|
| `main` | Professor | Estado estável / integração |
| `dev/bruno` | Bruno | Desenvolvimento do módulo Libras |

---

## Estrutura do módulo

```text
libras_bridge/
├── app/
│   └── src/main/
│       ├── java/br/edu/app/librasbridge/
│       │   ├── data/
│       │   ├── ui/
│       │   └── avatar/
│       └── assets/
│           └── databases/
│               └── libras.db
├── tools/
│   ├── setup_videos.sh
│   └── setup_videos.bat
└── README.md
```

---

## Problemas conhecidos

- Sinais compostos como `bom dia` e `boa noite` ainda ausentes no dataset de vídeos
- O dataset atual cobre aproximadamente 5641 sinais isolados
