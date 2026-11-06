import speech_recognition as sr
#Biblioteca pyaudio para processamento em tempo real
import pyaudio


#Função uusando straming em tempo real:
def reconhecimento_em_tempo_real():
    p = pyaudio.PyAudio()

    #Configurações para fluxo de áudio:
    stram = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=44100,
                   input=True, #Use input=True para gravar o microfone
                   output=True) #Use output=True para saída de audio


#Função para reconhecer fala em um arquivo de áudio e converter para texto
def audio_para_texto(caminho_arquivo):
    # Cria o reconhecedor
    r = sr.Recognizer()

    # Carrega o arquivo de áudio
    with sr.AudioFile(caminho_arquivo) as source:
        # Lê todo o conteúdo do áudio
        audio_data = r.record(source)

    try:
        # Usa a API gratuita do Google (necessita internet)
        texto = r.recognize_google(audio_data, language="pt-BR")
        return texto
    except sr.UnknownValueError:
        return "Não foi possível entender o áudio."
    except sr.RequestError as e:
        return f"Erro ao se comunicar com o serviço de reconhecimento: {e}"

if __name__ == "__main__":
    caminho = "venv/audio/F047-0003.wav"  # coloque aqui o caminho do seu arquivo
    resultado = audio_para_texto(caminho)
    print("Texto reconhecido:")
    print(resultado)
