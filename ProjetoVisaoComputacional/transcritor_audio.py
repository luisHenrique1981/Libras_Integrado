# transcritor_audio.py
# Trasncreve audio em tempo real do microfone para texto usando a biblioteca SpeechRecognition
# Requer conexão com a internet para usar a API do Google
import speech_recognition as sr
import threading
import queue
import time
import io

class RealTimeSpeechToText:
    def __init__(self, energy_threshold=300, record_timeout=5, phrase_timeout=3):
        self.r = sr.Recognizer()
        self.r.energy_threshold = energy_threshold
        self.r.pause_threshold = 0.8
        self.r.dynamic_energy_threshold = False

        self.audio_queue = queue.Queue()
        self.speaking = False
        self.last_speech_time = time.time()
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

        self.audio_source = sr.Microphone()
        self.stop_listening = None
        self.transcription_thread = None
        self.running = False

        print("Inicializando o reconhecedor de fala...")
        with self.audio_source as source:
            self.r.adjust_for_ambient_noise(source)
        print("Ajuste de ruído ambiente completo. Pronto para ouvir.")

    def _listen_continuously(self):
        def callback(recognizer, audio):
            self.audio_queue.put(audio)
            self.speaking = True
            self.last_speech_time = time.time()

        self.stop_listening = self.r.listen_in_background(self.audio_source, callback, phrase_time_limit=self.record_timeout)
        print("Começando a escutar em segundo plano...")
        self.running = True

        try:
            while self.running:
                if self.speaking and (time.time() - self.last_speech_time > self.phrase_timeout):
                    self.speaking = False
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupção do teclado. Parando de escutar.")
            self.stop()
        except Exception as e:
            print(f"Erro na escuta contínua: {e}")
            self.stop()

    def _process_audio_queue(self):
        while self.running or not self.audio_queue.empty():
            try:
                audio = self.audio_queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue

            try:
                text = self.r.recognize_google(audio, language="pt-BR")
                if text:
                    print(f"Você disse: {text}")
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Não foi possível solicitar resultados do serviço Google Speech; {e}")
            except Exception as e:
                print(f"Erro inesperado durante a transcrição: {e}")

            self.audio_queue.task_done()
        print("Thread de processamento de áudio encerrada.")

    def start(self):
        if self.running:
            print("Já está rodando.")
            return

        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_continuously)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        self.transcription_thread = threading.Thread(target=self._process_audio_queue)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()

        print("Sistema de reconhecimento de fala iniciado. Fale algo.")
        print("Pressione Ctrl+C para parar.")

    def stop(self):
        if not self.running:
            print("Já está parado.")
            return

        self.running = False
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            print("Escuta em segundo plano parada.")

        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2)

        print("Sistema de reconhecimento de fala parado.")

if __name__ == "__main__":
    stt_system = RealTimeSpeechToText()
    stt_system.start()

    try:
        while stt_system.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupção detectada. Parando o sistema...")
    finally:
        stt_system.stop()
        print("Programa encerrado.")