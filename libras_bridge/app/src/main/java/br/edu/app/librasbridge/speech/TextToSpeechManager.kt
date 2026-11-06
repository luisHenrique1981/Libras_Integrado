package br.edu.app.librasbridge.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class TextToSpeechManager(
    context: Context,
    private val onStatus: (String) -> Unit,
    private val onSpeakingChanged: (Boolean) -> Unit
) : TextToSpeech.OnInitListener {

    private val tts = TextToSpeech(context, this)
    private var ready = false

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            tts.language = Locale("pt", "BR")
            onStatus("Síntese de voz pronta.")
        } else {
            onStatus("Falha ao iniciar síntese de voz.")
        }
    }

    fun speak(text: String) {
        if (!ready || text.isBlank()) return
        onSpeakingChanged(true)
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "camera_sentence")
    }

    fun stop() {
        tts.stop()
        onSpeakingChanged(false)
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
        onSpeakingChanged(false)
    }
}
