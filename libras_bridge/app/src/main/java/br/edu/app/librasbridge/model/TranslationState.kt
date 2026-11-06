package br.edu.app.librasbridge.model

data class TranslationState(
    val recognizedGestureRaw: String = "",
    val recognizedGestureToken: String = "",
    val assembledSentence: String = "",
    val heardText: String = "",
    val avatarPlaylistUris: List<String> = emptyList(),
    val avatarPlaylistLabels: List<String> = emptyList(),
    val statusMessage: String = "Pronto.",
    val isListening: Boolean = false,
    val isModelReady: Boolean = false,
    val isSpeaking: Boolean = false
)
