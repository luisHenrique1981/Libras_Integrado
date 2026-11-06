package br.edu.app.librasbridge.viewmodel

import androidx.lifecycle.ViewModel
import br.edu.app.librasbridge.avatar.SignDictionary
import br.edu.app.librasbridge.model.TranslationState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class MainViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(TranslationState())
    val uiState: StateFlow<TranslationState> = _uiState.asStateFlow()

    fun setModelReady(ready: Boolean) {
        _uiState.value = _uiState.value.copy(
            isModelReady = ready,
            statusMessage = if (ready) "Modelo de gestos carregado." else "Modelo de gestos não encontrado; modo demonstração ativo."
        )
    }

    fun appendRecognizedToken(rawGesture: String, token: String) {
        val current = _uiState.value.assembledSentence.trim()
        val newSentence = when {
            token.isBlank() -> current
            current.endsWith(token) -> current
            current.isBlank() -> token
            else -> "$current $token"
        }
        _uiState.value = _uiState.value.copy(
            recognizedGestureRaw = rawGesture,
            recognizedGestureToken = token,
            assembledSentence = newSentence,
            statusMessage = if (token.isBlank()) "Gesto detectado sem mapeamento textual." else "Sinal reconhecido: $token"
        )
    }

    fun clearCameraSentence() {
        _uiState.value = _uiState.value.copy(
            recognizedGestureRaw = "",
            recognizedGestureToken = "",
            assembledSentence = "",
            statusMessage = "Texto limpo."
        )
    }

    fun setHeardText(text: String) {
        val playlist = SignDictionary.resolvePlaylistWithLabels(text)
        _uiState.value = _uiState.value.copy(
            heardText = text,
            avatarPlaylistLabels = playlist.map { it.first },
            avatarPlaylistUris = playlist.map { it.second },
            statusMessage = when {
                playlist.isNotEmpty() -> "Sequência de sinais preparada para o avatar."
                else -> "Frase reconhecida, mas sem animação cadastrada no dicionário local."
            }
        )
    }

    fun setListening(listening: Boolean) {
        _uiState.value = _uiState.value.copy(
            isListening = listening,
            statusMessage = if (listening) "Escutando..." else "Escuta finalizada."
        )
    }

    fun setSpeaking(speaking: Boolean) {
        _uiState.value = _uiState.value.copy(
            isSpeaking = speaking,
            statusMessage = if (speaking) "Reproduzindo voz sintetizada..." else _uiState.value.statusMessage
        )
    }

    fun setStatus(status: String) {
        _uiState.value = _uiState.value.copy(statusMessage = status)
    }
}
