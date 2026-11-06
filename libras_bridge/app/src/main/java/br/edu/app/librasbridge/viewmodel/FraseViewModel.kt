package br.edu.app.librasbridge.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import br.edu.app.librasbridge.data.FraseResolvida
import br.edu.app.librasbridge.data.LibrasRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

private const val TAG = "FraseViewModel"

sealed class EstadoVideo {
    object Inativo : EstadoVideo()

    data class Carregando(
        val atual: Int,
        val total: Int,
    ) : EstadoVideo()

    data class Pronto(
        val resultado: FraseResolvida,
    ) : EstadoVideo()

    data class ParcialmentePronto(
        val resultado: FraseResolvida,
    ) : EstadoVideo()

    data class Erro(
        val mensagem: String,
    ) : EstadoVideo()
}

class FraseViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = LibrasRepository(app.applicationContext)

    private val _estado = MutableStateFlow<EstadoVideo>(EstadoVideo.Inativo)
    val estado: StateFlow<EstadoVideo> = _estado

    fun montarFrase(frase: String) {
        val fraseNormalizada = frase.trim()

        if (fraseNormalizada.isBlank()) {
            Log.w(TAG, "[FRASE:BLANK] frase vazia ignorada.")
            _estado.value = EstadoVideo.Erro("Digite uma frase para traduzir.")
            return
        }

        _estado.value = EstadoVideo.Carregando(0, 0)
        Log.i(TAG, "[FRASE:REQUESTED] \"$fraseNormalizada\"")

        viewModelScope.launch {
            val resultado = repo.resolverFraseParaPlaylist(
                frase = fraseNormalizada,
                onProgresso = { atual, total ->
                    _estado.value = EstadoVideo.Carregando(atual, total)
                },
            )

            _estado.value = resultado.fold(
                onSuccess = { fraseResolvida ->
                    when {
                        fraseResolvida.totalmenteSucesso -> {
                            Log.i(
                                TAG,
                                "[FRASE:OK] ${fraseResolvida.arquivosParaReproducao.size} vídeos prontos."
                            )
                            EstadoVideo.Pronto(fraseResolvida)
                        }

                        fraseResolvida.parcialmenteSucesso -> {
                            Log.w(
                                TAG,
                                "[FRASE:PARTIAL] ${fraseResolvida.arquivosParaReproducao.size}/" +
                                        "${fraseResolvida.resolucoes.size} resolvidos. " +
                                        fraseResolvida.mensagemErroParaUI()
                            )
                            EstadoVideo.ParcialmentePronto(fraseResolvida)
                        }

                        else -> {
                            val msg = fraseResolvida.mensagemErroParaUI()
                            Log.e(TAG, "[FRASE:FAIL] $msg")
                            EstadoVideo.Erro(msg)
                        }
                    }
                },
                onFailure = { erro ->
                    val msg = erro.message ?: "Erro desconhecido ao resolver frase."
                    Log.e(TAG, "[FRASE:FAIL] $msg", erro)
                    EstadoVideo.Erro(msg)
                },
            )
        }
    }

    fun resetar() {
        Log.d(TAG, "[FRASE:RESET]")
        _estado.value = EstadoVideo.Inativo
    }
}