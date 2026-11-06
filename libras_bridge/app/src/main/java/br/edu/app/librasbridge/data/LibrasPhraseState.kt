package br.edu.app.librasbridge.data

import java.io.File

sealed class TokenResolucao {
    data class Resolvido(
        val token: String,
        val palavraEncontrada: String,
        val arquivo: File,
        val estrategia: String,
    ) : TokenResolucao()

    data class NaoResolvido(
        val token: String,
        val motivo: String,
        val caminhoEsperado: String?,
    ) : TokenResolucao()
}

data class FraseResolvida(
    val fraseOriginal: String,
    val tokensExtraidos: List<String>,
    val resolucoes: List<TokenResolucao>,
    val arquivosParaReproducao: List<File>,
    val logDiagnostico: List<String>,
) {
    val tokensResolvidos: List<TokenResolucao.Resolvido>
        get() = resolucoes.filterIsInstance<TokenResolucao.Resolvido>()

    val tokensNaoResolvidos: List<TokenResolucao.NaoResolvido>
        get() = resolucoes.filterIsInstance<TokenResolucao.NaoResolvido>()

    val totalmenteSucesso: Boolean
        get() = resolucoes.isNotEmpty() &&
                tokensNaoResolvidos.isEmpty() &&
                arquivosParaReproducao.isNotEmpty()

    val parcialmenteSucesso: Boolean
        get() = arquivosParaReproducao.isNotEmpty() && tokensNaoResolvidos.isNotEmpty()

    fun mensagemErroParaUI(): String {
        if (totalmenteSucesso) {
            return "Todos os tokens foram resolvidos com sucesso."
        }

        val faltas = tokensNaoResolvidos.joinToString("; ") { item ->
            buildString {
                append(item.token)
                append(" -> ")
                append(item.motivo)
                if (!item.caminhoEsperado.isNullOrBlank()) {
                    append(" (esperado: ")
                    append(item.caminhoEsperado)
                    append(")")
                }
            }
        }

        if (arquivosParaReproducao.isEmpty()) {
            return if (faltas.isBlank()) {
                "Nenhum vídeo resolvido."
            } else {
                "Nenhum vídeo resolvido. $faltas"
            }
        }

        return if (faltas.isBlank()) {
            "Resolução parcial sem detalhe adicional."
        } else {
            "Resolução parcial: ${arquivosParaReproducao.size}/${resolucoes.size} tokens com vídeo. Não resolvidos: $faltas"
        }
    }
}