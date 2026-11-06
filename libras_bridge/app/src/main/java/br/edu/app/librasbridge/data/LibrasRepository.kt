package br.edu.app.librasbridge.data

import android.content.Context
import android.util.Log
import br.edu.app.librasbridge.util.SynonymResolver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.io.File
import java.text.Normalizer
import java.util.Locale

private const val TAG = "LibrasRepository"

class LibrasRepository(private val context: Context) {

    private val dao = LibrasDatabase.getInstance(context).librasDao()

    fun buscarPorPrefixo(q: String): Flow<List<Palavra>> = dao.buscarPorPrefixo(q.trim())
    fun buscarFullText(q: String): Flow<List<Palavra>> = dao.buscarFullText(q.trim())
    fun porLetra(letra: String): Flow<List<Palavra>> = dao.porLetra(letra)
    fun porCategoria(cat: String): Flow<List<Palavra>> = dao.porCategoria(cat)

    suspend fun letras(): List<String> = dao.letrasDisponiveis()
    suspend fun categorias(): List<String> = dao.categoriasDisponiveis()
    suspend fun total(): Long = dao.total()
    suspend fun totalComVideo(): Long = dao.totalComVideoLocal()

    suspend fun obterArquivoVideo(palavra: Palavra): Result<File> =
        withContext(Dispatchers.IO) {
            when (val lookup = resolverArquivoVideo(palavra)) {
                is VideoLookup.Found -> Result.success(lookup.file)
                is VideoLookup.Missing -> {
                    val msg = buildString {
                        append("Vídeo não resolvido para '")
                        append(palavra.palavra)
                        append("'. Motivo=")
                        append(lookup.motivo)
                        if (!lookup.caminhoEsperado.isNullOrBlank()) {
                            append(", esperado=")
                            append(lookup.caminhoEsperado)
                        }
                    }
                    Result.failure(IllegalStateException(msg))
                }
            }
        }

    suspend fun resolverFraseParaPlaylist(
        frase: String,
        onProgresso: ((Int, Int) -> Unit)? = null,
    ): Result<FraseResolvida> = withContext(Dispatchers.IO) {
        val audit = mutableListOf<String>()

        fun info(msg: String) {
            audit += msg
            Log.i(TAG, msg)
        }

        fun debug(msg: String) {
            audit += msg
            Log.d(TAG, msg)
        }

        fun warn(msg: String) {
            audit += msg
            Log.w(TAG, msg)
        }

        fun fail(msg: String): Result<FraseResolvida> {
            audit += msg
            Log.e(TAG, msg)
            return Result.failure(IllegalStateException(msg))
        }

        val fraseEntrada = frase.trim()
        info("[FRASE:START] entrada=\"$fraseEntrada\"")

        val tokens = tokenizarFrase(fraseEntrada)
        debug("[FRASE:TOKENS] ${tokens.joinToString(", ")}")

        if (tokens.isEmpty()) {
            return@withContext fail("[FRASE:FAIL] Nenhum token válido extraído da frase.")
        }

        val resolucoes = mutableListOf<TokenResolucao>()
        val arquivos = mutableListOf<File>()

        var cursor = 0
        onProgresso?.invoke(0, tokens.size)

        while (cursor < tokens.size) {
            var houveMatch = false
            val maxJanela = minOf(3, tokens.size - cursor)

            for (janela in maxJanela downTo 1) {
                val trecho = tokens.subList(cursor, cursor + janela).joinToString(" ")
                val palavra = buscarPalavraCandidata(trecho)

                if (palavra != null) {
                    when (val lookup = resolverArquivoVideo(palavra)) {
                        is VideoLookup.Found -> {
                            resolucoes += TokenResolucao.Resolvido(
                                token = trecho,
                                palavraEncontrada = palavra.palavra,
                                arquivo = lookup.file,
                                estrategia = lookup.estrategia,
                            )
                            arquivos += lookup.file
                            debug(
                                "[FRASE:HIT] token=\"$trecho\" palavra=\"${palavra.palavra}\" " +
                                        "estrategia=${lookup.estrategia} arquivo=${lookup.file.absolutePath}"
                            )
                        }

                        is VideoLookup.Missing -> {
                            resolucoes += TokenResolucao.NaoResolvido(
                                token = trecho,
                                motivo = lookup.motivo,
                                caminhoEsperado = lookup.caminhoEsperado,
                            )
                            warn(
                                "[FRASE:MISS] token=\"$trecho\" motivo=${lookup.motivo} " +
                                        "esperado=${lookup.caminhoEsperado}"
                            )
                        }
                    }

                    cursor += janela
                    onProgresso?.invoke(cursor.coerceAtMost(tokens.size), tokens.size)
                    houveMatch = true
                    break
                }
            }

            if (!houveMatch) {
                val token = tokens[cursor]
                resolucoes += TokenResolucao.NaoResolvido(
                    token = token,
                    motivo = "NOT_IN_DB",
                    caminhoEsperado = null,
                )
                warn("[FRASE:MISS] token=\"$token\" motivo=NOT_IN_DB esperado=null")
                cursor += 1
                onProgresso?.invoke(cursor.coerceAtMost(tokens.size), tokens.size)
            }
        }

        val resultado = FraseResolvida(
            fraseOriginal = fraseEntrada,
            tokensExtraidos = tokens,
            resolucoes = resolucoes.toList(),
            arquivosParaReproducao = arquivos.toList(),
            logDiagnostico = audit.toList(),
        )

        info(
            "[FRASE:RESULT] resolvidos=${resultado.tokensResolvidos.size}/${resultado.resolucoes.size} " +
                    "sucesso=${resultado.totalmenteSucesso} parcial=${resultado.parcialmenteSucesso}"
        )

        if (resultado.arquivosParaReproducao.isEmpty()) {
            val msg = resultado.mensagemErroParaUI()
            warn("[FRASE:FAIL] $msg")
            return@withContext Result.failure(IllegalStateException(msg))
        }

        Result.success(resultado)
    }

    suspend fun montarVideoFrase(
        frase: String,
        nomeBase: String = "frase_${System.currentTimeMillis()}",
        onProgresso: ((Int, Int) -> Unit)? = null,
    ): Result<File> = withContext(Dispatchers.IO) {
        val playlist = resolverFraseParaPlaylist(frase, onProgresso)
        playlist.fold(
            onSuccess = { resolvida ->
                val arquivo = resolvida.arquivosParaReproducao.firstOrNull()
                if (arquivo != null) {
                    Result.success(arquivo)
                } else {
                    Result.failure(
                        IllegalStateException(
                            "Resolver frase produziu sucesso sem arquivos. nomeBase=$nomeBase"
                        )
                    )
                }
            },
            onFailure = { erro ->
                Result.failure(erro)
            },
        )
    }

    private suspend fun buscarPalavraCandidata(termoOriginal: String): Palavra? {
        val candidatos = gerarCandidatosBusca(termoOriginal)
        for (candidato in candidatos) {
            dao.buscarExata(candidato)?.let { return it }
        }
        return null
    }

    private fun resolverArquivoVideo(palavra: Palavra): VideoLookup {
        val videoPath = palavra.videoPath?.trim().orEmpty()
        if (videoPath.isNotBlank()) {
            val arquivo = if (File(videoPath).isAbsolute()) {
                File(videoPath)
            } else {
                File(context.filesDir, videoPath)
            }

            if (arquivo.exists() && arquivo.isFile && arquivo.length() > 0L) {
                return VideoLookup.Found(
                    file = arquivo,
                    estrategia = "VIDEO_PATH",
                )
            }

            Log.w(TAG, "[VIDEO:PATH_MISSING] ${arquivo.absolutePath}")
        }

        val videoNome = palavra.videoNome?.trim().orEmpty()
        if (videoNome.isNotBlank()) {
            val candidatos = resolverCandidatosLocais(palavra, videoNome)
            val encontrado = candidatos.firstOrNull { it.exists() && it.isFile && it.length() > 0L }
            if (encontrado != null) {
                return VideoLookup.Found(
                    file = encontrado,
                    estrategia = "RECONSTRUCAO_LOCAL",
                )
            }

            return VideoLookup.Missing(
                motivo = "VIDEO_MISSING",
                caminhoEsperado = candidatos.firstOrNull()?.absolutePath,
            )
        }

        return VideoLookup.Missing(
            motivo = "NO_VIDEO_METADATA",
            caminhoEsperado = null,
        )
    }

    private fun resolverCandidatosLocais(palavra: Palavra, videoNome: String): List<File> {
        val letra = palavra.letra
            .trim()
            .uppercase(Locale.ROOT)
            .take(1)
            .ifBlank { "_" }

        val palavraDir = sanitize(palavra.palavra)
        val base = File(context.filesDir, "videos/$letra/$palavraDir")

        return gerarVariantesNome(videoNome).map { File(base, it) }
    }

    private fun tokenizarFrase(frase: String): List<String> {
        val stopwords = setOf(
            "o", "a", "os", "as", "um", "uma", "uns", "umas",
            "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
            "ao", "aos", "à", "às", "por", "pelo", "pela", "pelos", "pelas",
            "com", "sem", "sob", "sobre", "entre", "até", "para",
            "que", "se", "e", "é"
        )

        return frase
            .lowercase(Locale.ROOT)
            .replace(Regex("[.,!?;:\"'()\\[\\]{}]"), " ")
            .split(Regex("\\s+"))
            .map { it.trim() }
            .filter { it.length >= 2 && it !in stopwords }
    }

    private fun gerarCandidatosBusca(termoOriginal: String): List<String> {
        val base = termoOriginal.trim()
        val semAcento = semAcento(base)
        val comEspacos = base.replace(Regex("[-_]+"), " ").replace(Regex("\\s+"), " ").trim()
        val comEspacosSemAcento = semAcento(comEspacos)
        val canonico = SynonymResolver.canonicalize(base)
        val canonicoSemAcento = semAcento(canonico)

        return listOf(
            base,
            semAcento,
            comEspacos,
            comEspacosSemAcento,
            canonico,
            canonicoSemAcento,
        ).filter { it.isNotBlank() }.distinct()
    }

    private fun gerarVariantesNome(videoNome: String): List<String> {
        val baseSemExt = if (videoNome.lowercase(Locale.ROOT).endsWith(".mp4")) {
            videoNome.dropLast(4)
        } else {
            videoNome
        }

        fun variantes(base: String): List<String> {
            val semAcento = semAcento(base)
            return listOf(
                base,
                semAcento,
                base.replace(' ', '_'),
                semAcento.replace(' ', '_'),
                base.replace(' ', '-'),
                semAcento.replace(' ', '-'),
                base.replace('-', '_'),
                semAcento.replace('-', '_'),
            ).distinct()
        }

        fun stripNumeroSm(base: String): String? {
            val m = Regex("^(.+?)(\\d+)(Sm_Prog001)$", RegexOption.IGNORE_CASE).find(base)
            return m?.let { it.groupValues[1] + it.groupValues[3] }
        }

        fun addNumerosSm(base: String): List<String> {
            val m = Regex("^(.+?)(Sm_Prog001)$", RegexOption.IGNORE_CASE).find(base)
            val jaTemNumero = Regex("^.+\\d+Sm_Prog001$", RegexOption.IGNORE_CASE).matches(base)
            return if (m != null && !jaTemNumero) {
                (1..4).map { m.groupValues[1] + it + m.groupValues[2] }
            } else {
                emptyList()
            }
        }

        val allBases = mutableListOf<String>()
        variantes(baseSemExt).forEach { b ->
            allBases += variantes(b)
            stripNumeroSm(b)?.let { allBases += variantes(it) }
            allBases += addNumerosSm(b)
        }

        val seen = mutableSetOf<String>()
        return allBases
            .map { "$it.mp4" }
            .filter { seen.add(it) }
    }

    private fun semAcento(s: String): String {
        val nfd = Normalizer.normalize(s, Normalizer.Form.NFD)
        return nfd.replace(Regex("\\p{Mn}+"), "")
    }

    private fun sanitize(s: String): String =
        semAcento(s.trim().uppercase(Locale.ROOT))
            .replace(Regex("[\\\\/*?\"<>|]"), "_")
            .replace(Regex("\\s+"), "_")
            .ifBlank { "SEM_NOME" }

    private sealed interface VideoLookup {
        data class Found(
            val file: File,
            val estrategia: String,
        ) : VideoLookup

        data class Missing(
            val motivo: String,
            val caminhoEsperado: String?,
        ) : VideoLookup
    }
}