package br.edu.app.librasbridge.avatar

import android.content.Context
import br.edu.app.librasbridge.util.SynonymResolver
import org.json.JSONObject

object SignDictionary {
    private val fallbackMap = linkedMapOf(
        "oi" to "asset:///libras/oi.mp4",
        "bom dia" to "asset:///libras/bom_dia.mp4",
        "boa tarde" to "asset:///libras/boa_tarde.mp4",
        "boa noite" to "asset:///libras/boa_noite.mp4",
        "tudo bem" to "asset:///libras/tudo_bem.mp4",
        "obrigado" to "asset:///libras/obrigado.mp4",
        "ajuda" to "asset:///libras/ajuda.mp4",
        "água" to "asset:///libras/agua.mp4",
        "comer" to "asset:///libras/comer.mp4"
    )

    private val dynamicMap = linkedMapOf<String, String>()
    private var initialized = false

    fun initialize(context: Context) {
        if (initialized) return
        runCatching {
            context.assets.open("dict/sign_dictionary.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                for (key in json.keys()) {
                    dynamicMap[SynonymResolver.canonicalize(key)] = json.getString(key)
                }
            }
        }
        initialized = true
    }

    fun resolveSingle(text: String): String? = resolvePlaylist(text).firstOrNull()

    fun resolvePlaylist(text: String): List<String> {
        val normalized = SynonymResolver.normalize(text)
        if (normalized.isBlank()) return emptyList()

        resolveExact(normalized)?.let { return listOf(it) }

        val words = normalized.split(" ").filter { it.isNotBlank() }
        if (words.isEmpty()) return emptyList()

        val result = mutableListOf<String>()
        var index = 0
        while (index < words.size) {
            var matched: String? = null
            var matchedSize = 0
            for (size in words.size - index downTo 1) {
                val phrase = words.subList(index, index + size).joinToString(" ")
                val uri = resolveExact(phrase)
                if (uri != null) {
                    matched = uri
                    matchedSize = size
                    break
                }
            }
            if (matched != null) {
                result.add(matched)
                index += matchedSize
            } else {
                index += 1
            }
        }
        return result.distinct()
    }

    fun resolvePlaylistWithLabels(text: String): List<Pair<String, String>> {
        val normalized = SynonymResolver.normalize(text)
        if (normalized.isBlank()) return emptyList()
        resolveExact(normalized)?.let { return listOf(normalized to it) }

        val words = normalized.split(" ").filter { it.isNotBlank() }
        val result = mutableListOf<Pair<String, String>>()
        var index = 0
        while (index < words.size) {
            var matched: Pair<String, String>? = null
            var matchedSize = 0
            for (size in words.size - index downTo 1) {
                val phrase = words.subList(index, index + size).joinToString(" ")
                val uri = resolveExact(phrase)
                if (uri != null) {
                    matched = phrase to uri
                    matchedSize = size
                    break
                }
            }
            if (matched != null) {
                result.add(matched)
                index += matchedSize
            } else {
                index += 1
            }
        }
        return result
    }

    private fun resolveExact(normalized: String): String? {
        val canonical = SynonymResolver.canonicalize(normalized)
        return dynamicMap[canonical] ?: fallbackMap[canonical]
    }
}
