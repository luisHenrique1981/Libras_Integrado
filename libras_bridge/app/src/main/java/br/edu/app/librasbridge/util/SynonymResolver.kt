package br.edu.app.librasbridge.util

import java.text.Normalizer
import java.util.Locale

object SynonymResolver {

    private val canonicalToAliases = mapOf(
        "oi" to setOf("oi", "olá", "ola", "alo", "alô", "e aí", "olaa"),
        "obrigado" to setOf("obrigado", "obrigada", "grato", "grata", "valeu", "thanks"),
        "ajuda" to setOf("ajuda", "auxilio", "auxílio", "socorro", "me ajuda", "preciso de ajuda"),
        "água" to setOf("agua", "água", "beber agua", "beber água", "quero agua", "quero água"),
        "tudo bem" to setOf("tudo bem", "como vai", "como você está", "como vc está", "tudo certo"),
    )

    private val aliasToCanonical: Map<String, String> = buildMap {
        for ((canonical, aliases) in canonicalToAliases) {
            val normalizedCanonical = normalize(canonical)
            put(normalizedCanonical, normalizedCanonical)
            for (alias in aliases) {
                put(normalize(alias), normalizedCanonical)
            }
        }
    }

    fun normalize(text: String): String {
        val lower = text.trim().lowercase(Locale.ROOT)
        val withoutPunctuation = lower.replace(Regex("""[^\p{L}\p{Nd}\s]"""), " ")
        val collapsed = withoutPunctuation.replace(Regex("""\s+"""), " ").trim()
        return Normalizer.normalize(collapsed, Normalizer.Form.NFD)
            .replace(Regex("""\p{InCombiningDiacriticalMarks}+"""), "")
    }

    fun canonicalize(text: String): String {
        val normalized = normalize(text)
        return aliasToCanonical[normalized] ?: normalized
    }
}