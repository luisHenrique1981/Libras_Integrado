package br.edu.app.librasbridge.camera

class GestureInterpreter {

    private val buffer = ArrayDeque<String>()
    private val maxBufferSize = 10

    fun stabilize(raw: String): String {
        if (raw.isBlank()) return ""
        if (buffer.size >= maxBufferSize) buffer.removeFirst()
        buffer.addLast(raw)
        return buffer.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key.orEmpty()
    }

    fun toToken(gestureName: String): String {
        return when (gestureName.trim().uppercase()) {
            "OI" -> "oi"
            "BOM_DIA" -> "bom dia"
            "OBRIGADO" -> "obrigado"
            "AJUDA" -> "ajuda"
            "AGUA" -> "água"
            "COMER" -> "comer"
            else -> ""
        }
    }
}
