package br.edu.app.librasbridge.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class CameraFrameAnalyzer(
    private val gestureRecognizer: LibrasGestureRecognizer,
    private val onGestureRecognized: (raw: String, token: String) -> Unit
) : ImageAnalysis.Analyzer {

    private val interpreter = GestureInterpreter()

    override fun analyze(image: ImageProxy) {
        try {
            val raw = gestureRecognizer.recognize(image)
            val stable = interpreter.stabilize(raw)
            if (stable.isNotBlank()) {
                val token = interpreter.toToken(stable)
                if (token.isNotBlank()) {
                    onGestureRecognized(stable, token)
                }
            }
        } catch (_: Exception) {
            // keep analyzer resilient during demos
        } finally {
            image.close()
        }
    }
}
